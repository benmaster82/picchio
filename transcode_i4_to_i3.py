#!/usr/bin/env python3
"""transcode_i4_to_i3.py — requantize an already-converted Picchio model's
experts from INT4 gs64 to INT3 gs64, locally, WITHOUT re-downloading.

It reads a model produced by convert.py (INT4 experts + INT8 head + F32 attn),
dequantizes each INT4 expert weight to F32 and requantizes it to INT3 gs64;
everything else (embed/lm_head INT8, attention/router/norms F32, biases, scales
of non-experts) is copied unchanged. The output config.json is marked
`picchio_expert_bits: 3`.

Quality note: INT4→INT3 compounds a little error vs converting from the original
model, but avoids the multi-GB re-download.

Usage:
  python transcode_i4_to_i3.py --input C:/models/gptoss20b_i8h --output C:/models/gptoss20b_i3
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


# ── INT3 gs64 packer — MUST match matmul_i3_gs in quant.h (and convert.py) ──
def quantize_int3(tensor: np.ndarray, group_size: int = 64):
    assert group_size == 64
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    O, I = tensor.shape
    ng = (I + 63) // 64
    scales = np.empty((O, ng), dtype=np.float32)
    packed = np.zeros((O, ng * 24), dtype=np.uint8)
    for g in range(ng):
        g0, g1 = g * 64, min(g * 64 + 64, I)
        n = g1 - g0
        group = tensor[:, g0:g1]
        amax = np.max(np.abs(group), axis=1)
        sc = np.where(amax > 1e-8, amax / 3.5, 1e-8).astype(np.float32)
        scales[:, g] = sc
        codes = np.clip(np.round(group / sc[:, None]), -4, 3).astype(np.int16) + 4
        if n < 64:
            codes = np.concatenate([codes, np.zeros((O, 64 - n), dtype=np.int16)], axis=1)
        low2 = (codes & 3).astype(np.uint8)
        high1 = ((codes >> 2) & 1).astype(np.uint8)
        base = g * 24
        lp = low2.reshape(O, 16, 4)
        packed[:, base:base + 16] = (lp[:, :, 0] | (lp[:, :, 1] << 2)
                                     | (lp[:, :, 2] << 4) | (lp[:, :, 3] << 6)).astype(np.uint8)
        hp = high1.reshape(O, 8, 8)
        hb = np.zeros((O, 8), dtype=np.uint8)
        for i in range(8):
            hb |= (hp[:, :, i] << i).astype(np.uint8)
        packed[:, base + 16:base + 24] = hb
    return packed.reshape(-1), scales.reshape(-1)


def dequant_int4_gs(packed_flat, scales_flat, O, I, gs=64):
    """INT4 gs64 (value=(nibble-8)*scale, 2/byte, lo=even) → F32 [O, I]."""
    rb = (I + 1) // 2
    packed = packed_flat.reshape(O, rb)
    ng = (I + gs - 1) // gs
    scales = scales_flat.reshape(O, ng)
    q = np.empty((O, I), dtype=np.int16)
    ev = I - (I % 2)
    lo = packed[:, :ev // 2]
    q[:, 0:ev:2] = (lo & 0xF).astype(np.int16) - 8
    q[:, 1:ev:2] = ((lo >> 4) & 0xF).astype(np.int16) - 8
    if I % 2 == 1:
        q[:, -1] = (packed[:, -1] & 0xF).astype(np.int16) - 8
    out = np.empty((O, I), dtype=np.float32)
    for g in range(ng):
        g0, g1 = g * gs, min(g * gs + gs, I)
        out[:, g0:g1] = q[:, g0:g1].astype(np.float32) * scales[:, g:g + 1]
    return out


def main():
    ap = argparse.ArgumentParser(description="Requantize experts INT4 → INT3 (no re-download)")
    ap.add_argument("--input", required=True, help="INT4 model dir (from convert.py)")
    ap.add_argument("--output", required=True, help="output INT3 model dir")
    args = ap.parse_args()

    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cfg = json.load(open(inp / "config.json"))
    D = int(cfg["hidden_size"])
    inter = int(cfg.get("intermediate_size", cfg.get("moe_intermediate_size", D)))
    print(f"  hidden={D} intermediate={inter}")

    # Side files: config (marked), tokenizer, vocab.
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "generation_config.json", "picchio_vocab.bin"):
        if (inp / fn).exists():
            shutil.copy2(inp / fn, out / fn)
    ocfg = json.load(open(out / "config.json"))
    ocfg["picchio_expert_bits"] = 3
    json.dump(ocfg, open(out / "config.json", "w"), indent=2)
    print("  config marked picchio_expert_bits=3")

    shards = sorted(inp.glob("model*.safetensors"))
    saved_in = saved_out = 0
    for sp in shards:
        tensors, n_exp = {}, 0
        with safe_open(str(sp), framework="numpy") as f:
            keys = list(f.keys())
            kset = set(keys)
            for k in keys:
                if k.endswith(".qs") and k[:-3] in kset:
                    t0 = f.get_slice(k[:-3])
                    # an expert weight's .qs is regenerated with its weight; skip here
                    if str(t0.get_dtype()) in ("U8", "uint8"):
                        continue
                t = f.get_tensor(k)
                is_expert = (t.dtype == np.uint8) and ((k + ".qs") in kset)
                if is_expert:
                    I = D if "gate_up" in k else inter
                    rb = (I + 1) // 2
                    O = t.size // rb
                    qs = f.get_tensor(k + ".qs")
                    W = dequant_int4_gs(t, qs, O, I)
                    packed, scales = quantize_int3(W)
                    tensors[k] = packed
                    tensors[k + ".qs"] = scales
                    n_exp += 1
                else:
                    tensors[k] = t
        save_file(tensors, str(out / sp.name))
        saved_in += sp.stat().st_size
        saved_out += (out / sp.name).stat().st_size
        print(f"  {sp.name}: {n_exp} experts → INT3  ({sp.stat().st_size/1e9:.2f} → "
              f"{(out/sp.name).stat().st_size/1e9:.2f} GB)")

    print(f"\n  done: {saved_in/1e9:.1f} GB (int4) → {saved_out/1e9:.1f} GB (int3), "
          f"−{100*(1-saved_out/saved_in):.0f}%")
    print(f"  run:  python chat.py --model {out}")


if __name__ == "__main__":
    main()
