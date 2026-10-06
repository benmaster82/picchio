#!/usr/bin/env python3
"""convert_minimax.py — GPTQ-INT4 MiniMax-M2 checkpoint -> Picchio-native INT4 gs64.

Produces a model directory that picchio.c loads and runs directly (the engine
gates MiniMax-M2 on model_type="minimax"; see section 13 of README.md). The
forward-pass math was validated separately by minimax_forward_check.c against
the real upstream MiniMaxM2ForCausalLM.

Pipeline per weight matrix:
  GPTQ (qweight/qzeros/scales/g_idx, group_size=32, sym, desc_act=False)
    -> dequantize to F32                              [unpack order validated
                                                         against ModelCloud/
                                                         GPTQModel's torch.py;
                                                         zero-point offset read
                                                         from checkpoint_format,
                                                         see dequant_gptq_linear]
    -> requantize with convert.py's own quantize_int4  (group_size=64, Picchio's
                                                         native gs64 format)

Tensor routing (mirrors convert_shard's Qwen3 branch), renaming MiniMax's
block_sparse_moe.experts.* to the mlp.experts.* names the runtime looks up:
  - norms, router gate.weight, e_score_correction_bias, biases: F32 passthrough
  - self_attn.{q,k,v,o}_proj: dequantize GPTQ -> F32 (or INT8 with --dense-bits 8)
  - self_attn.{q,k}_norm: F32 passthrough
  - embed_tokens / lm_head (BF16, not GPTQ-quantized in this checkpoint): INT8
  - experts.E.{w1,w3}: fused interleaved gate_up, INT4 gs64
    (w1=gate, w3=up, matching MiniMaxM2MLP: act_fn(w1(x)) * w3(x))
  - experts.E.w2 (down): INT4 gs64

Usage:
  python convert_minimax.py --input D:\\models\\MiniMax-M2-GPTQ-INT4 --output D:\\models\\minimax_m2_i4 [--dense-bits 8] [--layers 0,1]
"""
import argparse
import gc
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import quantize_int4, quantize_int8  # reuse Picchio's own quantizers

GROUP_SIZE_GPTQ = 32
BITS = 4
MAXQ = (1 << BITS) - 1


def dequant_gptq_linear(qweight: np.ndarray, qzeros: np.ndarray,
                         scales: np.ndarray, g_idx: np.ndarray,
                         zero_offset: int = 1) -> np.ndarray:
    """GPTQModel classic INT4 layout -> F32 nn.Linear.weight [OUT, IN].

    qweight: [IN/8, OUT] int32, 8 input-channel nibbles packed per row (shift 4*i)
    qzeros:  [n_groups, OUT/8] int32, 8 output-channel nibbles packed per row
    scales:  [n_groups, OUT] (any float dtype)
    g_idx:   [IN] int, group index per input channel (== i//32 here, desc_act=False)
    zero_offset: added to every unpacked zero-point. 1 for a "gptq" (v1)
                 checkpoint, 0 for "gptq_v2" — see below.

    Unpack order validated against ModelCloud/GPTQModel's torch.py.

    THE ZERO OFFSET IS NOT OPTIONAL on a v1 checkpoint. GPTQModel serialises v1
    with the zero-points pre-decremented and adds them back at load time, in
    convert_gptq_v1_to_v2_format_module() (utils/model.py), for bits=4 +
    pack_dtype=int32 as:
        qzeros += 0b00010001000100010001000100010001   # +1 per packed nibble
    with the comment "v1 checkpoint format used to do `qzeros -= 1` before
    serialization, thus the additions here do not overflow". Unpacked, that is
    exactly `zero_raw + 1`, which is what we do here (no overflow concern once
    the nibbles are widened to int32).

    Getting this wrong is quiet and devastating: omitting the +1 biases EVERY
    weight by exactly +1*scale. Per-weight that looks harmless (~30% of the
    weight std), but a constant offset c across a matmul's 3072 inputs adds
    c*sum(x) to every output — and since x comes out of an RMSNorm with
    positive gains, sum(x) is large and positive, so every projection acquires
    a big positive bias, which RMSNorm never recenters and which therefore
    compounds across all 62 layers into total garbage. Do not "verify" this
    against GPTQModel's dequantize_weight() without checking the qzeros state:
    that method runs AFTER the v1->v2 conversion has already mutated qzeros in
    place, so feeding it raw v1 qzeros reproduces the same bug and agrees with
    it perfectly.
    """
    IN = g_idx.shape[0]
    OUT = scales.shape[1]
    scales = scales.astype(np.float32)
    g_idx = g_idx.astype(np.int64)

    shifts = (np.arange(8, dtype=np.uint32) * BITS)
    qw = qweight.astype(np.uint32)[:, None, :]                     # [IN/8, 1, OUT]
    weight_raw = ((qw >> shifts[None, :, None]) & MAXQ).reshape(IN, OUT).astype(np.int32)

    qz = qzeros.astype(np.uint32)[:, :, None]                      # [n_groups, OUT/8, 1]
    zero_raw = ((qz >> shifts[None, None, :]) & MAXQ).reshape(qzeros.shape[0], OUT).astype(np.int32)
    zero_raw = zero_raw + zero_offset

    scale_per_row = scales[g_idx]        # [IN, OUT]
    zero_per_row = zero_raw[g_idx]       # [IN, OUT]
    dequant = (weight_raw - zero_per_row).astype(np.float32) * scale_per_row  # [IN, OUT]
    return dequant.T  # -> [OUT, IN], the nn.Linear.weight convention


#   config.json         picchio.c reads every dimension from it.
#   tokenizer.*, vocab, merges, chat_template, special/added tokens
#                       AutoTokenizer in chat_minimax.py needs the full set;
#                       without tokenizer_config.json and chat_template.jinja it
#                       finds no chat template and no EOS id.
#   generation_config.json  carried for completeness.
# Deliberately NOT copied: quantize_config.json (describes the *source* GPTQ
# packing, meaningless once requantized) and model.safetensors.index.json (maps
# the source shard layout, so it would actively mislead here).
_SIDECAR_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)


def copy_sidecar_files(inp: Path, out: Path) -> None:
    """Copy the metadata the runtime and the chat bridge need alongside the
    weights, so the output directory is self-contained and loadable as-is."""
    copied, missing = [], []
    for name in _SIDECAR_FILES:
        src = inp / name
        if src.is_file():
            shutil.copy2(src, out / name)
            copied.append(name)
        else:
            missing.append(name)
    print(f"\ncopied {len(copied)} metadata files: {', '.join(copied)}")
    if missing:
        print(f"  not present in the source (may be fine): {', '.join(missing)}")
    if not (out / "config.json").is_file():
        print("  WARNING: no config.json in the output — picchio.exe cannot load "
              "this directory until you place one there.")


def zero_offset_for_checkpoint(model_dir: Path) -> int:
    """1 for a v1 ("gptq") checkpoint, 0 for v2 ("gptq_v2"). See
    dequant_gptq_linear's docstring for why this matters so much."""
    for name in ("quantize_config.json", "config.json"):
        p = model_dir / name
        if not p.exists():
            continue
        cfg = json.loads(p.read_text())
        qc = cfg.get("quantization_config", cfg)
        fmt = qc.get("checkpoint_format")
        if fmt == "gptq_v2":
            return 0
        if fmt == "gptq":
            return 1
    raise SystemExit(
        f"cannot determine GPTQ checkpoint_format in {model_dir}; refusing to "
        f"guess the zero-point offset (see dequant_gptq_linear's docstring)")


_EXPERT_RE = re.compile(
    r"^(model\.layers\.\d+\.block_sparse_moe\.experts\.\d+)\.(w1|w2|w3)$")


_GPTQ_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")


def convert_shard(shard_path: str, output_tensors: dict, pending: dict,
                   raw_pending: dict, stats: dict, dense_bits: int = 4,
                   zero_offset: int = 1):
    with safe_open(shard_path, framework="numpy") as f:
        keys = set(f.keys())

        # The 4 GPTQ sidecars (qweight/qzeros/scales/g_idx) of one Linear are
        # NOT guaranteed to land in the same shard (HF's sharder cuts by byte
        # budget, not by tensor-group), so buffer them across shards exactly
        # like interleave_qwen_experts buffers whole experts.
        for k in keys:
            for suf in _GPTQ_SUFFIXES:
                if k.endswith(suf):
                    base = k[: -len(suf)]
                    raw_pending.setdefault(base, {})[suf[1:]] = f.get_tensor(k).copy()
                    break

        ready_bases = [b for b, p in raw_pending.items()
                       if {"qweight", "qzeros", "scales", "g_idx"} <= set(p)]
        for base in ready_bases:
            comp = raw_pending.pop(base)
            dequant = dequant_gptq_linear(comp["qweight"], comp["qzeros"],
                                          comp["scales"], comp["g_idx"],
                                          zero_offset)  # [OUT, IN] F32
            stats["gptq_dequant"] += 1

            m = _EXPERT_RE.match(base)
            if m:
                expert_base, proj = m.group(1), m.group(2)
                pending.setdefault(expert_base, {})[proj] = dequant
                continue

            if base.endswith("self_attn.q_proj") or base.endswith("self_attn.k_proj") \
               or base.endswith("self_attn.v_proj") or base.endswith("self_attn.o_proj"):
                if dense_bits == 8:
                    q8, sc = quantize_int8(dequant)
                    output_tensors[base + ".weight"] = q8
                    output_tensors[base + ".weight.qs"] = sc
                    stats["dense_i8"] += q8.nbytes + sc.nbytes
                else:
                    output_tensors[base + ".weight"] = dequant
                    stats["dense_f32"] += dequant.nbytes
            else:
                # Shouldn't happen for this checkpoint's known layout, but keep
                # F32 as a safe fallback rather than silently dropping data.
                output_tensors[base + ".weight"] = dequant
                stats["other"] += dequant.nbytes

        # Everything else in the shard is a plain (non-GPTQ) tensor: norms,
        # router, correction bias, embed/lm_head (BF16), rotary inv_freq.
        plain_keys = [k for k in keys
                      if not k.endswith((".qweight", ".qzeros", ".scales", ".g_idx"))]

    import torch
    with safe_open(shard_path, framework="pt") as f:
        for key in plain_keys:
            if "rotary_emb.inv_freq" in key:
                continue  # Picchio derives RoPE from rope_theta at runtime
            t = f.get_tensor(key)
            t_np = t.float().numpy() if t.dtype in (torch.bfloat16, torch.float16) else t.numpy()
            t_np = t_np.astype(np.float32)

            if "embed_tokens" in key or key == "lm_head.weight":
                q8, sc = quantize_int8(t_np.reshape(-1, t_np.shape[-1]))
                output_tensors[key] = q8
                output_tensors[key + ".qs"] = sc
                stats["dense_i8"] += q8.nbytes + sc.nbytes
            else:
                # norms, block_sparse_moe.gate.weight, e_score_correction_bias
                output_tensors[key] = t_np
                stats["other"] += t_np.nbytes


def flush_pending_experts(output_tensors: dict, pending: dict, stats: dict):
    """An expert is complete once w1 (gate), w2 (down) and w3 (up) have all
    arrived (possibly from different shards, hence the pending buffer)."""
    for base in [b for b, p in pending.items() if {"w1", "w2", "w3"} <= set(p)]:
        parts = pending.pop(base)
        gate, up, down = parts["w1"], parts["w3"], parts["w2"]  # [I,D],[I,D],[D,I]
        moe_i, D = gate.shape
        fused = np.empty((2 * moe_i, D), dtype=np.float32)
        fused[0::2] = gate
        fused[1::2] = up
        gp, gs = quantize_int4(fused)
        # Picchio's internal expert-tensor convention is "mlp.experts.E.*"
        # regardless of the source architecture's own naming (the engine's
        # expert loader hardcodes this; "block_sparse_moe" is MiniMax's HF
        # name for the same tensors, so rename it here at conversion time).
        out_base = base.replace(".block_sparse_moe.experts.", ".mlp.experts.")
        output_tensors[out_base + ".gate_up_proj"] = gp
        output_tensors[out_base + ".gate_up_proj.qs"] = gs
        dp, ds = quantize_int4(down)
        output_tensors[out_base + ".down_proj"] = dp
        output_tensors[out_base + ".down_proj.qs"] = ds
        stats["expert_i4"] += gp.nbytes + gs.nbytes + dp.nbytes + ds.nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dense-bits", type=int, default=4, choices=(4, 8))
    ap.add_argument("--layers", default=None,
                     help="comma-separated layer indices to convert (default: all present)")
    ap.add_argument("--delete-source", action="store_true",
                     help="delete each source shard right after its tensors are read into "
                          "memory, to free disk space as conversion proceeds. DESTRUCTIVE: "
                          "the original GPTQ checkpoint cannot be recovered afterward.")
    args = ap.parse_args()

    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    shards = sorted(inp.glob("model-*.safetensors"))
    if not shards:
        print(f"no shards found in {inp}"); sys.exit(1)

    wanted_layers = None
    if args.layers:
        wanted_layers = {int(x) for x in args.layers.split(",")}

    zero_offset = zero_offset_for_checkpoint(inp)
    print(f"GPTQ checkpoint zero-point offset: +{zero_offset} "
          f"({'v1 `gptq`' if zero_offset else 'v2 `gptq_v2`'} format)")

    stats = {"gptq_dequant": 0, "dense_f32": 0, "dense_i8": 0, "expert_i4": 0, "other": 0}
    pending = {}
    raw_pending = {}

    for si, sp in enumerate(shards):
        if wanted_layers is not None:
            with safe_open(str(sp), framework="numpy") as f:
                shard_layers = {int(m.group(1)) for k in f.keys()
                                 if (m := re.search(r"model\.layers\.(\d+)\.", k))}
            if not (shard_layers & wanted_layers):
                continue

        output_tensors = {}
        print(f"[{si+1}/{len(shards)}] {sp.name}")
        convert_shard(str(sp), output_tensors, pending, raw_pending, stats,
                      args.dense_bits, zero_offset)
        flush_pending_experts(output_tensors, pending, stats)

        if wanted_layers is not None:
            output_tensors = {
                k: v for k, v in output_tensors.items()
                if (m := re.search(r"model\.layers\.(\d+)\.", k)) is None
                or int(m.group(1)) in wanted_layers
            }

        if output_tensors:
            need = sum(v.nbytes for v in output_tensors.values())
            free = shutil.disk_usage(str(out)).free
            margin = 3 * 1024**3  # 3 GB safety margin: don't risk a truncated safetensors write
            if free - need < margin:
                print(f"\nABORT: only {free/1e9:.1f} GB free at destination, need "
                      f"~{need/1e9:.1f} GB for {sp.name} (+{margin/1e9:.0f} GB margin). "
                      f"Stopping before writing a possibly-truncated file. "
                      f"{si}/{len(shards)} shards converted so far.")
                sys.exit(1)
            # Write to a temp file and atomically replace: a kill mid-write
            # (external AV/EDR, OS, whatever) then leaves either the old
            # complete file or nothing under the final name, never a
            # truncated one under the final name (this bit us once already —
            # shard 10 had 7.6MB of trailing garbage from a non-atomic write).
            out_path = out / sp.name
            tmp_path = out_path.with_suffix(".tmp")
            save_file(output_tensors, str(tmp_path))
            tmp_path.replace(out_path)
            print(f"  -> {sp.name}: {len(output_tensors)} tensors")

        if args.delete_source:
            # All tensors from this shard have already been copied into memory
            # (dequantized F32 arrays, or explicit .copy() for buffered GPTQ
            # sidecars) — the file itself is no longer needed even for entries
            # still waiting in pending/raw_pending for a sibling in a later shard.
            gc.collect()
            try:
                freed = sp.stat().st_size
                sp.unlink()
                free_now = shutil.disk_usage(str(out)).free
                print(f"  deleted source {sp.name} ({freed/1e9:.1f} GB freed); "
                      f"{free_now/1e9:.1f} GB free at destination")
            except FileNotFoundError:
                # Something else (AV/EDR quarantine, manual cleanup, a prior
                # crashed run) already removed it — output for this shard is
                # already written safely, so this isn't fatal, just note it.
                print(f"  source {sp.name} already gone (not deleted by us) — continuing")

    if pending:
        print(f"  WARNING: {len(pending)} experts never completed "
              f"(missing w1/w2/w3): {list(pending)[:3]} ...")
    if raw_pending:
        print(f"  WARNING: {len(raw_pending)} GPTQ linears never completed "
              f"(missing qweight/qzeros/scales/g_idx): {list(raw_pending)[:3]} ...")

    copy_sidecar_files(inp, out)

    print(f"\ndone. gptq linears dequantized: {stats['gptq_dequant']}, "
          f"dense f32: {stats['dense_f32']/1e6:.1f} MB, dense i8: {stats['dense_i8']/1e6:.1f} MB, "
          f"expert i4: {stats['expert_i4']/1e6:.1f} MB, other: {stats['other']/1e6:.1f} MB")
    print(f"\nnext: python export_vocab.py {out}/tokenizer.json {out}/picchio_vocab.bin")
    print(f"      python chat_minimax.py --model {out} --ctx 4096 --pin-gb 20 --async-moe --direct")


if __name__ == "__main__":
    main()
