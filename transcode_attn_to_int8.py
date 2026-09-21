#!/usr/bin/env python3
"""Requantize attention weights F32 -> INT8, locally, WITHOUT re-downloading.

Reads a Picchio model (INT4/INT3 experts + F32 attention) and writes a copy in
which each self_attn q/k/v/o_proj.weight is INT8 (per-row scales, ".qs"); experts
and everything else are copied unchanged. The runtime loads INT8 attention via
matmul_q8 (fmt 1). Effect: ~4x fewer attention bytes -> t_attn roughly halved and
the resident dense part shrinks (freeing RAM for the expert cache). Near-lossless.

Resumable: an output shard already present (non-empty) is skipped.

  python transcode_attn_to_int8.py --input D:/gptoss_i3 --output D:/gptoss_i3_d8
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

from convert import quantize_int8  # reuse the exact INT8 quantizer

ATTN = ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight")


def main():
    ap = argparse.ArgumentParser(description="Requantize attention F32 -> INT8 (no re-download)")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    inp, out = Path(args.input), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "generation_config.json", "picchio_vocab.bin"):
        if (inp / fn).exists():
            shutil.copy2(inp / fn, out / fn)

    shards = sorted(inp.glob("model*.safetensors"))
    tin = tout = 0
    total_attn = 0
    for sp in shards:
        dst = out / sp.name
        if dst.exists() and dst.stat().st_size > 0:      # resumable
            print(f"  {sp.name}: already done, skip")
            tin += sp.stat().st_size
            tout += dst.stat().st_size
            continue
        tensors = {}
        nq = 0
        with safe_open(str(sp), framework="numpy") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                if any(k.endswith(a) for a in ATTN) and t.dtype == np.float32 and t.ndim == 2:
                    q8, scales = quantize_int8(t.reshape(-1, t.shape[-1]))
                    tensors[k] = q8
                    tensors[k + ".qs"] = scales
                    nq += 1
                else:
                    tensors[k] = t
        save_file(tensors, str(dst))
        tin += sp.stat().st_size
        tout += dst.stat().st_size
        total_attn += nq
        print(f"  {sp.name}: {nq} attn->INT8 "
              f"({sp.stat().st_size/1e9:.2f} -> {dst.stat().st_size/1e9:.2f} GB)")

    print(f"\n  done: {total_attn} attention matrices INT8. "
          f"{tin/1e9:.1f} -> {tout/1e9:.1f} GB (-{100*(1-tout/tin):.0f}%)")
    print(f"  run:  python chat_qwen.py --model {out} --direct --async-moe --pin-gb 8")


if __name__ == "__main__":
    main()
