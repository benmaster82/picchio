#!/usr/bin/env python3
"""Confronta INT4 e INT8 su embedding e lm_head.

La configurazione ufficiale GPT-OSS esclude embed_tokens, lm_head, self_attn e router
dalla quantizzazione (`modules_to_not_convert`). Questo script misura quanto costa
quantizzarli e quanto si guadagna passando da INT4 gs64 a INT8 per riga.
"""
import glob
import sys

import numpy as np
import torch
from safetensors import safe_open

from convert import quantize_int4, quantize_int8

RAW = sys.argv[1] if len(sys.argv) > 1 else r"D:\gptoss20b_i4_raw"
TARGETS = ("lm_head.weight", "model.embed_tokens.weight")


def find(name):
    """Legge con il framework torch: numpy non gestisce bfloat16."""
    for path in sorted(glob.glob(RAW + r"\model-*.safetensors")):
        with safe_open(path, framework="pt") as f:
            if name in f.keys():
                return f.get_tensor(name)
    raise KeyError(name)


def to_f32(tensor):
    return tensor.float().numpy()


def dequant_int4(packed, scales, rows, cols, group=64):
    packed = packed.reshape(rows, cols // 2)
    nib = np.empty((rows, cols), dtype=np.int16)
    nib[:, 0::2] = packed & 0x0F
    nib[:, 1::2] = packed >> 4
    values = nib.astype(np.float32) - 8.0
    values *= np.repeat(scales.reshape(rows, -1), group, axis=1)[:, :cols]
    return values


def main():
    rng = np.random.default_rng(0)
    sample_rows = 16384  # sufficiente statisticamente, evita copie da GB
    for name in TARGETS:
        w = to_f32(find(name))[:sample_rows]
        rows, cols = w.shape
        norm = np.linalg.norm(w)

        packed, s4 = quantize_int4(w)
        d4 = dequant_int4(packed, s4.astype(np.float32), rows, cols)
        q8, s8 = quantize_int8(w)
        d8 = q8.astype(np.float32) * s8.reshape(-1, 1)

        e4 = np.linalg.norm(d4 - w) / norm
        e8 = np.linalg.norm(d8 - w) / norm
        print(f"\n{name}  shape={w.shape}")
        print(f"  errore relativo INT4 gs64 : {e4:.3%}")
        print(f"  errore relativo INT8 riga : {e8:.3%}")
        print(f"  miglioramento             : {e4 / e8:.1f}x")

        if "lm_head" in name:
            hidden = rng.standard_normal((64, cols), dtype=np.float32)
            hidden /= np.linalg.norm(hidden, axis=1, keepdims=True)
            hidden *= np.sqrt(cols)
            exact = hidden @ w.T
            for label, approx in (("INT4", hidden @ d4.T), ("INT8", hidden @ d8.T)):
                same = (exact.argmax(1) == approx.argmax(1)).mean()
                noise = np.abs(exact - approx).mean() / exact.std()
                print(f"  {label}: argmax identico {same:.1%}, "
                      f"rumore logit {noise:.2%} della deviazione standard")


if __name__ == "__main__":
    main()
