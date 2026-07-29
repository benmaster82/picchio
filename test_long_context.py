#!/usr/bin/env python3
"""Confronta Picchio e Transformers su contesto lungo.

Le validazioni esistenti coprono 32 token greedy e la posizione 130. Il testo
degenerato osservato nelle generazioni lunghe rende necessario verificare
posizioni molto oltre la finestra scorrevole di 128.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from validate_tiny import boundary_dumps, compare_common, load_model, run_c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="test_tiny_hf")
    parser.add_argument("--picchio-model", default="test_tiny_picchio")
    parser.add_argument("--exe", default="picchio.exe")
    parser.add_argument("--positions", type=int, default=300)
    args = parser.parse_args()

    model = load_model(args.model)
    exe = Path(args.exe).resolve()
    picchio_model = Path(args.picchio_model).resolve()
    tokens = [1 + (i % 97) for i in range(args.positions)]

    hf_dir = Path("oracle_long_hf")
    c_dir = Path("oracle_long_c")
    shutil.rmtree(hf_dir, ignore_errors=True)
    shutil.rmtree(c_dir, ignore_errors=True)

    boundary_dumps(model, tokens, hf_dir)
    run_c(exe, picchio_model, tokens, 0, c_dir, ctx=args.positions + 8)
    compare_common(hf_dir, c_dir)

    ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        logits = model(ids, use_cache=False).logits[0, -1]
    hf_top = [int(x) for x in torch.topk(logits, 5).indices]
    c_logits = np.load(c_dir / "logits.npy", allow_pickle=False).reshape(-1)
    c_top = list(np.argsort(-c_logits)[:5].astype(int))
    print(f"top-5 HF={hf_top}")
    print(f"top-5 C ={c_top}")
    assert hf_top[0] == c_top[0], (hf_top, c_top)
    print(f"OK: contesto {args.positions} posizioni coerente con Transformers")


if __name__ == "__main__":
    main()
