#!/usr/bin/env python3
"""verify_minimax.py — diff c_output.json (C port) against oracle.json (real
HF MiniMaxM2ForCausalLM forward pass) logit-for-logit.

Usage: python3 verify_minimax.py [model_dir]
"""
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "minimax_test_model")

oracle = json.load(open(model_dir / "oracle.json"))
c_out = json.load(open("c_output.json"))

assert oracle["tokens_in"] == c_out["tokens_in"], "token sequence mismatch"

worst = 0.0
all_ok = True
for op, cp in zip(oracle["positions"], c_out["positions"]):
    assert op["pos"] == cp["pos"]
    ol = op["logits"]
    cl = cp["logits"]
    assert len(ol) == len(cl), f"pos {op['pos']}: vocab size mismatch {len(ol)} vs {len(cl)}"
    diffs = [abs(a - b) for a, b in zip(ol, cl)]
    maxdiff = max(diffs)
    worst = max(worst, maxdiff)
    argmax_match = op["argmax"] == cp["argmax"]
    status = "OK" if maxdiff < 1e-3 and argmax_match else "MISMATCH"
    if status != "OK":
        all_ok = False
    print(f"  pos={op['pos']:2d}  argmax {op['argmax']:2d} vs {cp['argmax']:2d} "
          f"({'match' if argmax_match else 'DIFFER'})  max|Δlogit|={maxdiff:.6f}  {status}")

print()
print(f"worst max|Δlogit| across all positions/vocab: {worst:.6f}")
if all_ok:
    print("PASS — the C forward pass matches the real HF MiniMaxM2ForCausalLM within float tolerance.")
    sys.exit(0)
else:
    print("FAIL")
    sys.exit(1)
