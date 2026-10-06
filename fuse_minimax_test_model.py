#!/usr/bin/env python3
"""fuse_minimax_test_model.py — rewrite minimax_test_model/ (real HF tensor
names, F32, unquantized) into Picchio's internal naming convention, so
picchio.c's real loader can load it directly (no GPTQ, no quantization —
just the fuse+rename convert_minimax.py does for the real checkpoint, minus
the GPTQ dequant step since this fixture is already plain F32).

This is a one-off test-fixture adapter, not part of the real conversion
pipeline. Output goes next to the input as <dir>_picchio/.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

src = Path(sys.argv[1] if len(sys.argv) > 1 else "minimax_test_model")
dst = Path(sys.argv[2] if len(sys.argv) > 2 else str(src) + "_picchio")
dst.mkdir(parents=True, exist_ok=True)

EXPERT_RE = re.compile(r"^(model\.layers\.\d+)\.block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$")

tensors = {}
pending = {}
with safe_open(str(src / "model.safetensors"), framework="numpy") as f:
    for k in f.keys():
        v = f.get_tensor(k)
        m = EXPERT_RE.match(k)
        if m:
            layer, eid, proj = m.group(1), m.group(2), m.group(3)
            pending.setdefault((layer, eid), {})[proj] = v
            continue
        tensors[k] = v

for (layer, eid), parts in pending.items():
    gate, up, down = parts["w1"], parts["w3"], parts["w2"]  # [I,D],[I,D],[D,I]
    moe_i, D = gate.shape
    fused = np.empty((2 * moe_i, D), dtype=np.float32)
    fused[0::2] = gate
    fused[1::2] = up
    base = f"{layer}.mlp.experts.{eid}"
    tensors[base + ".gate_up_proj"] = fused
    tensors[base + ".down_proj"] = down

save_file(tensors, str(dst / "model.safetensors"))
cfg = json.load(open(src / "config.json"))
json.dump(cfg, open(dst / "config.json", "w"), indent=2)
print(f"{len(tensors)} tensors -> {dst}")
