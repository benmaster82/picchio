#!/usr/bin/env python3
from pathlib import Path
import numpy as np
from safetensors import safe_open

files = [Path("C:/picchio/model-00012.safetensors")]
files += sorted(Path("D:/gptoss_i4").glob("model-*.safetensors"))

bias_rows = []
for path in files:
    with safe_open(str(path), framework="np") as handle:
        for name in handle.keys():
            if name.endswith("gate_up_proj_bias") or name.endswith("down_proj_bias"):
                value = handle.get_tensor(name)
                bias_rows.append((
                    max(abs(float(np.min(value))), abs(float(np.max(value)))),
                    path.name, name, str(value.dtype), value.shape,
                    float(np.min(value)), float(np.max(value)),
                    float(np.sqrt(np.mean(value.astype(np.float64) ** 2))),
                ))

print("Bias expert più estremi:")
for row in sorted(bias_rows, reverse=True)[:20]:
    print(row)

print("\nLayer 0:")
for path in files:
    with safe_open(str(path), framework="np") as handle:
        for name in handle.keys():
            if not name.startswith("model.layers.0."):
                continue
            if not (name.endswith("bias") or name.endswith("sinks") or name.endswith(".qs")):
                continue
            value = handle.get_tensor(name)
            print(path.name, name, value.dtype, value.shape,
                  float(np.min(value)), float(np.max(value)))
