#!/usr/bin/env python3
"""Scansiona i tensori numerici del modello distribuito C:/D: per NaN/Inf."""
from pathlib import Path

import numpy as np
from safetensors import safe_open

files = [Path("C:/picchio/model-00012.safetensors")]
files += sorted(Path("D:/gptoss_i4").glob("model-*.safetensors"))

bad_scales = []
bad_dense = []
scale_values = 0
max_scale = 0.0
for path in files:
    with safe_open(str(path), framework="np") as handle:
        for name in handle.keys():
            if name.endswith(".qs"):
                array = handle.get_tensor(name)
                scale_values += array.size
                nonfinite = int((~np.isfinite(array)).sum())
                if array.size:
                    max_scale = max(max_scale, float(np.nanmax(np.abs(array))))
                if nonfinite:
                    bad_scales.append((path.name, name, nonfinite, array.size))
            elif (
                name.endswith("bias")
                or name.endswith("sinks")
                or name.endswith("norm.weight")
                or name.endswith("router.weight")
            ):
                array = handle.get_tensor(name)
                nonfinite = int((~np.isfinite(array)).sum())
                if nonfinite:
                    bad_dense.append((path.name, name, nonfinite, array.size))

print(f"scale_values={scale_values} max_abs_scale={max_scale} bad_scales={len(bad_scales)}")
for row in bad_scales[:30]:
    print("BAD_SCALE", row)
print(f"bad_dense={len(bad_dense)}")
for row in bad_dense[:30]:
    print("BAD_DENSE", row)
