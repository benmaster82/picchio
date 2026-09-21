#!/usr/bin/env python3
"""Fast structural and sampled-payload verification for .picchioflat files.

Env: FLAT_OUT (required path), FLAT_VERIFY_SAMPLES (default: 16; 0 = all).
"""
import os
from pathlib import Path

import flat_common as fc


path = Path(os.environ.get("FLAT_OUT", "experts.picchioflat"))
samples = int(os.environ.get("FLAT_VERIFY_SAMPLES", "16"))
meta, index = fc.read_flat(path)
size = path.stat().st_size

if not index:
    raise SystemExit("invalid flat store: empty index")

cursor = meta["data_offset"]
for i, (offset, length, padded, _) in enumerate(index):
    if offset != cursor:
        raise SystemExit(f"entry {i}: non-contiguous offset {offset}, expected {cursor}")
    if offset % fc.BS or padded % fc.BS:
        raise SystemExit(f"entry {i}: unaligned offset or padded length")
    if length <= 0 or padded < length or padded - length >= fc.BS:
        raise SystemExit(f"entry {i}: invalid lengths {length}/{padded}")
    cursor += padded
if cursor != size:
    raise SystemExit(f"file size mismatch: index ends at {cursor}, file is {size}")

n = len(index)
if samples == 0 or samples >= n:
    selected = list(range(n))
else:
    samples = max(samples, 2)
    selected = sorted({round(i * (n - 1) / (samples - 1)) for i in range(samples)})

with path.open("rb") as f:
    for i in selected:
        offset, length, _, expected_hash = index[i]
        f.seek(offset)
        payload = f.read(length)
        if len(payload) != length or fc.hash64(payload) != expected_hash:
            raise SystemExit(f"entry {i}: payload hash mismatch")

print(f"OK: {path}")
print(f"  {n} entries, {size / (1024**3):.2f} GiB")
print(f"  index SHA-256, {n} layouts, and {len(selected)} payload hashes verified")
