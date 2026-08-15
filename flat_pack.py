#!/usr/bin/env python3
"""flat_pack.py — S1 prototype: repack converted experts into a .picchioflat store.

Model-agnostic (Qwen3 and GPT-OSS): reads Picchio's converted INT4 experts from
the safetensors container and writes them into a flat, block-aligned file: one
contiguous, 4 KiB-aligned payload per expert, plus a resident index. No
re-quantization, just a repack, so the bytes are identical (flat_bench.py checks).

Layout (little-endian):
  [superblock : one BS block]
     8s "PCHIOFL1", u32 version, u32 block_size,
     u32 n_layers, u32 n_experts, u32 n_entries,
     u64 index_offset, u64 index_len, u64 data_offset
  [payload : each expert BS-aligned]  gate_up | gate_up.qs | down | down.qs + pad
  [index : n_entries * 16 bytes]      u64 offset, u32 len, u32 padded_len
                                      (addressed by layer*n_experts + eid)

Env: FLAT_MODEL, FLAT_OUT, FLAT_LAYERS (layers to pack, default 3).
"""
import os
import struct
import time
from pathlib import Path

import flat_common as fc

MODEL = os.environ.get("FLAT_MODEL", "D:/qwen3_30b_i4")
OUT = os.environ.get("FLAT_OUT", "D:/qwen3_flat.picchioflat")
BS = fc.BS

cfg, NL, NE, name2shard, scheme = fc.load(MODEL)
PACK_LAYERS = min(int(os.environ.get("FLAT_LAYERS", "3")), NL)
print(f"model {MODEL}: {NL} layers x {NE} experts; packing {PACK_LAYERS} layers")

os.makedirs(Path(OUT).parent, exist_ok=True)
t0 = time.time()
index = []
with open(OUT, "wb") as f:
    f.write(b"\0" * BS)                     # superblock placeholder
    for L in range(PACK_LAYERS):
        for E in range(NE):
            payload = b"".join(fc.tbytes(name2shard, nm) for nm in scheme(L, E))
            off = f.tell()
            assert off % BS == 0, off
            f.write(payload)
            pad = (-len(payload)) % BS
            if pad:
                f.write(b"\0" * pad)
            index.append((off, len(payload), len(payload) + pad))
        print(f"  layer {L}: {NE} experts, {f.tell()/1e9:.2f} GB")

    index_off = f.tell()
    for off, ln, pd in index:
        f.write(struct.pack("<QII", off, ln, pd))
    index_len = f.tell() - index_off

    f.seek(0)
    f.write(struct.pack("<8sIIIIIQQQ", b"PCHIOFL1", 1, BS, NL, NE,
                        len(index), index_off, index_len, BS))

avg = index_off / len(index) / 1e6
print(f"done: {OUT}")
print(f"  {len(index)} experts, {avg:.2f} MB/expert, payload {index_off/1e9:.2f} GB, "
      f"{time.time()-t0:.0f}s")
