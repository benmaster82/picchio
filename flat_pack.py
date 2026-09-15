#!/usr/bin/env python3
"""flat_pack.py — S1 prototype: repack converted experts into a .picchioflat store.

Model-agnostic (Qwen3 and GPT-OSS): reads Picchio's converted INT4 experts from
the safetensors container and writes them into a flat, block-aligned file: one
contiguous, 4 KiB-aligned payload per expert, plus a resident index. No
re-quantization, just a repack, so the bytes are identical (flat_bench.py checks).

Layout (little-endian): a v2 superblock, resident SHA-256-protected index, then
4 KiB-aligned payloads. Each 24-byte index entry is offset, exact length, padded
length, and a truncated SHA-256 payload hash.

Env: FLAT_MODEL, FLAT_OUT, FLAT_LAYERS (default: every layer).
"""
import os
import hashlib
import struct
import time
from pathlib import Path

import flat_common as fc

MODEL = os.environ.get("FLAT_MODEL", "D:/qwen3_30b_i4")
OUT = os.environ.get("FLAT_OUT", str(Path(MODEL) / "experts.picchioflat"))
BS = fc.BS

cfg, NL, NE, name2shard, scheme = fc.load(MODEL)
hidden, cfg_nl, cfg_ne, topk, moe_inter = fc.architecture(cfg)
assert (cfg_nl, cfg_ne) == (NL, NE)
PACK_LAYERS = min(int(os.environ.get("FLAT_LAYERS", str(NL))), NL)
if PACK_LAYERS < 1:
    raise SystemExit("FLAT_LAYERS must be at least 1")
bits = int(cfg.get("picchio_expert_bits", 4))
if bits not in (3, 4):
    raise SystemExit(f"unsupported picchio_expert_bits={bits}")
down_inter = moe_inter // 2
if bits == 3:
    gu_qbytes = moe_inter * ((hidden + 63) // 64) * 24
    down_qbytes = hidden * ((down_inter + 63) // 64) * 24
else:
    gu_qbytes = moe_inter * ((hidden + 1) // 2)
    down_qbytes = hidden * ((down_inter + 1) // 2)
expected_payload = (gu_qbytes + moe_inter * ((hidden + 63) // 64) * 4 +
                    down_qbytes + hidden * ((down_inter + 63) // 64) * 4)
print(f"model {MODEL}: {NL} layers x {NE} experts; packing {PACK_LAYERS} layers")

os.makedirs(Path(OUT).parent, exist_ok=True)
t0 = time.time()
index = []
tmp = OUT + ".tmp"
n_entries = PACK_LAYERS * NE
index_off = BS
index_len = n_entries * fc.INDEX_SIZE
data_off = (index_off + index_len + BS - 1) // BS * BS
with open(tmp, "wb") as f:
    f.write(b"\0" * data_off)               # superblock + reserved index
    for L in range(PACK_LAYERS):
        for E in range(NE):
            payload = b"".join(fc.tbytes(name2shard, nm) for nm in scheme(L, E))
            if len(payload) != expected_payload:
                raise ValueError(
                    f"layer {L} expert {E}: payload {len(payload)} bytes, "
                    f"expected {expected_payload} for INT{bits} gs64")
            off = f.tell()
            assert off % BS == 0, off
            f.write(payload)
            pad = (-len(payload)) % BS
            if pad:
                f.write(b"\0" * pad)
            index.append((off, len(payload), len(payload) + pad,
                          fc.hash64(payload)))
        print(f"  layer {L}: {NE} experts, {f.tell()/1e9:.2f} GB")

    raw_index = b"".join(struct.pack(fc.INDEX_FMT, *entry) for entry in index)
    f.seek(index_off)
    f.write(raw_index)
    superblock = struct.pack(
        fc.SUPER_FMT, fc.MAGIC, fc.VERSION, BS, hidden, NL, NE, topk,
        moe_inter, len(index), index_off, index_len, data_off,
        hashlib.sha256(raw_index).digest())
    f.seek(0)
    f.write(superblock)

os.replace(tmp, OUT)

payload_bytes = sum(entry[2] for entry in index)
avg = payload_bytes / len(index) / 1e6
print(f"done: {OUT}")
print(f"  {len(index)} experts, {avg:.2f} MB/expert, payload {payload_bytes/1e9:.2f} GB, "
      f"{time.time()-t0:.0f}s")
