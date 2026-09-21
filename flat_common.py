"""flat_common.py — shared helpers for the .picchioflat S1 harness.

Model-agnostic: auto-detects the per-expert tensor naming so the same harness
works for Qwen3 (experts.{E}.gate_up_proj) and GPT-OSS (experts.gate_up_proj.{E}).
The streamed expert payload is the four weight tensors (gate_up + scales, down +
scales); aggregated/resident biases are not part of the streamed expert.
"""
import hashlib
import json
import os
import struct
import atexit
from pathlib import Path

BS = 4096
MAGIC = b"PCHIOFL1"
VERSION = 2
SUPER_FMT = "<8sIIIIIIIIQQQ32s"
INDEX_FMT = "<QIIQ"
INDEX_SIZE = struct.calcsize(INDEX_FMT)
_PROJS = ("gate_up_proj", "gate_up_proj.qs", "down_proj", "down_proj.qs")
_handles = {}
_tensor_locs = {}


def _close_handles():
    for handle in _handles.values():
        handle.close()
    _handles.clear()


atexit.register(_close_handles)


def _scan_safetensors(path):
    """Yield (name, absolute_offset, byte_length) without mmap'ing the shard."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"short SafeTensors header: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len > 64 * 1024 * 1024 or 8 + header_len > size:
            raise ValueError(f"invalid SafeTensors header length: {path}")
        raw_header = f.read(header_len)
    if len(raw_header) != header_len:
        raise ValueError(f"short SafeTensors header: {path}")
    header = json.loads(raw_header)
    data_base = 8 + header_len
    for name, info in header.items():
        if name == "__metadata__":
            continue
        start, end = info["data_offsets"]
        if start < 0 or end < start or data_base + end > size:
            raise ValueError(f"invalid tensor offsets for {name}: {path}")
        yield name, data_base + start, end - start


def load(model):
    """Return (cfg, n_layers, n_experts, name2shard, scheme). `scheme(L, E)` gives
    the ordered list of the four tensor names for that expert."""
    cfg = json.load(open(f"{model}/config.json"))
    nl = cfg["num_hidden_layers"]
    ne = cfg.get("num_experts", cfg.get("num_local_experts"))

    _close_handles()
    _tensor_locs.clear()
    name2shard = {}
    for s in sorted(Path(model).glob("model-*.safetensors")):
        path = str(s)
        for name, offset, length in _scan_safetensors(path):
            name2shard[name] = path
            _tensor_locs[name] = (path, offset, length)

    def qwen(L, E):
        return [f"model.layers.{L}.mlp.experts.{E}.{p}" for p in _PROJS]

    def gptoss(L, E):
        b = f"model.layers.{L}.mlp.experts"
        return [f"{b}.gate_up_proj.{E}", f"{b}.gate_up_proj.{E}.qs",
                f"{b}.down_proj.{E}", f"{b}.down_proj.{E}.qs"]

    scheme = qwen if qwen(0, 0)[0] in name2shard else gptoss
    missing = [n for n in scheme(0, 0) if n not in name2shard]
    if missing:
        raise SystemExit(f"unrecognized expert layout, missing: {missing}")
    return cfg, nl, ne, name2shard, scheme


def tbytes(name2shard, name):
    p, offset, length = _tensor_locs[name]
    if p not in _handles:
        _handles[p] = open(p, "rb", buffering=0)
    handle = _handles[p]
    handle.seek(offset)
    payload = handle.read(length)
    if len(payload) != length:
        raise OSError(f"short tensor read for {name}: {len(payload)} of {length} bytes")
    return payload


def architecture(cfg):
    """Return the dimensions echoed by the v2 superblock."""
    hidden = cfg["hidden_size"]
    layers = cfg["num_hidden_layers"]
    experts = cfg.get("num_experts", cfg.get("num_local_experts"))
    topk = cfg.get("num_experts_per_tok", cfg.get("experts_per_token", 4))
    intermediate = cfg.get("moe_intermediate_size", cfg.get("intermediate_size"))
    return hidden, layers, experts, topk, intermediate * 2


def hash64(payload):
    """Little-endian u64 made from the first eight SHA-256 digest bytes."""
    return struct.unpack("<Q", hashlib.sha256(payload).digest()[:8])[0]


def read_flat(path):
    """Return (metadata, index) after validating the resident index hash."""
    with open(path, "rb") as f:
        sb = f.read(BS)
        if len(sb) != BS:
            raise ValueError("short .picchioflat superblock")
        fields = struct.unpack_from(SUPER_FMT, sb)
        (magic, version, bs, hidden, layers, experts, topk, moe_inter, count,
         index_offset, index_len, data_offset, index_sha) = fields
        if magic != MAGIC or version != VERSION or bs != BS:
            raise ValueError("unsupported .picchioflat format")
        if index_len != count * INDEX_SIZE:
            raise ValueError("invalid .picchioflat index length")
        f.seek(index_offset)
        raw = f.read(index_len)
    if len(raw) != index_len or hashlib.sha256(raw).digest() != index_sha:
        raise ValueError(".picchioflat index SHA-256 mismatch")
    index = [struct.unpack_from(INDEX_FMT, raw, i * INDEX_SIZE)
             for i in range(count)]
    meta = dict(hidden=hidden, layers=layers, experts=experts, topk=topk,
                moe_inter=moe_inter, count=count, index_offset=index_offset,
                index_len=index_len, data_offset=data_offset)
    return meta, index
