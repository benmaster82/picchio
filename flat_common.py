"""flat_common.py — shared helpers for the .picchioflat S1 harness.

Model-agnostic: auto-detects the per-expert tensor naming so the same harness
works for Qwen3 (experts.{E}.gate_up_proj) and GPT-OSS (experts.gate_up_proj.{E}).
The streamed expert payload is the four weight tensors (gate_up + scales, down +
scales); aggregated/resident biases are not part of the streamed expert.
"""
import json
from pathlib import Path

from safetensors import safe_open

BS = 4096
_PROJS = ("gate_up_proj", "gate_up_proj.qs", "down_proj", "down_proj.qs")
_handles = {}


def load(model):
    """Return (cfg, n_layers, n_experts, name2shard, scheme). `scheme(L, E)` gives
    the ordered list of the four tensor names for that expert."""
    cfg = json.load(open(f"{model}/config.json"))
    nl = cfg["num_hidden_layers"]
    ne = cfg.get("num_experts", cfg.get("num_local_experts"))

    name2shard = {}
    for s in sorted(Path(model).glob("model-*.safetensors")):
        with safe_open(str(s), framework="numpy") as h:
            for k in h.keys():
                name2shard[k] = str(s)

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
    p = name2shard[name]
    if p not in _handles:
        _handles[p] = safe_open(p, framework="numpy")
    return _handles[p].get_tensor(name).tobytes()
