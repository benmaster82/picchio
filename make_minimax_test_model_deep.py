#!/usr/bin/env python3
"""Deep variant of make_minimax_test_model.py (L=16 instead of 2), F32, no
quantization at all. Used to check whether the real-model RMS explosion seen
in picchio.c is a pure depth/architecture effect (would already show up here)
or specific to GPTQ/quantization at real scale (would NOT show up here)."""
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM
from make_minimax_test_model import _require_compatible_transformers

_require_compatible_transformers()   # same partial-RoPE hazard as the shallow fixture

out = Path("minimax_test_model_deep")
out.mkdir(parents=True, exist_ok=True)

D = 32
L = 16          # <-- deep, was 2
H = 4
KVH = 2
hd = 16
rotary = hd // 2
E = 4
topk = 2
I = 16
V = 32

config = MiniMaxM2Config(
    vocab_size=V, hidden_size=D, intermediate_size=I, mlp_intermediate_size=I,
    num_hidden_layers=L, num_attention_heads=H, num_key_value_heads=KVH,
    head_dim=hd, num_local_experts=E, num_experts_per_tok=topk,
    attn_type_list=[1] * L, rms_norm_eps=1e-6, max_position_embeddings=64,
    rope_theta=10000.0, rotary_dim=rotary, use_qk_norm=True,
    qk_norm_type="per_layer", use_routing_bias=True, scoring_func="sigmoid",
    use_grouped_topk=False, num_expert_group=None, topk_group=None,
    routed_scaling_factor=1.0, shared_intermediate_size=0, use_mtp=False,
    attn_window_size=None, sliding_window=None, initializer_range=0.02,
    tie_word_embeddings=False, eos_token_id=2, pad_token_id=0, use_cache=False,
)

torch.manual_seed(42)
model = MiniMaxM2ForCausalLM(config)
model.eval()
print(f"✓ model built (D={D}, L={L}, H={H}, KVH={KVH}, hd={hd}, rotary={rotary}, E={E}, top{topk}, V={V})")

tokens = [1, 5, 3, 7, 2, 9]
input_ids = torch.tensor([tokens], dtype=torch.long)
with torch.no_grad():
    out_fwd = model(input_ids=input_ids, use_cache=False, return_dict=True)
logits = out_fwd.logits[0]

oracle = {"tokens_in": tokens, "config": {
    "hidden_size": D, "num_hidden_layers": L, "num_attention_heads": H,
    "num_key_value_heads": KVH, "head_dim": hd, "rotary_dim": rotary,
    "num_local_experts": E, "num_experts_per_tok": topk,
    "intermediate_size": I, "vocab_size": V, "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0, "use_qk_norm": True, "scoring_func": "sigmoid",
    "use_routing_bias": True}, "positions": []}
for pos in range(len(tokens)):
    row = logits[pos]
    top5 = torch.topk(row, 5)
    oracle["positions"].append({
        "pos": pos, "tok_in": tokens[pos], "argmax": int(row.argmax()),
        "top5_ids": [int(i) for i in top5.indices],
        "top5_logits": [float(v) for v in top5.values],
        "logits": [float(v) for v in row],
    })
    print(f"  pos={pos} tok_in={tokens[pos]} -> argmax={oracle['positions'][-1]['argmax']} (logit={row.max():.4f})")

with open(out / "oracle.json", "w") as f:
    json.dump(oracle, f, indent=2)

state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(state_dict, str(out / "model.safetensors"))
total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
print(f"✓ model.safetensors ({len(state_dict)} tensors, {total_bytes/1024:.1f} KB)")

config_dict = json.loads(config.to_json_string())
config_dict["architectures"] = ["MiniMaxM2ForCausalLM"]
with open(out / "config.json", "w") as f:
    json.dump(config_dict, f, indent=2)
print("✓ config.json")
