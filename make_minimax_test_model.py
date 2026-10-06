#!/usr/bin/env python3
"""make_minimax_test_model.py — Generate a synthetic mini MiniMax-M2 model + oracle.

Porting scaffold for a future MiniMax-M2 backend in Picchio. Unlike
make_test_model.py (GPT-OSS, hand-built random weights + hand-written numpy
oracle), this one instantiates the REAL upstream MiniMaxM2ForCausalLM class
(vendored under reference/minimax_m2/, from ModelCloud/MiniMax-M2-GPTQMODEL-W4A16,
Apache-2.0) at tiny dimensions, runs its actual forward pass in PyTorch, and
saves both the random weights and the exact reference logits it produced. That
makes the oracle authoritative: it is not a reimplementation that could share a
bug with a hand-rolled reference, it IS the reference.

The tiny config keeps two real architectural quirks of the full checkpoint:
  - head_dim * num_attention_heads > hidden_size (attention widens then
    o_proj narrows back: 2x on the real model, kept here)
  - partial RoPE: rotary_dim = head_dim / 2 (only half of each head rotates)

Requires torch + transformers (the real ones; no re-implementation). Not needed
to run Picchio itself — this exists to validate and regression-test the
MiniMax-M2 backend, not to serve it.

Usage:
  python3 make_minimax_test_model.py [output_dir]
  # Default: ./minimax_test_model/
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reference.minimax_m2 import MiniMaxM2Config, MiniMaxM2ForCausalLM


def _require_compatible_transformers():
    """Refuse to run on transformers >= 5.0, where this oracle comes out WRONG.

    The vendored MiniMax code builds its RoPE with LlamaRotaryEmbedding. In
    transformers 5.x that class's compute_default_rope_parameters() derives
    inv_freq from the full head_dim and ignores partial_rotary_factor entirely
    (modeling_rope_utils.py honors it, but the Llama class-local function that
    rope_type="default" dispatches to does not). MiniMax-M2 uses partial RoPE —
    rotary_dim 64 of head_dim 128 — so the frequencies would be spaced for a
    128-wide rotation and the resulting oracle would silently encode the same
    defect reported in huggingface/transformers#48241.

    That failure mode is worse than no oracle at all: the committed fixture is
    the reference Picchio's own forward pass is checked against, so a quietly
    wrong one would make a correct implementation look broken. Hence a hard
    stop instead of a shim.
    """
    import transformers
    major = int(transformers.__version__.split(".")[0])
    if major >= 5:
        sys.exit(
            f"transformers {transformers.__version__} would produce an INCORRECT "
            f"oracle for MiniMax-M2's partial RoPE (see this function's docstring "
            f"and huggingface/transformers#48241).\n"
            f"The committed minimax_test_model/ fixture was generated on "
            f"transformers 4.57 and is the reference to use.\n"
            f"To regenerate it, pin the library: pip install 'transformers<5.0'")


def make_minimax_test_model(output_dir: str = "minimax_test_model"):
    _require_compatible_transformers()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Mini configuration (proportions borrowed from the real 230B checkpoint) ──
    D = 32          # hidden_size
    L = 2           # num_hidden_layers
    H = 4           # num_attention_heads
    KVH = 2         # num_key_value_heads
    hd = 16         # head_dim (note: H*hd = 64 = 2*D, same 2x widen as the real model)
    rotary = hd // 2  # rotary_dim: partial RoPE, only half of head_dim rotates
    E = 4           # num_local_experts
    topk = 2        # num_experts_per_tok
    I = 16          # intermediate_size (per-expert MLP inner dim)
    V = 32          # vocab_size

    config = MiniMaxM2Config(
        vocab_size=V,
        hidden_size=D,
        intermediate_size=I,
        mlp_intermediate_size=I,   # unused by this modeling code path; kept present for shape parity
        num_hidden_layers=L,
        num_attention_heads=H,
        num_key_value_heads=KVH,
        head_dim=hd,
        num_local_experts=E,
        num_experts_per_tok=topk,
        attn_type_list=[1] * L,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        rope_theta=10000.0,
        rotary_dim=rotary,
        use_qk_norm=True,
        qk_norm_type="per_layer",
        use_routing_bias=True,
        scoring_func="sigmoid",
        use_grouped_topk=False,     # real model sets this True but num_expert_group=None
        num_expert_group=None,      # degenerates to the same ungrouped top-k either way
        topk_group=None,
        routed_scaling_factor=1.0,
        shared_intermediate_size=0,  # no shared expert, matches the real checkpoint
        use_mtp=False,               # MTP weights are not read by this modeling code anyway
        attn_window_size=None,
        sliding_window=None,         # real checkpoint is full-attention on every layer
        initializer_range=0.02,
        tie_word_embeddings=False,
        eos_token_id=2,
        pad_token_id=0,
        use_cache=False,
    )

    torch.manual_seed(42)
    model = MiniMaxM2ForCausalLM(config)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ model built ({D=}, {L=}, {H=}, {KVH=}, {hd=}, {rotary=}, {E=}, top{topk}, {V=})")
    print(f"  {total_params:,} parameters")

    # ── Reference forward pass (teacher-forced, one shot — causal mask makes this
    #    equivalent to incremental decode for a standard quadratic-attention model) ──
    tokens = [1, 5, 3, 7, 2, 9]
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        out_fwd = model(input_ids=input_ids, use_cache=False, return_dict=True)
    logits = out_fwd.logits[0]  # [seq_len, V]

    oracle = {
        "tokens_in": tokens,
        "config": {
            "hidden_size": D, "num_hidden_layers": L, "num_attention_heads": H,
            "num_key_value_heads": KVH, "head_dim": hd, "rotary_dim": rotary,
            "num_local_experts": E, "num_experts_per_tok": topk,
            "intermediate_size": I, "vocab_size": V,
            "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "use_qk_norm": True, "scoring_func": "sigmoid", "use_routing_bias": True,
        },
        "positions": [],
    }
    for pos in range(len(tokens)):
        row = logits[pos]
        top5 = torch.topk(row, 5)
        oracle["positions"].append({
            "pos": pos,
            "tok_in": tokens[pos],
            "argmax": int(row.argmax()),
            "top5_ids": [int(i) for i in top5.indices],
            "top5_logits": [float(v) for v in top5.values],
            "logits": [float(v) for v in row],  # full vector — V is tiny (32), cheap to keep
        })
        print(f"  pos={pos} tok_in={tokens[pos]} → argmax={oracle['positions'][-1]['argmax']} "
              f"(logit={row.max():.4f})")

    with open(out / "oracle.json", "w") as f:
        json.dump(oracle, f, indent=2)
    print(f"✓ oracle.json ({len(tokens)} positions, full {V}-wide logits each)")

    # ── Save weights (native MiniMax tensor names, F32) + config ──
    state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state_dict, str(out / "model.safetensors"))
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    print(f"✓ model.safetensors ({len(state_dict)} tensors, {total_bytes/1024:.1f} KB)")

    config_dict = json.loads(config.to_json_string())
    config_dict["architectures"] = ["MiniMaxM2ForCausalLM"]
    with open(out / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"✓ config.json")

    print(f"\nThis is a VALIDATION FIXTURE, in HF tensor naming. oracle.json is")
    print(f"  the reference the C forward pass must match. To use it:")
    print(f"    python fuse_minimax_test_model.py {output_dir}   # -> Picchio naming")
    print(f"    .\\build_minimax_check.bat && .\\minimax_forward_check.exe")
    print(f"    python verify_minimax.py                      # diffs the logits")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "minimax_test_model"
    make_minimax_test_model(output)
