#!/usr/bin/env python3
"""make_test_model.py — Genera un mini-modello sintetico in formato safetensors.

Crea una directory con config.json + model.safetensors contenenti pesi random
con le dimensioni corrette per testare il loader di Picchio end-to-end.

Il modello è PICCOLO (D=64, 2 layer, 4 expert) — pochi MB.
Non genera testo sensato, ma serve a validare che:
  1. Il config.json viene letto correttamente
  2. I safetensors vengono aperti e parsati
  3. I pesi vengono caricati nelle struct giuste
  4. Il forward pass gira senza crash

Uso:
  python3 make_test_model.py [output_dir]
  # Default: ./test_model/
"""

import json
import sys
import os
from pathlib import Path

import numpy as np

try:
    from safetensors.numpy import save_file
except ImportError:
    print("Installa safetensors: pip install safetensors numpy")
    sys.exit(1)


def make_test_model(output_dir: str = "test_model"):
    """Genera un mini-modello GPT-OSS-like per testing."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Configurazione mini ──
    D = 64           # hidden_size
    L = 2            # num_hidden_layers
    H = 4            # num_attention_heads
    KVH = 2          # num_key_value_heads
    hd = 16          # head_dim = D / H
    E = 4            # num_local_experts
    topk = 2         # num_experts_per_tok
    I = D * 2        # intermediate_size (gate_up fused = 2*I = 4*D)
    V = 32           # vocab_size (piccolo per test)

    config = {
        "architectures": ["GptOssForCausalLM"],
        "model_type": "gpt_oss",
        "hidden_size": D,
        "num_hidden_layers": L,
        "num_attention_heads": H,
        "num_key_value_heads": KVH,
        "head_dim": hd,
        "intermediate_size": I,
        "num_local_experts": E,
        "num_experts_per_tok": topk,
        "vocab_size": V,
        "max_position_embeddings": 64,
        "sliding_window": 16,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "attention_bias": True,
        "tie_word_embeddings": False,
        "layer_types": ["sliding_attention", "full_attention"],
        "eos_token_id": 2,
        "pad_token_id": 0,
        "routed_scaling_factor": 1.0,
    }

    with open(out / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ config.json ({D=}, {L=}, {H=}, {KVH=}, {E=}, {V=})")

    # ── Genera pesi random ──
    np.random.seed(42)
    tensors = {}

    # Embedding e LM head
    tensors["model.embed_tokens.weight"] = np.random.randn(V, D).astype(np.float32) * 0.02
    tensors["lm_head.weight"] = np.random.randn(V, D).astype(np.float32) * 0.02

    # Final norm
    tensors["model.norm.weight"] = np.ones(D, dtype=np.float32)

    moe_inter = I * 2  # gate_up fused

    for l in range(L):
        prefix = f"model.layers.{l}"

        # RMSNorm
        tensors[f"{prefix}.input_layernorm.weight"] = np.ones(D, dtype=np.float32)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = np.ones(D, dtype=np.float32)

        # Attention Q/K/V/O weights
        tensors[f"{prefix}.self_attn.q_proj.weight"] = np.random.randn(H * hd, D).astype(np.float32) * 0.02
        tensors[f"{prefix}.self_attn.k_proj.weight"] = np.random.randn(KVH * hd, D).astype(np.float32) * 0.02
        tensors[f"{prefix}.self_attn.v_proj.weight"] = np.random.randn(KVH * hd, D).astype(np.float32) * 0.02
        tensors[f"{prefix}.self_attn.o_proj.weight"] = np.random.randn(D, H * hd).astype(np.float32) * 0.02

        # Attention bias
        tensors[f"{prefix}.self_attn.q_proj.bias"] = np.zeros(H * hd, dtype=np.float32)
        tensors[f"{prefix}.self_attn.k_proj.bias"] = np.zeros(KVH * hd, dtype=np.float32)
        tensors[f"{prefix}.self_attn.v_proj.bias"] = np.zeros(KVH * hd, dtype=np.float32)
        tensors[f"{prefix}.self_attn.o_proj.bias"] = np.zeros(D, dtype=np.float32)

        # Router
        tensors[f"{prefix}.mlp.router.weight"] = np.random.randn(E, D).astype(np.float32) * 0.1
        tensors[f"{prefix}.mlp.router.bias"] = np.zeros(E, dtype=np.float32)

        # Expert weights (gate_up fused + down)
        for e in range(E):
            ep = f"{prefix}.mlp.experts.{e}"
            # gate_up_proj: [moe_inter, D] = [2*I, D]
            tensors[f"{ep}.gate_up_proj"] = np.random.randn(moe_inter, D).astype(np.float32) * 0.02
            tensors[f"{ep}.gate_up_proj_bias"] = np.zeros(moe_inter, dtype=np.float32)
            # down_proj: [D, I] — input dimension is moe_inter/2 = I
            tensors[f"{ep}.down_proj"] = np.random.randn(D, I).astype(np.float32) * 0.02
            tensors[f"{ep}.down_proj_bias"] = np.zeros(D, dtype=np.float32)

    # ── Salva in safetensors ──
    save_file(tensors, str(out / "model.safetensors"))

    total_params = sum(t.size for t in tensors.values())
    total_bytes = sum(t.nbytes for t in tensors.values())
    print(f"✓ model.safetensors ({total_params:,} params, {total_bytes/1024:.1f} KB)")
    print(f"✓ {len(tensors)} tensori")
    print(f"\nPer testare:")
    print(f"  ./picchio.exe {output_dir}")
    print(f"  oppure: MODEL={output_dir} ./picchio.exe")


if __name__ == "__main__":
    output = sys.argv[1] if len(sys.argv) > 1 else "test_model"
    make_test_model(output)
