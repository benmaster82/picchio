#!/usr/bin/env python3
"""test_forward.py — Valida il forward pass di Picchio in Python.

Carica il mini-modello di test e esegue un forward pass identico a quello
del codice C, poi verifica che ogni componente funziona correttamente.

Questo serve come:
  1. Oracle: genera output di riferimento per validare il C
  2. Test strutturale: verifica che l'architettura è corretta
  3. Validazione prima di avere il compilatore

Uso:
  python3 test_forward.py [test_model]
"""

import json
import sys
import numpy as np
from pathlib import Path

try:
    from safetensors.numpy import load_file
except ImportError:
    print("pip install safetensors numpy")
    sys.exit(1)


def rmsnorm(x, w, eps=1e-5):
    """RMSNorm: x * w / sqrt(mean(x^2) + eps)"""
    ms = np.mean(x * x)
    return x * w / np.sqrt(ms + eps)


def silu(x):
    """SiLU activation: x * sigmoid(x)"""
    return x / (1.0 + np.exp(-x))


def softmax(x):
    """Numerically stable softmax"""
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def rope(v, pos, head_dim, theta=10000.0):
    """Apply RoPE to a vector (non-interleaved)"""
    half = head_dim // 2
    out = v.copy()
    for j in range(half):
        freq = 1.0 / (theta ** (2.0 * j / head_dim))
        ang = pos * freq
        cs, sn = np.cos(ang), np.sin(ang)
        a, b = v[j], v[j + half]
        out[j] = a * cs - b * sn
        out[j + half] = a * sn + b * cs
    return out


def forward_pass(model_dir):
    """Esegue un forward pass completo sul mini-modello."""

    model_dir = Path(model_dir)

    # Carica config
    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    D = cfg["hidden_size"]
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    KVH = cfg["num_key_value_heads"]
    hd = cfg["head_dim"]
    E = cfg["num_local_experts"]
    topk = cfg["num_experts_per_tok"]
    I = cfg["intermediate_size"]
    V = cfg["vocab_size"]
    sw = cfg["sliding_window"]
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_theta"]
    layer_types = cfg["layer_types"]

    moe_inter = I * 2  # gate_up fused

    print(f"Config: D={D} L={L} H={H} KVH={KVH} hd={hd} E={E} top{topk} V={V}")
    print(f"        moe_inter={moe_inter} sw={sw} theta={theta}")

    # Carica pesi
    tensors = load_file(str(model_dir / "model.safetensors"))
    print(f"Tensori caricati: {len(tensors)}")

    # ── Test componenti ──
    print("\n── Test componenti ──")

    # Test RMSNorm
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    w = np.ones(4, dtype=np.float32)
    out = rmsnorm(x, w, eps)
    rms = np.sqrt(np.mean(x*x) + eps)
    assert abs(out[0] - 1.0/rms) < 1e-5, f"RMSNorm fallito: {out[0]} vs {1.0/rms}"
    print("  RMSNorm ✓")

    # Test softmax
    x = np.array([1.0, 2.0, 3.0])
    s = softmax(x)
    assert abs(np.sum(s) - 1.0) < 1e-6, "Softmax somma != 1"
    assert s[2] > s[1] > s[0], "Softmax ordine sbagliato"
    print("  Softmax ✓")

    # Test SiLU
    assert abs(silu(0.0)) < 1e-6, "SiLU(0) != 0"
    expected = 1.0 / (1.0 + np.exp(-1.0))
    assert abs(silu(1.0) - expected) < 1e-5, f"SiLU(1) = {silu(1.0)} vs {expected}"
    print("  SiLU ✓")

    # Test RoPE pos=0
    v = np.arange(1, hd+1, dtype=np.float32)
    v_rope = rope(v, 0, hd, theta)
    assert np.allclose(v, v_rope, atol=1e-5), "RoPE pos=0 ha modificato il vettore"
    print("  RoPE (pos=0) ✓")

    # ── Forward pass su 4 token ──
    print("\n── Forward pass: 4 token ──")

    tokens = [1, 5, 3, 7]

    embed = tensors["model.embed_tokens.weight"]   # [V, D]
    lm_head = tensors["lm_head.weight"]            # [V, D]
    final_norm_w = tensors["model.norm.weight"]    # [D]

    # KV cache: [L][max_pos][KVH * hd]
    kv_dim = KVH * hd
    max_pos = 64
    K_cache = np.zeros((L, max_pos, kv_dim), dtype=np.float32)
    V_cache = np.zeros((L, max_pos, kv_dim), dtype=np.float32)

    group = H // KVH

    for pos, tok in enumerate(tokens):
        # Embedding
        h = embed[tok].copy()

        # Layer loop
        for l in range(L):
            prefix = f"model.layers.{l}"
            lt = layer_types[l % len(layer_types)]

            # Pre-attention norm
            in_ln = tensors[f"{prefix}.input_layernorm.weight"]
            hn = rmsnorm(h, in_ln, eps)

            # Q/K/V projections
            wq = tensors[f"{prefix}.self_attn.q_proj.weight"]  # [H*hd, D]
            wk = tensors[f"{prefix}.self_attn.k_proj.weight"]  # [KVH*hd, D]
            wv = tensors[f"{prefix}.self_attn.v_proj.weight"]  # [KVH*hd, D]
            wo = tensors[f"{prefix}.self_attn.o_proj.weight"]  # [D, H*hd]

            bq = tensors[f"{prefix}.self_attn.q_proj.bias"]
            bk = tensors[f"{prefix}.self_attn.k_proj.bias"]
            bv = tensors[f"{prefix}.self_attn.v_proj.bias"]
            bo = tensors[f"{prefix}.self_attn.o_proj.bias"]

            q = wq @ hn + bq
            k = wk @ hn + bk
            v = wv @ hn + bv

            # RoPE
            for head in range(H):
                q[head*hd:(head+1)*hd] = rope(q[head*hd:(head+1)*hd], pos, hd, theta)
            for head in range(KVH):
                k[head*hd:(head+1)*hd] = rope(k[head*hd:(head+1)*hd], pos, hd, theta)

            # Store in KV cache
            K_cache[l, pos, :] = k
            V_cache[l, pos, :] = v

            # Attention window
            if lt == "sliding_attention":
                start_pos = max(0, pos - sw)
            else:
                start_pos = 0

            # Compute attention
            attn_out = np.zeros(H * hd, dtype=np.float32)
            for head in range(H):
                kv_h = head // group
                qh = q[head*hd:(head+1)*hd]

                n_pos = pos - start_pos + 1
                scores = np.zeros(n_pos, dtype=np.float32)
                scale = 1.0 / np.sqrt(hd)

                for t in range(start_pos, pos + 1):
                    kt = K_cache[l, t, kv_h*hd:(kv_h+1)*hd]
                    scores[t - start_pos] = np.dot(qh, kt) * scale

                weights = softmax(scores)

                for t in range(start_pos, pos + 1):
                    vt = V_cache[l, t, kv_h*hd:(kv_h+1)*hd]
                    attn_out[head*hd:(head+1)*hd] += weights[t - start_pos] * vt

            # Output projection
            attn_result = wo @ attn_out + bo

            # Residual
            h = h + attn_result

            # Pre-FFN norm
            post_ln = tensors[f"{prefix}.post_attention_layernorm.weight"]
            hn = rmsnorm(h, post_ln, eps)

            # MoE routing
            router_w = tensors[f"{prefix}.mlp.router.weight"]  # [E, D]
            router_b = tensors[f"{prefix}.mlp.router.bias"]    # [E]
            scores = router_w @ hn + router_b

            # Top-K selection
            top_indices = np.argsort(-scores)[:topk]
            top_scores = scores[top_indices]
            top_weights = softmax(top_scores)

            # Expert computation
            moe_out = np.zeros(D, dtype=np.float32)
            for ki in range(topk):
                eid = top_indices[ki]
                w_k = top_weights[ki]

                ep = f"{prefix}.mlp.experts.{eid}"
                gu_w = tensors[f"{ep}.gate_up_proj"]        # [moe_inter, D]
                gu_b = tensors[f"{ep}.gate_up_proj_bias"]   # [moe_inter]
                d_w = tensors[f"{ep}.down_proj"]            # [D, D]
                d_b = tensors[f"{ep}.down_proj_bias"]       # [D]

                # gate_up fused
                gu = gu_w @ hn + gu_b
                half = moe_inter // 2
                gate = gu[:half]
                up = gu[half:]
                activated = silu(gate) * up

                # down proj
                expert_out = d_w @ activated + d_b
                moe_out += w_k * expert_out

            # Residual
            h = h + moe_out

        # Final norm
        hn = rmsnorm(h, final_norm_w, eps)

        # LM head
        logits = lm_head @ hn

        # Greedy
        next_tok = np.argmax(logits)
        print(f"  pos={pos} tok_in={tok} → tok_out={next_tok} (logit_max={logits[next_tok]:.4f})")

    print("\n── Forward pass COMPLETATO ──")
    print("Il codice Python produce output coerente (pesi random = output random ma valido).")
    print("Quando il C compila, il self-test dovrebbe dare risultati analoghi.")
    return 0


if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "test_model"
    sys.exit(forward_pass(model_dir))
