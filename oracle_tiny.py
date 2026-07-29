#!/usr/bin/env python3
"""Genera checkpoint deterministici dal vero GptOssForCausalLM Transformers."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.gpt_oss.modeling_gpt_oss import (
    apply_rotary_pos_emb,
    repeat_kv,
)

VERSIONS = {
    "torch": torch.__version__,
    "transformers": __import__("transformers").__version__,
}


def save_array(root, name, tensor, manifest):
    array = tensor.detach().float().cpu().contiguous().numpy()
    path = root / f"{name}.npy"
    np.save(path, array, allow_pickle=False)
    raw = array.tobytes()
    manifest[name] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "min": float(array.min()) if array.size else 0.0,
        "max": float(array.max()) if array.size else 0.0,
        "mean": float(array.mean()) if array.size else 0.0,
    }


def build_oracle(model_dir, output_dir, token_id):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    arrays = {}
    ids = torch.tensor([[token_id]], dtype=torch.long)
    position_ids = torch.tensor([[0]], dtype=torch.long)

    with torch.no_grad():
        h = model.model.embed_tokens(ids)
        save_array(root, "embedding", h, arrays)
        cos, sin = model.model.rotary_emb(h, position_ids)
        save_array(root, "rope_cos", cos, arrays)
        save_array(root, "rope_sin", sin, arrays)

        for layer_index, layer in enumerate(model.model.layers):
            base = f"layer{layer_index}"
            save_array(root, f"{base}.input", h, arrays)
            residual = h
            hn = layer.input_layernorm(h)
            save_array(root, f"{base}.pre_attn_norm", hn, arrays)
            attn = layer.self_attn
            q0 = F.linear(hn, attn.q_proj.weight, None)
            k0 = F.linear(hn, attn.k_proj.weight, None)
            v0 = F.linear(hn, attn.v_proj.weight, None)
            q = attn.q_proj(hn)
            k = attn.k_proj(hn)
            v = attn.v_proj(hn)
            save_array(root, f"{base}.q_nobias", q0, arrays)
            save_array(root, f"{base}.k_nobias", k0, arrays)
            save_array(root, f"{base}.v_nobias", v0, arrays)
            save_array(root, f"{base}.q_bias", q, arrays)
            save_array(root, f"{base}.k_bias", k, arrays)
            save_array(root, f"{base}.v_bias", v, arrays)

            shape = (*hn.shape[:-1], -1, attn.head_dim)
            q = q.view(shape).transpose(1, 2)
            k = k.view(shape).transpose(1, 2)
            v = v.view(shape).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            save_array(root, f"{base}.q_rope", q, arrays)
            save_array(root, f"{base}.k_rope", k, arrays)
            save_array(root, f"{base}.v_heads", v, arrays)

            kr = repeat_kv(k, attn.num_key_value_groups)
            vr = repeat_kv(v, attn.num_key_value_groups)
            scores = torch.matmul(q, kr.transpose(2, 3)) * attn.scaling
            sinks = attn.sinks.reshape(1, -1, 1, 1).expand(1, -1, 1, -1)
            combined = torch.cat((scores, sinks), dim=-1)
            probs = F.softmax(combined - combined.max(dim=-1, keepdim=True).values, dim=-1)
            save_array(root, f"{base}.attn_scores", scores, arrays)
            save_array(root, f"{base}.sink_logits", sinks, arrays)
            save_array(root, f"{base}.attn_probs_with_sink", probs, arrays)
            heads = torch.matmul(probs[..., :-1], vr).transpose(1, 2).contiguous()
            concat = heads.reshape(*hn.shape[:-1], -1)
            attn_out = attn.o_proj(concat)
            save_array(root, f"{base}.attn_concat", concat, arrays)
            save_array(root, f"{base}.attn_out", attn_out, arrays)
            h = residual + attn_out
            save_array(root, f"{base}.post_attn_residual", h, arrays)

            residual = h
            hn = layer.post_attention_layernorm(h)
            save_array(root, f"{base}.pre_moe_norm", hn, arrays)
            router = layer.mlp.router
            router_logits = F.linear(hn.reshape(-1, hn.shape[-1]), router.weight, router.bias)
            top_values, top_indices = torch.topk(router_logits, router.top_k, dim=-1)
            top_weights = F.softmax(top_values, dim=-1)
            save_array(root, f"{base}.router_logits", router_logits, arrays)
            save_array(root, f"{base}.top_indices", top_indices.float(), arrays)
            save_array(root, f"{base}.top_weights", top_weights, arrays)

            experts = layer.mlp.experts
            moe = torch.zeros_like(hn.reshape(-1, hn.shape[-1]))
            for rank, expert_id in enumerate(top_indices[0].tolist()):
                gu = hn.reshape(-1, hn.shape[-1]) @ experts.gate_up_proj[expert_id]
                gu = gu + experts.gate_up_proj_bias[expert_id]
                gate, up = gu[..., 0::2], gu[..., 1::2]
                gate_clamped = gate.clamp(max=experts.limit)
                up_clamped = up.clamp(min=-experts.limit, max=experts.limit)
                activated = (up_clamped + 1) * gate_clamped * torch.sigmoid(
                    gate_clamped * experts.alpha
                )
                expert_out = activated @ experts.down_proj[expert_id]
                expert_out = expert_out + experts.down_proj_bias[expert_id]
                contribution = expert_out * top_weights[0, rank]
                save_array(root, f"{base}.expert{rank}.gate_up", gu, arrays)
                save_array(root, f"{base}.expert{rank}.activated", activated, arrays)
                save_array(root, f"{base}.expert{rank}.output", expert_out, arrays)
                save_array(root, f"{base}.expert{rank}.contribution", contribution, arrays)
                moe += contribution
            moe = moe.view_as(hn)
            save_array(root, f"{base}.moe_out", moe, arrays)
            h = residual + moe
            save_array(root, f"{base}.output", h, arrays)

        final_norm = model.model.norm(h)
        logits = model.lm_head(final_norm)
        save_array(root, "final_norm", final_norm, arrays)
        save_array(root, "logits", logits, arrays)
        top_values, top_indices = torch.topk(logits[0, -1], 10)
        save_array(root, "logits_top10_values", top_values, arrays)
        save_array(root, "logits_top10_indices", top_indices.float(), arrays)

        official = model(ids, use_cache=False).logits
        max_error = float((official - logits).abs().max())
        if max_error > 1e-5:
            raise RuntimeError(f"oracle manuale diverge dal modello: {max_error}")

    metadata = {
        "format": "picchio-oracle-v1",
        "model_dir": str(model_dir),
        "versions": VERSIONS,
        "input_ids": [token_id],
        "position": 0,
        "manual_vs_transformers_max_abs": max_error,
        "argmax": int(torch.argmax(logits[0, -1]).item()),
        "top1_top2_margin": float(top_values[0] - top_values[1]),
        "arrays": arrays,
    }
    (root / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Oracle scritto in {root}; argmax={metadata['argmax']}; max_abs={max_error:.3g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="test_tiny_hf")
    parser.add_argument("--output", default="oracle_hf")
    parser.add_argument("--token", type=int, default=1)
    args = parser.parse_args()
    build_oracle(args.model, args.output, args.token)


if __name__ == "__main__":
    main()
