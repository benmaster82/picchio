#!/usr/bin/env python3
"""Valida Picchio tiny: token greedy e checkpoint al confine sliding."""
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def save(root, name, tensor):
    array = tensor.detach().float().cpu().contiguous().numpy()
    np.save(root / f"{name}.npy", array, allow_pickle=False)


def load_model(path):
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    return AutoModelForCausalLM.from_pretrained(
        path, local_files_only=True, dtype=torch.float32,
        attn_implementation="eager",
    ).eval()


def hf_greedy(model, prompt, count):
    ids = torch.tensor([prompt], dtype=torch.long)
    generated = []
    with torch.no_grad():
        out = model(ids, use_cache=True)
        cache = out.past_key_values
        for _ in range(count):
            token = int(out.logits[0, -1].argmax())
            generated.append(token)
            one = torch.tensor([[token]], dtype=torch.long)
            out = model(one, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
    return generated


def run_c(exe, model_dir, prompt, count, oracle_dir=None, ctx=256):
    env = os.environ.copy()
    for name in ("OUTPUT", "INPUT_FILE", "MODEL_AUX", "TRACE_NUMERIC", "ORACLE_DIR"):
        env.pop(name, None)
    env.update({"RAW": "1", "INPUT": " ".join(map(str, prompt)),
                "MAX": str(count), "TEMPERATURE": "0", "REP": "1",
                "PIN_GB": "1", "CTX": str(ctx)})
    if oracle_dir:
        env["ORACLE_DIR"] = str(oracle_dir)
    proc = subprocess.run([str(exe), str(model_dir), str(count)], env=env,
                          capture_output=True, text=True, errors="replace",
                          timeout=300, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr[-4000:])
    tokens = [int(x) for x in re.findall(r"\[(\d+)\]", proc.stdout)]
    match = re.search(r"next=(\d+)", proc.stderr)
    return tokens, int(match.group(1)) if match else None, proc.stderr

def boundary_dumps(model, tokens, root):
    root.mkdir(parents=True, exist_ok=True)
    hooks = []
    layer0 = model.model.layers[0]
    captured = {}

    def capture(name):
        def hook(_module, _args, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[name] = value[:, -1:].detach()
        return hook

    hooks.append(model.model.embed_tokens.register_forward_hook(capture("embedding")))
    hooks.append(layer0.input_layernorm.register_forward_hook(capture("layer0.pre_attn_norm")))
    hooks.append(layer0.self_attn.register_forward_hook(capture("layer0.attn_out")))
    hooks.append(layer0.post_attention_layernorm.register_forward_hook(capture("layer0.pre_moe_norm")))
    hooks.append(layer0.mlp.register_forward_hook(capture("layer0.moe_out")))
    hooks.append(model.model.norm.register_forward_hook(capture("final_norm")))
    with torch.no_grad():
        logits = model(torch.tensor([tokens], dtype=torch.long), use_cache=False).logits
    for hook in hooks:
        hook.remove()
    for name, value in captured.items():
        save(root, name, value)
    save(root, "logits", logits[:, -1:])


def compare_common(reference, candidate, atol=1e-5, rtol=1e-5):
    names = sorted({p.stem for p in reference.glob("*.npy")} &
                   {p.stem for p in candidate.glob("*.npy")})
    for name in names:
        a = np.load(reference / f"{name}.npy", allow_pickle=False)
        b = np.load(candidate / f"{name}.npy", allow_pickle=False)
        if a.shape != b.shape or not np.allclose(a, b, atol=atol, rtol=rtol):
            diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            raise AssertionError(f"{name}: shape {b.shape}/{a.shape}, max_abs={diff}")
        print(f"OK boundary {name}: max_abs={np.max(np.abs(a-b)):.3g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="test_tiny_hf")
    parser.add_argument("--picchio-model", default="test_tiny_picchio")
    parser.add_argument("--exe", default="picchio.exe")
    args = parser.parse_args()
    model = load_model(args.model)
    exe = Path(args.exe).resolve()
    picchio_model = Path(args.picchio_model).resolve()

    prompt = [1]
    expected = hf_greedy(model, prompt, 32)
    got, first, _ = run_c(exe, picchio_model, prompt, 32)
    if first != expected[0] or got != expected:
        for i, (a, b) in enumerate(zip(expected, got)):
            if a != b:
                raise AssertionError(f"greedy diverge a {i}: HF={a}, C={b}")
        raise AssertionError(f"greedy lunghezza/first: HF={expected}, C={got}, first={first}")
    print(f"OK greedy 32 token: {expected}")

    boundary_tokens = [1 + (i % 97) for i in range(130)]
    hf_dir = Path("oracle_boundary_hf")
    c_dir = Path("oracle_boundary_c")
    shutil.rmtree(hf_dir, ignore_errors=True)
    shutil.rmtree(c_dir, ignore_errors=True)
    boundary_dumps(model, boundary_tokens, hf_dir)
    _, _, _ = run_c(exe, picchio_model, boundary_tokens, 0, c_dir, ctx=132)
    compare_common(hf_dir, c_dir)
    print("VALIDAZIONE TINY SEQUENZIALE SUPERATA")


if __name__ == "__main__":
    main()
