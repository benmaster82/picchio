#!/usr/bin/env python3
"""End-to-end Qwen3-MoE smoke test (requires torch + transformers).

Creates a tiny all-MoE checkpoint through the official Transformers model class,
converts it, packs a flat expert store, and checks that the safetensors, aligned
async, and persistent-service paths emit the same greedy token IDs.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM


ROOT = Path(__file__).resolve().parent
EXE = ROOT / ("picchio.exe" if os.name == "nt" else "picchio")


def invoke(args, env=None, input_text=None, capture=False):
    return subprocess.run(
        [str(a) for a in args], cwd=ROOT, env=env, input=input_text,
        text=True, encoding="utf-8", errors="replace", check=True,
        capture_output=capture)


def token_ids(stdout):
    return [int(line) for line in stdout.splitlines() if line.strip().isdigit()]


def engine_env():
    env = os.environ.copy()
    for key in ("MODEL", "MODEL_AUX", "SERVICE", "FLAT", "FLAT_VERIFY",
                "ASYNC_MOE", "DIRECT", "INPUT_FILE", "PROMPT"):
        env.pop(key, None)
    env.update({
        "RAW": "1", "INPUT": "1 3 5", "MAX": "8", "OUTPUT": "ids",
        "TEMPERATURE": "0", "REP": "1", "CTX": "32", "ECAP": "2",
        "OMP_NUM_THREADS": "2", "IO_THREADS": "2", "PREFETCH": "0",
    })
    return env


def main():
    if not EXE.is_file():
        raise SystemExit(f"build Picchio first; executable not found: {EXE}")

    # A deterministic workspace directory also works under restrictive Windows
    # sandbox profiles whose ACLs can make tempfile-created children unwritable.
    tmp = ROOT / ".qwen_smoke_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    try:
        raw = tmp / "raw"
        converted = tmp / "i4"
        flat = converted / "experts.picchioflat"

        torch.manual_seed(1234)
        cfg = Qwen3MoeConfig(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            moe_intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            num_experts=4, num_experts_per_tok=2, decoder_sparse_step=1,
            mlp_only_layers=[], max_position_embeddings=256,
            rope_theta=10000.0, rms_norm_eps=1e-6, attention_bias=False,
            norm_topk_prob=True, tie_word_embeddings=False,
            bos_token_id=1, eos_token_id=2)
        Qwen3MoeForCausalLM(cfg).save_pretrained(raw, safe_serialization=True)

        py_env = os.environ.copy()
        py_env["PYTHONUTF8"] = "1"
        invoke([sys.executable, "convert.py", "--model", raw,
                "--output", converted], env=py_env)
        py_env.update({"FLAT_MODEL": str(converted), "FLAT_OUT": str(flat)})
        invoke([sys.executable, "flat_pack.py"], env=py_env)

        base = engine_env()
        base.update({"FLAT": "0", "ASYNC_MOE": "0", "DIRECT": "0"})
        reference = invoke([EXE, converted], env=base, capture=True)
        ref_ids = token_ids(reference.stdout)
        if len(ref_ids) != 8:
            raise AssertionError(f"expected 8 baseline IDs, got {ref_ids}")

        fast = engine_env()
        fast.update({"FLAT": str(flat), "FLAT_VERIFY": "1",
                     "ASYNC_MOE": "1", "DIRECT": "1"})
        accelerated = invoke([EXE, converted], env=fast, capture=True)
        fast_ids = token_ids(accelerated.stdout)
        if fast_ids != ref_ids:
            raise AssertionError(f"Qwen flat/async mismatch: {ref_ids} != {fast_ids}")

        service = fast.copy()
        for key in ("RAW", "INPUT", "MAX", "OUTPUT"):
            service.pop(key, None)
        service["SERVICE"] = "1"
        frames = invoke(
            [EXE, converted], env=service,
            input_text="TURN 4 0 0 0.95 50 3 1 3 5\nSHUTDOWN\n",
            capture=True).stdout.splitlines()
        service_ids = [int(line.split()[1]) for line in frames
                       if line.startswith("TOKEN ")]
        if service_ids != ref_ids[:4] or not any(line.startswith("READY ") for line in frames):
            raise AssertionError(f"Qwen service mismatch: {service_ids} != {ref_ids[:4]}")

        print("Qwen3-MoE smoke PASSED")
        print("  converter + safetensors + flat/DIRECT/ASYNC_MOE + SERVICE")
        print("  greedy IDs:", " ".join(map(str, ref_ids)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
