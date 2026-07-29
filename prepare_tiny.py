#!/usr/bin/env python3
"""Prepara una fixture GPT-OSS tiny F32 nel layout leggibile da Picchio."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

REPO = "tiny-random/gpt-oss"
REVISION = "02ba5c61f879b5a38a8b1f7a8e0409b8e1bb8f38"
FILES = ["config.json", "generation_config.json", "model.safetensors"]


def tensor_meta(tensor):
    raw = tensor.detach().contiguous().cpu().numpy().tobytes()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def prepare(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    tensors = load_file(str(source / "model.safetensors"), device="cpu")
    converted = {}

    for name, value in tensors.items():
        if ".mlp.experts." not in name or name.endswith(
            (".gate_up_proj_bias", ".down_proj_bias")
        ):
            converted[name] = value.float().contiguous()

    for name, value in tensors.items():
        if not name.endswith(".mlp.experts.gate_up_proj"):
            continue
        prefix = name.removesuffix("gate_up_proj")
        gate_up = value.float()  # [E, D, 2I], gate/up interleaved
        down = tensors[prefix + "down_proj"].float()  # [E, I, D]
        for expert in range(gate_up.shape[0]):
            # Picchio conserva il layout GPT-OSS interleaved [g0,u0,g1,u1,...].
            gu = gate_up[expert].transpose(0, 1).contiguous()
            converted[f"{prefix}gate_up_proj.{expert}"] = gu
            converted[f"{prefix}down_proj.{expert}"] = (
                down[expert].transpose(0, 1).contiguous()
            )

    # Il checkpoint usa embedding legati e non contiene lm_head.weight.
    converted["lm_head.weight"] = converted["model.embed_tokens.weight"].clone()
    save_file(converted, str(output / "model.safetensors"))
    shutil.copy2(source / "config.json", output / "config.json")
    if (source / "generation_config.json").exists():
        shutil.copy2(source / "generation_config.json", output / "generation_config.json")

    manifest = {
        "format": "picchio-tiny-f32-v1",
        "source_repo": REPO,
        "source_revision": REVISION,
        "layout_only": True,
        "tensors": {name: tensor_meta(t) for name, t in sorted(converted.items())},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Fixture Picchio: {output} ({len(converted)} tensori)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="test_tiny_hf")
    parser.add_argument("--output", default="test_tiny_picchio")
    args = parser.parse_args()
    source = Path(args.source)
    if not (source / "model.safetensors").exists():
        snapshot_download(
            REPO, revision=REVISION, local_dir=source, allow_patterns=FILES
        )
    prepare(source, Path(args.output))


if __name__ == "__main__":
    main()
