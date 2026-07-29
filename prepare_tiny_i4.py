#!/usr/bin/env python3
"""Crea fixture Picchio INT4 e riferimento HF con gli stessi pesi dequantizzati."""
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from convert import quantize_int4
from prepare_tiny import prepare

SOURCE = Path("test_tiny_hf")
F32 = Path("test_tiny_picchio")
I4 = Path("test_tiny_picchio_i4")
HF_REF = Path("test_tiny_hf_i4ref")
GROUP_SIZE = 16  # più gruppi sul tiny D=32; stesso kernel gs64 del 120B


def dequant(packed, scales, rows, cols, group_size):
    rb = (cols + 1) // 2
    data = packed.reshape(rows, rb)
    q = np.empty((rows, cols), dtype=np.float32)
    low = (data & 0x0F).astype(np.int16) - 8
    high = (data >> 4).astype(np.int16) - 8
    q[:, 0::2] = low[:, : q[:, 0::2].shape[1]]
    if cols > 1:
        q[:, 1::2] = high[:, : q[:, 1::2].shape[1]]
    ng = (cols + group_size - 1) // group_size
    sc = scales.reshape(rows, ng)
    out = np.empty_like(q)
    for group in range(ng):
        start = group * group_size
        end = min(start + group_size, cols)
        out[:, start:end] = q[:, start:end] * sc[:, group, None]
    return torch.from_numpy(out)


def main():
    if not (F32 / "model.safetensors").exists():
        prepare(SOURCE, F32)
    dense = load_file(str(F32 / "model.safetensors"), device="cpu")
    original = load_file(str(SOURCE / "model.safetensors"), device="cpu")
    output = {}
    dequantized = {}

    for name, tensor in dense.items():
        is_matrix = name in {"model.embed_tokens.weight", "lm_head.weight"}
        is_expert = ".mlp.experts.gate_up_proj." in name or ".mlp.experts.down_proj." in name
        if tensor.ndim == 2 and (is_matrix or is_expert):
            array = tensor.float().numpy()
            packed, scales = quantize_int4(array, group_size=GROUP_SIZE)
            output[name] = torch.from_numpy(packed.copy())
            output[name + ".qs"] = torch.from_numpy(scales.copy())
            dequantized[name] = dequant(packed, scales, *array.shape, GROUP_SIZE)
        else:
            output[name] = tensor.float().contiguous()

    I4.mkdir(parents=True, exist_ok=True)
    save_file(output, str(I4 / "model.safetensors"))
    shutil.copy2(F32 / "config.json", I4 / "config.json")
    (I4 / "quantization.json").write_text(json.dumps({
        "format": "int4-symmetric-grouped", "group_size": GROUP_SIZE,
        "value": "(nibble - 8) * scale", "source": str(F32)
    }, indent=2), encoding="utf-8")

    reference = {name: tensor.float().contiguous() for name, tensor in original.items()}
    reference["model.embed_tokens.weight"] = dequantized["model.embed_tokens.weight"]
    cfg = json.loads((SOURCE / "config.json").read_text(encoding="utf-8"))
    experts = cfg["num_local_experts"]
    intermediate = cfg["intermediate_size"]
    hidden = cfg["hidden_size"]
    for layer in range(cfg["num_hidden_layers"]):
        prefix = f"model.layers.{layer}.mlp.experts."
        gu_all = torch.empty((experts, hidden, 2 * intermediate), dtype=torch.float32)
        down_all = torch.empty((experts, intermediate, hidden), dtype=torch.float32)
        for expert in range(experts):
            gu = dequantized[f"{prefix}gate_up_proj.{expert}"]
            gu_all[expert] = gu.T
            down_all[expert] = dequantized[f"{prefix}down_proj.{expert}"].T
        reference[prefix + "gate_up_proj"] = gu_all
        reference[prefix + "down_proj"] = down_all

    HF_REF.mkdir(parents=True, exist_ok=True)
    save_file(reference, str(HF_REF / "model.safetensors"))
    shutil.copy2(SOURCE / "config.json", HF_REF / "config.json")
    if (SOURCE / "generation_config.json").exists():
        shutil.copy2(SOURCE / "generation_config.json", HF_REF / "generation_config.json")
    print(f"Fixture INT4: {I4}; riferimento dequantizzato: {HF_REF}")


if __name__ == "__main__":
    main()
