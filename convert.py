#!/usr/bin/env python3
"""convert.py — Convert GPT-OSS-120B from HuggingFace to INT4 for Picchio.

The original model has:
  - Dense part (attention, embed, norms, router): BF16
  - MoE experts (gate_up_proj, down_proj): MXFP4

This converter:
  1. Downloads shard by shard (not everything in RAM)
  2. Dense part BF16 → quantize to per-row symmetric INT4
  3. Experts MXFP4 → dequantize to F32 → requantize to INT4
  4. Norms and biases: kept as F32
  5. Writes safetensors with the Picchio layout

Estimated output: ~57 GB (INT4 for everything).
RAM required: ~4 GB (one shard at a time).

Usage:
  python convert.py --model openai/gpt-oss-120b --output D:/gptoss_i4
  python convert.py --model D:/gptoss_orig --output D:/gptoss_i4  # from local

Requirements:
  pip install safetensors numpy torch huggingface_hub
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    from safetensors import safe_open
    from safetensors.numpy import save_file
except ImportError:
    print("pip install safetensors numpy")
    sys.exit(1)


# ── INT4 per-row symmetric quantization ──

def quantize_int4(tensor: np.ndarray, group_size: int = 64) -> tuple:
    """Quantize F32 [O, I] → (packed_uint8, scales_f32).

    If group_size > 0: one scale every group_size values (gs64 = +9pp quality).
    Scheme: value = (nibble - 8) * scale
    Range: [-8, 7], packed 2 per byte (lo nibble first).
    """
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    
    O, I = tensor.shape
    
    if group_size <= 0 or group_size >= I:
        # Per-row scaling (old, less precise)
        amax = np.max(np.abs(tensor), axis=1)
        scales = np.where(amax > 1e-8, amax / 7.0, 1e-8).astype(np.float32)
        quantized = np.clip(np.round(tensor / scales[:, None]), -8, 7).astype(np.int8)
    else:
        # Group-scaled: one scale every group_size values
        n_groups = (I + group_size - 1) // group_size
        scales = np.empty((O, n_groups), dtype=np.float32)
        quantized = np.empty((O, I), dtype=np.int8)
        
        for g in range(n_groups):
            g_start = g * group_size
            g_end = min(g_start + group_size, I)
            group = tensor[:, g_start:g_end]
            amax = np.max(np.abs(group), axis=1)
            sc = np.where(amax > 1e-8, amax / 7.0, 1e-8).astype(np.float32)
            scales[:, g] = sc
            quantized[:, g_start:g_end] = np.clip(
                np.round(group / sc[:, None]), -8, 7).astype(np.int8)
        
        scales = scales.reshape(-1)  # flatten [O * n_groups]
    
    # Pack 2 values per byte
    rb = (I + 1) // 2
    packed = np.zeros((O, rb), dtype=np.uint8)
    
    for i in range(0, I - 1, 2):
        lo = (quantized[:, i].astype(np.int16) + 8).astype(np.uint8) & 0xF
        hi = (quantized[:, i + 1].astype(np.int16) + 8).astype(np.uint8) & 0xF
        packed[:, i // 2] = lo | (hi << 4)
    
    if I % 2 == 1:
        lo = (quantized[:, -1].astype(np.int16) + 8).astype(np.uint8) & 0xF
        packed[:, rb - 1] = lo
    
    return packed.reshape(-1), scales


# ── MXFP4 dequantization ──

# FP4 E2M1 table: 4 bits → float value
# Bit layout: [sign(1)][exp(2)][mantissa(1)]
FP4_LUT = np.array([
    0.0,    0.5,   1.0,   1.5,   2.0,   3.0,   4.0,   6.0,    # 0000-0111
   -0.0,   -0.5,  -1.0,  -1.5,  -2.0,  -3.0,  -4.0,  -6.0,   # 1000-1111
], dtype=np.float32)


def quantize_int8(tensor: np.ndarray) -> tuple:
    """Symmetric INT8 with a per-row scale.

    Used for embedding and lm_head, which the official GPT-OSS configuration
    excludes from MXFP4 quantization (`modules_to_not_convert`). At INT4 the noise
    lands directly on the logits of a 201k-entry vocabulary; INT8 costs twice the
    space but reduces the error by about 16×.
    """
    rows, cols = tensor.shape
    q = np.empty((rows, cols), dtype=np.int8)
    scales = np.empty(rows, dtype=np.float32)
    # In blocks: on [201088, 2880] a full copy would cost over 2 GB.
    block = max(1, 8_000_000 // max(cols, 1))
    for start in range(0, rows, block):
        end = min(start + block, rows)
        chunk = tensor[start:end].astype(np.float32, copy=False)
        amax = np.abs(chunk).max(axis=1)
        s = np.where(amax > 0, amax / 127.0, 1.0).astype(np.float32)
        q[start:end] = np.clip(np.rint(chunk / s[:, None]), -127, 127).astype(np.int8)
        scales[start:end] = s
    return q, scales


def decode_e8m0(raw: np.ndarray) -> np.ndarray:
    """Decode E8M0 scale bytes to float32.
    E8M0: 8-bit exponent, no mantissa. value = 2^(raw - 127)
    """
    return np.ldexp(np.ones_like(raw, dtype=np.float32), raw.astype(np.int32) - 127)


def dequant_mxfp4(blocks: np.ndarray, scales: np.ndarray, 
                   shape: tuple, block_size: int = 32) -> np.ndarray:
    """Dequantize an MXFP4 tensor to F32.

    blocks: uint8 packed (2 FP4 per byte), flat shape
    scales: E8M0 or float scale per block
    shape: (rows, cols) output
    block_size: elements per block (typically 32)
    """
    rows, cols = shape
    n_blocks_per_row = (cols + block_size - 1) // block_size
    out = np.zeros(shape, dtype=np.float32)
    
    bytes_per_block = block_size // 2
    
    for r in range(rows):
        for b in range(n_blocks_per_row):
            col_start = b * block_size
            col_end = min(col_start + block_size, cols)
            n_vals = col_end - col_start
            
            # Scale for this block
            scale_idx = r * n_blocks_per_row + b
            if scale_idx < len(scales):
                sc = float(scales[scale_idx])
            else:
                sc = 1.0

            # Unpacked values
            block_offset = r * n_blocks_per_row * bytes_per_block + b * bytes_per_block
            for i in range(0, n_vals, 2):
                byte_idx = block_offset + i // 2
                if byte_idx >= len(blocks):
                    break
                byte = blocks[byte_idx]
                lo = byte & 0x0F
                hi = (byte >> 4) & 0x0F
                out[r, col_start + i] = FP4_LUT[lo] * sc
                if col_start + i + 1 < cols:
                    out[r, col_start + i + 1] = FP4_LUT[hi] * sc
    
    return out


def dequant_mxfp4_fast(blocks: np.ndarray, scales: np.ndarray,
                       shape: tuple, block_size: int = 32) -> np.ndarray:
    """Vectorized (much faster) version of the MXFP4 dequantization."""
    rows, cols = shape
    n_blocks_per_row = (cols + block_size - 1) // block_size
    bytes_per_block = block_size // 2

    # Unpack all the nibbles
    lo_nibbles = blocks & 0x0F
    hi_nibbles = (blocks >> 4) & 0x0F
    
    # Interleave: [lo0, hi0, lo1, hi1, ...]
    all_nibbles = np.empty(len(blocks) * 2, dtype=np.uint8)
    all_nibbles[0::2] = lo_nibbles
    all_nibbles[1::2] = hi_nibbles
    
    # Lookup FP4
    all_values = FP4_LUT[all_nibbles]
    
    # Reshape to [rows, n_blocks_per_row, block_size]
    total_vals = rows * n_blocks_per_row * block_size
    if len(all_values) < total_vals:
        all_values = np.pad(all_values, (0, total_vals - len(all_values)))
    all_values = all_values[:total_vals].reshape(rows, n_blocks_per_row, block_size)
    
    # Apply scales: broadcast [rows, n_blocks_per_row, 1]
    scales_reshaped = scales[:rows * n_blocks_per_row].reshape(rows, n_blocks_per_row, 1)
    all_values *= scales_reshaped
    
    # Reshape to [rows, cols] (trim padding)
    out = all_values.reshape(rows, -1)[:, :cols]
    return out.astype(np.float32)


# ── Main conversion ──

def convert_shard(shard_path: str, output_tensors: dict, cfg: dict,
                  stats: dict, dense_bits: int = 4):
    """Convert a single safetensors shard."""
    
    D = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    moe_inter = I * 2  # gate_up fused
    
    with safe_open(shard_path, framework="numpy") as f:
        keys = list(f.keys())
        
        for key in keys:
            # Read metadata for dtype without loading the tensor
            # safe_open with numpy does not support bfloat16, use torch
            pass

    # Reopen with torch to handle bfloat16
    try:
        import torch
        with safe_open(shard_path, framework="pt") as f:
            keys = list(f.keys())
            
            for key in keys:
                tensor = f.get_tensor(key)  # torch tensor
                original_dtype = tensor.dtype
                
                # Convert to float32 numpy
                if tensor.dtype == torch.bfloat16:
                    tensor_np = tensor.float().numpy()
                elif tensor.dtype == torch.float16:
                    tensor_np = tensor.float().numpy()
                elif tensor.dtype == torch.float32:
                    tensor_np = tensor.numpy()
                elif tensor.dtype == torch.uint8:
                    tensor_np = tensor.numpy()
                elif tensor.dtype == torch.int8:
                    tensor_np = tensor.numpy()
                else:
                    tensor_np = tensor.float().numpy()
                
                original_shape = tensor_np.shape
                
                # Determine the tensor type
                is_norm = "layernorm" in key or "norm.weight" in key
                is_bias = ".bias" in key or key.endswith("_bias")
                is_embed = "embed_tokens" in key or "lm_head" in key
                is_expert = "experts" in key
                is_router = "router" in key
                is_attn = "self_attn" in key
                
                if is_norm:
                    output_tensors[key] = tensor_np.astype(np.float32)
                    stats["norms"] += tensor_np.nbytes
                    
                elif is_bias and not is_expert:
                    output_tensors[key] = tensor_np.astype(np.float32)
                    stats["bias"] += tensor_np.nbytes
                    
                elif is_router:
                    output_tensors[key] = tensor_np.astype(np.float32)
                    stats["router"] += tensor_np.size * 4
                    
                elif is_expert:
                    if is_bias:
                        # Expert biases are aggregated BF16/F16/F32 [E,...]:
                        # do not quantize them and do not produce .qs.
                        output_tensors[key] = tensor_np.astype(np.float32)
                        stats["bias"] += tensor_np.size * 4
                    elif "blocks" in key:
                        # MXFP4 blocks — dequantized with the associated scales
                        # Save temporarily, we will process it later with the scales
                        output_tensors[key] = tensor_np
                        stats["expert_i4"] += 0  # counted later
                    elif "scales" in key:
                        # MXFP4 scales E8M0 — save to combine with blocks
                        output_tensors[key] = tensor_np
                    else:
                        # Expert weight F32 — quantize to INT4
                        t_f32 = tensor_np.astype(np.float32)
                        if t_f32.ndim >= 2:
                            t_2d = t_f32.reshape(-1, t_f32.shape[-1])
                            packed, scales = quantize_int4(t_2d)
                            output_tensors[key] = packed
                            output_tensors[key + ".qs"] = scales
                            stats["expert_i4"] += packed.nbytes + scales.nbytes
                        else:
                            output_tensors[key] = t_f32
                            stats["other"] += t_f32.nbytes
                        
                elif is_embed or is_attn:
                    t_f32 = tensor_np.astype(np.float32)
                    if t_f32.ndim == 1:
                        output_tensors[key] = t_f32
                        stats["bias"] += t_f32.nbytes
                    elif is_attn:
                        # Attention: do NOT quantize (modules_to_not_convert)
                        # Keep F32 to preserve quality
                        output_tensors[key] = t_f32
                        stats["dense_i4"] += t_f32.nbytes
                    else:
                        # Embedding/lm_head: INT8 per row, not INT4.
                        # They are excluded from the official quantization: at INT4
                        # the logit noise degrades long generations.
                        t_2d = t_f32.reshape(-1, t_f32.shape[-1])
                        q8, scales = quantize_int8(t_2d)
                        output_tensors[key] = q8
                        output_tensors[key + ".qs"] = scales
                        stats["dense_i4"] += q8.nbytes + scales.nbytes
                else:
                    output_tensors[key] = tensor_np.astype(np.float32)
                    stats["other"] += tensor_np.size * 4
                
                stats["total_tensors"] += 1
    except ImportError:
        print("  ✗ PyTorch required to read BF16 tensors")
        print("    pip install torch")
        sys.exit(1)


def convert_model(model_path: str, output_path: str, dense_bits: int = 4):
    """Convert the GPT-OSS-120B model."""

    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy config and tokenizer
    import shutil
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            print(f"  ✓ {fname}")
    
    # Load config
    with open(model_path / "config.json") as f:
        cfg = json.load(f)

    print(f"\n  Model: GPT-OSS-120B")
    print(f"  D={cfg['hidden_size']} L={cfg['num_hidden_layers']} "
          f"E={cfg['num_local_experts']} top{cfg['num_experts_per_tok']}")
    print(f"  Input format: MXFP4 (experts) + BF16 (dense)")
    print(f"  Output format: per-row symmetric INT4")

    # Find all safetensors files
    shard_files = sorted(model_path.glob("*.safetensors"))
    if not shard_files:
        # Try the "original" subdirectory
        shard_files = sorted((model_path / "original").glob("*.safetensors"))

    if not shard_files:
        print(f"\n  ✗ No .safetensors file found in {model_path}")
        print(f"  Download with: huggingface-cli download openai/gpt-oss-120b --local-dir {model_path}")
        return
    
    print(f"\n  {len(shard_files)} shards to convert")

    # Convert shard by shard
    stats = {
        "total_tensors": 0, "norms": 0, "bias": 0, "router": 0,
        "dense_i4": 0, "expert_i4": 0, "expert_blocks": 0,
        "expert_scales": 0, "other": 0
    }
    
    total_output_bytes = 0
    t_start = time.time()
    
    for si, shard_path in enumerate(shard_files):
        print(f"\n  [{si+1}/{len(shard_files)}] {shard_path.name}...")
        t0 = time.time()
        
        output_tensors = {}
        convert_shard(str(shard_path), output_tensors, cfg, stats, dense_bits)
        
        # ── Post-processing: dequantize MXFP4 blocks+scales → INT4 ──
        blocks_keys = [k for k in list(output_tensors.keys()) if k.endswith("_blocks")]
        for bk in blocks_keys:
            base = bk[:-7]  # remove "_blocks"
            sk = base + "_scales"
            
            if sk not in output_tensors:
                continue
            
            blocks = output_tensors[bk]   # uint8 [n_experts, rows, n_blocks, 16]
            scales = output_tensors[sk]    # uint8 [n_experts, rows, n_blocks] E8M0
            
            n_exp = blocks.shape[0]
            rows = blocks.shape[1]
            n_blk = blocks.shape[2]
            block_size = 32  # 16 bytes × 2 values/byte
            cols = n_blk * block_size  # = 2880
            
            print(f"    MXFP4: {base} [{n_exp}×{rows}×{cols}]...", end="", flush=True)
            
            # Dequant per expert, row by row (vectorized per block)
            for e in range(n_exp):
                # blocks[e]: [rows, n_blk, 16] → unpack nibbles
                blk_flat = blocks[e].reshape(rows, n_blk * 16)  # [rows, n_blk*16]
                
                # Unpack: 2 FP4 per byte
                lo = blk_flat & 0x0F
                hi = (blk_flat >> 4) & 0x0F
                # Interleave: [lo0, hi0, lo1, hi1, ...]
                nibbles = np.empty((rows, n_blk * 32), dtype=np.uint8)
                nibbles[:, 0::2] = lo
                nibbles[:, 1::2] = hi
                nibbles = nibbles[:, :cols]  # trim
                
                # FP4 → float via LUT
                values = FP4_LUT[nibbles]  # [rows, cols]
                
                # Scale E8M0: value = 2^(raw - 127)
                sc = scales[e]  # [rows, n_blk]
                sc_float = np.ldexp(np.ones_like(sc, dtype=np.float32),
                                    sc.astype(np.int32) - 127)  # [rows, n_blk]
                
                # Broadcast the scale per block of 32
                sc_expanded = np.repeat(sc_float, block_size, axis=1)[:, :cols]
                values *= sc_expanded

                # Quantize to INT4
                packed, row_scales = quantize_int4(values)

                # Save with the Picchio name:
                # model.layers.X.mlp.experts.Y.{gate_up_proj|down_proj}
                # We extract the layer and projection name from the key
                # base = "model.layers.X.mlp.experts.{gate_up_proj|down_proj}"
                out_key = f"{base}.{e}"
                output_tensors[out_key] = packed
                output_tensors[out_key + ".qs"] = row_scales
                stats["expert_i4"] += packed.nbytes + row_scales.nbytes
            
            # Remove the original blocks and scales
            del output_tensors[bk]
            del output_tensors[sk]
            print(f" ✓ ({n_exp} experts)")

        # Convert expert biases from BF16 to F32
        bias_keys = [k for k in list(output_tensors.keys()) 
                     if "experts" in k and "bias" in k and isinstance(output_tensors[k], np.ndarray)]
        for bk in bias_keys:
            t = output_tensors[bk]
            if t.dtype != np.float32:
                output_tensors[bk] = t.astype(np.float32)
        
        # Save the output shard
        out_name = f"model-{si:05d}.safetensors"
        out_path = output_path / out_name

        # save_file wants np arrays
        save_file(output_tensors, str(out_path))

        shard_bytes = out_path.stat().st_size
        total_output_bytes += shard_bytes
        elapsed = time.time() - t0

        print(f"    → {out_name} ({shard_bytes/1e9:.2f} GB, "
              f"{len(output_tensors)} tensors, {elapsed:.1f}s)")

    total_elapsed = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print(f"  Conversion completed in {total_elapsed:.0f}s")
    print(f"  Output: {output_path}")
    print(f"  Total size: {total_output_bytes/1e9:.1f} GB")
    print(f"  Tensors: {stats['total_tensors']}")
    print(f"  Dense INT4: {stats['dense_i4']/1e9:.2f} GB")
    print(f"  Expert INT4: {stats['expert_i4']/1e9:.2f} GB")
    print(f"  Norms (F32): {stats['norms']/1e6:.1f} MB")
    print(f"  Router (F32): {stats['router']/1e6:.1f} MB")
    print(f"  Bias (F32): {stats['bias']/1e6:.1f} MB")
    print(f"\n  To use: picchio.exe {output_path}")


# ── Download + conversion ──

def download_and_convert(repo_id: str, output_path: str, dense_bits: int = 4):
    """Download the model from HuggingFace and convert it in a single pass."""

    try:
        from huggingface_hub import snapshot_download, list_repo_files
    except ImportError:
        print("pip install huggingface_hub")
        sys.exit(1)

    print(f"  Repository: {repo_id}")

    # List the files to gauge the size
    files = list_repo_files(repo_id)
    st_files = [f for f in files if f.endswith(".safetensors")]
    print(f"  {len(st_files)} safetensors files in the repo")

    # Download everything (with resume) into the output directory
    # HF Hub handles caching and resume automatically
    print(f"\n  Downloading (resumable, ~57 GB)...")
    print(f"  Destination: {output_path}")
    print(f"  (You can interrupt and resume at any time)\n")

    local_dir = snapshot_download(
        repo_id,
        local_dir=output_path + "_raw",
        allow_patterns=["*.safetensors", "*.json"],
    )

    print(f"\n  ✓ Download complete: {local_dir}")

    # Convert
    convert_model(local_dir, output_path, dense_bits)


# ── Entry point ──

def main():
    parser = argparse.ArgumentParser(
        description="Convert GPT-OSS-120B from MXFP4/BF16 to INT4 for Picchio"
    )
    parser.add_argument("--model", required=True,
                        help="Local path to the model or HuggingFace repo (e.g. openai/gpt-oss-120b)")
    parser.add_argument("--output", required=True,
                        help="Output directory for the converted model")
    parser.add_argument("--dense-bits", type=int, default=4, choices=[4, 8],
                        help="Bits for the dense part (4 or 8, default: 4)")
    parser.add_argument("--download", action="store_true",
                        help="Download from HuggingFace before converting")
    
    args = parser.parse_args()
    
    print(f"🪶 picchio convert — GPT-OSS-120B → INT4\n")
    
    if args.download or "/" in args.model and not Path(args.model).exists():
        download_and_convert(args.model, args.output, args.dense_bits)
    else:
        if not Path(args.model).exists():
            print(f"  ✗ Path not found: {args.model}")
            print(f"  Use --download to download from HuggingFace")
            sys.exit(1)
        convert_model(args.model, args.output, args.dense_bits)


if __name__ == "__main__":
    main()
