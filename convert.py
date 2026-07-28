#!/usr/bin/env python3
"""convert.py — Converte GPT-OSS-120B da HuggingFace a INT4 per Picchio.

Il modello originale ha:
  - Parte densa (attention, embed, norms, router): BF16
  - Expert MoE (gate_up_proj, down_proj): MXFP4

Questo convertitore:
  1. Scarica shard per shard (non tutto in RAM)
  2. Parte densa BF16 → quantizza a INT4 simmetrico per-riga
  3. Expert MXFP4 → dequantizza a F32 → requantizza a INT4
  4. Norms e bias: mantiene F32
  5. Scrive in safetensors con layout Picchio

Output stimato: ~57 GB (INT4 per tutto).
RAM richiesta: ~4 GB (un shard alla volta).

Uso:
  python convert.py --model openai/gpt-oss-120b --output D:/gptoss_i4
  python convert.py --model D:/gptoss_orig --output D:/gptoss_i4  # da locale

Requisiti:
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


# ── INT4 quantizzazione simmetrica per-riga ──

def quantize_int4(tensor: np.ndarray) -> tuple:
    """Quantizza F32 [O, I] → (packed_uint8, scales_f32).
    
    Schema: value = (nibble - 8) * scale
    Range: [-8, 7], packed 2 per byte (lo nibble first).
    """
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    
    O, I = tensor.shape
    
    # Scala per riga
    amax = np.max(np.abs(tensor), axis=1)
    scales = np.where(amax > 1e-8, amax / 7.0, 1e-8).astype(np.float32)
    
    # Quantizza
    quantized = np.clip(np.round(tensor / scales[:, None]), -8, 7).astype(np.int8)
    
    # Pack 2 valori per byte
    rb = (I + 1) // 2
    packed = np.zeros((O, rb), dtype=np.uint8)
    
    for i in range(0, I - 1, 2):
        lo = (quantized[:, i].astype(np.int16) + 8).astype(np.uint8) & 0xF
        hi = (quantized[:, i + 1].astype(np.int16) + 8).astype(np.uint8) & 0xF
        packed[:, i // 2] = lo | (hi << 4)
    
    # Ultimo elemento se I è dispari
    if I % 2 == 1:
        lo = (quantized[:, -1].astype(np.int16) + 8).astype(np.uint8) & 0xF
        packed[:, rb - 1] = lo
    
    return packed.reshape(-1), scales


# ── MXFP4 dequantizzazione ──

# Tabella FP4 E2M1: 4 bit → float value
# Bit layout: [sign(1)][exp(2)][mantissa(1)]
FP4_LUT = np.array([
    0.0,    0.5,   1.0,   1.5,   2.0,   3.0,   4.0,   6.0,    # 0000-0111
   -0.0,   -0.5,  -1.0,  -1.5,  -2.0,  -3.0,  -4.0,  -6.0,   # 1000-1111
], dtype=np.float32)


def decode_e8m0(raw: np.ndarray) -> np.ndarray:
    """Decode E8M0 scale bytes to float32.
    E8M0: 8-bit exponent, no mantissa. value = 2^(raw - 127)
    """
    return np.ldexp(np.ones_like(raw, dtype=np.float32), raw.astype(np.int32) - 127)


def dequant_mxfp4(blocks: np.ndarray, scales: np.ndarray, 
                   shape: tuple, block_size: int = 32) -> np.ndarray:
    """Dequantizza un tensore MXFP4 a F32.
    
    blocks: uint8 packed (2 FP4 per byte), shape flat
    scales: E8M0 o float scale per blocco
    shape: (rows, cols) output
    block_size: elementi per blocco (tipicamente 32)
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
            
            # Scale per questo blocco
            scale_idx = r * n_blocks_per_row + b
            if scale_idx < len(scales):
                sc = float(scales[scale_idx])
            else:
                sc = 1.0
            
            # Unpacked valori
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
    """Versione vettorizzata (molto più veloce) della dequantizzazione MXFP4."""
    rows, cols = shape
    n_blocks_per_row = (cols + block_size - 1) // block_size
    bytes_per_block = block_size // 2
    
    # Unpack tutti i nibble
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


# ── Conversione principale ──

def convert_shard(shard_path: str, output_tensors: dict, cfg: dict, 
                  stats: dict, dense_bits: int = 4):
    """Converte un singolo shard safetensors."""
    
    D = cfg["hidden_size"]
    I = cfg["intermediate_size"]
    moe_inter = I * 2  # gate_up fused
    
    with safe_open(shard_path, framework="numpy") as f:
        keys = list(f.keys())
        
        for key in keys:
            # Leggi metadata per dtype senza caricare il tensore
            # safe_open con numpy non supporta bfloat16, usiamo torch
            pass
    
    # Riapri con torch per gestire bfloat16
    try:
        import torch
        with safe_open(shard_path, framework="pt") as f:
            keys = list(f.keys())
            
            for key in keys:
                tensor = f.get_tensor(key)  # torch tensor
                original_dtype = tensor.dtype
                
                # Converti a float32 numpy
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
                
                # Determina il tipo di tensore
                is_norm = "layernorm" in key or "norm.weight" in key
                is_bias = ".bias" in key
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
                        output_tensors[key] = tensor_np.astype(np.float32)
                        stats["bias"] += tensor_np.size * 4
                    elif "blocks" in key:
                        # MXFP4 blocks — dequantizziamo con le scale associate
                        # Salva temporaneamente, processeremo dopo con le scale
                        output_tensors[key] = tensor_np
                        stats["expert_i4"] += 0  # conteggio dopo
                    elif "scales" in key:
                        # MXFP4 scales E8M0 — salva per combinare con blocks
                        output_tensors[key] = tensor_np
                    else:
                        # Expert weight F32 — quantizza a INT4
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
                    else:
                        t_2d = t_f32.reshape(-1, t_f32.shape[-1])
                        packed, scales = quantize_int4(t_2d)
                        output_tensors[key] = packed
                        output_tensors[key + ".qs"] = scales
                        stats["dense_i4"] += packed.nbytes + scales.nbytes
                else:
                    output_tensors[key] = tensor_np.astype(np.float32)
                    stats["other"] += tensor_np.size * 4
                
                stats["total_tensors"] += 1
    except ImportError:
        print("  ✗ PyTorch richiesto per leggere tensori BF16")
        print("    pip install torch")
        sys.exit(1)


def convert_model(model_path: str, output_path: str, dense_bits: int = 4):
    """Converte il modello GPT-OSS-120B."""
    
    model_path = Path(model_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copia config e tokenizer
    import shutil
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json"]:
        src = model_path / fname
        if src.exists():
            shutil.copy2(src, output_path / fname)
            print(f"  ✓ {fname}")
    
    # Carica config
    with open(model_path / "config.json") as f:
        cfg = json.load(f)
    
    print(f"\n  Modello: GPT-OSS-120B")
    print(f"  D={cfg['hidden_size']} L={cfg['num_hidden_layers']} "
          f"E={cfg['num_local_experts']} top{cfg['num_experts_per_tok']}")
    print(f"  Formato input: MXFP4 (expert) + BF16 (denso)")
    print(f"  Formato output: INT4 simmetrico per-riga")
    
    # Trova tutti i file safetensors
    shard_files = sorted(model_path.glob("*.safetensors"))
    if not shard_files:
        # Prova nella subdirectory "original"
        shard_files = sorted((model_path / "original").glob("*.safetensors"))
    
    if not shard_files:
        print(f"\n  ✗ Nessun file .safetensors trovato in {model_path}")
        print(f"  Scarica con: huggingface-cli download openai/gpt-oss-120b --local-dir {model_path}")
        return
    
    print(f"\n  {len(shard_files)} shard da convertire")
    
    # Converti shard per shard
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
        
        # ── Post-processing: dequantizza MXFP4 blocks+scales → INT4 ──
        blocks_keys = [k for k in list(output_tensors.keys()) if k.endswith("_blocks")]
        for bk in blocks_keys:
            base = bk[:-7]  # rimuovi "_blocks"
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
            
            # Dequant per expert, riga per riga (vettorizzato per blocco)
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
                
                # Broadcast scale per blocco di 32
                sc_expanded = np.repeat(sc_float, block_size, axis=1)[:, :cols]
                values *= sc_expanded
                
                # Quantizza a INT4
                packed, row_scales = quantize_int4(values)
                
                # Salva con nome per Picchio:
                # model.layers.X.mlp.experts.Y.{gate_up_proj|down_proj}
                # Estraiamo layer e nome proiezione dal key
                # base = "model.layers.X.mlp.experts.{gate_up_proj|down_proj}"
                out_key = f"{base}.{e}"
                output_tensors[out_key] = packed
                output_tensors[out_key + ".qs"] = row_scales
                stats["expert_i4"] += packed.nbytes + row_scales.nbytes
            
            # Rimuovi blocks e scales originali
            del output_tensors[bk]
            del output_tensors[sk]
            print(f" ✓ ({n_exp} expert)")
        
        # Converti bias expert da BF16 a F32
        bias_keys = [k for k in list(output_tensors.keys()) 
                     if "experts" in k and "bias" in k and isinstance(output_tensors[k], np.ndarray)]
        for bk in bias_keys:
            t = output_tensors[bk]
            if t.dtype != np.float32:
                output_tensors[bk] = t.astype(np.float32)
        
        # Salva output shard
        out_name = f"model-{si:05d}.safetensors"
        out_path = output_path / out_name
        
        # save_file vuole np arrays
        save_file(output_tensors, str(out_path))
        
        shard_bytes = out_path.stat().st_size
        total_output_bytes += shard_bytes
        elapsed = time.time() - t0
        
        print(f"    → {out_name} ({shard_bytes/1e9:.2f} GB, "
              f"{len(output_tensors)} tensori, {elapsed:.1f}s)")
    
    total_elapsed = time.time() - t_start
    
    # Riepilogo
    print(f"\n{'='*60}")
    print(f"  Conversione completata in {total_elapsed:.0f}s")
    print(f"  Output: {output_path}")
    print(f"  Dimensione totale: {total_output_bytes/1e9:.1f} GB")
    print(f"  Tensori: {stats['total_tensors']}")
    print(f"  Dense INT4: {stats['dense_i4']/1e9:.2f} GB")
    print(f"  Expert INT4: {stats['expert_i4']/1e9:.2f} GB")
    print(f"  Norms (F32): {stats['norms']/1e6:.1f} MB")
    print(f"  Router (F32): {stats['router']/1e6:.1f} MB")
    print(f"  Bias (F32): {stats['bias']/1e6:.1f} MB")
    print(f"\n  Per usare: picchio.exe {output_path}")


# ── Download + conversione ──

def download_and_convert(repo_id: str, output_path: str, dense_bits: int = 4):
    """Scarica il modello da HuggingFace e converte in un unico passaggio."""
    
    try:
        from huggingface_hub import snapshot_download, list_repo_files
    except ImportError:
        print("pip install huggingface_hub")
        sys.exit(1)
    
    print(f"  Repository: {repo_id}")
    
    # Lista file per capire la dimensione
    files = list_repo_files(repo_id)
    st_files = [f for f in files if f.endswith(".safetensors")]
    print(f"  {len(st_files)} file safetensors nel repo")
    
    # Scarica tutto (con resume) nella directory output
    # HF Hub gestisce il caching e il resume automaticamente
    print(f"\n  Scaricamento in corso (resumable, ~57 GB)...")
    print(f"  Destinazione: {output_path}")
    print(f"  (Puoi interrompere e riprendere in qualsiasi momento)\n")
    
    local_dir = snapshot_download(
        repo_id,
        local_dir=output_path + "_raw",
        allow_patterns=["*.safetensors", "*.json"],
    )
    
    print(f"\n  ✓ Download completato: {local_dir}")
    
    # Converti
    convert_model(local_dir, output_path, dense_bits)


# ── Entry point ──

def main():
    parser = argparse.ArgumentParser(
        description="Converte GPT-OSS-120B da MXFP4/BF16 a INT4 per Picchio"
    )
    parser.add_argument("--model", required=True,
                        help="Percorso locale al modello o repo HuggingFace (es. openai/gpt-oss-120b)")
    parser.add_argument("--output", required=True,
                        help="Directory output per il modello convertito")
    parser.add_argument("--dense-bits", type=int, default=4, choices=[4, 8],
                        help="Bit per la parte densa (4 o 8, default: 4)")
    parser.add_argument("--download", action="store_true",
                        help="Scarica da HuggingFace prima di convertire")
    
    args = parser.parse_args()
    
    print(f"🪶 picchio convert — GPT-OSS-120B → INT4\n")
    
    if args.download or "/" in args.model and not Path(args.model).exists():
        download_and_convert(args.model, args.output, args.dense_bits)
    else:
        if not Path(args.model).exists():
            print(f"  ✗ Percorso non trovato: {args.model}")
            print(f"  Usa --download per scaricare da HuggingFace")
            sys.exit(1)
        convert_model(args.model, args.output, args.dense_bits)


if __name__ == "__main__":
    main()
