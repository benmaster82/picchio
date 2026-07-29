#!/usr/bin/env python3
"""convert_streaming.py — Scarica e converte GPT-OSS-120B un shard alla volta.

Non tiene mai più di 1 shard raw su disco (~4.6 GB).
Scarica → converte → cancella raw → prossimo shard.
"""
import sys, time, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import *
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "openai/gpt-oss-120b"
OUTPUT = "D:/gptoss_i4"
RAW_DIR = "D:/gptoss_tmp"
N_SHARDS = 15  # model-00000-of-00014 to model-00014-of-00014

os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# Scarica config e tokenizer
print("=== Scaricamento config/tokenizer ===")
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "generation_config.json"]:
    try:
        p = hf_hub_download(REPO, fname, local_dir=OUTPUT)
        print(f"  ✓ {fname}")
    except Exception as e:
        print(f"  ⚠ {fname}: {e}")

cfg = json.load(open(f"{OUTPUT}/config.json"))
print(f"\n  D={cfg['hidden_size']} L={cfg['num_hidden_layers']} E={cfg['num_local_experts']}")

# Rigenera vocab binario
print("\n=== Export vocab binario ===")
from export_vocab import export_vocab
export_vocab(f"{OUTPUT}/tokenizer.json", f"{OUTPUT}/picchio_vocab.bin")

# Converte shard per shard
print(f"\n=== Conversione {N_SHARDS} shard (attention=F32, expert=INT4 gs64) ===\n")
t_total = time.time()

for si in range(N_SHARDS):
    shard_name = f"model-{si:05d}-of-{N_SHARDS-1:05d}.safetensors"
    out_path = Path(OUTPUT) / f"model-{si:05d}.safetensors"
    
    # Skip se già convertito
    if out_path.exists() and out_path.stat().st_size > 1e9:
        print(f"  [{si+1}/{N_SHARDS}] {shard_name} — già convertito, skip")
        continue
    
    print(f"  [{si+1}/{N_SHARDS}] {shard_name}")
    
    # 1. Scarica
    t0 = time.time()
    print(f"    scaricamento...", end="", flush=True)
    raw_path = hf_hub_download(REPO, shard_name, local_dir=RAW_DIR)
    t_dl = time.time() - t0
    raw_size = Path(raw_path).stat().st_size
    print(f" {raw_size/1e9:.1f} GB in {t_dl:.0f}s")
    
    # 2. Converti
    t0 = time.time()
    stats = {'total_tensors':0,'norms':0,'bias':0,'router':0,'dense_i4':0,
             'expert_i4':0,'expert_blocks':0,'expert_scales':0,'other':0}
    output_tensors = {}
    
    convert_shard(raw_path, output_tensors, cfg, stats)
    
    # Post-process MXFP4
    blocks_keys = [k for k in list(output_tensors.keys()) if k.endswith('_blocks')]
    for bk in blocks_keys:
        base = bk[:-7]
        sk = base + '_scales'
        if sk not in output_tensors: continue
        blocks = output_tensors[bk]
        scales = output_tensors[sk]
        n_exp, rows, n_blk = blocks.shape[0], blocks.shape[1], blocks.shape[2]
        cols = n_blk * 32
        print(f"    MXFP4: {base.split('.')[-1]} [{n_exp}×{rows}×{cols}]...", end="", flush=True)
        for e in range(n_exp):
            blk_flat = blocks[e].reshape(rows, n_blk * 16)
            lo = blk_flat & 0x0F
            hi = (blk_flat >> 4) & 0x0F
            nibbles = np.empty((rows, n_blk * 32), dtype=np.uint8)
            nibbles[:, 0::2] = lo
            nibbles[:, 1::2] = hi
            nibbles = nibbles[:, :cols]
            values = FP4_LUT[nibbles]
            sc = scales[e]
            sc_float = np.ldexp(np.ones_like(sc, dtype=np.float32),
                                sc.astype(np.int32) - 127)
            sc_expanded = np.repeat(sc_float, 32, axis=1)[:, :cols]
            values *= sc_expanded
            packed, row_scales = quantize_int4(values)
            output_tensors[f'{base}.{e}'] = packed
            output_tensors[f'{base}.{e}.qs'] = row_scales
        del output_tensors[bk]
        del output_tensors[sk]
        print(f" ✓")
    
    # Fix bias dtype
    for bk in [k for k in list(output_tensors.keys()) if 'experts' in k and 'bias' in k]:
        t = output_tensors[bk]
        if t.dtype != np.float32: output_tensors[bk] = t.astype(np.float32)
    
    # Salva
    save_file(output_tensors, str(out_path))
    t_conv = time.time() - t0
    out_size = out_path.stat().st_size
    
    # Verifica attention
    attn_f32 = sum(1 for k in output_tensors if 'self_attn' in k and 'weight' in k 
                   and output_tensors[k].dtype == np.float32 and output_tensors[k].ndim > 0)
    
    print(f"    → {out_path.name}: {out_size/1e9:.2f} GB, {len(output_tensors)} tensori, "
          f"attn_f32={attn_f32}, {t_conv:.0f}s")
    
    # 3. Cancella raw (libera spazio per il prossimo)
    try:
        os.remove(raw_path)
    except:
        pass
    
    del output_tensors

# Cleanup
import shutil
shutil.rmtree(RAW_DIR, ignore_errors=True)

elapsed = time.time() - t_total
print(f"\n{'='*60}")
print(f"  Conversione completata in {elapsed/60:.0f} min")
print(f"  Output: {OUTPUT}")
files = list(Path(OUTPUT).glob("model-*.safetensors"))
total = sum(f.stat().st_size for f in files)
print(f"  {len(files)} shard, {total/1e9:.1f} GB totale")
print(f"\n  Per usare: picchio.exe {OUTPUT}")
