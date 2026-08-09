#!/usr/bin/env python3
"""convert_streaming.py — Download and convert GPT-OSS-120B one shard at a time.

Never keeps more than 1 raw shard on disk (~4.6 GB).
Download → convert → delete raw → next shard.
"""
import sys, time, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert import *
from pathlib import Path
from huggingface_hub import hf_hub_download

# Paths and repo are overridable via environment (handy for a different disk):
#   PICCHIO_OUTPUT  where the converted shards go
#   PICCHIO_RAW     scratch dir for the one raw shard at a time
#   PICCHIO_REPO    Hugging Face repo id
# Downloads honour HF_ENDPOINT (e.g. https://hf-mirror.com) for a faster mirror.
REPO = os.environ.get("PICCHIO_REPO", "openai/gpt-oss-120b")
OUTPUT = os.environ.get("PICCHIO_OUTPUT", "C:/models/gptoss_i4")
RAW_DIR = os.environ.get("PICCHIO_RAW", "C:/models/gptoss_tmp")
N_SHARDS = 15  # model-00000-of-00014 to model-00014-of-00014

os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)

# Download config and tokenizer
print("=== Downloading config/tokenizer ===")
for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "generation_config.json"]:
    try:
        p = hf_hub_download(REPO, fname, local_dir=OUTPUT)
        print(f"  ✓ {fname}")
    except Exception as e:
        print(f"  ⚠ {fname}: {e}")

cfg = json.load(open(f"{OUTPUT}/config.json"))
print(f"\n  D={cfg['hidden_size']} L={cfg['num_hidden_layers']} E={cfg['num_local_experts']}")

# Regenerate the binary vocab
print("\n=== Exporting binary vocab ===")
from export_vocab import export_vocab
export_vocab(f"{OUTPUT}/tokenizer.json", f"{OUTPUT}/picchio_vocab.bin")

# Convert shard by shard
print(f"\n=== Converting {N_SHARDS} shards (attention=F32, expert=INT4 gs64) ===\n")
t_total = time.time()

for si in range(N_SHARDS):
    shard_name = f"model-{si:05d}-of-{N_SHARDS-1:05d}.safetensors"
    out_path = Path(OUTPUT) / f"model-{si:05d}.safetensors"
    
    # Skip if already converted
    if out_path.exists() and out_path.stat().st_size > 1e9:
        print(f"  [{si+1}/{N_SHARDS}] {shard_name} — already converted, skip")
        continue

    print(f"  [{si+1}/{N_SHARDS}] {shard_name}")

    # 1. Download
    t0 = time.time()
    print(f"    downloading...", end="", flush=True)
    raw_path = hf_hub_download(REPO, shard_name, local_dir=RAW_DIR)
    t_dl = time.time() - t0
    raw_size = Path(raw_path).stat().st_size
    print(f" {raw_size/1e9:.1f} GB in {t_dl:.0f}s")
    
    # 2. Convert
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
    
    # Save
    save_file(output_tensors, str(out_path))
    t_conv = time.time() - t0
    out_size = out_path.stat().st_size

    # Verify attention
    attn_f32 = sum(1 for k in output_tensors if 'self_attn' in k and 'weight' in k
                   and output_tensors[k].dtype == np.float32 and output_tensors[k].ndim > 0)

    print(f"    → {out_path.name}: {out_size/1e9:.2f} GB, {len(output_tensors)} tensors, "
          f"attn_f32={attn_f32}, {t_conv:.0f}s")

    # 3. Delete the raw file (frees space for the next one)
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
print(f"  Conversion completed in {elapsed/60:.0f} min")
print(f"  Output: {OUTPUT}")
files = list(Path(OUTPUT).glob("model-*.safetensors"))
total = sum(f.stat().st_size for f in files)
print(f"  {len(files)} shards, {total/1e9:.1f} GB total")
print(f"\n  To use: picchio.exe {OUTPUT}")
