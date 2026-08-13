#!/usr/bin/env python3
"""Stream-download + convert a Qwen3-MoE model one shard at a time.

Never keeps the whole raw checkpoint on disk: download shard -> convert to
Picchio INT4 -> delete raw -> next. Only the ~16 GB converted model stays.

Reuses convert.py's Qwen path (convert_shard + interleave_qwen_experts). The
cross-shard `pending` buffer holds half-seen experts in RAM (11 experts in
Qwen3-30B-A3B are split across a shard boundary), so deleting each raw shard
right after processing is safe within a run.

Config via environment (defaults target D: without touching the 120B):
  PICCHIO_REPO    HF repo id           (default Qwen/Qwen3-30B-A3B-Instruct-2507)
  PICCHIO_OUTPUT  converted model dir  (default D:/qwen3_30b_i4)
  PICCHIO_RAW     scratch for one raw shard + HF cache (default D:/qwen_tmp)
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = os.environ.get("PICCHIO_REPO", "Qwen/Qwen3-30B-A3B-Instruct-2507")
OUTPUT = os.environ.get("PICCHIO_OUTPUT", "D:/qwen3_30b_i4")
RAW_DIR = os.environ.get("PICCHIO_RAW", "D:/qwen_tmp")

# Keep every byte of HF's cache on D: — C: has almost no free space.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ["HF_HOME"] = str(Path(RAW_DIR) / "hf_home")
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

import numpy as np  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402
from safetensors.numpy import save_file  # noqa: E402
from convert import convert_shard, interleave_qwen_experts  # noqa: E402


def _reclaim(raw_path):
    """Delete the raw shard and any HF blob copy left inside RAW_DIR/.cache."""
    try:
        os.remove(raw_path)
    except OSError:
        pass
    shutil.rmtree(Path(RAW_DIR) / ".cache", ignore_errors=True)


def main():
    # ── config + tokenizer (small files) ──
    print(f"=== {REPO} ===")
    print("  downloading config/tokenizer...")
    tok_files = ["config.json", "generation_config.json", "tokenizer.json",
                 "tokenizer_config.json", "vocab.json", "merges.txt",
                 "special_tokens_map.json"]
    for fname in tok_files:
        try:
            hf_hub_download(REPO, fname, local_dir=OUTPUT)
            print(f"    ok {fname}")
        except Exception as e:
            print(f"    -- {fname}: {e}")

    cfg = json.load(open(f"{OUTPUT}/config.json"))
    mt = cfg.get("model_type", "?")
    if "qwen" not in mt.lower():
        print(f"  ! model_type={mt} is not Qwen — use convert.py for GPT-OSS.")
        sys.exit(1)
    print(f"  D={cfg['hidden_size']} L={cfg['num_hidden_layers']} "
          f"E={cfg.get('num_experts', cfg.get('num_local_experts'))} "
          f"top{cfg['num_experts_per_tok']} moe_i={cfg.get('moe_intermediate_size')}")

    # ── shard list from the safetensors index ──
    idx_path = hf_hub_download(REPO, "model.safetensors.index.json", local_dir=OUTPUT)
    weight_map = json.load(open(idx_path))["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    print(f"  {len(shard_names)} shards\n")

    stats = {"total_tensors": 0, "norms": 0, "bias": 0, "router": 0,
             "dense_i4": 0, "expert_i4": 0, "expert_blocks": 0,
             "expert_scales": 0, "other": 0}
    pending = {}          # cross-shard buffer for half-seen experts (kept in RAM)
    t_total = time.time()

    for si, shard_name in enumerate(shard_names):
        out_path = Path(OUTPUT) / f"model-{si:05d}.safetensors"
        # Resume: skip only if this shard left no expert half pending last time.
        # (A completed run has empty `pending` at every boundary; the 11 split
        # experts all resolve within the run.) If a skip ever strands a half,
        # the final check below reports it and asks for a clean rerun.
        if out_path.exists() and out_path.stat().st_size > 1e8 and not pending:
            print(f"  [{si+1}/{len(shard_names)}] {shard_name} — already converted, skip")
            continue

        print(f"  [{si+1}/{len(shard_names)}] {shard_name}")
        t0 = time.time()
        print("    downloading...", end="", flush=True)
        raw_path = hf_hub_download(REPO, shard_name, local_dir=RAW_DIR)
        raw_gb = Path(raw_path).stat().st_size / 1e9
        print(f" {raw_gb:.1f} GB in {time.time()-t0:.0f}s")

        t0 = time.time()
        output_tensors = {}
        convert_shard(raw_path, output_tensors, cfg, stats, is_qwen=True)
        interleave_qwen_experts(output_tensors, pending, stats)

        save_file(output_tensors, str(out_path))
        attn_f32 = sum(1 for k in output_tensors if "self_attn" in k and "weight" in k
                       and getattr(output_tensors[k], "dtype", None) == np.float32
                       and output_tensors[k].ndim > 1)
        print(f"    -> {out_path.name}: {out_path.stat().st_size/1e9:.2f} GB, "
              f"{len(output_tensors)} tensors, attn_f32={attn_f32}, "
              f"pending={len(pending)}, {time.time()-t0:.0f}s")

        _reclaim(raw_path)
        del output_tensors

    if pending:
        print(f"\n  ! {len(pending)} experts still incomplete: {list(pending)[:5]}")
        print("    A skipped shard stranded an expert half. Delete the last output")
        print("    shard(s) and rerun to reprocess without skipping.")
        sys.exit(1)

    shutil.rmtree(RAW_DIR, ignore_errors=True)
    files = sorted(Path(OUTPUT).glob("model-*.safetensors"))
    total = sum(f.stat().st_size for f in files)
    print(f"\n{'='*60}")
    print(f"  done in {(time.time()-t_total)/60:.0f} min")
    print(f"  {len(files)} shards, {total/1e9:.1f} GB -> {OUTPUT}")
    print(f"  run: python chat_qwen.py --model {OUTPUT} --no-reasoning --ctx 2048 --pin-gb 8")


if __name__ == "__main__":
    main()
