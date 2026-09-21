# Picchio v0.7.0 — INT8 attention and format/engine co-design

Picchio v0.7.0 focuses on **byte reduction**, the dominant cost on a CPU that
streams a MoE model larger than its RAM. The headline is **INT8 attention**: the
attention weights, previously stored in F32, were the single largest chunk of
per-token byte movement. Storing them near-losslessly in INT8 roughly halves the
attention time and frees several gigabytes of resident RAM for the expert cache.
This release also adds dense-model conversion, speculative-decoding scaffolding,
and a VRAM-resident expert read-skip, on top of the multi-model streaming core
from 0.6.0.

## Highlights

### INT8 attention (`--dense-bits 8`)

- Added end-to-end INT8 storage for the attention projections (Q/K/V/O): per-row
  scales, run through the existing INT8 matmul kernel. Near-lossless in quality,
  it roughly halves `t_attn` and shrinks the resident dense part.
- Wired the previously inert `--dense-bits 8` path in `convert.py` and taught
  `convert_streaming_qwen.py` to honor `DENSE_BITS` during shard-by-shard
  conversion.
- Added `transcode_attn_to_int8.py`: requantizes the attention weights of an
  existing converted model (INT4/INT3 experts + F32 attention) to INT8 in place,
  without re-downloading or re-converting the original checkpoint. Resumable.
- Applied and quality-checked across gpt-oss-20b, Qwen3-30B-A3B, and
  gpt-oss-120b: correct facts and arithmetic, coherent multi-turn prose.

### Dense-model conversion

- A dense (non-MoE) checkpoint now converts as a **1-expert MoE**: the single MLP
  becomes expert 0 with a zeroed router and a rewritten config
  (`num_local_experts=1`, `num_experts_per_tok=1`), so the streaming engine runs
  it unchanged.
- Added `tie_word_embeddings` handling (e.g. small Qwen3 dense checkpoints).
- Primary use is providing a small, RAM-resident draft model for speculative
  decoding.

### Speculative decoding (experimental)

- Added a batched verification path (`forward_verify`, per-position argmax over a
  prefill range) and a draft-model loop (`DRAFT_MODEL`, `SPEC_K`) that proposes K
  tokens, verifies them in one pass, and accepts the longest correct prefix plus
  one bonus token.
- The draft model is loaded in isolation so two open model databases do not
  collide on the per-thread file-handle cache; output is **byte-identical** to
  greedy decode.
- Added `SPEC_PROBE` and `SELF_DRAFT_PROBE` diagnostics.

  This path is a **net win only when the target model is memory-resident and the
  draft has a high acceptance rate**. On a disk-bound target where `ASYNC_MOE`
  already hides the expert I/O, the extra draft and verification work is a net
  loss, so it is shipped as **scaffolding, disabled by default**, for larger-RAM
  and GPU configurations.

### Streaming and cache refinements

- Skip the disk read for experts already resident in the VRAM expert cache
  (`pgpu_moe_resident`), with a safe CPU fallback if a resident-only expert is
  evicted.
- Fixed the safetensors per-thread file-handle cache so two open model databases
  (target + draft) no longer share handles keyed only by file index.
- Added `ECAP` to pin the entire expert tier resident when the model fits in RAM.

### Documentation

- Reworked `README.md`: at-a-glance badges, a headline **Measured performance**
  table right after the intro, an ASCII **engine schema** (per-token flow and
  memory hierarchy), documentation for INT8 attention and the new environment
  variables, and an honest GPU note for small (4 GB) cards.
- Accuracy fix: routing is described as top-k with per-model expert counts
  (previously "top-4 of 128", which is GPT-OSS-specific; Qwen3-30B-A3B is top-8).

## Measured results

Warm, greedy decode on one six-core AVX2 Windows laptop, 16 GB RAM, internal
NVMe, GTX 1650 4 GB (left idle). These describe one machine and are not hardware
guarantees.

| Model | experts | F32 attention | INT8 attention | |
|---|---|---:|---:|---|
| gpt-oss-20b | INT4 | 1.4 tok/s | **3.3 tok/s** | **+136%** |
| Qwen3-30B-A3B | INT4 | 2.2 tok/s | **2.9 tok/s** | **+32%**; dense resident 4.3 → 1.6 GB |
| gpt-oss-120b | INT3 | 0.5 tok/s | **1.24 tok/s** | 66 GB streamed on 16 GB RAM |

The gain is largest where attention dominates (the 20B). The 120B remains
storage-bound: its decode also scales with drive speed, and moving it to a faster
internal NVMe roughly doubled throughput, consistent with the streaming design.
On the 4 GB GTX 1650 the GPU paths did not beat the optimized CPU path; the
effective levers on this hardware are RAM residency and byte reduction (INT8
attention, INT3 experts).

## Compatibility and upgrade notes

- Existing INT4/INT3 model directories from 0.6.0 remain usable unchanged; their
  attention stays F32.
- INT8 attention is **opt-in**. To use it, either convert with `--dense-bits 8`
  (or `DENSE_BITS=8` for the streaming Qwen converter) or retrofit an existing
  model with `transcode_attn_to_int8.py`, then run against the new directory.
- `DRAFT_MODEL`, `SPEC_K`, `SPEC_PROBE`, `SELF_DRAFT_PROBE`, and `ECAP` are opt-in;
  the standard single-model CPU path remains the reference default.
- The GPT-OSS F32-attention path is unchanged; nothing in the default runtime
  behaves differently unless the new flags/env vars are set.
- Model weights are not included. Users must obtain and convert checkpoints in
  accordance with their original licenses.

## Validation performed for this release

- Clean Windows x64 AVX2/FMA build.
- Built-in forward and math self-tests, including the INT8 matmul path and the
  speculative-verify byte-identity self-test.
- Quality spot-checks with INT8 attention on gpt-oss-20b, Qwen3-30B-A3B, and
  gpt-oss-120b (factual, arithmetic, and multi-turn coherence).
- Python syntax validation for `convert.py`, `convert_streaming_qwen.py`, and
  `transcode_attn_to_int8.py`.

## Known limitations

- Speculative decoding is scaffolding: on a single disk-bound machine it does not
  improve throughput and is off by default.
- INT8 attention quality was validated by spot-checks, not a full per-checkpoint
  logit comparison against the original full-precision model.
- GPU paths remain experimental and, on small (4 GB) cards, do not beat the CPU
  path.
- The server uses one model process and one KV cache; requests are serialized.

## Downloads

- `picchio.exe`: self-contained Windows x64 CPU build. Requires AVX2/FMA. The
  binary identifies itself as v0.7.0 at startup and is unsigned, so Windows
  SmartScreen may display a warning.
- `SHA256SUMS.txt`: checksum for the downloadable binary.
- GitHub-generated source archives: build with `build.bat` on Windows or `make`
  on supported Unix-like systems.

Binary SHA-256:

```text
<fill in after building picchio.exe: sha256sum picchio.exe>
```

## Full change history

Compare: https://github.com/benmaster82/picchio/compare/v0.6.0...v0.7.0
