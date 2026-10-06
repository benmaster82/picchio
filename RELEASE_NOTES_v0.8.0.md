# Picchio v0.8.0 — MiniMax-M2, a third model family

Picchio v0.8.0 adds **MiniMax-M2** (230 B total, ~10 B active, 62 layers ×
256 experts) alongside GPT-OSS and Qwen3-MoE. At ~122 GB converted it is by far
the largest model the engine has run: it does not fit in RAM on any consumer
machine and streams from disk end to end, which is precisely the case Picchio was
built for.

Support is config-gated on `model_type="minimax"`, like Qwen3 before it, so the
existing families are untouched — verified by the built-in self-test plus
end-to-end generation on all three.

## Highlights

### MiniMax-M2 support

Three architectural switches, each on only for MiniMax checkpoints:

- **Partial RoPE** — only the first `rotary_dim` (64) channels of each 128-wide
  head rotate; the rest pass through untouched.
- **Whole-vector QK-Norm** — a single RMSNorm across the entire concatenated
  multi-head Q (or K) vector, rather than per head as in Qwen3.
- **Sigmoid routing** — `e_score_correction_bias` selects the top-k experts, but
  the mixing weight is the *unbiased* sigmoid score, renormalised over the
  selection. The bias never reaches the weights. Applied in both top-k sites
  (single-token decode and the batched prefill loop); the GPU router is bypassed
  for this path, since its kernel assumes a pre-sigmoid linear bias.

### Converter and chat bridge

- `convert_minimax.py`: GPTQ-INT4 → Picchio INT4 gs64, with `--dense-bits 8` for
  INT8 attention, atomic per-shard writes, and an optional `--delete-source` that
  frees each source shard as it is consumed (destructive; the original checkpoint
  is gone afterwards).
- `chat_minimax.py`: MiniMax reasons unconditionally — its template's generation
  prompt ends in a literal `<think>`, so replies begin *inside* a reasoning block
  and there is no `enable_thinking` switch. The bridge splits the reply on the
  `</think>` token, hides reasoning behind a progress spinner by default
  (`--show-thinking` streams it), and records it through the template's own
  `reasoning_content` field.

### Engine fixes

- `ST_MAX_TENSORS` raised from 32 768 to 131 072. That budget covers the **whole**
  tensor database rather than a single shard; MiniMax's expert tensors alone are
  ~63 k, and the old ceiling stopped registering tensors mid-load without
  reporting an error.
- `cfg_rotary_dim()` falls back to `head_dim` when `rotary_dim` is unset, so
  hand-built configurations (the synthetic self-test model, whose `Cfg` is
  zero-initialised) keep full rotation instead of silently skipping RoPE
  entirely — a defect the self-test could not have caught on its own, since both
  forward paths would have skipped it identically.
- `chat_qwen.py` repaired for `transformers` 5.x, where
  `apply_chat_template(tokenize=True)` returns a `BatchEncoding` instead of a
  list. The bridge was shipping the literal text `input_ids attention_mask` as
  token IDs, and the engine exited on it.

## Measured performance

MiniMax-M2 on a **12-core AVX2 laptop, 32 GB RAM, model on an entry-level NVMe**.
This is a different, larger machine than the 16 GB laptop used for the figures in
the README's main table, so these numbers are not comparable with those:

| Configuration | tok/s | Expert-cache hit |
|---|---:|---:|
| `--pin-gb 12` | 0.34 | 42.9% |
| `+ --async-moe --direct` | 0.40 | 42.9% |
| `+ --pin-gb 20` | **0.48** | 54.6% |

The expert cache is the dominant lever: the experts total ~119 GB, so even 20 GB
holds only ~17% of them while every token touches 496 experts across 62 layers.
Raising `--io-threads` past the default made no difference — the NVMe is not
queue-depth limited. `IDOT=1` gained ~5% but visibly changed the output, which is
expected of the approximate integer kernel and not a good trade. Unlike the other
families, repeat runs do **not** speed up: the learned hot-store converges
immediately and the resident expert set stops changing.

For scale, `gpt-oss-120b` reaches ~2 tok/s on this same machine, because it
streams roughly 4× less expert data per token.

## Upgrade notes

- No action required for existing GPT-OSS or Qwen3 models: converted checkpoints,
  flat stores, and command lines are unchanged.
- The startup banner now reads `GPT-OSS/Qwen3-MoE/MiniMax-M2`.
- Anyone running the chat bridges on `transformers` 5.x should take this release:
  `chat_qwen.py` was broken there before it.
- Model weights are not included. Users must obtain and convert checkpoints in
  accordance with their original licenses.

## Validation performed for this release

- Clean Windows x64 AVX2/FMA build.
- Built-in forward and math self-tests, including the RoPE, INT3/INT4 matmul, and
  pipeline byte-identity checks.
- End-to-end generation regression on all three families after the shared-code
  changes: GPT-OSS-120B (INT3 + INT8 attention), Qwen3-30B-A3B (INT4 + INT8
  attention) and MiniMax-M2 each answered a factual prompt correctly.
- `minimax_forward_check.c` matches the real upstream `MiniMaxM2ForCausalLM` to
  `max|Δlogit| = 1e-6` on the committed fixture (`verify_minimax.py`).
- The MiniMax architecture was additionally cross-checked line by line against
  llama.cpp's own `minimax-m2.cpp`, which agrees on all three switches.
- The converted 230 B checkpoint loads all 64 349 tensors and generates coherent
  text.
- Python syntax validation and import/CLI smoke tests across the project's entry
  points.

## Known limitations

- MiniMax-M2 is I/O-bound on consumer hardware; expect well under 1 tok/s unless
  a large fraction of the experts fits in the cache.
- `convert_minimax.py` emits INT4 experts only. The engine already supports INT3
  (`picchio_expert_bits: 3`), which would cut expert bytes ~22%, but the
  converter does not offer it yet.
- No full-model numeric oracle comparison against `transformers` for MiniMax-M2.
  This is blocked on the library itself: its in-tree MiniMax-M2 support drops the
  checkpoints' `rotary_dim` and applies full-head RoPE
  ([huggingface/transformers#48241](https://github.com/huggingface/transformers/issues/48241)),
  so it is not a trustworthy reference for this architecture today. For the same
  reason the validation fixture is **committed** rather than regenerated, and
  `make_minimax_test_model.py` refuses to run on `transformers >= 5.0` instead of
  emitting a silently wrong oracle.
- Carried over from 0.7.0: speculative decoding remains scaffolding and is off by
  default; GPU paths remain experimental and do not beat the CPU path on small
  (4 GB) cards; the server uses one model process and one KV cache, so requests
  are serialized.

## Third-party code

`reference/minimax_m2/` vendors the upstream MiniMax-M2 modeling code
(Apache-2.0, © 2024-2025 ModelCloud.ai) **byte-identical and unmodified**, used
only to build the validation fixture. It is never compiled into the engine. See
[`reference/README.md`](reference/README.md) for attribution.

## Downloads

- `picchio.exe`: self-contained Windows x64 CPU build. Requires AVX2/FMA. The
  binary identifies itself as v0.8.0 at startup and is unsigned, so Windows
  SmartScreen may display a warning.
- `SHA256SUMS.txt`: checksum for the downloadable binary.
- GitHub-generated source archives: build with `build.bat` on Windows or `make`
  on supported Unix-like systems.

Binary SHA-256:

```text
72767a6d140d111ef97bad42c36a5891e3a89552ca0129614183129c465d4a28  picchio.exe
```

## Full change history

Compare: https://github.com/benmaster82/picchio/compare/v0.7.0...v0.8.0
