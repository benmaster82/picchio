# Picchio v0.6.0 — multi-model MoE streaming and storage-bypass I/O

Picchio v0.6.0 expands the project from a GPT-OSS streaming prototype into a
multi-model MoE runtime. This release adds Qwen3-MoE and INT3 support, introduces
an aligned expert store with direct and completion-driven asynchronous I/O, and
adds a two-node pipeline mode. It also includes substantial multi-turn,
conversion, correctness, and observability improvements.

## Highlights

### Qwen3-MoE support

- Added runtime and converter support for Qwen3-MoE checkpoints, including
  QK-Norm, plain SwiGLU, softmax-normalized top-k routing, and the Qwen rotary
  embedding layout.
- Added `chat_qwen.py`, which renders Qwen's native ChatML template through the
  Transformers tokenizer and uses Picchio's persistent `SERVICE` protocol.
- Added resumable shard-by-shard conversion with
  `convert_streaming_qwen.py`.
- Added an end-to-end synthetic Qwen3-MoE regression test covering conversion,
  safetensors inference, PicchioFlat, direct I/O, asynchronous MoE, and service
  framing.
- Validated the runtime on a converted Qwen3-30B-A3B checkpoint: all 25,013
  tensors loaded and synchronous/asynchronous greedy runs produced identical
  token IDs.

### INT3 expert storage and kernels

- Added group-scaled INT3 expert storage end to end: quantization, conversion,
  loading, self-tests, and AVX2/FMA decode kernels.
- Added `transcode_i4_to_i3.py` to convert existing Picchio INT4 experts to INT3
  without downloading the original checkpoint again.
- Dense tensors retain the precision required by the existing runtime; INT3 is
  applied to the large streamed expert weights.

### PicchioFlat and asynchronous storage pipeline

- Added the version 2 `.picchioflat` format: one contiguous, 4 KiB-aligned
  payload per expert, a resident SHA-256-protected index, and optional per-expert
  payload verification.
- Added `flat_pack.py`, `flat_bench.py`, and `flat_bench_qd.py` for packing,
  byte-verification, and queue-depth benchmarking.
- Added cross-platform direct expert reads with `O_DIRECT` on Linux and
  `FILE_FLAG_NO_BUFFERING` on Windows, with a safe buffered fallback.
- Added completion-driven `ASYNC_MOE`: ready experts can be computed while the
  remaining routed experts are still being read. Contributions are reduced in
  canonical router order, preserving deterministic output.
- Added configurable I/O concurrency through `IO_THREADS` and frontend flags
  `--io-threads`, `--async-moe`, and `--direct`.

### Two-node distributed inference

- Added a persistent TCP pipeline that splits the transformer at a layer
  boundary across two machines.
- Added batched distributed prefill, avoiding one network round trip per prompt
  token.
- Added coordinator-driven sampling and persistent service/chat integration.
- Added loopback and byte-identity tests proving that the distributed layer split
  produces the same tokens as the monolithic forward path.

The current split is primarily a capacity feature. Both nodes need access to the
converted model files, and a network round trip is added to each decode step, so
single-stream speed is not guaranteed to improve.

### Optional GPU and CPU experimental paths

- Added a runtime-loaded CUDA backend that leaves the default CPU build
  dependency-free.
- Added optional GPU `lm_head` and MoE expert-cache prototypes, with automatic
  CPU fallback.
- Added an optional INT8-activation × INT4-weight CPU kernel with AVX-VNNI
  runtime dispatch and an AVX2 fallback (`IDOT=1`).

These paths are experimental and remain disabled by default. Measurements on a
4 GB GTX 1650 Max-Q did not beat the optimized CPU path; GPUs able to keep a much
larger expert set resident are the intended target.

### Chat, server, and runtime reliability

- Fixed long multi-turn GPT-OSS handling and improved KV-prefix reuse.
- Added live reasoning display and improved terminal status reporting.
- Added per-request repetition penalty support and robust expert reads.
- Kept the OpenAI-compatible HTTP server on the same persistent service path.
- Expanded runtime timing, cache, I/O, prefetch, and async-wait statistics.
- Corrected Qwen stop-token handling and service-mode tokenizer behavior.

## Measured results

These measurements describe one six-core Windows test machine and are not
hardware guarantees.

- GPT-OSS-20B INT4, internal NVMe, constrained expert cache: complete
  PicchioFlat plus direct I/O and asynchronous MoE averaged **1.206 tok/s**,
  versus **1.104 tok/s** for asynchronous safetensors (**+9.2%**). Compared with
  one synchronous safetensors reference at 0.842 tok/s, the complete pipeline
  was approximately **43% faster**. All generated token IDs matched.
- Qwen3-30B-A3B, short four-token greedy regression: asynchronous safetensors
  completed in 10.77 seconds versus 12.75 seconds synchronously, approximately
  **+18.4% tok/s**, with identical token IDs.
- GPT-OSS-120B remains strongly storage-bound on an external SSD; previously
  measured steady-state decode was approximately **0.25 tok/s**. Faster storage
  and a larger RAM expert cache remain the most effective improvements.

See `README.md`, `DESIGN.md`, and `DESIGN_STREAMING_IO.md` for the complete test
conditions and interpretation.

## Compatibility and upgrade notes

- Existing supported Picchio INT4 model directories remain usable.
- `.picchioflat` version 1 prototype files are not compatible with this release.
  Regenerate them with the current `flat_pack.py`.
- Models converted by older revisions that stored `embed_tokens` or `lm_head` in
  the legacy packed-U8 layout must be reconverted with the current converter.
- `ASYNC_MOE`, `DIRECT`, `IDOT`, and the CUDA backend are opt-in. The synchronous
  CPU path remains the reference default.
- PicchioFlat files are model-specific generated artifacts and are intentionally
  not included in the release.

## Validation performed for this release

- Clean Windows x64 AVX2/FMA build.
- Built-in forward and math self-test, including INT4, INT3, canonical async-MoE
  reduction, and monolithic-versus-split byte identity.
- Two-stage pipeline self-test over a real loopback TCP socket.
- Python syntax validation for the converters, chat frontends, server, flat-store
  tools, distributed launcher, and smoke tests.
- Synthetic Qwen3-MoE conversion/runtime/service smoke test.
- Complete PicchioFlat expert payload verification against the source
  safetensors containers during development testing.

## Known limitations

- Qwen3-MoE has synthetic oracle coverage and real-checkpoint runtime coverage;
  a full per-checkpoint numerical/logit comparison against the original
  full-precision Transformers model remains future validation work.
- Long Qwen multi-turn prefix-reuse sessions need broader testing.
- Distributed inference reduces resident layer memory per node but introduces
  network latency.
- The server uses one model process and one KV cache; requests are serialized.
- CUDA support requires a separately built `picchio_cuda` library and is not
  included in the standard CPU binary.
- Model weights are not included. Users must obtain and convert checkpoints in
  accordance with their original licenses.

## Downloads

- `picchio.exe`: self-contained Windows x64 CPU build. Requires AVX2/FMA. The
  binary identifies itself as v0.6.0 at startup and is unsigned, so Windows
  SmartScreen may display a warning.
- `SHA256SUMS.txt`: checksum for the downloadable binary.
- GitHub-generated source archives: build with `build.bat` on Windows or `make`
  on supported Unix-like systems.

Binary SHA-256:

```text
ca13b50d4fb1bddc4cd25ddef2247c774214313840eee5415f94051db7cdb51f  picchio.exe
```

## Full change history

Compare: https://github.com/benmaster82/picchio/compare/v0.5.0...v0.6.0
