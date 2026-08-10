# Picchio: Streaming MoE Engine for GPT-OSS (20B/120B)

> *The woodpecker (picchio) drums a hundred times a second on a huge trunk,
> we drum 128 experts on a huge disk.*

**Goal:** run GPT-OSS-120B (117B parameters, MoE) on consumer hardware
(16 GB RAM, NVMe SSD, optional GPU) in pure C, with experts streamed from disk.

License: MIT.

## 0. Status and normative contract (July 2026 revision)

The historical sections that follow describe the initial idea and may contain
outdated estimates. In case of conflict, this section takes precedence. The
actual architecture used by Picchio is: hidden size 2880, 36 layers all MoE,
64 query heads, 8 KV heads, head dimension 64, 128 experts per layer, top-4,
intermediate size 2880, fused gate/up 5760, sliding-window-128 attention
alternated with full attention, `rope_theta=150000`, and a vocabulary of about
201K. GPT-OSS additionally uses attention sinks, YaRN, and a clipped SwiGLU
variant: none of these details can be replaced with an approximation without an
oracle proof.

The current status is **end-to-end runtime and forward pass numerically
validated** against Transformers on the tiny model, both in F32 and in
group-scaled INT4. The real GPT-OSS-120B has also completed multiple
autoregressive steps with KV-cache and finite logits. This validation covers the
runtime and forward pass on raw token IDs; the o200k tokenizer and Harmony are
not yet token-exact and remain a separate suite.

### 0.1 Official fixture

- Model: `tiny-random/gpt-oss`
- Pinned revision: `02ba5c61f879b5a38a8b1f7a8e0409b8e1bb8f38`
- Purpose: architectural debugging; random weights, not an evaluation of language
  quality.
- Primary input: fixed raw token IDs. Tokenizer and Harmony are a separate suite.
- Reference: `transformers==4.57.1`, `torch==2.6.0`, `safetensors==0.6.2`.
- Execution: batch 1, `eval()`, no gradient, no sampling, PILOT disabled,
  repetition penalty 1.0, and deterministic argmax.

### 0.2 Three levels of validation

**L0: Container-exact.** The converter must produce a manifest with name, dtype,
shape, byte count, quantization scheme, group size, and hash. Every mandatory
weight, including expert biases and attention sinks, must exist. Packed INT4,
scales, and sample dequantized rows must match between Python and C within the
declared tolerance. A short read, an unexpected shape, or a silent fallback are
fatal errors.

**L1: Implementation-exact.** Picchio C is compared against a Python oracle that
reads the same Picchio-converted files and reproduces the same dequantization.
The top-k indices and argmax must be identical; for F32 tensors we record max-abs,
max-rel, and cosine error. This level separates runtime bugs from the effects of
lossy quantization.

**L2: Transformers token-exact.** With the same input IDs, Picchio is compared
against Transformers on the original checkpoint. Greedy IDs must match. Because
MXFP4/BF16 → INT4 gs64 is lossy, an L2 mismatch after L0 and L1 have passed must
also report top-1/top-2 margin and logit difference: on its own it does not prove
a C bug.

### 0.3 Checkpoints and mandatory order

For a token at position 0 and layer 0, compare, in order:

1. embedding and pre-attention RMSNorm;
2. Q/K/V before and after bias;
3. Q/K after RoPE/YaRN;
4. KV written, mask range, and attention scores;
5. sink logit, sink softmax mass, head concat, and output projection;
6. residual and pre-MoE RMSNorm;
7. router logits, sorted top-k, and their weights;
8. gate/up with bias, split, clipped SwiGLU, down with bias, and weighted
   contributions;
9. end-of-layer residual;
10. final norm, logits, top-10, top-1/top-2 margin, and argmax.

After the single token: a short sequence on layer 0; positions 127/128/129 for the
sliding window; one token through all layers; a fixed raw prefill; finally 8–32
greedy tokens. Only the first divergent checkpoint is fixed, then the process is
repeated from the start.

Dumps are F32 little-endian or `.npy`, accompanied by JSON with version, revision,
input IDs, position, layer, shape, dtype, hash, and statistics. On the tiny model
full tensors can be saved; on the 120B, hashes and probes are used. OpenMP/SIMD,
cache eviction, hot-store, PILOT, tokenizer, and Harmony are re-enabled and
verified only after L1.

### 0.4 Acceptance criteria for the "correct token" milestone

- Fixture reproducible from the pinned revision, without depending on the 120B
  model.
- No mandatory weight missing or replaced with default values.
- Isolated tests passed for INT4 gs64, embedding, RMSNorm, RoPE/YaRN, softmax with
  sink, routing, and clipped SwiGLU.
- All L1 checkpoints within the recorded tolerances and top-k/argmax identical.
- Identical L1 greedy sequence for at least 32 raw tokens.
- L2 result documented separately; only afterwards are the validated fixes applied
  to the conversion and runtime of GPT-OSS-120B.

### 0.5 Validation results

Validation completed on 29 July 2026:

- layer-by-layer F32 checkpoints within `atol=rtol=1e-5`;
- top-k, router weights, and argmax identical;
- 32 greedy tokens identical between Picchio and Transformers;
- KV-cache and sliding boundary verified at position 130;
- group-scaled INT4 container compared against an identical dequantized reference;
- all 15 real shards validated with the official safetensors parser;
- shard 13, found truncated, rebuilt expert-by-expert with bit-for-bit
  verification of the scales already present;
- GPT-OSS-120B run for multiple autoregressive steps without NaN/Inf.

The old conversion had mistakenly quantized the expert biases. The runtime
supports that format for backward compatibility, but the recommended mode uses the
F32 sidecar produced by `download_expert_biases.py`. Future conversions preserve
the expert biases directly in F32.

### 0.6 Single-turn token-exact chat milestone

The first chat path does not duplicate o200k/Harmony in C. A Python bridge, based
on the official `openai-harmony` library at a pinned version, owns prompt
rendering, tokenization, channel parsing, and decoding. Picchio receives and
produces only raw token IDs; `tok.h` remains an approximate interactive fallback
and is not part of the token-exact contract.

The runtime protocol must offer:

- input from a file of decimal IDs, to avoid depending on the Windows environment
  variable limit; `RAW=1` and `INPUT` remain compatible;
- machine-readable output containing every generated ID, including Harmony and
  stop tokens, with no C-side filtering or decoding;
- stdout reserved for IDs in that mode, and diagnostics on stderr;
- explicit stop on `<|return|>` and `<|call|>`, also reporting the terminator;
  `<|end|>` closes a message but not the entire assistant response;
- reproducible greedy sampling for the first validation.

Acceptance criteria: the IDs rendered by the bridge match official Harmony; the
file→C transport is exact; the full output sequence is parsable by Harmony; the
existing raw mode and the tiny F32/INT4 suites do not regress.

Result: milestone reached on 29 July 2026. The real GPT-OSS-120B answered "Ciao!"
with a 77-token Harmony prompt, 930.34 s prefill, 23 tokens in 251.34 s, about
0.09 tokens/s and 20.3% expert cache hit.

### 0.7 Persistent multi-turn chat milestone

Goal: do not reload the model and do not recompute the prefix already processed.

Measured constraint: Harmony rendering is **not** prefix-preserving between turns.
In the canonical re-render the `analysis` channel is dropped and the final
`<|return|>` becomes `<|end|>`. By keeping the analysis, the divergence is reduced
to just the terminator, with about 88% of positions reusable. The protocol
therefore cannot simply append a delta: the bridge computes the longest common
prefix and the runtime restarts from that position, overwriting the subsequent KV.

Mandatory invariant: every transmitted token must also be consumed by the forward
pass, including `<|return|>` and `<|call|>`, so that `pos` always matches the
number of valid positions in KV. A token emitted but not consumed would make the
next turn numerically wrong.

Service protocol, text lines over a pipe, stdout reserved for the protocol and
stderr for diagnostics:

- `READY <ctx_capacity> <vocab> <stop_ids...>` at startup;
- `TURN <max_new> <keep> <temp> <top_p> <top_k> <n_ids> <ids...>`: reuse `keep`
  positions and consume the new IDs; the three sampling parameters apply to the
  turn (defensive clamp on the C side), making temperature per-request without a
  restart;
- `TOKEN <id>` for each generated token, terminator included;
- `DONE <RETURN|CALL|MAX_TOKENS|CONTEXT_FULL> <n_output> <pos>`;
- `ERROR <code> <fatal> <message>` with validation before mutating the KV;
- `RESET` sets `pos` back to 0 keeping the model and expert cache; `SHUTDOWN`
  exits cleanly.

The real KV capacity is `CTX`: no prompt may exceed it, because beyond that limit
writes would be silently ignored, producing wrong results.

Results from 29 July 2026, verified on the tiny model:

- prefix reuse produces tokens identical to the full prefill, with consistent
  `pos`;
- an out-of-range `keep` is rejected without corrupting the session;
- two consecutive turns reused 68 of 77 positions;
- the tiny F32 and INT4 suites did not regress.

On the real GPT-OSS-120B the persistent session answered correctly, with the
`analysis` and `final` channels separated by the official parser.

### 0.8 Performance: measurements and hardware constraints

Profile of a real turn with a 77-token prompt and 24 generated:

- `t_moe` 983.70 s, of which **565.73 s spent waiting on disk** over 11,569 reads;
- `t_attn` 134.20 s, `t_head` 71.29 s;
- expert cache hit 20.5% over 14,544 requests;
- about 0.09 tokens/s.

Measured memory constraint: 15.83 GB total with about 6.86 GB free. The dense part
occupies 4.46 GB, so the expert cache cannot exceed about 2.4 GB. `PIN_GB` 1 and 2
both produce the minimum of 4 slots per layer; `PIN_GB=3` would bring the total to
7.6 GB, beyond the free memory, causing paging. Increasing the cache therefore
requires more RAM, not just a different parameter.

I/O constraint: the model resides on an external **USB** SSD with a JMicron
bridge, not on the internal NVMe, consistent with the ~253 MB/s observed.

Prefetch on Windows: the PILOT thread does not modify the LRU cache and does not
share the main thread's handles. It opens its own handles, copies the routing
input, computes the top-k of the next layer, and reads the predicted expert ranges
to bring them into the operating system's cache. Shared structures are only read,
so there is no data race that could alter the numeric result; the cache-presence
check is deliberately lock-free and an imprecise outcome costs at most one useless
read. Correctness with prefetch enabled is verified by the tiny suites.

**OpenMP.** The binary was being compiled without `-fopenmp`, so the `#pragma omp`
directives of the matmul kernels in `quant.h` were ignored and all computation
stayed on a single core, on a CPU with 6 physical cores. The `-Wno-unknown-pragmas`
option was hiding the warning. Both were fixed: `-fopenmp` is mandatory and the
warning must remain visible.

Comparative measurements on the same turn, 77-token prompt and 24 generated:

| Configuration | `t_attn` | `t_moe` | `t_head` | disk | phase total |
|---|---|---|---|---|---|
| Baseline, single core | 134.20 s | 983.70 s | 71.29 s | 565.73 s | ~1,189 s |
| OpenMP | 30.98 s | 649.91 s | 19.93 s | 534.69 s | ~701 s |
| OpenMP + PILOT | 30.64 s | 730.05 s | 19.30 s | 614.21 s | ~780 s |

Conclusions: OpenMP is worth about 1.7×, and the MoE net of disk drops from about
418 s to about 115 s. PILOT as conceived here **makes things about 11% worse**,
because it warms the operating system's cache, which has no room here: pages are
evicted and the disk is read twice. That is why prefetch remains optional and off
by default.

Necessary redesign of prefetch: insert the expert directly into the LRU cache with
synchronization, eliminating the second read, and measure how many predicted
experts are actually used, given that the next layer's routing is estimated from
the current layer's hidden state and is therefore approximate.

**Batched prefill (batch-union).** With a 4-slot-per-layer cache and top-4, the
token-by-token path evicts the entire cache at every position, so the prompt
re-reads the same experts many times. Prefill now processes positions in blocks:
for each layer, attention is run in position order, then the routing of all
positions is computed and each expert of the union is read **only once**, reusing
it for all positions that selected it. The math does not change: only the order of
reads changes. The sequential path stays active with oracle dumps, tracing, or
repetition penalty, so the validation suites are unaffected. `test_prefill_batch.py`
verifies that the two paths produce identical tokens.

### 0.10 Precision of embedding and lm_head

The initial container quantized `embed_tokens` and `lm_head` to INT4 gs64, whereas
the official GPT-OSS configuration explicitly excludes them from quantization along
with `self_attn` and `router` (`modules_to_not_convert`). With short generations
the defect stays invisible; on long texts the response collapsed into a mix of
languages and repetitions.

Measurements on a sample of 16,384 rows of the original weights:

| Tensor | INT4 gs64 | INT8 per row |
|---|---|---|
| `lm_head` | 11.04% relative error | 0.99% |
| `embed_tokens` | 15.67% | 3.35% |

With `lm_head` at INT4 the argmax matches only 76.6% of the time and the logit
noise is 8.7% of their standard deviation: the most probable token changes in about
one case out of four. At INT8 the argmax matches 100% and the noise drops to 0.77%.

Adopted fix: INT8 with per-row scale for both tensors, verified by
`check_head_quality.py`. It costs about 0.5 GB of dense part, from 3.20 to 3.71 GB,
versus the ~4.6 GB that would be needed to keep them in F32. The loader recognizes
`I8` tensors with their scales; the INT8 kernels were already present.

Experimental confirmation: the request that previously degenerated now produces a
correct, well-formatted list in Italian with `--temperature 0.7`, and the internal
reasoning drops from 94 to 14 tokens. The defect was not in the sampling nor in the
runtime, which had meanwhile been verified with the oracle up to 300 positions.

**Rule.** Respect the model's quantization exclusion list. The output head projects
onto 201,088 entries: compressing it to 4 bits shifts the token distribution far
more than the average weight error would suggest.

### 0.9 GPT-OSS-20B

Same code, no changes: dimensions, layers, experts, and attention patterns are read
from `config.json`. The 20B has 24 layers and 32 experts per layer, top-4, hidden
2880. The conversion produced 14.0 GB in 856 s, with 3.20 GB dense, 10.75 GB of
experts, and the expert biases already in F32, so without a sidecar.

Comparison on the same Harmony prompt and the same hardware:

| Metric | 120B | 20B |
|---|---|---|
| Cache hit | 20.5% | 56.7% |
| Disk wait | 534.69 s | 97.83 s |
| `t_moe` | 649.91 s | 181.88 s |
| `t_attn` | 30.98 s | 19.63 s |
| `t_head` | 19.93 s | 6.94 s |
| Phase total | ~701 s | ~208 s |

About 1.7 s per token versus about 7 s, so a factor of 4. The structural reason is
cache coverage: 4 slots per layer are worth 3% of 128 experts in the 120B but a
much larger share of the 32 experts in the 20B.

Effect of cache size on the 20B, with a 3.14 GB dense part and 8.37 GB of measured
free RAM:

| `PIN_GB` | slots/layer | cache | hit | disk | phase total |
|---|---|---|---|---|---|
| 3 | 10 | 3.0 GB | 56.7% | 97.83 s | ~208 s |
| 4 | 14 | 4.2 GB | 67.1% | 76.75 s | ~190 s |

Beyond `PIN_GB=4` the resident total would exceed free RAM and cause paging.

**SIMD.** As with OpenMP, the AVX2 paths in `quant.h` are guarded by
`#ifdef __AVX2__` and the build passed no architecture flags, so the kernels used
only scalar code. Added `-mavx2 -mfma`; the binary requires a CPU with AVX2. FMA
changes the rounding order, but the tiny F32 and INT4 suites stay within tolerance
with a maximum error of `3.73e-08`.

Effect on the 20B at `PIN_GB=4`, same prompt and same 43-token output:

| Metric | Without SIMD | With SIMD |
|---|---|---|
| `t_moe` | 162.70 s | 88.49 s |
| `t_attn` | 20.55 s | 13.93 s |
| `t_head` | 6.91 s | 1.33 s |
| disk | 76.75 s | 71.14 s |
| phase total | ~190 s | ~104 s |

Computation net of disk drops from about 113 s to about 32.6 s, and the pure MoE
from about 85.9 s to about 17.4 s. The bottleneck therefore returns to I/O, which
weighs 71 s out of 104, i.e. 68%: the next useful interventions are more RAM for
the expert cache, the model on internal NVMe, and a prefetch that populates the LRU
cache directly.

**Methodological lesson.** Two of the biggest gains came not from new code but from
missing compilation flags, `-fopenmp` and `-mavx2 -mfma`, with the warnings
silenced. Before optimizing, verify that the existing code is actually compiled.

**Sampling and conversational use.** Greedy decoding, used for the determinism of
validations, can enter loops: on an open-ended question the 20B consumed 200 tokens
in the `analysis` channel without reaching `final`. With `--temperature 0.7` the
same question produced a correct answer. `chat.py` therefore exposes
`--temperature`, `--top-p`, `--top-k`, and `--seed`, keeping the greedy default so
as not to alter the tests.

Real two-turn conversation on the 20B: the second turn reused 279 of 294 positions,
i.e. 95%, processing only 15, and the answer was correct. The cache hit rises with
use thanks to the hot-store, from 67.1% to 79.2% in the same session.

**Model location.** Copying the container from the external SSD measured 52.7 MB/s
sequential, a value consistent with a USB 2.0 link and not with the drive's
performance. Moving the 20B to the internal NVMe, all else equal (same 120 forwards,
43 tokens, and 67.1% cache hit):

| Metric | USB SSD | Internal NVMe |
|---|---|---|
| disk | 71.14 s | 37.22 s |
| `t_moe` | 88.49 s | 55.35 s |
| phase total | ~104 s | ~73.5 s |
| load | 8.4 s | 4.3 s |

The MoE net of disk stays unchanged, about 18 s, so the gain is entirely I/O. The
disk drops from 68% to 51% of the time. Summary of the 20B path: from ~208 s
(without SIMD, on USB) to ~73.5 s, i.e. 2.8×. With 768 total experts, about 9.5 GB,
full residency is reachable with more RAM, a condition in which the disk leaves the
critical path. The 20B response ends with `RETURN`, so the full Harmony loop,
terminator included, is verified on the real model.

### 0.11 Parallel expert reads (queue depth > 1)

Loading experts from disk was entirely serial: both decode (`moe_forward`) and
prefill (`forward_prefill`) read one expert at a time, keeping the disk queue at
depth 1. A parallel read path was introduced:

- `st.h` uses per-thread (TLS) handles on Windows, so multiple concurrent
  `ReadFile` calls are safe; the main thread keeps the sequential handle for dense
  loading. On POSIX, `pread` on a shared fd is already thread-safe.
- `cache_load_batch` first resolves the hits (marking them as used, so they are not
  evicted), serially reserves one slot for each miss, then reads the misses in
  parallel with OpenMP. Same math, same bytes in the same slots: only the
  concurrent order of the reads changes. Controls: `IO_THREADS` (default 4),
  `PIPE=0` forces serial, `ECAP` overrides the slots per layer.

Validated token-exact on the tiny F32/INT4 suites (even under eviction with
`ECAP=4`) and on the real GPT-OSS-20B (the exact same number of reads, 1355, in
every configuration).

Warm-vs-warm benchmark, 3 pairs, 71-token Harmony prompt, 6 generated, greedy:

| Disk | Serial disk wait | Parallel disk wait | serial t_moe | parallel t_moe |
|---|---|---|---|---|
| USB SSD (JMicron) | ~54.7 s | ~49.3 s (−10%) | ~68.6 s | ~63.4 s |
| NVMe (Kingston) | ~27.5 s | ~23.4 s (−15%) | ~41.7 s | ~38.2 s |

Parallelism pays off more on NVMe (−15% vs −10%): real command queuing benefits
from QD>1, the SATA-USB bridge does not. The gain remains limited, however, because
the workload is **bandwidth-bound**: ~1355 misses × ~13 MB ≈ 17.6 GB read at
~320 MB/s (USB) or ~494 MB/s (NVMe), i.e. at the sequential ceiling of the
respective disks. Reducing the *bytes* read therefore matters more than
parallelizing or coalescing them; see the `PIN_GB` sweep below.

### 0.12 Real RSS on Windows and PIN_GB sweep

`rss_gb` returned 0 on Windows, the primary platform: the tuning of `PIN_GB` was
blind. With `GetProcessMemoryInfo` (WorkingSetSize) the measurement is now real,
and it revealed that the historical estimates greatly overstated residency. In
particular the claim in §0.8 that `PIN_GB=3` would cause paging was based on the
broken measurement: on the 20B the RSS stays well below the 15.83 GB physical even
with much larger caches.

Measured sweep on the 20B (D=2880, 24 layers, 32 experts/layer), same greedy
prompt; the clean metric is the number of misses (deterministic, independent of the
OS page cache):

| PIN_GB | slots/layer | RSS | hit % | disk reads |
|---|---|---|---|---|
| 4 (old default) | 14 | 7.2 GB | 81.7% | 1355 |
| 5 | 17 | 8.4 GB | 85.6% | 1065 |
| 6 | 21 | 9.3 GB | 88.9% | 821 |
| 7 | 25 | 9.2 GB | 90.7% | 684 |
| 8 | 28 | 8.5 GB | 91.2% | 648 |
| **9** | **32 (full)** | **7.7 GB** | **91.4%** | **634** |
| 10–12 | 35–43 | ~7.5 GB | 91.4% | 634 |

From `PIN_GB=4` to `9` the misses collapse by 53% (1355→634). The plateau is at
`PIN_GB=9`, where the cache holds all 32 experts/layer: beyond that, the extra slots
stay empty because there are only 32 experts per layer. The floor of 634 reads is
the incompressible cost of the cold-load of each expert routed once; it is amortized
only by keeping the process alive (multi-turn SERVICE mode), where subsequent turns
approach 100% hit.

Adopted fixes: the `PIN_GB` default is now **adaptive to physical RAM**
(`GlobalMemoryStatusEx` on Windows, `sysconf` on Linux). Without an override,
experts get all the RAM except a 6 GB reserve for dense/KV/OS; the cache
self-limits to `n_experts` slots per layer (never more slots than experts). On this
machine (15.83 GiB) the budget comes out to ~10.6 GB → 32 slots/layer, i.e. **full
residency of the 20B automatically**, with RSS ~6.6 GB. It scales up on larger
machines and lowers itself on small ones; an explicit `PIN_GB` is still honored for
tuning.

**Scaling.** Streaming exists because the model does not fit in RAM; it is
therefore the *hard case*, not a limit of the architecture. On better hardware every
axis improves independently: more RAM raises the hit rate (for the 120B, ~66 GB of
experts, 64–128 GB is needed to approach the 95%+ estimated in §7.2); a Gen4/5 NVMe
(5–7 GB/s vs ~320 MB/s of USB) cuts the residual cold-loads by 15–20×; more cores
scale computation; the GPU tier (§5) is planned. Dimensions are read from
`config.json`: 20B and 120B run with the same code.

### 0.13 Prefetch → LRU: prediction, mechanism, and results

The old PILOT only warmed the OS cache (double read, −11% with little RAM). It was
rewritten to **populate the LRU directly**, with a single-writer design that avoids
any race: the main thread (the only one that mutates the LRU) predicts the experts
of layer L+1, reserves the slots, and hands them to a thread that reads them from
disk during the attention of L+1; `slot->loading` (release/acquire) synchronizes
the buffer and its visibility. `prefetch_issue` reserves at most `ecap − topk`
slots, so there are always enough non-loading slots for the real routing. Enabled
with `PREFETCH=1` (default off), verified token-exact vs prefetch off on the tiny
F32/INT4 suites, even under eviction.

**Prediction accuracy** (`PREDICT_PROBE=1`, answers the open question in §0.8).
Predicting the top-4 of L+1 from the hidden state of L, measured on the 20B (8,372
samples, random ~12.5%): proxy A (pre-MoE norm of L) **81.1%**, proxy B (post-MoE
state of L normalized with the post_ln of L+1, missing only the attention of L+1)
**89.4%**. Proxy B is used: ~3.58 of 4 experts correct, ~10% of reads wasted.

**Results on decode** (20B, `PIN_GB=4`, constrained cache, greedy). The accounting
metrics improve a lot: hit 83→94%, `t_edisk` on the main thread from 50 to 19 s
(USB) and from 26 to 8 s (NVMe), because the reads migrate to the prefetch thread.
But the **wall-clock `t_moe` does not follow**:

| Disk | `t_moe` OFF | `t_moe` ON |
|---|---|---|
| USB SSD | ~64 s | ~68 s (worse, +6%) |
| NVMe | ~42.9 s | ~37–43 s (0 to −14%) |

Two structural causes: (1) in **decode the attention of a single token is too
short** to hide a ~13 MB load, so the main thread waits anyway (`slot_wait_ready`)
and the wait stays in the wall-clock; (2) on **single-queue USB** the two concurrent
readers get in each other's way (penalty), whereas on NVMe the real queue cancels it
(neutral/slight gain). The mechanism is therefore correct and safe, but its value on
decode is limited by the short overlap windows, on both disks.

**Where the value is unlocked: prefill.** Prefill processes many positions per layer
→ large overlap windows. Moreover the set of experts of the layer (`uniq[]`) is
**known before** loading them: the prefetch is therefore EXACT, without prediction.
`forward_prefill` now uses a double-buffer pipeline: halved blocks (two in cache),
and while the current block is being computed (matmul over n positions, disk idle)
the next block is loaded by the prefetch thread (with internal IO_THREADS
parallelism). Token-exact vs prefetch off on the tiny suites (even under eviction,
multiple blocks).

Measurement on the 20B, 108-token prompt, `ECAP=8` (constrained cache, a proxy for
the 120B's pressure), `REP=1`, **on USB disk**:

| | t_moe (≈ prefill) | disk reads (main) |
|---|---|---|
| prefetch OFF | ~47.2 s | 1084 |
| prefetch ON  | ~40.2 s (**−15%**) | ~535 |

Consistent gain (both ON runs below both OFF runs) **even on USB**, unlike decode:
in prefill the compute window per block is large enough to hide the load of the next
block. The theoretical floor is `max(disk, compute)`; the single-level pipeline
captures part of it, with room to go deeper.

**Note.** Batched prefill (and therefore the pipeline) is enabled only with `REP=1`:
with the default repetition penalty (1.1) prefill falls back to the sequential path.
The penalty has no effect on the encoding of a fixed prompt, so enabling batched
even with the penalty active is a future optimization. On the 120B (128 experts,
small ecap → many blocks) this pipeline is the primary candidate to bring down the
~930 s prefill.

The other lever that gets around the bandwidth wall on any disk remains **reducing
the bytes read** (cold experts in a more compressed format).

---

## 1. Analysis of the GPT-OSS-120B architecture

> **Historical note (v0.1).** Sections 1–10 are the original founding document.
> Several figures below are *early estimates* made before the real `config.json`
> was read and are now known to be wrong (e.g. hidden size is 2880, not 6144;
> experts are ~12.4 MB, not ~113 MB). They are kept for the historical record; the
> authoritative numbers are in Section 0.

### 1.1 Fundamental numbers

| Property                      | Value                            |
|-------------------------------|----------------------------------|
| Total parameters              | 116.8B                           |
| Active parameters per token   | 5.1B                             |
| Layers                        | 36                               |
| Total experts per MoE layer   | 128                              |
| Active experts per token      | 4                                |
| Hidden dimension (D)          | 6144 (estimated from param count)|
| MoE intermediate (I_moe)      | ~12288 (estimated, 2×D)          |
| Dense intermediate (I_dense)  | ~24576 (estimated, 4×D)          |
| Attention heads               | 48 (estimated)                   |
| GQA groups                    | 8                                |
| KV heads                      | 6 (48/8)                         |
| Head dim                      | 128 (estimated, D/heads)         |
| Context length                | 128K                             |
| Vocabulary                    | ~200K (o200k_harmony)            |
| Native quantization           | MXFP4 (MoE), BF16 (rest)         |

> **Note:** the exact dimensions (hidden, heads, intermediate) are read from the
> model's `config.json` at runtime, as Colibri does. The values above are
> reasonable estimates based on the parameter count and the documentation.

### 1.2 Layer structure

GPT-OSS-120B uses an architecture with **alternating dense and MoE layers**.
The first N layers are dense (classic FFN), the rest are MoE.
From the documentation, the attention pattern alternates between:
- **Dense attention** (full causal)
- **Locally banded sparse attention** (local window)

This is similar to GPT-3 and different from GLM (which uses MLA + DSA).

### 1.3 Attention: GQA (not MLA)

Unlike GLM which uses Multi-head Latent Attention (MLA) with KV compression to 576
float/token, GPT-OSS uses standard **Grouped Query Attention**:
- 48 query heads
- 6 KV heads (group size 8)
- Head dim 128
- Standard RoPE (not interleaved)

**KV-cache per token:** 2 × 6 × 128 = 1536 float × 36 layers = 55,296 float
At BF16: 55,296 × 2 bytes = ~110 KB/token. For 4K tokens: ~430 MB.
For 128K: ~13.5 GB (significant, careful management needed).

### 1.4 MXFP4 weight format

The MoE weights use **MXFP4** (Microscaling FP4, OCP standard):
- Each value is an FP4 (4 bits: 1 sign + 2 exp + 1 mantissa), range ±6.0
- 2 values packed per byte (`tensor.blocks`, uint8)
- A scale shared per block of 32 elements (`tensor.scales`, E8M0 or FP8)
- The scale is along the last dimension

This differs from Colibri's symmetric INT4 (which uses the range [-8,7] with an
offset). Dedicated MXFP4 dequantization kernels are needed.

**Alternative:** convert MXFP4 → symmetric INT4 at conversion time, losing
~0.1-0.3% of quality but reusing Colibri's fast kernels. This is the recommended
choice for the first version.

### 1.5 Expert size

An MoE expert has 3 matrices (gate_proj, up_proj, down_proj):
- gate: [I_moe, D] = [12288, 6144]
- up:   [I_moe, D] = [12288, 6144]
- down: [D, I_moe] = [6144, 12288]

Parameters per expert: 3 × 12288 × 6144 = ~226M parameters.
At INT4 (0.5 byte/param): ~113 MB per expert.
At MXFP4 (0.5 byte/param + scale): ~113 MB + scale.

**Total experts on disk:** 128 experts × ~30 MoE layers × 113 MB ≈ **430 GB** at
MXFP4.

> If the MoE intermediate dimension is smaller (e.g. 8192), the experts drop to
> ~75 MB each and the total to ~290 GB. The config.json will tell.

### 1.6 Resident dense part

The dense part includes:
- Embedding + lm_head: ~200K × 6144 × 2 ≈ 2.4B params
- Attention for 36 layers: Q/K/V/O projections
- Dense layers (classic FFN in the first layers)
- Shared experts (if present, to be verified in the config)
- LayerNorm / RMSNorm weights

Estimate at INT4: **~3-5 GB** resident in RAM.
At BF16: **~8-10 GB**.

---

## 2. Engine architecture

### 2.1 Memory hierarchy (like Colibri)

```
┌─────────────┐
│   GPU VRAM   │  Tier 0: "hottest" experts + dense (optional)
├─────────────┤
│   RAM        │  Tier 1: resident dense + expert LRU cache
├─────────────┤
│   NVMe/SSD   │  Tier 2: all experts (cold storage, streaming)
└─────────────┘
```

**Fundamental policy (from Colibri):** placement decides ONLY speed, never the
precision or the routing semantics. The output is identical regardless of where the
experts reside.

### 2.2 Per-token pipeline

```
For each layer l in [0, N_layers):
  1. RMSNorm(input)
  2. GQA Attention
     a. Q = x @ Wq               (48 heads × 128 dim)
     b. K = x @ Wk               (6 KV heads × 128 dim)
     c. V = x @ Wv               (6 KV heads × 128 dim)
     d. RoPE(Q, K, pos)
     e. Update KV-cache[l]
     f. scores = Q @ K^T / sqrt(128)
     g. If sparse layer: apply banded mask
     h. attn = softmax(scores) @ V
     i. out = attn @ Wo
  3. Residual: h = h + out
  4. RMSNorm(h)
  5. If DENSE layer:
     a. FFN: SiLU(gate(x)) * up(x) → down → out
  6. If MoE layer:
     a. ROUTE: router(x) → top-4 experts with weights
     b. UNION: gather unique set of experts (per batch)
     c. PLACE: look up in VRAM → RAM cache → disk
     d. LOAD: load missing experts (coalesced pread)
     e. COMPUTE: SiLU(gate_e(x)) * up_e(x) → down_e → weighted
     f. SHARED: shared expert (if present) always resident
     g. out = weighted sum of experts + shared
  7. Residual: h = h + out

Final head:
  RMSNorm(h) → lm_head → logits → sampling
```

### 2.3 Piloted prefetch (PILOT)

Like Colibri, a separate thread runs the routing of layer L+1 while layer L is
computing, and issues `pread`/`posix_fadvise` on the predicted experts. With 4
active experts per token (vs 8 in GLM), predictability may differ, to be measured.

### 2.4 Per-layer LRU cache

Each MoE layer has a pool of reusable `ESlot`s (expert slots). The policy is LRU:
the least recently used expert is evicted. In addition, a "learned" hot-store
(.picchio_usage) tracks per-expert frequency and automatically pins the hottest
ones.

### 2.5 Dual-SSD

Same concept as Colibri: if there is a second SSD, the experts are distributed
across the two drives with a deterministic hash weighted by bandwidth.

---

## 3. Main data structures (C)

```c
/* ── Configuration (read from config.json) ── */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, n_experts, topk;
    int moe_inter, dense_inter, head_dim;
    int first_dense;          /* first N dense layers (no MoE) */
    int vocab;
    int ctx_len;              /* max context (128K) */
    int stop_ids[8], n_stop;
    float eps, theta;         /* RMSNorm epsilon, RoPE theta */
    float routed_scale;       /* scale factor for expert routing */
    int8_t attn_type[128];    /* per layer: 0=dense, 1=banded sparse */
} Cfg;

/* ── Quantized tensor [O, I] ── */
/* fmt: 0=F32, 1=INT8, 2=INT4, 3=MXFP4, 4=BF16 */
typedef struct {
    int fmt;
    float *qf;               /* F32 data */
    int8_t *q8;              /* INT8 data */
    uint8_t *q4;             /* INT4/MXFP4 packed data */
    float *s;                /* scale per row (INT8/INT4) or per block (MXFP4) */
    int O, I;                /* output × input dimensions */
    int block_size;          /* MXFP4: scale block size (32) */
} QT;

/* ── Layer ── */
typedef struct {
    float *in_ln, *post_ln;  /* RMSNorm weights */

    /* GQA Attention */
    QT wq, wk, wv, wo;      /* Q/K/V/O projections */

    int sparse;              /* 0=dense layer, 1=MoE layer */

    /* Dense FFN (sparse==0) */
    QT gate_proj, up_proj, down_proj;

    /* MoE (sparse==1) */
    float *router;           /* router weights [n_experts, D] */
    float *router_bias;      /* correction bias (if present) */
    QT sh_gate, sh_up, sh_down; /* shared expert (if present) */
} Layer;

/* ── Expert slot (reusable, LRU cache) ── */
typedef struct {
    int eid;                 /* expert ID (-1 = empty) */
    QT g, u, d;             /* gate/up/down projections */
    uint8_t *slab;           /* coalesced buffer for pread */
    float *fslab;            /* scale buffer */
    int64_t slab_cap;
    uint64_t last_used;      /* timestamp for LRU */
} ESlot;

/* ── KV-Cache ── */
typedef struct {
    /* GQA: K and V for each KV head, per layer */
    /* K[layer][pos][kv_head * head_dim] */
    /* V[layer][pos][kv_head * head_dim] */
    float **K, **V;          /* [n_layers][max_pos * n_kv_heads * head_dim] */
    int max_pos;             /* allocated positions */
    int cur_pos;             /* current position */
} KVCache;

/* ── Model ── */
typedef struct {
    Cfg c;
    /* shards S; */          /* safetensors reader */

    QT embed, lm_head;
    float *final_norm;
    Layer *L;                /* [n_layers] */

    KVCache kv;

    /* Per-layer expert cache */
    ESlot **ecache;          /* [n_layers][ecap] */
    int *ecn;                /* experts cached per layer */
    int ecap;                /* cache capacity per layer */

    /* Learned hot-store */
    ESlot **pin;             /* pinned experts per layer */
    int *npin;
    uint32_t **eusage;       /* persistent counters */
    uint32_t **eheat;        /* recent heat */

    /* Current working set */
    ESlot ws[32];            /* max topk * batch experts in flight */

    /* Statistics */
    uint64_t eclock, hits, miss, ereq;
    uint64_t n_fw, n_emit;
    double t_edisk, t_emm, t_attn, t_head;
    int64_t resident_bytes;
} Model;
```

---

## 4. File layout of the converted model

```
/path/to/gptoss_i4/
├── config.json              ← copied from the original
├── tokenizer.json           ← o200k_harmony
├── params.json              ← conversion metadata
│
├── dense.safetensors        ← embed + lm_head + attention + dense FFN
│                               (all at INT4 or BF16, ~4 GB)
│
├── experts-00.safetensors   ← experts of layers 0..5 (sharded for parallelism)
├── experts-01.safetensors   ← experts of layers 6..11
├── ...                      ← ~6 shards of ~50-70 GB
│
├── .picchio_usage           ← routing counters (updated every turn)
└── .picchio_kv              ← persistent KV-cache (optional)
```

Each expert is stored as 3 contiguous tensors:
```
model.layers.{L}.mlp.experts.{E}.gate_proj.weight     → uint8 packed
model.layers.{L}.mlp.experts.{E}.gate_proj.weight.qs   → float32 scale
model.layers.{L}.mlp.experts.{E}.up_proj.weight        → uint8 packed
model.layers.{L}.mlp.experts.{E}.up_proj.weight.qs     → float32 scale
model.layers.{L}.mlp.experts.{E}.down_proj.weight      → uint8 packed
model.layers.{L}.mlp.experts.{E}.down_proj.weight.qs   → float32 scale
```

Contiguity in the file is crucial: a single `pread` loads the whole expert.

---

## 5. Project modules

> **Historical note.** The tree below is the original v0.1 plan. The real project
> is a flatter layout (see the "Files in this repository" section of `README.md`);
> several files listed here (`tier.h`, `attn.h`, `pilot.h`, `backend_cuda.*`, the
> `web/` dashboard) were never split out (their logic lives inside `picchio.c`)
> or remain unbuilt future work.

```
picchio/
├── DESIGN.md                ← this document
├── Makefile                 ← build + check + clean
├── c/
│   ├── picchio.c            ← main engine (forward pass, MoE loop, decode)
│   ├── st.h                 ← safetensors reader (like Colibri)
│   ├── tok.h                ← o200k_harmony tokenizer
│   ├── json.h               ← minimal JSON parser
│   ├── tier.h               ← VRAM/RAM/disk hierarchy, LRU, hot-store
│   ├── quant.h              ← quantization kernels: INT4, INT8, MXFP4, IDOT
│   ├── attn.h               ← GQA attention + RoPE + KV-cache
│   ├── simd.h               ← SIMD primitives: AVX2, AVX-512, NEON
│   ├── pilot.h              ← piloted prefetch (separate thread)
│   ├── backend_cuda.h/.cu   ← optional VRAM tier
│   ├── convert.py           ← HF MXFP4 → INT4 container conversion
│   ├── openai_server.py     ← OpenAI-compatible API gateway
│   ├── setup.sh             ← build + self-test
│   └── tests/
│       ├── test_matmul.c    ← kernel validation vs float reference
│       ├── test_attn.c      ← GQA validation vs torch
│       └── oracle.py        ← generates reference tokens from transformers
├── web/                     ← browser dashboard (optional, phase 2)
└── docs/
    ├── benchmarks.md
    └── tuning.md
```

---

## 6. Incremental development plan

### Phase 1: "Correct token" (weeks 1-3)
- [ ] safetensors reader (`st.h`): reusable from Colibri with adaptations
- [ ] config.json parser → `Cfg` struct
- [ ] o200k_harmony tokenizer (wrapper of the .json with BPE)
- [ ] Load dense part (embed, attention, FFN) at BF16/INT4
- [ ] Dense forward pass: RMSNorm → GQA → FFN → residual
- [ ] GQA KV-cache
- [ ] Standard RoPE
- [ ] Validation: token-exact vs `transformers` on a test prompt
- [ ] **Milestone:** generate the first correct token

### Phase 2: "Working MoE" (weeks 3-5)
- [ ] Router: top-4 expert selection (sigmoid/softmax + top-k)
- [ ] Expert loading from disk (coalesced `pread`)
- [ ] Per-layer LRU cache
- [ ] Resident shared expert (if the model has one)
- [ ] MoE validation: expert routing identical to transformers
- [ ] **Milestone:** generate coherent text (even if slow)

### Phase 3: "Speed" (weeks 5-8)
- [ ] SIMD kernels: AVX2/NEON for INT4/INT8 matmul
- [ ] IDOT (integer dot-product: quantize activations → int8)
- [ ] Piloted prefetch (PILOT thread)
- [ ] Asynchronous I/O (PIPE: thread pool for parallel pread)
- [ ] Batch-union (each unique expert read only once per batch)
- [ ] Optional O_DIRECT
- [ ] Learned hot-store (.picchio_usage)
- [ ] **Milestone:** >0.5 tok/s on consumer NVMe

### Phase 4: "Production" (weeks 8-12)
- [ ] MXFP4 → INT4 converter (Python, shard-by-shard)
- [ ] OpenAI-compatible API server
- [ ] Sampling: temperature, top-p, top-k, repetition penalty
- [ ] Harmony chat template
- [ ] Persistent KV-cache on disk
- [ ] Dual-SSD
- [ ] Profiling and tuning dashboard
- [ ] **Milestone:** end-to-end demo, interactive chat

### Phase 5: "GPU and beyond" (optional)
- [ ] VRAM tier with CUDA backend
- [ ] Experts resident in GPU
- [ ] Metal backend for Apple Silicon
- [ ] Speculative decoding (if the model has a draft head)

---

## 7. Performance estimates

> **Historical note.** These estimates use the early (wrong) ~113 MB expert size.
> With the real ~12.4 MB experts the I/O per token is roughly 9× smaller; see
> Section 0 for measured numbers.

### 7.1 Scenario: Desktop 32 GB RAM, NVMe 3.5 GB/s

**Resident dense part:** ~4 GB (INT4), leaves ~25 GB for expert cache.
**Experts per token:** 4 experts × ~113 MB = ~452 MB per MoE layer.
**MoE layers:** ~30. Unique experts per token: up to 4×30 = 120 (but with overlap).

Cold case (everything from disk):
- 120 experts × 113 MB = ~13.5 GB of reads per token
- At 3.5 GB/s: ~3.9 seconds per token → **~0.26 tok/s**

Warm case (50% cache hit):
- ~6.8 GB of reads → ~1.9 s/tok → **~0.5 tok/s**

Hot case (90% cache hit, after warm-up):
- ~1.35 GB of reads → ~0.4 s/tok → **~2.5 tok/s**

With PILOT prefetch and asynchronous I/O: +30-50% → **~1-3.5 tok/s**

### 7.2 Scenario: 64 GB RAM, everything pinned

Total experts for 30 MoE layers × 128 = 3840 experts × 113 MB = ~430 GB.
They don't all fit, but with 60 GB available ~530 experts (the hottest) are pinned.
If the routing is concentrated (as measured for GLM: a few dominant experts), the
cache hit rate can exceed 95% → **3-5 tok/s CPU-only**.

### 7.3 Scenario: GPU RTX 4090 (24 GB VRAM) + 32 GB RAM

~210 experts in VRAM + dense on GPU.
For most tokens: 0 disk accesses → **5-10 tok/s** (estimated).

### 7.4 Comparison with GLM (Colibri)

GPT-OSS-120B is much more favorable for streaming:
- Experts ~7× smaller (113 MB vs 19 MB, to be verified with real dims)
- Only 4 active experts vs ~8
- Fewer MoE layers (~30 vs 75)
- **Cache miss ~15× cheaper** in terms of I/O

> If the real MoE intermediate dimensions are ~8192 instead of 12288, the experts
> drop to ~75 MB and everything improves further.

---

## 8. Key differences from Colibri

| Aspect | Colibri (GLM) | Picchio (GPT-OSS-120B) |
|---|---|---|
| Attention | MLA (compressed KV, 576 float/tok) | Standard GQA (1536 float/tok) |
| KV-cache | Ultra-compressed, on-the-fly reconstruction | Standard GQA, larger but simpler |
| RoPE | Partial interleaved | Standard |
| Router | Sigmoid + noaux_tc + bias | To be determined (probably softmax top-k) |
| Experts/token | ~8 | 4 |
| Experts/layer | 256 | 128 |
| Weight format | Symmetric INT4 per-row | Native MXFP4 (we'll convert to INT4) |
| Speculative | Native MTP head | To be verified (may not have one) |
| Sparse attention | DSA lightning indexer | Banded (local window, simpler) |
| Model size | ~370 GB (int4) | ~60-430 GB (depends on intermediate dim) |

---

## 9. Open design decisions

> **Historical note.** Most of these were resolved during development; see Section 0
> for the actual architecture (hidden 2880, softmax-with-bias router, sliding-window
> 128, Harmony chat, no shared expert relied upon).

1. **Conversion format:** native MXFP4 or symmetric INT4?
   - Recommendation: INT4 for v1 (reuse Colibri kernels), native MXFP4 for v2

2. **Real model dimensions:** the exact numbers from config.json are needed.
   Download and inspect `openai/gpt-oss-120b` on HuggingFace.

3. **Shared expert:** GPT-OSS may not have a shared expert like GLM. To be verified
   in the model code.

4. **Router type:** sigmoid with bias (like GLM) or classic softmax? To be read in
   the `gpt_oss/torch/model.py` code.

5. **Banded attention pattern:** what is the window size? Exact alternation? Impacts
   the KV-cache (we can discard tokens beyond the window).

6. **Chat format:** the model requires harmony encoding; the correct prompt
   templates are needed.

---

## 10. Dependencies and build requirements

**Runtime (zero dependencies):**
- C compiler: gcc ≥ 9 or clang ≥ 12, with OpenMP
- POSIX: pread, posix_fadvise, mmap, pthread
- Optional: CUDA toolkit (for the VRAM tier)

**Build-time:**
- make
- Python 3.10+ (only for the converter and the API server)

**Target platforms:**
- Linux x86_64 (primary)
- macOS arm64 (Apple Silicon, NEON)
- Windows x86_64 (MinGW/MSVC, secondary)

---

*Founding document v0.1, Picchio Project, July 2026*
