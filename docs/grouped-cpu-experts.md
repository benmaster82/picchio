# Grouped CPU expert decode

`MOE_GROUPED=1` enables an optional CPU scheduling path inspired by the grouped
matrix-job scheduling in [Pithos](https://github.com/benmaster82/Pithos/blob/a20a66e/src/model.c).
Picchio retains its own quantized formats and arithmetic kernels. No Pithos
kernel code or GGUF format support is imported.

## Scope

In synchronous single-token decode, selected experts normally run one after
another, with an OpenMP parallel region for each gate/up and down matrix. On
Qwen top-8 across 48 layers, that is 768 expert matrix parallel regions per token.
OpenMP may already retain worker threads; the target is dispatch/barrier overhead,
not a claim that every call creates OS threads.

The new path uses one OpenMP team per MoE layer:

1. Compute gate/up row tiles for all selected experts, including their biases.
2. Run the existing clipped GPT-OSS or plain Qwen/MiniMax SwiGLU activation.
3. Compute down row tiles for all selected experts, including their biases.
4. Sum weighted outputs in the original router top-k order on the caller.

Row tiles contain up to 64 output rows. Per-row dot products keep the same
SIMD/scalar accumulation order. In-place activation is sequential within each
expert to avoid overwriting interleaved gate/up inputs still needed by another
activation element. Separate experts can activate concurrently.

F32, INT4 per-row, INT4 gs64 and INT3 gs64 expert weights use the existing
arithmetic through serial row-range entry points. The ordinary matrix APIs
remain parallel and continue to handle batched prefill. INT8 row ranges are also
available to the dispatcher, but this does not add INT8 expert-file loading.

The model owns reusable routing, gate/up and expert-output scratch. Its capacity
is `E + topk * (moe_inter + hidden)` floats, allocated lazily and reused across
layers, turns and RESET. Qwen3-30B-A3B uses 112.5 KiB. Scratch belongs to the
single forward caller; I/O workers do not access it, and separate draft models
have their own allocation. Like the existing model weights it lives until the
model/process is released.

## Interaction with existing paths

- Default is off, and no checkpoint conversion is needed.
- `ASYNC_MOE=1` keeps completion-driven CPU compute so grouping does not impose
  a global wait for the remaining disk reads. Decode scratch still gets reused.
- GPU expert execution, numeric tracing and unsupported expert formats use the
  established compute paths. CPU expert grouping can coexist with GPU attention.
- `IDOT=1` with INT4 experts keeps the existing activation preparation/kernel;
  this change does not introduce a new approximation or shared INT8 activation.
- Oracle dumping remains supported by the grouped path, including intermediate
  expert tensors and weighted contributions, so numerical comparison exercises
  the actual new scheduler.
- Prefetch still uses normal protected cache slots; grouped compute begins only
  after the synchronous loader has finished the selected batch.

`STATS` exposes `moe_grouped_calls`, `moe_scratch_allocs` and `moe_scratch_bytes`.
The exit summary prints grouped layer count and scratch allocation/size when
the grouped compute path ran. An enabled flag with zero grouped calls can mean
an existing async, GPU or IDOT compute path took precedence.

## Validation

```powershell
.\build.bat
python -X utf8 test_moe_grouped.py --exe .\picchio.exe
```

On Linux/macOS use your Python environment and `--exe ./picchio`. The test needs
numpy and safetensors. It creates only temporary synthetic model files.

Tests compare legacy and grouped execution for synthetic GPT-OSS, Qwen and
MiniMax configurations, F32/INT3/INT4 gs64/INT4 per-row experts, and 1/2/4/6 CPU
threads. They require byte-identical oracle tensors (including logits and
routing), greedy token/finish frames, scratch reuse, eviction with a small cache,
KV prefix truncation, RESET, and the production prefill/async/IDOT fallback paths.
The engine self-test also passes with `MOE_GROUPED=1`.

## Local real-Qwen comparison (2026-11-12)

Qwen3-30B-A3B-Instruct-2507, Picchio INT4 gs64 experts, F32 attention, INT8
embedding/head, Windows, i7-10750H, 32 GB RAM. GPU disabled, 4 CPU threads,
4 I/O threads, 4 GiB expert budget, context 512, buffered safetensors, no prefetch
or async MoE, greedy sampling, 64 generated tokens. Four fresh processes ran
in the order reference / grouped / grouped / reference. OS page-cache state was
not reset. Existing applications remained open. Both modes used the same new
executable, differing only in `MOE_GROUPED`.

| Mode | Generation interval reported by the chat bridge | Internal MoE time | Process wall time |
| --- | --- | --- | --- |
| Reference, first | 47.3 s | 30.03 s | 57.61 s |
| Grouped, first | 42.7 s | 25.08 s | 52.20 s |
| Grouped, second | 43.0 s | 25.11 s | 52.61 s |
| Reference, second | 46.5 s | 29.29 s | 55.87 s |

All four responses were identical. Mean generation interval decreased from
46.9 to 42.85 seconds (about 8.6% less time, or 9.5% more tokens per second).
Grouped runs recorded 3,072 grouped layers, one 112.5 KiB scratch allocation,
and 8.02 GB final RSS, the same reported final RSS as the reference. Internal
timers must not be summed as independent wall-clock intervals.

These are two runs per mode on one prompt and one machine, not a broad
performance guarantee. The option stays disabled by default. Additional models,
longer contexts and storage-limited/async workloads still need repeated performance
measurements; the expanded validation below covers more exact CPU settings.

## Expanded release validation (2026-11-12)

The same exact CPU implementation was tested with 4/6 threads and 2/4 GiB expert
budgets on the machine above, now with context 1024. All modes generated identical
raw token IDs, finish reasons and positions. Validation included:

- A 34-token prompt with 32 generated tokens under each thread/cache setting.
- A 393-token prompt with 24 generated tokens at 4 threads / 4 GiB.
- Three conversational turns at 4 threads / 4 GiB. Both modes answered `OK`,
  `ORIONE-42`, and `verde`, reusing 39 and then 75 KV-prefix tokens.
- RESET between workloads and unchanged model-owned scratch across turns.

The matrix contains 16 real-model turns, plus two turns in a repeated short pair.
Prefill and decode were measured separately. Fresh processes were used per mode
and setting; the long and conversational workloads shared their process after
RESET, retaining its expert cache. OS page-cache state was uncontrolled.

| Short workload: threads / cache | Reference total | Grouped total | Reference decode | Grouped decode |
| --- | --- | --- | --- | --- |
| 4 / 4 GiB, repeated pair | 30.94 s | 28.85 s | 15.74 s | 13.57 s |
| 6 / 4 GiB | 27.27 s | 26.25 s | 13.55 s | 12.36 s |
| 4 / 2 GiB | 32.27 s | 29.82 s | 17.09 s | 14.92 s |
| 6 / 2 GiB | 28.21 s | 27.20 s | 15.09 s | 13.83 s |

These are exploratory pairs, not averages from a large benchmark. Six threads
with 4 GiB was fastest among the measured short configurations on this i7;
the result does not override the contributor's different Linux hardware in #8.

Two large timing outliers are retained in the raw records: the initial 4-thread /
4-GiB reference short turn took 240.40 s (224.85 s decode), and the grouped long
turn took 570.57 s (556.93 s prefill). Their causes were not established. The short
pair was repeated, yielding the first table row. No speedup is calculated from
the anomalous original short pair.

For the long workload, reference/grouped decode was 15.10 / 13.63 s, but the full
turn was 186.12 / 570.57 s because of the prefill outlier. Grouped expert decode
does not execute during that batched prefill. The slower full-turn observation
is not hidden or represented as an end-to-end gain. The option stays off by default.

Raw IDs, timings, counters and caveats are in the [validation record](benchmarks/grouped-qwen-windows-2026-11-12.json).
To reproduce with an existing converted model and local tokenizer, activate your
Python model environment and run:

```powershell
python -X utf8 benchmark_qwen_cpu.py --exe .\picchio.exe --model ..\models\qwen3_30b_i4 --output grouped-validation.json
```

On Linux use your Python environment and `--exe ./picchio`. The harness requires
Transformers/tokenizers, uses only local model/tokenizer files, and explicitly
disables IDOT, GPU, prefetch, direct I/O and async MoE. `--threads` and `--cache-gib`
accept lists; `--short-only` omits long/conversational workloads. It retains logs
and partial results if a run fails, with protocol timeouts and owned-process
cleanup. Later runs also record optional Windows process CPU time to help
investigate wall-time stalls; that metric was not present in the original matrix.
Linux/macOS runtime coverage remains outstanding.

## Trying it locally

From the repository in PowerShell, with your Python model environment activated:

```powershell
$env:MOE_GROUPED = '1'
python -X utf8 chat_qwen.py --model ..\models\qwen3_30b_i4 --ctx 512 --pin-gb 4 --threads 4 --io-threads 4 --no-reasoning
```

This uses the standard rebuilt executable. Local developers can select a separate
test executable with `--exe PATH` while an existing chat holds `picchio.exe` open.
Set `MOE_GROUPED=0` for the reference scheduler.

Measure identical prompts/tokens, thread count, cache budget and storage settings.
Alternate the two modes and report startup, generation, attention, MoE, memory
and loader metrics separately. A warm in-RAM compute improvement may disappear
when expert reads dominate; this option is not an unconditional speedup claim.
