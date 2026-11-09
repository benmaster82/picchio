# Issue #8: routing statistics and expert-cache behaviour

Investigation of [issue #8](https://github.com/benmaster82/picchio/issues/8),
against source commit `a86393b`, 2026-11-09. This is a source review and a
benchmark protocol, not a new performance measurement.

## Confirmed behaviour

In `picchio.c`:

- `hotstore_save` writes per-layer `eusage` counters to `.picchio_usage`.
- `hotstore_load` restores counters and reports their sum. It does not call
  an expert loader or populate cache slots.
- Decode and batched prefill increment `eusage` and `eheat`. No cache-placement
  decision reads either counter. `eheat` is not decayed or used for promotion.
- `pin` and `npin` are allocated, and lookup paths check them, but no path
  populates the pinned set. `npin` remains zero.
- `cache_lookup` evicts by `last_used`. Batch loading protects selected slots;
  optional prefetch inserts predicted experts into the ordinary LRU. Neither
  mechanism chooses experts using persisted frequency counts.
- Startup creates empty expert slots before loading the routing statistics.

The issue's hot-store observation was correct in the reviewed commit. Documentation
and code comments described intended preloading/pinning as existing behaviour.
`--pin-gb` configures an LRU capacity budget in GiB, not a quota
of permanently pinned experts.

## Implemented optional warm start

`HOTSTORE_PRELOAD=N` now loads up to N previously used experts per owned layer
before worker threads start. Default is zero; N is clamped to existing LRU
capacity. Higher frequency wins, with lower expert ID breaking ties. Zero-count
experts are skipped. Loading colder selected entries first leaves the hottest
entry most recent. Normal LRU eviction remains active and no pinned slots are
allocated. Startup time and cumulative loader counters include this work; its
elapsed time and expert count are also logged separately.

The loader validates version and dimensions and reads the entire counter payload
before restoring any counters. Missing or incomplete files cause no warm start.
Version 1 has no checkpoint fingerprint: same-dimension stale statistics cannot
be detected and can cause unnecessary reads, so keep usage files with their
original model. Preloading uses the existing expert loader's fatal short-read
semantics if the actual weight files are damaged.

Run `python test_hotstore.py --exe ./picchio` (or `./picchio.exe` on Windows)
for generated-fixture regressions. Repeat with `--bits 3` and `--bits 4` for
packed INT3/INT4 weights. Dependencies: numpy and safetensors. All three formats
passed locally on Windows with MinGW GCC 16.2.0, including identical tokens and
byte-identical logit dumps, reset, eviction and sync/async execution. The engine's
`--self-test` also passed. These are synthetic checks, not Qwen throughput results.

## What the reported measurements establish

The [full contributor report](https://github.com/robertosantosx2/LEONES/blob/main/docs/subprojects/REDIL-IA-distribuida/PICCHIO-QWEN3-30B-A3B-MSI-2026-11-07.md)
reports CPU-only Qwen3-30B-A3B-Instruct-2507 INT4 inference on an i7-12650H,
14 GiB RAM and NVMe under Ubuntu. The RTX 3050 was not used for inference.

With 4 CPU threads, 4 I/O threads, a 4 GiB expert-cache budget and context 512,
three runs took 57.0, 65.2 and 70.3 seconds: median 65.2 seconds, reported
throughput 1.8–2.2 tokens/s. This is an experimental reference for that machine,
not a portable default or a guaranteed speed. More CPU/I/O threads did not
improve the reported results consistently. A higher cache hit ratio by itself
does not establish lower latency.

## Reproducible follow-up on the contributor's Linux machine

Use the same converted checkpoint and tokenizer; record their revision,
conversion settings, Picchio commit/compiler, Python packages, CPU, RAM, kernel,
storage and power mode. Do not change attention quantisation between runs.

Run from the repository with the existing Python environment and binary:

```bash
mkdir -p benchmark-results/issue-8
model="$HOME/picchio-data/qwen3-30b-a3b-int4"
prompt='Responde en español: explica brevemente qué es un modelo Mixture of Experts y cómo selecciona los expertos que utiliza.'
for run in 1 2 3 4 5; do
  env -u MODEL -u ECAP -u PIPE \
    FLAT=0 PREFETCH=0 PILOT=0 ASYNC_MOE=0 DIRECT=0 HOTSTORE_PRELOAD=0 \
    GPU=0 GPU_DENSE=0 GPU_ROUTER=0 GPU_PREFETCH=0 GPU_EXPERTS=0 GPU_LMHEAD=0 \
    GPU_DENSE_RELEASE_HOST=0 EXPERT_REUSE=1 TENSOR_INDEX=1 \
    .venv/bin/python -u chat_qwen.py "$prompt" \
      --model "$model" --ctx 512 --pin-gb 4 --threads 4 --io-threads 4 \
      --max-tokens 128 --no-reasoning --temperature 0 --top-p 1 --top-k 1 \
      > "benchmark-results/issue-8/baseline-$run.log" 2>&1 || break
done
```

Greedy sampling is intentional for comparison; this is a new controlled baseline,
not an exact reproduction of the original default-temperature runs. Start each
run as a fresh process, preserve the output and check token counts and text match.
An early EOS may yield fewer than 128 tokens; report actual counts.

These are **fresh-process runs with uncontrolled OS page-cache state**, not
cold-disk measurements. Do not drop system caches or attribute repeated-run
speedups to `.picchio_usage`. In-process warm runs and multi-turn KV reuse must
be labelled separately. The script leaves existing routing statistics in place;
the baseline cache policy does not consume them. For a warm-start comparison,
change only `HOTSTORE_PRELOAD` to a positive slot count and restore the same
saved usage file before each baseline/candidate run (shutdown updates it).
Keep a backup outside the model directory and include startup/preload cost in
total elapsed time. Compare both repeated and unrelated prompts.

After the baseline, change one variable per comparison: cache budgets 2/3/4 GiB,
CPU threads 2/4/8, then I/O threads 2/4/8. Repeat each configuration at least
five times, interleaving baseline and candidate runs to reduce ordering effects.
Increase cache above 4 GiB only if observed memory headroom permits it.

Keep raw logs and report median and range for elapsed time, actual generated
tokens, throughput, cache hits/misses, expert loads, attention/MoE/head times,
read times and RSS. Record available RAM, swap activity, CPU frequency and
temperature alongside each run. Runtime read counters measure loader activity,
not necessarily physical disk reads: buffered reads can hit the OS page cache.
Internal timers may overlap and must not be summed as wall-clock intervals.
Final RSS is not peak RSS. Record failures rather than excluding them silently.

## Gate before enabling warm start by default

The opt-in preload uses the existing evictable LRU within its current slot
budget. Permanent pinning and adaptive eviction are separate changes. Before
enabling warm start by default, require:

- Identical greedy token IDs/logits to the baseline on synthetic fixtures.
- Tests for missing, truncated, incompatible and stale usage files.
- Bounded slots/memory, including prefetch and asynchronous reads.
- Separate startup/preload time and generation time; preload must not hide I/O
  outside the reported interval.
- Both repeated and unrelated prompts, to expose stale-frequency cache pollution.
- Measured total latency and memory on the real Qwen checkpoint, not just hit rate.

No real-model speedup from this policy has been measured or claimed here.
