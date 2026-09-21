#!/usr/bin/env python3
"""Reproducible multi-turn benchmark for GPT-OSS on Picchio.

The model is loaded once. Each turn reuses the longest valid Harmony/KV prefix,
while cumulative engine counters are sampled before and after the turn. Results
are written as both JSON (complete record) and CSV (one row per turn).
"""

import argparse
import csv
import json
import os
import platform
import statistics
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

from chat import HarmonyChat, PicchioSession, resolve_aux


DEFAULT_PROMPTS = [
    "Memorizza: progetto Picchio, codice ORIONE-42, colore verde. "
    "Rispondi soltanto: OK.",
    "Qual e il codice del progetto? Rispondi soltanto con il codice.",
    "Quale colore era associato al progetto? Rispondi con una sola parola.",
]
DEFAULT_EXPECTED = ["ok", "orione-42", "verde"]

COUNTERS = (
    "forward", "tokens_emitted", "cache_hits", "cache_misses",
    "cache_requests", "expert_loads", "async_batches", "gpu_router_calls",
    "gpu_dense_calls", "gpu_dense_fallbacks",
    "prefetch_issued", "predict_hits_a", "predict_hits_b", "predict_total",
    "prefill_tokens", "decode_tokens",
    "expert_buffer_allocs", "expert_buffer_reuses",
)
TIMERS = (
    "disk_seconds", "async_wait_seconds", "attention_seconds", "moe_seconds",
    "head_seconds", "gpu_router_seconds", "gpu_dense_seconds",
    "prefetch_read_seconds",
    "prefill_seconds", "decode_seconds",
    "expert_compute_seconds",
)


def load_prompts(path):
    if path is None:
        return list(DEFAULT_PROMPTS)
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value):
        raise ValueError("the prompts file must be a non-empty JSON array of strings")
    return value


def engine_delta(before, after):
    delta = {key: after[key] - before[key] for key in COUNTERS + TIMERS}
    delta["position"] = after["position"]
    delta["rss_gb"] = after["rss_gb"]
    delta["resident_gb"] = after["resident_gb"]
    delta["expert_bytes_estimate"] = after["expert_bytes_estimate"]
    for key in ("gpu_dense_host_released_bytes", "expert_slots_per_layer", "available_ram_gib"):
        delta[key] = after[key]

    requests = delta["cache_hits"] + delta["cache_misses"]
    delta["cache_hit_pct"] = (
        100.0 * delta["cache_hits"] / requests if requests else 0.0)
    predicted = delta["predict_total"]
    delta["predict_a_pct"] = (
        100.0 * delta["predict_hits_a"] / predicted if predicted else 0.0)
    delta["predict_b_pct"] = (
        100.0 * delta["predict_hits_b"] / predicted if predicted else 0.0)
    loads = delta["expert_loads"] + delta["prefetch_issued"]
    delta["expert_io_estimate_gb"] = (
        loads * after["expert_bytes_estimate"] / 1e9)
    delta["io_seconds"] = delta["disk_seconds"] + delta["prefetch_read_seconds"]
    return delta


def serialise_messages(messages):
    return [message.to_dict() for message in messages]


def final_text(messages):
    chunks = []
    for message in serialise_messages(messages):
        if message.get("channel") != "final":
            continue
        for content in message.get("content", []):
            if content.get("type") == "text":
                chunks.append(content.get("text", ""))
    return "".join(chunks).strip()


def csv_row(turn):
    generation = turn["generation"]
    engine = turn["engine"]
    return {
        "turn": turn["turn"],
        "prompt": turn["prompt"],
        "answer": turn["answer"],
        "expected_contains": turn["expected_contains"],
        "semantic_pass": turn["semantic_pass"],
        "finish_reason": generation["reason"],
        "completion_tokens": generation["tokens"],
        "elapsed_s": generation["elapsed"],
        "ttft_s": generation["ttft"],
        "tokens_per_s": generation["tokens_per_s"],
        "decode_tokens_per_s": generation.get("decode_tokens_per_s"),
        "prefill_seconds": generation.get("prefill_seconds"),
        "decode_seconds": generation.get("decode_seconds"),
        "prompt_tokens": generation["prompt_tokens"],
        "reused_tokens": generation["reused"],
        "reuse_pct": generation["reuse_pct"],
        "position": generation["pos"],
        "forwards": engine["forward"],
        "cache_hit_pct": engine["cache_hit_pct"],
        "expert_loads": engine["expert_loads"],
        "prefetch_issued": engine["prefetch_issued"],
        "expert_io_estimate_gb": engine["expert_io_estimate_gb"],
        "io_seconds": engine["io_seconds"],
        "async_wait_seconds": engine["async_wait_seconds"],
        "attention_seconds": engine["attention_seconds"],
        "moe_seconds": engine["moe_seconds"],
        "head_seconds": engine["head_seconds"],
        "gpu_router_calls": engine["gpu_router_calls"],
        "gpu_router_ms_per_call": (
            1000.0 * engine["gpu_router_seconds"] / engine["gpu_router_calls"]
            if engine["gpu_router_calls"] else 0.0),
        "gpu_dense_calls": engine["gpu_dense_calls"],
        "gpu_dense_ms_per_call": (
            1000.0 * engine["gpu_dense_seconds"] / engine["gpu_dense_calls"]
            if engine["gpu_dense_calls"] else 0.0),
        "gpu_dense_fallbacks": engine["gpu_dense_fallbacks"],
        "predict_a_pct": engine["predict_a_pct"],
        "predict_b_pct": engine["predict_b_pct"],
        "rss_gb": engine["rss_gb"],
    }


def parse_args():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Persistent GPT-OSS multi-turn benchmark with engine counters")
    parser.add_argument("--model", default=str(root / "models" / "gptoss_i3"))
    parser.add_argument("--exe", default=str(Path(__file__).with_name("picchio.exe")))
    parser.add_argument("--output-dir", default=str(root / "benchmarks"))
    parser.add_argument("--prompts", help="JSON array of prompts (default: 3-turn memory test)")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--pin-gb", type=float, default=2.0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--io-threads", type=int, default=4)
    parser.add_argument("--reasoning", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--with-reasoning", action="store_true",
                        help="allow the analysis channel (default pre-commits final)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--flat", default="0")
    parser.add_argument("--model-aux")
    parser.add_argument("--no-async-moe", action="store_true")
    parser.add_argument("--no-direct", action="store_true")
    parser.add_argument("--no-gpu-prefetch", action="store_true")
    parser.add_argument("--gpu-dense", action="store_true")
    parser.add_argument("--gpu-dense-release-host", action="store_true")
    parser.add_argument("--no-expert-reuse", action="store_true")
    parser.add_argument("--no-tensor-index", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.prompts)
    expected = DEFAULT_EXPECTED if args.prompts is None else [None] * len(prompts)
    if args.turns < 1 or args.turns > len(prompts):
        raise SystemExit(f"--turns must be between 1 and {len(prompts)}")
    if args.max_tokens < 0:
        raise SystemExit("--max-tokens cannot be negative")

    model = Path(args.model).resolve()
    exe = Path(args.exe).resolve()
    if not model.is_dir():
        raise SystemExit(f"model not found: {model}")
    if not exe.is_file():
        raise SystemExit(f"executable not found: {exe}")

    started = datetime.now(timezone.utc)
    sampling = {
        "TEMPERATURE": args.temperature,
        "TOPP": args.top_p,
        "TOPK": args.top_k,
        "REP": 1.0,
        "SEED": args.seed,
    }
    load_started = time.perf_counter()
    session = PicchioSession(
        exe, model, args.ctx, args.pin_gb, args.threads,
        resolve_aux(model, args.model_aux), sampling,
        async_moe=not args.no_async_moe,
        direct=not args.no_direct,
        io_threads=args.io_threads,
        flat=args.flat,
        gpu_prefetch=not args.no_gpu_prefetch,
        gpu_dense=args.gpu_dense,
        gpu_dense_release_host=args.gpu_dense_release_host,
        expert_reuse=not args.no_expert_reuse,
        tensor_index=not args.no_tensor_index,
    )
    load_seconds = time.perf_counter() - load_started
    turns = []
    try:
        chat = HarmonyChat(session, args.reasoning, args.date,
                           no_reasoning=not args.with_reasoning)
        initial_engine = session.stats()
        for index, prompt in enumerate(prompts[:args.turns], 1):
            print(f"\n[benchmark turn {index}/{args.turns}] {prompt}", file=sys.stderr)
            before = session.stats()
            replies, reason, _ = chat.ask(prompt, args.max_tokens, live=False)
            after = session.stats()

            generation = dict(chat.last_stats)
            generation["tokens_per_s"] = (
                generation["tokens"] / generation["elapsed"]
                if generation["elapsed"] else 0.0)
            generation["reuse_pct"] = (
                100.0 * generation["reused"] / generation["prompt_tokens"]
                if generation["prompt_tokens"] else 0.0)
            answer = final_text(replies)
            expected_text = expected[index - 1]
            turns.append({
                "turn": index,
                "prompt": prompt,
                "answer": answer,
                "expected_contains": expected_text,
                "semantic_pass": (expected_text in answer.casefold()
                                  if expected_text is not None else None),
                "generation": generation,
                "engine": engine_delta(before, after),
                "messages": serialise_messages(replies),
            })
            print(
                f"  {generation['tokens']} tokens in {generation['elapsed']:.2f}s; "
                f"TTFT={generation['ttft'] if generation['ttft'] is not None else 0:.2f}s; "
                f"reuse={generation['reuse_pct']:.1f}%", file=sys.stderr)
        final_engine = session.stats()
    finally:
        session.close()

    total_tokens = sum(turn["generation"]["tokens"] for turn in turns)
    total_elapsed = sum(turn["generation"]["elapsed"] for turn in turns)
    ttfts = [turn["generation"]["ttft"] for turn in turns
             if turn["generation"]["ttft"] is not None]
    checked = [turn["semantic_pass"] for turn in turns
               if turn["semantic_pass"] is not None]
    report = {
        "schema": "picchio.chat-benchmark.v2",
        "started_utc": started.isoformat(),
        "host": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
        },
        "configuration": {
            "model": str(model), "exe": str(exe), "ctx": args.ctx,
            "pin_gb": args.pin_gb, "threads": args.threads,
            "io_threads": args.io_threads, "max_tokens": args.max_tokens,
            "turns": args.turns, "reasoning": args.reasoning,
            "no_reasoning": not args.with_reasoning,
            "temperature": args.temperature, "top_p": args.top_p,
            "top_k": args.top_k, "seed": args.seed, "flat": args.flat,
            "async_moe": not args.no_async_moe,
            "direct": not args.no_direct,
            "gpu_prefetch": not args.no_gpu_prefetch,
            "gpu_dense": args.gpu_dense,
            "gpu_dense_release_host": args.gpu_dense_release_host,
            "expert_reuse": not args.no_expert_reuse,
            "tensor_index": not args.no_tensor_index,
        },
        "load_seconds": load_seconds,
        "summary": {
            "completion_tokens": total_tokens,
            "turn_elapsed_seconds": total_elapsed,
            "tokens_per_second": total_tokens / total_elapsed if total_elapsed else 0.0,
            "decode_tokens_per_second": (
                (final_engine["decode_tokens"] - initial_engine["decode_tokens"]) /
                (final_engine["decode_seconds"] - initial_engine["decode_seconds"])
                if final_engine["decode_seconds"] > initial_engine["decode_seconds"] else None),
            "mean_ttft_seconds": statistics.fmean(ttfts) if ttfts else None,
            "semantic_passes": sum(checked),
            "semantic_checks": len(checked),
            "engine": engine_delta(initial_engine, final_engine),
        },
        "turns": turns,
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    stem = output_dir / f"chat-{stamp}"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    rows = [csv_row(turn) for turn in turns]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nJSON: {json_path}", file=sys.stderr)
    print(f"CSV:  {csv_path}", file=sys.stderr)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path),
                      "summary": report["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
