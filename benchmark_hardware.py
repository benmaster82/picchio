#!/usr/bin/env python3
"""Sequential hardware sweep. JSON results; native prefill/decode timings.

Each case starts a fresh process and uses the same short Harmony prompt and
greedy sampling. No OS cache purge: DIRECT expert reads bypass the page cache.
This is a short tuning workload, not a claim about all conversation lengths.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from datetime import datetime, timezone

from openai_harmony import (Conversation, Message, Role, HarmonyEncodingName,
                           load_harmony_encoding)
from chat import PicchioSession
from benchmark_chat import engine_delta


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
        (name, ctypes.c_ulonglong) for name in
        ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual",
         "avail_virtual", "avail_extended")]


def available_gib():
    if os.name != "nt":
        return None
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError()
    return status.avail_phys / 2**30


class MemoryMonitor:
    """Sample global available memory, plus Windows system-wide paging reads.

    Page Reads/sec is global, so it cannot attribute disk paging to Picchio.
    Process page-fault counters in STATS include soft faults and are separate.
    """
    def __init__(self):
        self.minimum = available_gib()
        self.stop_event = threading.Event()
        self.page_read_rates = []
        self.pdh = None
        self.query = ctypes.c_void_p()
        self.counter = ctypes.c_void_p()
        if os.name == "nt":
            pdh = ctypes.WinDLL("pdh")
            if pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) == 0:
                status = pdh.PdhAddEnglishCounterW(
                    self.query, ctypes.c_wchar_p(r"\Memory\Page Reads/sec"),
                    0, ctypes.byref(self.counter))
                if status == 0:
                    self.pdh = pdh
                    pdh.PdhCollectQueryData(self.query)
                else:
                    pdh.PdhCloseQuery(self.query)
                    self.query = ctypes.c_void_p()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        class Value(ctypes.Structure):
            _fields_ = [("status", ctypes.c_ulong), ("value", ctypes.c_double)]
        while not self.stop_event.wait(0.5):
            free = available_gib()
            if free is not None:
                self.minimum = min(self.minimum, free)
            if self.pdh:
                value = Value()
                if self.pdh.PdhCollectQueryData(self.query) == 0:
                    rc = self.pdh.PdhGetFormattedCounterValue(
                        self.counter, 0x200, None, ctypes.byref(value))  # PDH_FMT_DOUBLE
                    if rc == 0 and value.status in (0, 1):
                        self.page_read_rates.append(value.value)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        if self.pdh:
            self.pdh.PdhCloseQuery(self.query)

    def result(self):
        return {"minimum_available_ram_gib": self.minimum,
                "system_page_reads_per_s_mean": (
                    sum(self.page_read_rates) / len(self.page_read_rates)
                    if self.page_read_rates else None),
                "system_page_reads_per_s_max": max(self.page_read_rates, default=None)}


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(root / "models/gptoss_i3"))
    parser.add_argument("--exe", default=str(Path(__file__).with_name("picchio.exe")))
    parser.add_argument("--output", default=str(root / "benchmarks" /
        ("hardware-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + ".json")))
    parser.add_argument("--cases", help="JSON file containing case objects")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.tokens < 2 or args.repeat < 1:
        parser.error("--tokens >= 2 and --repeat >= 1 are required")
    cases = json.loads(Path(args.cases).read_text(encoding="utf-8")) if args.cases else [
        {"pin_gb": 2, "release_host": False},
        {"pin_gb": 3, "release_host": False},
        {"pin_gb": 4, "release_host": False},
        {"pin_gb": 2, "release_host": True},
        {"pin_gb": 4, "release_host": True},
        {"pin_gb": 6, "release_host": True},
    ]
    # Prevent inherited tuning/debug settings from silently changing this workload.
    for name in ("ECAP", "PREDICT_PROBE", "PREFETCH", "PILOT", "PIPE_ROLE",
                 "PIPE_CUT", "IDOT", "DROP", "HOT", "HEAT", "LOOKAHEAD"):
        os.environ.pop(name, None)
    os.environ["PREFILL_BATCH"] = "64"
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    prompt = "Conta da uno a venti, separando i numeri con virgole."
    ids = encoding.render_conversation_for_completion(Conversation.from_messages([
        Message.from_role_and_content(Role.USER, prompt)]), Role.ASSISTANT)
    ids += encoding.encode("<|channel|>final<|message|>", allowed_special="all")
    report = {"schema": "picchio.hardware-sweep.v1", "prompt": prompt,
              "prompt_ids": ids, "max_tokens": args.tokens,
              "exe_sha256": hashlib.sha256(Path(args.exe).read_bytes()).hexdigest(),
              "cases": [], "note": "Short greedy workload; no cold OS cache purge. "
              "Decode counts native forward steps including the final KV commit. "
              "Paging reads are system-wide, not attributable to one process."}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reference = None
    for repetition in range(args.repeat):
        for case in (cases if repetition % 2 == 0 else list(reversed(cases))):
            config = {"threads": 8, "io_threads": 4, "gpu_prefetch": True,
                      "gpu_dense": True, "expert_reuse": True, "tensor_index": True, **case}
            row = {"configuration": config, "repetition": repetition + 1}
            report["cases"].append(row)
            free = available_gib()
            # Host dense weights load before release; leave headroom for OS/KV/I/O.
            required = max(5.7, (1.5 if config["release_host"] else 5.1) + config["pin_gb"] + 0.75)
            if free is not None and free < required:
                row.update(status="skipped_memory", available_gib=free, required_gib=required)
                print(json.dumps(row), flush=True)
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                continue
            print(f"CASE {repetition + 1}: {config}", flush=True)
            started = time.perf_counter()
            session = None
            with MemoryMonitor() as monitor:
                try:
                    session = PicchioSession(
                        args.exe, args.model, 512, config["pin_gb"], config["threads"], None,
                        {"TEMPERATURE": 0, "TOPP": 1, "TOPK": 1, "REP": 1, "SEED": 1234},
                        async_moe=True, direct=True, io_threads=config["io_threads"], flat="0",
                        gpu_prefetch=config["gpu_prefetch"], gpu_dense=config["gpu_dense"],
                        gpu_dense_release_host=config["release_host"],
                        expert_reuse=config["expert_reuse"], tensor_index=config["tensor_index"])
                    row["load_seconds"] = time.perf_counter() - started
                    before = session.stats()
                    token_times = []
                    turn_start = time.perf_counter()
                    produced, reason, pos = session.turn(ids, args.tokens, 0,
                        lambda _: token_times.append(time.perf_counter()))
                    after = session.stats()
                    if config["release_host"] and not after["gpu_dense_host_released_bytes"]:
                        raise RuntimeError("requested host release was not active")
                    if config["gpu_dense"] and after["gpu_dense_calls"] == before["gpu_dense_calls"]:
                        raise RuntimeError("requested GPU dense backend was not active")
                    if reference is None:
                        reference = produced
                    row.update(status="ok", produced_ids=produced,
                        same_tokens_as_first=produced == reference,
                        completion_text=encoding.decode_utf8(produced), reason=reason,
                        position=pos, timing=session.last_timing,
                        turn_seconds=time.perf_counter() - turn_start,
                        ttft_seconds=token_times[0] - turn_start if token_times else None,
                        engine=engine_delta(before, after),
                        private_gib=after["private_gib"], peak_rss_gib=after["peak_rss_gib"],
                        process_page_faults=after["process_page_faults"] - before["process_page_faults"])
                except Exception as exc:
                    row.update(status="error", error=str(exc))
                finally:
                    if session:
                        session.close()
            row.update(monitor.result())
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    print(f"REPORT {output}", flush=True)
    if not any(row.get("status") == "ok" for row in report["cases"]):
        raise SystemExit("No configuration completed; inspect the report.")


if __name__ == "__main__":
    main()
