#!/usr/bin/env python3
"""Token-exact GPT-OSS chat: official Harmony + persistent Picchio session.

Rendering, tokenization, and parsing belong to `openai-harmony`. Picchio only
receives and returns raw token IDs. Between turns the common prefix of the
KV-cache is reused, because the Harmony re-render is not prefix-preserving
(the analysis is dropped and `<|return|>` becomes `<|end|>`).
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from openai_harmony import (
    Conversation, HarmonyEncodingName, Message, ReasoningEffort,
    RenderConversationConfig, Role, StreamableParser, SystemContent,
    load_harmony_encoding,
)

import chat_ui as ui

KEEP_ANALYSIS = RenderConversationConfig(auto_drop_analysis=False)


# ── Terminal UI ──────────────────────────────────────────────────────
# All decoration is written to stderr; stdout carries only the model's text,
# so piping or redirecting the answer stays clean. Colour is used only when
# stderr is an interactive terminal (and NO_COLOR is not set).

def _resolve_exe(exe):
    """Resolve exe to an absolute path. Windows' CreateProcess (used by
    subprocess.Popen with an explicit executable) doesn't reliably find a
    bare relative name like "picchio.exe" without a "./" prefix, even when
    it sits in the current directory. """
    p = Path(exe)
    if p.is_file():
        return str(p.resolve())
    found = shutil.which(str(exe))
    return found if found else exe


class PicchioSession:
    """Persistent Picchio process in SERVICE mode."""

    def __init__(self, exe, model, ctx, pin_gb, threads, model_aux, sampling=None,
                 async_moe=False, direct=False, io_threads=None, flat=None,
                 gpu_router=False, gpu_prefetch=False, gpu_dense=False,
                 gpu_experts=False, gpu_dense_release_host=False, expert_reuse=True,
                 tensor_index=True):
        if gpu_dense_release_host and not gpu_dense:
            raise ValueError("host weight release requires gpu_dense=True")
        env = os.environ.copy()
        for name in ("INPUT", "PROMPT", "INPUT_FILE", "OUTPUT", "MODEL_AUX",
                     "TRACE_NUMERIC", "ORACLE_DIR", "FLAT", "GPU",
                     "GPU_ROUTER", "GPU_PREFETCH", "GPU_DENSE", "GPU_EXPERTS",
                     "GPU_LMHEAD", "GPU_DENSE_RELEASE_HOST", "ASYNC_MOE",
                     "DIRECT", "IO_THREADS"):
            env.pop(name, None)
        env.update({"SERVICE": "1", "TEMPERATURE": "0", "REP": "1",
                    "CTX": str(ctx), "PIN_GB": str(pin_gb),
                    "OMP_NUM_THREADS": str(threads), "EXPERT_REUSE": "1" if expert_reuse else "0",
                    "TENSOR_INDEX": "1" if tensor_index else "0"})
        if sampling:
            env.update({k: str(v) for k, v in sampling.items() if v is not None})
        if async_moe:
            env["ASYNC_MOE"] = "1"
        if direct:
            env["DIRECT"] = "1"
        if io_threads is not None:
            env["IO_THREADS"] = str(io_threads)
        if flat is not None:
            env["FLAT"] = str(flat)
        if gpu_router:
            env["GPU_ROUTER"] = "1"
        if gpu_prefetch:
            env["GPU_PREFETCH"] = "1"
        if gpu_dense:
            env["GPU_DENSE"] = "1"
        if gpu_dense_release_host:
            env["GPU_DENSE_RELEASE_HOST"] = "1"
        if gpu_experts:
            env["GPU_EXPERTS"] = "1"
        # Default sampling sent with every TURN (a per-turn override is possible).
        s = sampling or {}
        self.temperature = 1.0 if s.get("TEMPERATURE") is None else float(s["TEMPERATURE"])
        self.top_p = 0.95 if s.get("TOPP") is None else float(s["TOPP"])
        self.top_k = 50 if s.get("TOPK") is None else int(s["TOPK"])
        if model_aux:
            env["MODEL_AUX"] = model_aux
        self.proc = subprocess.Popen(
            [_resolve_exe(exe), str(model)], env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, encoding="ascii",
            errors="replace", bufsize=1)
        try:
            ready = self._line()
            if not ready.startswith("READY "):
                raise RuntimeError(f"service startup failed: {ready}")
            fields = ready.split()
            self.ctx = int(fields[1])
            self.vocab = int(fields[2])
            self.stop_ids = [int(x) for x in fields[3:]]
        except Exception:
            self.proc.kill()
            self.proc.wait()
            raise

    def _line(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("the Picchio service closed unexpectedly")
        return line.strip()

    def turn(self, ids, max_new, keep, on_token,
             temperature=None, top_p=None, top_k=None):
        before = self.stats()
        self.last_timing = {}
        temp = self.temperature if temperature is None else temperature
        topp = self.top_p if top_p is None else top_p
        topk = self.top_k if top_k is None else top_k
        payload = (f"TURN {max_new} {keep} {temp} {topp} {topk} {len(ids)} "
                   + " ".join(map(str, ids)) + "\n")
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()
        produced = []
        while True:
            line = self._line()
            if line.startswith("TOKEN "):
                token = int(line.split()[1])
                produced.append(token)
                on_token(token)
            elif line.startswith("DONE "):
                _, reason, _, pos = line.split()
                after = self.stats()
                for phase in ("prefill", "decode"):
                    seconds = phase + "_seconds"
                    tokens = phase + "_tokens"
                    if seconds in before and seconds in after:
                        duration = after[seconds] - before[seconds]
                        count = after[tokens] - before[tokens]
                        self.last_timing.update({seconds: duration, tokens: count,
                            phase + "_tokens_per_s": count / duration if duration > 0 else None})
                return produced, reason, int(pos)
            elif line.startswith("ERROR "):
                raise RuntimeError(line)
            else:
                raise RuntimeError(f"unexpected protocol frame: {line}")

    def close(self):
        try:
            self.proc.stdin.write("SHUTDOWN\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def reset(self):
        self.proc.stdin.write("RESET\n")
        self.proc.stdin.flush()
        line = self._line()
        if line != "DONE RESET 0 0":
            raise RuntimeError(f"reset failed: {line}")

    def stats(self):
        """Return a cumulative, machine-readable snapshot from the C engine."""
        self.proc.stdin.write("STATS\n")
        self.proc.stdin.flush()
        line = self._line()
        if not line.startswith("STATS "):
            raise RuntimeError(f"stats failed: {line}")
        try:
            return json.loads(line[6:])
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid stats frame: {line}") from exc


class HarmonyChat:
    """Harmony conversation with reuse of the prefix already consumed by the model."""

    def __init__(self, session, reasoning, current_date, no_reasoning=False):
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.session = session
        system = (SystemContent.new()
                  .with_reasoning_effort(ReasoningEffort(reasoning.capitalize()))
                  .with_conversation_start_date(current_date))
        self.system_message = Message.from_role_and_content(Role.SYSTEM, system)
        self.messages = [self.system_message]
        self.committed = []
        self.last_stats = None
        # No-reasoning mode: pre-commit the `final` channel, so the assistant's
        # turn cannot emit an `analysis` message. The model consumes these tokens
        # (it does not regenerate them), so they must be pre-fed to the parser and
        # included in the parsing/committed set.
        self.no_reasoning = no_reasoning
        self.final_prefill = (
            self.encoding.encode("<|channel|>final<|message|>", allowed_special="all")
            if no_reasoning else [])
        if session is not None and set(session.stop_ids) - set(self.encoding.stop_tokens()):
            raise RuntimeError(f"inconsistent runtime stop tokens: {session.stop_ids}")

    def reset(self):
        self.session.reset()
        self.messages = [self.system_message]
        self.committed = []
        self.last_stats = None

    def render(self, user_text):
        messages = self.messages + [Message.from_role_and_content(Role.USER, user_text)]
        return self.encoding.render_conversation_for_completion(
            Conversation.from_messages(messages), Role.ASSISTANT, KEEP_ANALYSIS)

    def ask(self, user_text, max_new, live=True):
        full = self.render(user_text)
        if len(full) > self.session.ctx:
            raise RuntimeError(
                f"insufficient context: {len(full)} positions needed out of {self.session.ctx}")
        keep = 0
        for a, b in zip(full, self.committed):
            if a != b:
                break
            keep += 1
        delta = full[keep:] + self.final_prefill
        if not delta:
            raise RuntimeError("empty delta: nothing to process")

        parser = StreamableParser(self.encoding, Role.ASSISTANT, strict=False)
        # Pre-feed the `final` channel so the parser is already in the right channel
        # when the generated tokens arrive (which start from the content).
        for tok in self.final_prefill:
            parser.process(tok)

        t0 = time.perf_counter()
        # Stream BOTH channels live so the user always sees what is happening: the
        # reasoning prints dimmed on stderr under a "thinking ❯" header, the answer
        # prints on stdout under "picchio ❯". Keeping the answer alone on stdout
        # means redirecting/piping the command still yields a clean answer file.
        state = {"n": 0, "channel": None, "first_token_at": None}

        def _header(ch):
            if ch == "analysis":
                ui.begin_thinking()
            else:
                ui.begin_answer()

        def on_token(token):
            if state["first_token_at"] is None:
                state["first_token_at"] = time.perf_counter()
            parser.process(token)
            state["n"] += 1
            chunk = parser.last_content_delta
            ch = parser.current_channel
            if not (live and chunk and ch in ("analysis", "final")):
                # Header/role tokens, or non-live (JSON) mode: just animate the spinner.
                if state["channel"] is None:
                    ui.thinking(state["n"], time.perf_counter() - t0)
                return
            if state["channel"] != ch:
                # Channel switch: close the previous line on its own stream, then
                # print the new header. The first header just clears the spinner.
                if state["channel"] == "final":
                    sys.stdout.write("\n"); sys.stdout.flush()
                elif state["channel"] == "analysis":
                    sys.stderr.write("\n"); sys.stderr.flush()
                _header(ch)
                state["channel"] = ch
            if ch == "final":
                sys.stdout.write(chunk); sys.stdout.flush()
            else:
                sys.stderr.write(ui.DIM(chunk)); sys.stderr.flush()

        produced, reason, pos = self.session.turn(delta, max_new, keep, on_token)
        self.committed = full + self.final_prefill + produced

        # Close the last streamed line on whichever stream it used.
        if state["channel"] == "final":
            sys.stdout.write("\n"); sys.stdout.flush()
        elif state["channel"] == "analysis":
            sys.stderr.write("\n"); sys.stderr.flush()
        else:
            ui.clear_status()

        dt = time.perf_counter() - t0
        n = len(produced)
        ttft = (state["first_token_at"] - t0
                if state["first_token_at"] is not None else None)
        self.last_stats = {
            "tokens": n, "elapsed": dt, "pos": pos, "ctx": self.session.ctx,
            "reused": keep, "prompt_tokens": len(full), "reason": reason,
            "ttft": ttft,
            **self.session.last_timing,
        }
        ui.metrics(**self.last_stats)

        try:
            replies = self.encoding.parse_messages_from_completion_tokens(
                self.final_prefill + produced, Role.ASSISTANT)
        except Exception as exc:
            replies = parser.messages
            ui.write(ui.DIM(f"  Incomplete response ({reason}: {exc})"))
        self.messages.append(Message.from_role_and_content(Role.USER, user_text))
        self.messages.extend(replies)
        return replies, reason, pos


def resolve_aux(model, override):
    # Fresh conversions bake the expert biases into the shards, so no auxiliary
    # files are needed. Pass extra files explicitly with --model-aux when the
    # model is split across disks (or needs a legacy bias sidecar).
    return override


def main():
    ui.configure_utf8()

    parser = argparse.ArgumentParser(description="GPT-OSS chat with Picchio")
    parser.add_argument("prompt", nargs="?", help="single question; omit to start the chat")
    parser.add_argument("--model", default=r"C:\models\gptoss_i4")
    parser.add_argument("--exe", default="picchio.exe")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--reasoning", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="skip the analysis channel: pre-commit the final channel, "
                             "the model answers without reasoning (faster)")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--pin-gb", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--io-threads", type=int,
                        help="parallel expert reads (engine default: 4)")
    parser.add_argument("--async-moe", action="store_true",
                        help="overlap routed expert reads with CPU expert compute")
    parser.add_argument("--direct", action="store_true",
                        help="use unbuffered expert reads (best paired with --async-moe)")
    parser.add_argument("--flat",
                        help="flat expert store path, or 0 to force SafeTensors")
    parser.add_argument("--gpu-router", action="store_true",
                        help="keep all routers resident and execute them on the GPU")
    parser.add_argument("--gpu-prefetch", action="store_true",
                        help="predict L+1 on the GPU and prefetch experts during current MoE")
    parser.add_argument("--gpu-dense", action="store_true",
                        help="keep attention Q/K/V/O as FP16 in VRAM and run projections on GPU")
    parser.add_argument("--gpu-dense-release-host", action="store_true",
                        help="free uploaded Q/K/V/O host weights; GPU errors terminate the session")
    parser.add_argument("--no-expert-reuse", action="store_true",
                        help="use the legacy allocating expert reader for comparisons")
    parser.add_argument("--no-tensor-index", action="store_true",
                        help="use linear tensor-name lookup for comparisons")
    parser.add_argument("--gpu-experts", action="store_true",
                        help="experimental expert compute offload (not recommended on 4 GB GPUs)")
    parser.add_argument("--model-aux")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = deterministic greedy; values ~0.7-1.0 avoid loops")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rep", type=float, default=1.0,
                        help="repetition penalty (1.0=off; try 1.1 to stop "
                             "degenerate loops like '......')")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--show-analysis", action="store_true",
                        help="deprecated: reasoning is now always shown live "
                             "(dimmed) alongside the answer")
    parser.add_argument("--dry-run", action="store_true",
                        help="render and verify the IDs without starting Picchio")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.gpu_dense_release_host and not args.gpu_dense:
        parser.error("--gpu-dense-release-host requires --gpu-dense")

    if args.dry_run:
        chat = HarmonyChat(None, args.reasoning, args.date, args.no_reasoning)
        text = args.prompt if args.prompt is not None else input("You: ")
        ids = chat.render(text) + chat.final_prefill
        rendered = chat.encoding.decode_utf8(ids)
        if chat.encoding.encode(rendered, allowed_special="all") != ids:
            raise SystemExit("Harmony ID round-trip failed")
        print(json.dumps({"token_count": len(ids), "ids": ids, "rendered": rendered},
                         ensure_ascii=False, indent=2))
        return

    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    if not exe.is_file():
        parser.error(f"executable not found: {exe}")
    if not model.is_dir():
        parser.error(f"model not found: {model}")

    sampling = {"TEMPERATURE": args.temperature, "TOPP": args.top_p,
                "TOPK": args.top_k, "REP": args.rep, "SEED": args.seed}
    session = PicchioSession(
        exe, model, args.ctx, args.pin_gb, args.threads,
        resolve_aux(model, args.model_aux), sampling,
        async_moe=args.async_moe, direct=args.direct,
        io_threads=args.io_threads, flat=args.flat,
        gpu_router=args.gpu_router, gpu_prefetch=args.gpu_prefetch,
        gpu_dense=args.gpu_dense,
        gpu_dense_release_host=args.gpu_dense_release_host,
        expert_reuse=not args.no_expert_reuse,
        tensor_index=not args.no_tensor_index,
        gpu_experts=args.gpu_experts)
    chat = HarmonyChat(session, args.reasoning, args.date, args.no_reasoning)
    single = args.prompt is not None

    def show_header():
        reasoning = "off" if args.no_reasoning else args.reasoning
        gpu_features = []
        if args.gpu_dense:
            gpu_features.append("dense")
        if args.gpu_prefetch:
            gpu_features.append("prefetch")
        elif args.gpu_router:
            gpu_features.append("router")
        if args.gpu_experts:
            gpu_features.append("experts")
        gpu_mode = "GPU " + "+".join(gpu_features) if gpu_features else None
        ui.header("GPT-OSS", model, session.ctx, args.pin_gb, args.temperature,
                  args.threads, not single, reasoning, args.async_moe,
                  args.direct, args.io_threads, gpu_mode)

    show_header()

    try:
        while True:
            if single:
                text = args.prompt
                ui.show_user(text)
            else:
                try:
                    text = ui.prompt()
                except EOFError:
                    ui.write()
                    break
                if not text:
                    continue
                command = text.lower()
                if command in ("/exit", "/quit"):
                    break
                if command == "/help":
                    ui.help_text()
                    continue
                if command == "/clear":
                    ui.clear_screen()
                    show_header()
                    continue
                if command == "/reset":
                    chat.reset()
                    ui.write(ui.DIM("  Conversation and KV cache reset."))
                    continue
                if command == "/stats":
                    ui.show_stats(chat.last_stats)
                    continue
                if command == "/settings":
                    show_header()
                    continue
                if command.startswith("/"):
                    ui.warning(f"Unknown command: {text} · use /help")
                    continue
            replies, reason, pos = chat.ask(text, args.max_tokens,
                                            live=not args.json)
            if args.json:
                print(json.dumps({"reason": reason, "pos": pos,
                                  "messages": [m.to_dict() for m in replies]},
                                 ensure_ascii=False, default=str, indent=2))
            if single:
                break
    except KeyboardInterrupt:
        ui.interrupted()
    finally:
        session.close()


if __name__ == "__main__":
    main()
