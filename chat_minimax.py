#!/usr/bin/env python3
"""Chat bridge for MiniMax-M2 models running on Picchio.

Mirrors chat_qwen.py — same SERVICE pipe protocol (READY / TURN / TOKEN / DONE /
SHUTDOWN), same KV-prefix reuse, same UI — and reuses its PicchioSession rather
than duplicating the transport. What differs is MiniMax-M2's prompt format, in
two ways that matter:

  1. Reasoning is not optional. The chat template's generation prompt ends with
     a literal '<think>\\n', so every reply *starts inside* a thinking block:
     the model emits its reasoning, then '</think>', then the user-facing answer.
     There is no enable_thinking switch to turn this off (unlike Qwen3). This
     bridge therefore splits each reply on the '</think>' token id and shows the
     two halves separately.

  2. Interleaved thinking means the rendered prefix changes retroactively. The
     template only keeps an assistant turn's reasoning while no user message
     follows it (see chat_template.jinja: `loop.index0 > ns.last_user_index`),
     so as soon as you send a new message, the previous turn's reasoning
     disappears from the render. The longest-common-prefix with what the engine
     already holds then ends at that point, which caps KV reuse at roughly the
     system prompt plus the first exchange. That is inherent to the format, not
     a bug — the reused% in the metrics line reports the truth.

Requires: pip install transformers
Convert first (see the MiniMax-M2 notes in convert_minimax.py):
  python convert_minimax.py --input D:/models/MiniMax-M2-GPTQ-INT4 \
      --output D:/models/minimax_m2_i4 --dense-bits 8

Run:
  python chat_minimax.py --model D:/models/minimax_m2_i4 \
      --ctx 4096 --pin-gb 12 --max-tokens 512
"""
import argparse
import os
import sys
import time

try:
    from transformers import AutoTokenizer
except ImportError:
    print("pip install transformers", file=sys.stderr)
    sys.exit(1)

import chat_ui as ui
from chat_qwen import PicchioSession, as_id_list


THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _special_id(tokenizer, piece, fallback):
    """Resolve a special token to its id, preferring the tokenizer's own view."""
    tid = tokenizer.convert_tokens_to_ids(piece)
    if isinstance(tid, int) and tid >= 0 and tid != getattr(tokenizer, "unk_token_id", None):
        return tid
    encoded = tokenizer.encode(piece, add_special_tokens=False)
    if len(encoded) == 1:
        return encoded[0]
    return fallback


class MiniMaxChat:
    """MiniMax-M2 conversation with KV-prefix reuse and reasoning separation."""

    def __init__(self, session, tokenizer, system=None, show_thinking=False):
        self.s = session
        self.tok = tokenizer
        self.show_thinking = show_thinking
        self.system = system
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})
        self.committed = []          # exact token stream the engine has consumed
        self.last_stats = None
        self.think_close_id = _special_id(tokenizer, THINK_CLOSE, 200051)

        eos = tokenizer.eos_token_id
        if session is not None and eos is not None and eos not in session.stop_ids:
            ui.warning(f"Runtime stop IDs {session.stop_ids} do not include "
                       f"tokenizer EOS {eos} — generation may not stop on its own")

    def reset(self):
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": self.system})
        # Dropping committed forces keep=0, so the engine re-prefills from scratch.
        self.committed = []
        self.last_stats = None

    def _render(self):
        return as_id_list(self.tok.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=True))

    def ask(self, user_text, max_new, temperature=None, live=True):
        self.messages.append({"role": "user", "content": user_text})
        full = self._render()
        if len(full) > self.s.ctx:
            raise RuntimeError(
                f"insufficient context: {len(full)} positions needed of {self.s.ctx}")

        keep = 0
        for a, b in zip(full, self.committed):
            if a != b:
                break
            keep += 1
        delta = full[keep:]
        if not delta:
            raise RuntimeError("empty delta: nothing to process")

        t0 = time.time()
        acc = []                 # every id produced this turn
        split = {"at": None}     # index in acc of the </think> token
        shown = {"think": False, "answer": False}
        printed = {"think": 0, "answer": 0}
        ttft = {"value": None}

        def _stream(phase, text):
            new = text[printed[phase]:]
            if not new:
                return
            if not shown[phase]:
                if not new.strip():
                    return
                (ui.begin_thinking if phase == "think" else ui.begin_answer)()
                shown[phase] = True
                # The template emits '</think>\n\n' before the answer; skipping
                # that leading whitespace keeps the reply flush with its label.
                printed[phase] = len(text) - len(text.lstrip())
                new = text[printed[phase]:]
            sys.stdout.write(new)
            sys.stdout.flush()
            printed[phase] = len(text)

        def on_token(token):
            acc.append(token)
            if token == self.think_close_id and split["at"] is None:
                split["at"] = len(acc) - 1
                if live and shown["think"]:
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                return
            if not live:
                return
            if split["at"] is None:
                if ttft["value"] is None:
                    ttft["value"] = time.time() - t0
                if self.show_thinking:
                    _stream("think", self.tok.decode(acc, skip_special_tokens=True))
                else:
                    ui.thinking(len(acc), time.time() - t0)
                return
            answer_ids = acc[split["at"] + 1:]
            _stream("answer", self.tok.decode(answer_ids, skip_special_tokens=True))

        produced, reason, pos = self.s.turn(
            delta, max_new, keep, on_token, temperature=temperature)

        if split["at"] is None:
            reasoning = self.tok.decode(produced, skip_special_tokens=True).strip()
            answer = ""
        else:
            reasoning = self.tok.decode(
                produced[:split["at"]], skip_special_tokens=True).strip()
            answer = self.tok.decode(
                produced[split["at"] + 1:], skip_special_tokens=True).strip()

        # reasoning_content is the field the MiniMax template reads directly, so
        # we never have to re-embed <think> markers into the content ourselves;
        # the template also drops it automatically on the next user turn.
        self.messages.append({"role": "assistant", "content": answer,
                              "reasoning_content": reasoning})
        self.committed = full + produced

        if live and (shown["answer"] or shown["think"]):
            sys.stdout.write("\n")
            sys.stdout.flush()
        else:
            ui.clear_status()

        if split["at"] is None:
            ui.warning(f"stopped while still reasoning after {len(produced)} tokens "
                       f"({reason.lower()}) — no answer yet; raise --max-tokens")

        dt = time.time() - t0
        self.last_stats = {
            "tokens": len(produced), "elapsed": dt, "pos": pos, "ctx": self.s.ctx,
            "reused": keep, "prompt_tokens": len(full), "reason": reason,
            "ttft": ttft["value"],
        }
        ui.metrics(**self.last_stats)
        return answer


def main():
    ap = argparse.ArgumentParser(description="MiniMax-M2 chat bridge for Picchio")
    ap.add_argument("prompt", nargs="?", help="single-shot prompt (omit for interactive)")
    ap.add_argument("--model", required=True, help="converted model folder")
    ap.add_argument("--exe", default="./picchio.exe" if os.name == "nt" else "./picchio")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer path/repo (default: --model folder)")
    ap.add_argument("--ctx", type=int, default=4096,
                    help="context positions; MiniMax reasons at length, so this "
                         "defaults higher than the Qwen bridge")
    ap.add_argument("--pin-gb", type=float, default=12)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 6)
    ap.add_argument("--io-threads", type=int,
                    help="parallel expert reads (engine default: 4)")
    ap.add_argument("--async-moe", action="store_true",
                    help="overlap routed expert reads with CPU expert compute")
    ap.add_argument("--direct", action="store_true",
                    help="use unbuffered expert reads (best paired with --async-moe)")
    ap.add_argument("--model-aux", default=None)
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="budget per reply; it has to cover the reasoning too")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="MiniMax-M2's own recommendation is 1.0")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--show-thinking", action="store_true",
                    help="stream the reasoning block instead of a progress spinner")
    ap.add_argument("--system", default=None,
                    help="optional system prompt (the template supplies a default)")
    args = ap.parse_args()

    ui.configure_utf8()

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model,
                                        trust_remote_code=True)
    sampling = {"TEMPERATURE": args.temperature, "TOPP": args.top_p, "TOPK": args.top_k}
    session = PicchioSession(args.exe, args.model, args.ctx, args.pin_gb,
                             args.threads, args.model_aux, sampling,
                             async_moe=args.async_moe, direct=args.direct,
                             io_threads=args.io_threads)
    chat = MiniMaxChat(session, tok, system=args.system,
                       show_thinking=args.show_thinking)
    single = args.prompt is not None

    def show_header():
        ui.header("MiniMax-M2", args.model, session.ctx, args.pin_gb,
                  args.temperature, args.threads, not single,
                  "always on" + ("" if args.show_thinking else " (hidden)"),
                  args.async_moe, args.direct, args.io_threads)

    show_header()

    try:
        if single:
            ui.show_user(args.prompt)
            chat.ask(args.prompt, args.max_tokens, temperature=args.temperature)
        else:
            while True:
                try:
                    user = ui.prompt()
                except (EOFError, KeyboardInterrupt):
                    ui.write()
                    break
                command = user.lower()
                if command in ("/exit", "/quit"):
                    break
                if not user:
                    continue
                if command == "/help":
                    ui.help_text()
                    continue
                if command == "/clear":
                    ui.clear_screen()
                    show_header()
                    continue
                if command == "/reset":
                    chat.reset()
                    ui.write("  " + ui.DIM("Conversation and KV cache reset."))
                    continue
                if command == "/stats":
                    ui.show_stats(chat.last_stats)
                    continue
                if command == "/settings":
                    show_header()
                    continue
                if command.startswith("/"):
                    ui.warning(f"Unknown command: {user} · use /help")
                    continue
                try:
                    chat.ask(user, args.max_tokens, temperature=args.temperature)
                except KeyboardInterrupt:
                    ui.interrupted()
    finally:
        session.close()


if __name__ == "__main__":
    main()
