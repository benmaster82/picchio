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
import subprocess
import sys
from datetime import date
from pathlib import Path

from openai_harmony import (
    Conversation, HarmonyEncodingName, Message, ReasoningEffort,
    RenderConversationConfig, Role, StreamableParser, SystemContent,
    load_harmony_encoding,
)

KEEP_ANALYSIS = RenderConversationConfig(auto_drop_analysis=False)


class PicchioSession:
    """Persistent Picchio process in SERVICE mode."""

    def __init__(self, exe, model, ctx, pin_gb, threads, model_aux, sampling=None):
        env = os.environ.copy()
        for name in ("INPUT", "PROMPT", "INPUT_FILE", "OUTPUT", "MODEL_AUX",
                     "TRACE_NUMERIC", "ORACLE_DIR"):
            env.pop(name, None)
        env.update({"SERVICE": "1", "TEMPERATURE": "0", "REP": "1",
                    "CTX": str(ctx), "PIN_GB": str(pin_gb),
                    "OMP_NUM_THREADS": str(threads)})
        if sampling:
            env.update({k: str(v) for k, v in sampling.items() if v is not None})
        # Default sampling sent with every TURN (a per-turn override is possible).
        s = sampling or {}
        self.temperature = 1.0 if s.get("TEMPERATURE") is None else float(s["TEMPERATURE"])
        self.top_p = 0.95 if s.get("TOPP") is None else float(s["TOPP"])
        self.top_k = 50 if s.get("TOPK") is None else int(s["TOPK"])
        if model_aux:
            env["MODEL_AUX"] = model_aux
        self.proc = subprocess.Popen(
            [str(exe), str(model)], env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, encoding="ascii",
            errors="replace", bufsize=1)
        ready = self._line()
        if not ready.startswith("READY "):
            raise RuntimeError(f"service startup failed: {ready}")
        fields = ready.split()
        self.ctx = int(fields[1])
        self.vocab = int(fields[2])
        self.stop_ids = [int(x) for x in fields[3:]]

    def _line(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("the Picchio service closed unexpectedly")
        return line.strip()

    def turn(self, ids, max_new, keep, on_token,
             temperature=None, top_p=None, top_k=None):
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


class HarmonyChat:
    """Harmony conversation with reuse of the prefix already consumed by the model."""

    def __init__(self, session, reasoning, current_date, no_reasoning=False):
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.session = session
        system = (SystemContent.new()
                  .with_reasoning_effort(ReasoningEffort(reasoning.capitalize()))
                  .with_conversation_start_date(current_date))
        self.messages = [Message.from_role_and_content(Role.SYSTEM, system)]
        self.committed = []
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

    def render(self, user_text):
        messages = self.messages + [Message.from_role_and_content(Role.USER, user_text)]
        return self.encoding.render_conversation_for_completion(
            Conversation.from_messages(messages), Role.ASSISTANT, KEEP_ANALYSIS)

    def ask(self, user_text, max_new, show_channel="final"):
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

        state = {"n": 0, "shown": False}

        def on_token(token):
            parser.process(token)
            state["n"] += 1
            chunk = parser.last_content_delta
            if chunk and parser.current_channel == show_channel:
                if not state["shown"]:
                    print("", file=sys.stderr)  # end the progress line
                    state["shown"] = True
                print(chunk, end="", flush=True)
            elif not state["shown"]:
                # The final channel has not started yet: show that it is working.
                print(f"\r[{parser.current_channel or 'header'}: {state['n']} tokens]",
                      end="", file=sys.stderr, flush=True)

        print(f"[reuse {keep}/{len(full)} positions, {len(delta)} to process]",
              file=sys.stderr)
        produced, reason, pos = self.session.turn(delta, max_new, keep, on_token)
        print()
        self.committed = full + self.final_prefill + produced

        try:
            replies = self.encoding.parse_messages_from_completion_tokens(
                self.final_prefill + produced, Role.ASSISTANT)
        except Exception as exc:
            replies = parser.messages
            print(f"[incomplete response ({reason}): {exc}]", file=sys.stderr)
        self.messages.append(Message.from_role_and_content(Role.USER, user_text))
        self.messages.extend(replies)
        return replies, reason, pos


def resolve_aux(model, override):
    if override is not None:
        return override
    if model != Path(r"D:\gptoss_i4").resolve():
        return None
    root = Path(__file__).resolve().parent
    candidates = [root / "expert_biases.safetensors", root / "model-00012.safetensors"]
    return ";".join(str(path) for path in candidates if path.is_file())


def main():
    parser = argparse.ArgumentParser(description="GPT-OSS chat with Picchio")
    parser.add_argument("prompt", nargs="?", help="single question; omit to start the chat")
    parser.add_argument("--model", default=r"D:\gptoss_i4")
    parser.add_argument("--exe", default="picchio.exe")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--reasoning", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--no-reasoning", action="store_true",
                        help="skip the analysis channel: pre-commit the final channel, "
                             "the model answers without reasoning (faster)")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--pin-gb", type=int, default=1)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--model-aux")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = deterministic greedy; values ~0.7-1.0 avoid loops")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--show-analysis", action="store_true",
                        help="show the analysis channel instead of final")
    parser.add_argument("--dry-run", action="store_true",
                        help="render and verify the IDs without starting Picchio")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

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
                "TOPK": args.top_k, "SEED": args.seed}
    session = PicchioSession(exe, model, args.ctx, args.pin_gb, args.threads,
                             resolve_aux(model, args.model_aux), sampling)
    chat = HarmonyChat(session, args.reasoning, args.date, args.no_reasoning)
    channel = "analysis" if args.show_analysis else "final"
    single = args.prompt is not None

    try:
        while True:
            if single:
                text = args.prompt
            else:
                try:
                    text = input("\nYou: ").strip()
                except EOFError:
                    break
                if not text:
                    continue
                if text in ("/exit", "/quit"):
                    break
            print("Assistant: ", end="", flush=True)
            replies, reason, pos = chat.ask(text, args.max_tokens, channel)
            if args.json:
                print(json.dumps({"reason": reason, "pos": pos,
                                  "messages": [m.to_dict() for m in replies]},
                                 ensure_ascii=False, default=str, indent=2))
            elif reason not in ("RETURN", "CALL"):
                print(f"[interrupted: {reason}]", file=sys.stderr)
            if single:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    finally:
        session.close()


if __name__ == "__main__":
    main()
