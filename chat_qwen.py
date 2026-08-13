#!/usr/bin/env python3
"""Chat bridge for Qwen3-MoE models running on Picchio.

Mirrors chat.py but swaps the GPT-OSS "Harmony" pipeline for Qwen3's native
ChatML template, rendered by transformers' AutoTokenizer. Picchio only exchanges
raw token IDs; between turns the common KV prefix is reused. ChatML is
prefix-preserving (it just appends <|im_start|>...<|im_end|> blocks), so the
reuse is cleaner than Harmony's.

The pipe protocol (READY / TURN / TOKEN / DONE / SHUTDOWN) is identical to the
one chat.py uses, so the same Picchio SERVICE build drives both model families.

Requires: pip install transformers
Build/convert first (see PORTING_QWEN3.md):
  python convert.py --model Qwen/Qwen3-30B-A3B --output C:/models/qwen3_30b_i4 --download

Run:
  python chat_qwen.py --model C:/models/qwen3_30b_i4 --no-reasoning \
      --ctx 2048 --pin-gb 8 --max-tokens 256 --temperature 0.7
"""
import argparse
import os
import subprocess
import sys
import time

try:
    from transformers import AutoTokenizer
except ImportError:
    print("pip install transformers", file=sys.stderr)
    sys.exit(1)


class PicchioSession:
    """Persistent Picchio process in SERVICE mode (identical transport to chat.py)."""

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


class Qwen3Chat:
    """Qwen3 ChatML conversation with KV-prefix reuse between turns."""

    def __init__(self, session, tokenizer, no_reasoning=False, system=None):
        self.s = session
        self.tok = tokenizer
        self.no_reasoning = no_reasoning
        self.messages = []
        if system:
            self.messages.append({"role": "system", "content": system})
        self.committed = []          # exact token stream the model has consumed
        # Sanity: the runtime's stop id should be the tokenizer's eos.
        eos = tokenizer.eos_token_id
        if session is not None and eos is not None and eos not in session.stop_ids:
            print(f"  ! warning: runtime stop_ids {session.stop_ids} do not include "
                  f"tokenizer eos {eos}", file=sys.stderr)

    def _render(self):
        """Render the whole conversation to token IDs, adding the assistant prompt."""
        kwargs = dict(add_generation_prompt=True, tokenize=True)
        # enable_thinking is honored by Qwen3 templates; guard for older versions.
        try:
            return self.tok.apply_chat_template(
                self.messages, enable_thinking=not self.no_reasoning, **kwargs)
        except TypeError:
            return self.tok.apply_chat_template(self.messages, **kwargs)

    def ask(self, user_text, max_new, temperature=None, live=True):
        self.messages.append({"role": "user", "content": user_text})
        full = self._render()
        if len(full) > self.s.ctx:
            raise RuntimeError(
                f"insufficient context: {len(full)} positions needed of {self.s.ctx}")

        # Longest common prefix with what the model already holds in its KV.
        keep = 0
        for a, b in zip(full, self.committed):
            if a != b:
                break
            keep += 1
        delta = full[keep:]
        if not delta:
            raise RuntimeError("empty delta: nothing to process")

        t0 = time.time()
        acc = []          # ids produced so far
        printed = 0       # chars already streamed to stdout

        def on_token(token):
            # Incremental decode: re-decode the tail and print only the new suffix.
            # Robust to multibyte tokens — a partial UTF-8 sequence just prints on
            # the next token, once the codepoint is complete.
            nonlocal printed
            acc.append(token)
            if not live:
                return
            text = self.tok.decode(acc, skip_special_tokens=True)
            if len(text) > printed:
                sys.stdout.write(text[printed:])
                sys.stdout.flush()
                printed = len(text)

        produced, reason, pos = self.s.turn(
            delta, max_new, keep, on_token, temperature=temperature)

        text = self.tok.decode(produced, skip_special_tokens=True).strip()
        self.messages.append({"role": "assistant", "content": text})
        # committed is the actual consumed stream, so next turn's prefix reuse is exact.
        self.committed = full + produced

        if live:
            sys.stdout.write("\n")
            sys.stdout.flush()
        dt = time.time() - t0
        n = len(produced)
        tps = n / dt if dt > 0 else 0.0
        print(f"  {n} tokens · {dt:.1f}s · {tps:.1f} tok/s · "
              f"reused {keep}/{len(full)} · {reason}", file=sys.stderr)
        return text


def main():
    ap = argparse.ArgumentParser(description="Qwen3-MoE chat bridge for Picchio")
    ap.add_argument("prompt", nargs="?", help="single-shot prompt (omit for interactive)")
    ap.add_argument("--model", required=True, help="converted model folder")
    ap.add_argument("--exe", default="./picchio.exe" if os.name == "nt" else "./picchio")
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer path/repo (default: --model folder)")
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--pin-gb", type=float, default=8)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 6)
    ap.add_argument("--model-aux", default=None)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--no-reasoning", action="store_true",
                    help="disable Qwen3 thinking (enable_thinking=False)")
    ap.add_argument("--system", default=None, help="optional system prompt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    sampling = {"TEMPERATURE": args.temperature, "TOPP": args.top_p, "TOPK": args.top_k}
    session = PicchioSession(args.exe, args.model, args.ctx, args.pin_gb,
                             args.threads, args.model_aux, sampling)
    chat = Qwen3Chat(session, tok, no_reasoning=args.no_reasoning, system=args.system)

    try:
        if args.prompt:
            chat.ask(args.prompt, args.max_tokens, temperature=args.temperature)
        else:
            print("  Qwen3 chat ready. Type /exit to quit.", file=sys.stderr)
            while True:
                try:
                    user = input("you > ")
                except (EOFError, KeyboardInterrupt):
                    break
                if user.strip() in ("/exit", "/quit"):
                    break
                if not user.strip():
                    continue
                chat.ask(user, args.max_tokens, temperature=args.temperature)
    finally:
        session.close()


if __name__ == "__main__":
    main()
