#!/usr/bin/env python3
"""Chat GPT-OSS token-exact: Harmony ufficiale + sessione Picchio persistente.

Il rendering, la tokenizzazione e il parsing appartengono a `openai-harmony`.
Picchio riceve e restituisce soltanto ID token raw. Tra turni viene riusato il
prefisso comune della KV-cache, perché il re-render Harmony non è prefix-preserving
(l'analysis viene scartato e `<|return|>` diventa `<|end|>`).
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
    """Processo Picchio persistente in modalità SERVICE."""

    def __init__(self, exe, model, ctx, pin_gb, threads, model_aux):
        env = os.environ.copy()
        for name in ("INPUT", "PROMPT", "INPUT_FILE", "OUTPUT", "MODEL_AUX",
                     "TRACE_NUMERIC", "ORACLE_DIR"):
            env.pop(name, None)
        env.update({"SERVICE": "1", "TEMPERATURE": "0", "REP": "1",
                    "CTX": str(ctx), "PIN_GB": str(pin_gb),
                    "OMP_NUM_THREADS": str(threads)})
        if model_aux:
            env["MODEL_AUX"] = model_aux
        self.proc = subprocess.Popen(
            [str(exe), str(model)], env=env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, text=True, encoding="ascii",
            errors="replace", bufsize=1)
        ready = self._line()
        if not ready.startswith("READY "):
            raise RuntimeError(f"avvio servizio fallito: {ready}")
        fields = ready.split()
        self.ctx = int(fields[1])
        self.vocab = int(fields[2])
        self.stop_ids = [int(x) for x in fields[3:]]

    def _line(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("il servizio Picchio si è chiuso inaspettatamente")
        return line.strip()

    def turn(self, ids, max_new, keep, on_token):
        payload = f"TURN {max_new} {keep} {len(ids)} " + " ".join(map(str, ids)) + "\n"
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
                raise RuntimeError(f"frame di protocollo inatteso: {line}")

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
    """Conversazione Harmony con riuso del prefisso già consumato dal modello."""

    def __init__(self, session, reasoning, current_date):
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.session = session
        system = (SystemContent.new()
                  .with_reasoning_effort(ReasoningEffort(reasoning.capitalize()))
                  .with_conversation_start_date(current_date))
        self.messages = [Message.from_role_and_content(Role.SYSTEM, system)]
        self.committed = []
        if session is not None and set(session.stop_ids) - set(self.encoding.stop_tokens()):
            raise RuntimeError(f"stop token del runtime incoerenti: {session.stop_ids}")

    def render(self, user_text):
        messages = self.messages + [Message.from_role_and_content(Role.USER, user_text)]
        return self.encoding.render_conversation_for_completion(
            Conversation.from_messages(messages), Role.ASSISTANT, KEEP_ANALYSIS)

    def ask(self, user_text, max_new, show_channel="final"):
        full = self.render(user_text)
        if len(full) > self.session.ctx:
            raise RuntimeError(
                f"contesto insufficiente: servono {len(full)} posizioni su {self.session.ctx}")
        keep = 0
        for a, b in zip(full, self.committed):
            if a != b:
                break
            keep += 1
        delta = full[keep:]
        if not delta:
            raise RuntimeError("delta vuoto: nulla da elaborare")

        parser = StreamableParser(self.encoding, Role.ASSISTANT, strict=False)

        def on_token(token):
            parser.process(token)
            chunk = parser.last_content_delta
            if chunk and parser.current_channel == show_channel:
                print(chunk, end="", flush=True)

        print(f"[riuso {keep}/{len(full)} posizioni, {len(delta)} da elaborare]",
              file=sys.stderr)
        produced, reason, pos = self.session.turn(delta, max_new, keep, on_token)
        print()
        self.committed = full + produced

        try:
            replies = self.encoding.parse_messages_from_completion_tokens(
                produced, Role.ASSISTANT)
        except Exception as exc:
            replies = parser.messages
            print(f"[risposta incompleta ({reason}): {exc}]", file=sys.stderr)
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
    parser = argparse.ArgumentParser(description="Chat GPT-OSS con Picchio")
    parser.add_argument("prompt", nargs="?", help="domanda singola; omessa avvia la chat")
    parser.add_argument("--model", default=r"D:\gptoss_i4")
    parser.add_argument("--exe", default="picchio.exe")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--ctx", type=int, default=512)
    parser.add_argument("--reasoning", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--pin-gb", type=int, default=1)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--model-aux")
    parser.add_argument("--show-analysis", action="store_true",
                        help="mostra il canale analysis invece di final")
    parser.add_argument("--dry-run", action="store_true",
                        help="renderizza e verifica gli ID senza avviare Picchio")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        chat = HarmonyChat(None, args.reasoning, args.date)
        text = args.prompt if args.prompt is not None else input("Tu: ")
        ids = chat.render(text)
        rendered = chat.encoding.decode_utf8(ids)
        if chat.encoding.encode(rendered, allowed_special="all") != ids:
            raise SystemExit("round-trip Harmony degli ID fallito")
        print(json.dumps({"token_count": len(ids), "ids": ids, "rendered": rendered},
                         ensure_ascii=False, indent=2))
        return

    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    if not exe.is_file():
        parser.error(f"eseguibile non trovato: {exe}")
    if not model.is_dir():
        parser.error(f"modello non trovato: {model}")

    session = PicchioSession(exe, model, args.ctx, args.pin_gb, args.threads,
                             resolve_aux(model, args.model_aux))
    chat = HarmonyChat(session, args.reasoning, args.date)
    channel = "analysis" if args.show_analysis else "final"
    single = args.prompt is not None

    try:
        while True:
            if single:
                text = args.prompt
            else:
                try:
                    text = input("\nTu: ").strip()
                except EOFError:
                    break
                if not text:
                    continue
                if text in ("/exit", "/quit"):
                    break
            print("Assistente: ", end="", flush=True)
            replies, reason, pos = chat.ask(text, args.max_tokens, channel)
            if args.json:
                print(json.dumps({"reason": reason, "pos": pos,
                                  "messages": [m.to_dict() for m in replies]},
                                 ensure_ascii=False, default=str, indent=2))
            elif reason not in ("RETURN", "CALL"):
                print(f"[interrotto: {reason}]", file=sys.stderr)
            if single:
                break
    except KeyboardInterrupt:
        print("\nInterrotto.", file=sys.stderr)
    finally:
        session.close()


if __name__ == "__main__":
    main()
