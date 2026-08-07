#!/usr/bin/env python3
"""server.py — API HTTP OpenAI-compatible per Picchio.

Espone `POST /v1/chat/completions` (streaming SSE e non), `GET /v1/models` e
`GET /health`. Si appoggia interamente all'infrastruttura già token-exact di
`chat.py`: rendering/parsing con `openai-harmony`, un solo processo Picchio in
modalità SERVICE con riuso del prefisso KV.

Il modello è un unico processo con una sola KV-cache: le richieste vengono
serializzate con un lock. Il riuso del prefisso lavora sui token, quindi fra
richieste diverse recupera comunque l'overlap e ricalcola solo alla divergenza.

Sampling per-richiesta: `max_tokens`, `reasoning_effort` e `no_reasoning` sono
gestiti a ogni richiesta (lato Harmony/protocollo TURN). temperature/top-p/top-k
sono fissati all'avvio del server (letti da env dal C una sola volta): un
override per-richiesta richiederebbe di estendere il protocollo TURN.
"""
import argparse
import json
import sys
import threading
import time
import uuid
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai_harmony import (
    Conversation, DeveloperContent, HarmonyEncodingName, Message,
    ReasoningEffort, RenderConversationConfig, Role, StreamableParser,
    SystemContent, load_harmony_encoding,
)

from chat import PicchioSession, resolve_aux

KEEP_ANALYSIS = RenderConversationConfig(auto_drop_analysis=False)

# reason del frame DONE del C -> finish_reason OpenAI
FINISH_REASON = {
    "RETURN": "stop",
    "CALL": "tool_calls",
    "MAX_TOKENS": "length",
    "CONTEXT_FULL": "length",
}


class Engine:
    """Sessione Picchio condivisa, serializzata, con riuso del prefisso KV."""

    def __init__(self, session, model_name, default_reasoning, default_date):
        self.session = session
        self.model_name = model_name
        self.default_reasoning = default_reasoning
        self.default_date = default_date
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.committed = []          # ultimi ID token consumati dal modello
        self.lock = threading.Lock()  # una generazione alla volta

    def build_ids(self, oai_messages, reasoning, req_date, no_reasoning):
        """Costruisce gli ID Harmony da una lista di messaggi stile OpenAI."""
        system = (SystemContent.new()
                  .with_reasoning_effort(ReasoningEffort(reasoning.capitalize()))
                  .with_conversation_start_date(req_date))
        messages = [Message.from_role_and_content(Role.SYSTEM, system)]

        # I messaggi system/developer OpenAI diventano istruzioni del developer.
        dev_text = "\n\n".join(
            m["content"] for m in oai_messages
            if m.get("role") in ("system", "developer") and m.get("content"))
        if dev_text:
            dev = DeveloperContent.new().with_instructions(dev_text)
            messages.append(Message.from_role_and_content(Role.DEVELOPER, dev))

        for m in oai_messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "user":
                messages.append(Message.from_role_and_content(Role.USER, content))
            elif role == "assistant":
                # Lo storico dell'assistant dal client ha solo il canale final.
                messages.append(Message.from_role_and_content(Role.ASSISTANT, content))
            # system/developer già gestiti sopra; altri ruoli ignorati.

        full = self.encoding.render_conversation_for_completion(
            Conversation.from_messages(messages), Role.ASSISTANT, KEEP_ANALYSIS)

        final_prefill = (
            self.encoding.encode("<|channel|>final<|message|>", allowed_special="all")
            if no_reasoning else [])
        return list(full) + final_prefill, final_prefill

    def generate(self, ids, final_prefill, max_new, on_delta):
        """Esegue un turno. `on_delta(channel, text)` per ogni frammento.

        Ritorna (final_text, analysis_text, finish_reason, prompt_tokens,
        completion_tokens).
        """
        if len(ids) > self.session.ctx:
            raise ValueError(
                f"contesto insufficiente: {len(ids)} posizioni su {self.session.ctx}")

        # prefisso comune con quanto già consumato dal modello
        keep = 0
        for a, b in zip(ids, self.committed):
            if a != b:
                break
            keep += 1
        delta = ids[keep:]
        if not delta:
            raise ValueError("delta vuoto: nulla da elaborare")
        print(f"[turno: riuso {keep}/{len(ids)} posizioni, {len(delta)} da elaborare]",
              file=sys.stderr)

        parser = StreamableParser(self.encoding, Role.ASSISTANT, strict=False)
        for tok in final_prefill:
            parser.process(tok)

        collected = {"final": [], "analysis": []}

        def on_token(token):
            parser.process(token)
            chunk = parser.last_content_delta
            if chunk:
                ch = parser.current_channel
                if ch in collected:
                    collected[ch].append(chunk)
                on_delta(ch, chunk)

        produced, reason, _pos = self.session.turn(delta, max_new, keep, on_token)
        self.committed = ids + produced

        return (
            "".join(collected["final"]),
            "".join(collected["analysis"]),
            FINISH_REASON.get(reason, "stop"),
            len(ids),
            len(final_prefill) + len(produced),
        )


class Handler(BaseHTTPRequestHandler):
    engine = None  # iniettato in main()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # --- utilità di risposta ---
    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message, err_type="invalid_request_error"):
        self._json(code, {"error": {"message": message, "type": err_type}})

    def _sse_open(self):
        # Streaming senza Content-Length: si chiude la connessione a fine stream
        # così il client vede l'EOF (niente chunked manuale su BaseHTTPRequestHandler).
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse_send(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n\n")
        self.wfile.flush()

    # --- routing ---
    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok"})
        elif self.path in ("/v1/models", "/models"):
            self._json(200, {
                "object": "list",
                "data": [{"id": self.engine.model_name, "object": "model",
                          "created": 0, "owned_by": "picchio"}],
            })
        else:
            self._error(404, f"non trovato: {self.path}", "not_found")

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, f"non trovato: {self.path}", "not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, f"JSON non valido: {exc}")
            return

        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            self._error(400, "'messages' mancante o vuoto")
            return

        eng = self.engine
        reasoning = req.get("reasoning_effort", eng.default_reasoning)
        if reasoning not in ("low", "medium", "high"):
            reasoning = eng.default_reasoning
        no_reasoning = bool(req.get("no_reasoning", False))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 512)
        stream = bool(req.get("stream", False))
        req_date = req.get("date", eng.default_date)

        try:
            ids, final_prefill = eng.build_ids(messages, reasoning, req_date, no_reasoning)
        except Exception as exc:
            self._error(400, f"rendering Harmony fallito: {exc}")
            return

        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if stream:
            self._handle_stream(eng, ids, final_prefill, max_new, cid, created)
        else:
            self._handle_blocking(eng, ids, final_prefill, max_new, cid, created)

    # --- non streaming ---
    def _handle_blocking(self, eng, ids, final_prefill, max_new, cid, created):
        try:
            with eng.lock:
                final, analysis, finish, ptok, ctok = eng.generate(
                    ids, final_prefill, max_new, lambda ch, txt: None)
        except ValueError as exc:
            self._error(400, str(exc))
            return
        except Exception as exc:
            self._error(500, f"generazione fallita: {exc}", "server_error")
            return

        message = {"role": "assistant", "content": final}
        if analysis:
            message["reasoning_content"] = analysis
        self._json(200, {
            "id": cid, "object": "chat.completion", "created": created,
            "model": eng.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": ptok, "completion_tokens": ctok,
                      "total_tokens": ptok + ctok},
        })

    # --- streaming SSE ---
    def _handle_stream(self, eng, ids, final_prefill, max_new, cid, created):
        self._sse_open()

        def frame(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": eng.model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        # primo frame: ruolo
        self._sse_send(frame({"role": "assistant"}))

        def on_delta(channel, text):
            if channel == "final":
                self._sse_send(frame({"content": text}))
            elif channel == "analysis":
                self._sse_send(frame({"reasoning_content": text}))

        try:
            with eng.lock:
                _final, _analysis, finish, _p, _c = eng.generate(
                    ids, final_prefill, max_new, on_delta)
        except ValueError as exc:
            self._sse_send(frame({"content": f"[errore: {exc}]"}, "stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        except Exception as exc:
            self._sse_send(frame({"content": f"[errore interno: {exc}]"}, "stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self._sse_send(frame({}, finish))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser(description="Server API OpenAI-compatible per Picchio")
    ap.add_argument("--model", default=r"D:\gptoss20b_i4")
    ap.add_argument("--exe", default="picchio.exe")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--pin-gb", type=int, default=4)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--model-aux")
    ap.add_argument("--reasoning", choices=("low", "medium", "high"), default="medium")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    if not exe.is_file():
        ap.error(f"eseguibile non trovato: {exe}")
    if not model.is_dir():
        ap.error(f"modello non trovato: {model}")

    sampling = {"TEMPERATURE": args.temperature, "TOPP": args.top_p,
                "TOPK": args.top_k, "SEED": args.seed}
    print(f"[avvio Picchio SERVICE: {model}]", file=sys.stderr)
    session = PicchioSession(exe, model, args.ctx, args.pin_gb, args.threads,
                             resolve_aux(model, args.model_aux), sampling)
    print(f"[pronto: ctx={session.ctx} vocab={session.vocab}]", file=sys.stderr)

    Handler.engine = Engine(session, model.name, args.reasoning, args.date)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server in ascolto su http://{args.host}:{args.port}  "
          f"(POST /v1/chat/completions)]", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[arresto]", file=sys.stderr)
    finally:
        httpd.server_close()
        session.close()


if __name__ == "__main__":
    main()
