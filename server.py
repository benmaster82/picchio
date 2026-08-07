#!/usr/bin/env python3
"""server.py — OpenAI-compatible HTTP API for Picchio.

Exposes `POST /v1/chat/completions` (streaming SSE and non-streaming),
`GET /v1/models`, and `GET /health`. It relies entirely on the already
token-exact infrastructure of `chat.py`: rendering/parsing with `openai-harmony`,
a single Picchio process in SERVICE mode with KV prefix reuse.

The model is a single process with one KV-cache: requests are serialized with a
lock. The prefix reuse works on tokens, so across different requests it still
recovers the overlap and recomputes only at the divergence.

Per-request parameters: `max_tokens`, `temperature`, `top_p`, `top_k`,
`reasoning_effort`, and `no_reasoning`. The sampling values travel in the extended
TURN header, so they change per request without restarting the server; if omitted,
the defaults passed at startup are used.
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

# C DONE-frame reason -> OpenAI finish_reason
FINISH_REASON = {
    "RETURN": "stop",
    "CALL": "tool_calls",
    "MAX_TOKENS": "length",
    "CONTEXT_FULL": "length",
}


class Engine:
    """Shared, serialized Picchio session with KV prefix reuse."""

    def __init__(self, session, model_name, default_reasoning, default_date):
        self.session = session
        self.model_name = model_name
        self.default_reasoning = default_reasoning
        self.default_date = default_date
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.committed = []          # last token IDs consumed by the model
        self.lock = threading.Lock()  # one generation at a time

    def build_ids(self, oai_messages, reasoning, req_date, no_reasoning):
        """Build the Harmony IDs from a list of OpenAI-style messages."""
        system = (SystemContent.new()
                  .with_reasoning_effort(ReasoningEffort(reasoning.capitalize()))
                  .with_conversation_start_date(req_date))
        messages = [Message.from_role_and_content(Role.SYSTEM, system)]

        # OpenAI system/developer messages become developer instructions.
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
                # The assistant history from the client has only the final channel.
                messages.append(Message.from_role_and_content(Role.ASSISTANT, content))
            # system/developer already handled above; other roles ignored.

        full = self.encoding.render_conversation_for_completion(
            Conversation.from_messages(messages), Role.ASSISTANT, KEEP_ANALYSIS)

        final_prefill = (
            self.encoding.encode("<|channel|>final<|message|>", allowed_special="all")
            if no_reasoning else [])
        return list(full) + final_prefill, final_prefill

    def generate(self, ids, final_prefill, max_new, on_delta,
                 temperature=None, top_p=None, top_k=None):
        """Run one turn. `on_delta(channel, text)` for each fragment.

        temperature/top_p/top_k: per-request override (None = session default).
        Returns (final_text, analysis_text, finish_reason, prompt_tokens,
        completion_tokens).
        """
        if len(ids) > self.session.ctx:
            raise ValueError(
                f"insufficient context: {len(ids)} positions out of {self.session.ctx}")

        # common prefix with what the model has already consumed
        keep = 0
        for a, b in zip(ids, self.committed):
            if a != b:
                break
            keep += 1
        # Always reprocess at least the last token: if the prompt is identical or a
        # prefix of what was already consumed, keep would cover everything and the
        # delta would be empty, with no token to restart generation from.
        if keep >= len(ids):
            keep = len(ids) - 1
        delta = ids[keep:]
        print(f"[turn: reuse {keep}/{len(ids)} positions, {len(delta)} to process]",
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

        produced, reason, _pos = self.session.turn(
            delta, max_new, keep, on_token,
            temperature=temperature, top_p=top_p, top_k=top_k)
        self.committed = ids + produced

        return (
            "".join(collected["final"]),
            "".join(collected["analysis"]),
            FINISH_REASON.get(reason, "stop"),
            len(ids),
            len(final_prefill) + len(produced),
        )


class Handler(BaseHTTPRequestHandler):
    engine = None  # injected in main()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # --- response helpers ---
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
        # Streaming without Content-Length: close the connection at the end of the
        # stream so the client sees the EOF (no manual chunking on BaseHTTPRequestHandler).
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
            self._error(404, f"not found: {self.path}", "not_found")

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, f"not found: {self.path}", "not_found")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as exc:
            self._error(400, f"invalid JSON: {exc}")
            return

        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            self._error(400, "'messages' missing or empty")
            return

        eng = self.engine
        reasoning = req.get("reasoning_effort", eng.default_reasoning)
        if reasoning not in ("low", "medium", "high"):
            reasoning = eng.default_reasoning
        no_reasoning = bool(req.get("no_reasoning", False))
        max_new = int(req.get("max_tokens") or req.get("max_completion_tokens") or 512)
        stream = bool(req.get("stream", False))
        req_date = req.get("date", eng.default_date)
        # Per-request sampling (None = session default).
        sampling = {"temperature": req.get("temperature"),
                    "top_p": req.get("top_p"),
                    "top_k": req.get("top_k")}

        try:
            ids, final_prefill = eng.build_ids(messages, reasoning, req_date, no_reasoning)
        except Exception as exc:
            self._error(400, f"Harmony rendering failed: {exc}")
            return

        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        if stream:
            self._handle_stream(eng, ids, final_prefill, max_new, cid, created, sampling)
        else:
            self._handle_blocking(eng, ids, final_prefill, max_new, cid, created, sampling)

    # --- non streaming ---
    def _handle_blocking(self, eng, ids, final_prefill, max_new, cid, created, sampling):
        try:
            with eng.lock:
                final, analysis, finish, ptok, ctok = eng.generate(
                    ids, final_prefill, max_new, lambda ch, txt: None, **sampling)
        except ValueError as exc:
            self._error(400, str(exc))
            return
        except Exception as exc:
            self._error(500, f"generation failed: {exc}", "server_error")
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
    def _handle_stream(self, eng, ids, final_prefill, max_new, cid, created, sampling):
        self._sse_open()

        def frame(delta, finish=None):
            return {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": eng.model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        # first frame: role
        self._sse_send(frame({"role": "assistant"}))

        def on_delta(channel, text):
            if channel == "final":
                self._sse_send(frame({"content": text}))
            elif channel == "analysis":
                self._sse_send(frame({"reasoning_content": text}))

        try:
            with eng.lock:
                _final, _analysis, finish, _p, _c = eng.generate(
                    ids, final_prefill, max_new, on_delta, **sampling)
        except ValueError as exc:
            self._sse_send(frame({"content": f"[error: {exc}]"}, "stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        except Exception as exc:
            self._sse_send(frame({"content": f"[internal error: {exc}]"}, "stop"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return

        self._sse_send(frame({}, finish))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main():
    ap = argparse.ArgumentParser(description="OpenAI-compatible API server for Picchio")
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
        ap.error(f"executable not found: {exe}")
    if not model.is_dir():
        ap.error(f"model not found: {model}")

    sampling = {"TEMPERATURE": args.temperature, "TOPP": args.top_p,
                "TOPK": args.top_k, "SEED": args.seed}
    print(f"[starting Picchio SERVICE: {model}]", file=sys.stderr)
    session = PicchioSession(exe, model, args.ctx, args.pin_gb, args.threads,
                             resolve_aux(model, args.model_aux), sampling)
    print(f"[ready: ctx={session.ctx} vocab={session.vocab}]", file=sys.stderr)

    Handler.engine = Engine(session, model.name, args.reasoning, args.date)
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[server listening on http://{args.host}:{args.port}  "
          f"(POST /v1/chat/completions)]", file=sys.stderr)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[shutting down]", file=sys.stderr)
    finally:
        httpd.server_close()
        session.close()


if __name__ == "__main__":
    main()
