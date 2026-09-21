"""Shared terminal presentation for Picchio's chat bridges.

All decoration is written to stderr. Model content stays on stdout so piping and
redirecting a response continues to produce clean text.
"""
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import picchio_logo


def configure_utf8():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _enable_ansi():
    if os.environ.get("NO_COLOR") or not sys.stderr.isatty():
        return False
    if sys.platform == "win32":
        try:
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.SetConsoleOutputCP(65001)
            handle = kernel.GetStdHandle(-12)  # STDERR
            mode = ctypes.c_uint32()
            if not kernel.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            kernel.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            return False
    return True


configure_utf8()
ANSI = _enable_ansi()


def paint(code):
    return (lambda text: f"\x1b[{code}m{text}\x1b[0m") if ANSI else str


GREEN = paint("38;2;55;161;89")
RED = paint("38;2;226;74;58")
BLUE = paint("38;2;84;160;255")
DIM = paint("2")
BOLD = paint("1")
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def write(text="", end="\n"):
    sys.stderr.write(str(text) + end)
    sys.stderr.flush()


def status(text):
    if ANSI:
        sys.stderr.write("\r\x1b[2K" + text)
        sys.stderr.flush()


def clear_status():
    if ANSI:
        sys.stderr.write("\r\x1b[2K")
        sys.stderr.flush()


def _width():
    return max(36, min(shutil.get_terminal_size((80, 24)).columns - 4, 76))


def rule():
    write("  " + DIM("─" * _width()))


def _labeled(key, value):
    available = max(20, _width() - 11)
    lines = textwrap.wrap(str(value), width=available,
                          break_long_words=False, break_on_hyphens=False) or [""]
    write(f"  {DIM(key.ljust(11))}{lines[0]}")
    for line in lines[1:]:
        write(" " * 13 + line)


def model_precision(model):
    try:
        config = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
        return f"INT{int(config.get('picchio_expert_bits', 4))}"
    except (OSError, ValueError, TypeError):
        return "quantized"


def header(family, model, ctx, pin_gb, temperature, threads, interactive,
           reasoning=None, async_moe=False, direct=False, io_threads=None,
           gpu_mode=None):
    precision = model_precision(model)
    execution = "hybrid CPU/GPU" if gpu_mode else "CPU streaming"
    picchio_logo.banner(f"{family} · {precision} · {execution}", compact=True)

    runtime = f"{threads} CPU threads · {pin_gb:g} GB expert cache"
    io = []
    if async_moe:
        io.append("async")
    if direct:
        io.append("direct I/O")
    if io_threads:
        io.append(f"{io_threads} I/O threads")
    if gpu_mode:
        io.append(gpu_mode)
    if io:
        runtime += " · " + " / ".join(io)

    generation = f"context {ctx:,} · temperature {temperature:g}"
    if reasoning:
        generation += f" · reasoning {reasoning}"

    rows = (("MODEL", Path(model).name), ("RUNTIME", runtime),
            ("GENERATION", generation))
    write()
    for key, value in rows:
        _labeled(key, value)
    write()
    write(f"  {GREEN('●')} {BOLD('Ready')}" +
          (DIM("  ·  /help for commands") if interactive else ""))
    rule()


def begin_message(role):
    clear_status()
    color = BLUE if role == "YOU" else GREEN
    if role == "THINKING":
        write(f"\n  {DIM(BOLD(role))}")
    else:
        write(f"\n  {color(BOLD(role))}")


def show_user(text):
    begin_message("YOU")
    write(f"  {text}")


def prompt():
    begin_message("YOU")
    sys.stderr.write("  " + BLUE("›") + " ")
    sys.stderr.flush()
    return input().strip()


def begin_answer():
    begin_message("PICCHIO")
    write("  ", end="")


def begin_thinking():
    begin_message("THINKING")
    write("  ", end="")


def thinking(token_count, elapsed):
    glyph = SPINNER[token_count % len(SPINNER)]
    status(DIM(f"  {glyph} Thinking · {token_count} tokens · {elapsed:.1f}s"))


def metrics(tokens, elapsed, pos, ctx, reused, prompt_tokens, reason=None,
            ttft=None, prefill_seconds=None, prefill_tokens=None,
            prefill_tokens_per_s=None, decode_seconds=None, decode_tokens=None,
            decode_tokens_per_s=None):
    tps = tokens / elapsed if elapsed > 0 else 0.0
    tps_text = f"{tps:.2f}" if tps < 1.0 else f"{tps:.1f}"
    used = min(pos, ctx)
    reuse = (100.0 * reused / prompt_tokens) if prompt_tokens else 0.0
    parts = [f"{tokens} tokens", f"{elapsed:.1f}s", f"{tps_text} tok/s total",
             f"context {used:,}/{ctx:,}", f"reused {reuse:.0f}%"]
    if decode_tokens_per_s is not None:
        parts.insert(2, f"decode {decode_tokens_per_s:.2f} tok/s")
    if prefill_seconds is not None:
        parts.insert(2, f"prefill {prefill_seconds:.1f}s")
    if ttft is not None:
        parts.insert(2, f"TTFT {ttft:.1f}s")
    if reason and reason not in ("RETURN", "CALL", "EOS"):
        parts.append(reason.lower())
    lines = []
    current = ""
    for part in parts:
        candidate = part if not current else current + " · " + part
        if current and len(candidate) > _width():
            lines.append(current)
            current = part
        else:
            current = candidate
    if current:
        lines.append(current)
    for line in lines:
        write("  " + DIM(line))


def warning(message):
    write(f"  {RED('!')} {message}")


def interrupted():
    clear_status()
    write("\n  " + DIM("Interrupted"))


def clear_screen():
    if ANSI:
        sys.stderr.write("\x1b[2J\x1b[H")
    else:
        sys.stderr.write("\n" * 3)
    sys.stderr.flush()


def help_text():
    write()
    write(f"  {BOLD('COMMANDS')}")
    write("  /help       Show this list")
    write("  /clear      Clear the terminal")
    write("  /reset      Reset conversation and KV cache")
    write("  /stats      Show the latest generation metrics")
    write("  /settings   Show the active model settings")
    write("  /exit       Close the chat")


def show_stats(stats):
    if not stats:
        write("  " + DIM("No response has been generated yet."))
        return
    write()
    write(f"  {BOLD('LATEST RESPONSE')}")
    metrics(**stats)
