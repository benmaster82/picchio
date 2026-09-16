"""Shared terminal logo/banner for Picchio's Python bridges (chat.py, chat_qwen.py).

Renders the pixel-art woodpecker from assets/picchio.svg in ANSI truecolor, with
the wordmark to its right. Degrades to plain blocks/text when colour is off
(NO_COLOR set, or the stream is not a TTY).
"""
import os
import sys

# Palette taken straight from assets/picchio.svg.
_PAL = {
    "R": (226, 74, 58),    # crown / vent (red)
    "G": (55, 161, 89),    # body (green)
    "L": (126, 203, 146),  # breast (light green)
    "D": (44, 122, 70),    # tail (dark green)
    "g": (110, 110, 116),  # beak (grey)
    "E": (35, 35, 40),     # eye
}
_ART = [
    "..RR.....",
    ".RRR.....",
    "GGEGG....",
    "GGGGGggg.",
    "GLLLG....",
    "LLLLG....",
    ".GLRG....",
    ".GDG.....",
    "..DD.....",
    "...D.....",
]
_WORDMARK = (47, 143, 87)   # #2f8f57
_GREY = (140, 140, 140)


def _color_on(stream):
    return not os.environ.get("NO_COLOR") and getattr(stream, "isatty", lambda: False)()


def _fg(rgb, s):
    r, g, b = rgb
    return f"\x1b[38;2;{r};{g};{b}m{s}\x1b[0m"


def _compact_art(color):
    """Compress the 10-row pixel bird into five terminal half-block rows."""
    def cell(top, bottom):
        if top == "." and bottom == ".":
            return " "
        if not color:
            return "█"
        if top == bottom:
            return _fg(_PAL[top], "█")
        if bottom == ".":
            return _fg(_PAL[top], "▀")
        if top == ".":
            return _fg(_PAL[bottom], "▄")
        tr, tg, tb = _PAL[top]
        br, bg, bb = _PAL[bottom]
        return f"\x1b[38;2;{tr};{tg};{tb};48;2;{br};{bg};{bb}m▀\x1b[0m"

    return ["".join(cell(_ART[row][col], _ART[row + 1][col])
                    for col in range(len(_ART[row])))
            for row in range(0, len(_ART), 2)]


def banner(spec="int4 · streaming CPU", stream=sys.stderr, compact=False):
    """Print the woodpecker logo and wordmark to `stream` (stderr by default)."""
    color = _color_on(stream)

    def px(ch):
        if ch == ".":
            return "  "
        return _fg(_PAL[ch], "██") if color else "██"

    art = _compact_art(color) if compact else ["".join(px(c) for c in row) for row in _ART]
    if color and compact:
        text = {
            1: "\x1b[1m" + _fg(_WORDMARK, "PICCHIO") + "\x1b[0m",
            2: _fg(_GREY, "it drums the model off the disk"),
            3: _fg(_GREY, spec),
        }
    elif color:
        text = {
            3: "\x1b[1m" + _fg(_WORDMARK, "picchio") + "\x1b[0m",
            4: _fg(_GREY, "\x1b[3mit drums the model off the disk\x1b[0m"),
            5: _fg(_GREY, spec),
        }
    else:
        text = ({1: "PICCHIO", 2: "it drums the model off the disk", 3: spec}
                if compact else
                {3: "PICCHIO", 4: "it drums the model off the disk", 5: spec})

    lines = ["  " + row + ("   " + text[i] if i in text else "")
             for i, row in enumerate(art)]
    stream.write("\n" + "\n".join(lines) + "\n")
    stream.flush()
