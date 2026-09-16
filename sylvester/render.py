from __future__ import annotations

import os
import sys
from fractions import Fraction

from .exact import Surd

_ASCII = False
_COLOR = True


def enable_windows_ansi():
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel = ctypes.windll.kernel32
        for handle in (-11, -12):
            mode = ctypes.c_ulong()
            if kernel.GetConsoleMode(kernel.GetStdHandle(handle), ctypes.byref(mode)):
                kernel.SetConsoleMode(kernel.GetStdHandle(handle), mode.value | 0x0004)
        return True
    except Exception:
        return False


def autodetect():
    global _ASCII, _COLOR
    enable_windows_ansi()
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    _ASCII = "utf" not in encoding and encoding not in ("cp65001",)
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        _COLOR = False


def set_ascii(flag):
    global _ASCII
    _ASCII = flag


def set_color(flag):
    global _COLOR
    _COLOR = flag


def glyph(unicode_char, fallback):
    return fallback if _ASCII else unicode_char


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GREY = "\033[90m"
    ON_BLUE = "\033[44m"
    ON_GREY = "\033[100m"


def paint(text, *codes):
    if not _COLOR or not codes:
        return text
    return "".join(codes) + text + C.RESET


def dim(text):
    return paint(text, C.DIM)


def visible_len(text):
    out, i, n = 0, 0, len(text)
    while i < n:
        if text[i] == "\033":
            while i < n and text[i] != "m":
                i += 1
            i += 1
        else:
            out += 1
            i += 1
    return out


def slice_visible(text, start, width):
    if start <= 0 and width >= visible_len(text):
        return text
    out = []
    seen = 0
    taken = 0
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "":
            j = i
            while j < n and text[j] != "m":
                j += 1
            out.append(text[i:j + 1])
            i = j + 1
            continue
        if seen >= start:
            if taken >= width:
                break
            out.append(text[i])
            taken += 1
        seen += 1
        i += 1
    return "".join(out)


def pad(text, width, align="<"):
    gap = width - visible_len(text)
    if gap <= 0:
        return text
    if align == ">":
        return " " * gap + text
    if align == "^":
        left = gap // 2
        return " " * left + text + " " * (gap - left)
    return text + " " * gap


def radical(d):
    if d == -1:
        return "i"
    if d < 0:
        return "i" + glyph("√", "sqrt") + str(-d)
    return glyph("√", "sqrt") + str(d)


def _int_term(m, rad):
    if m == 1:
        return rad
    if m == -1:
        return "-" + rad
    return "%d%s" % (m, rad)


def fmt(value):
    if isinstance(value, Surd):
        return _fmt_surd(value)
    if isinstance(value, complex):
        if abs(value.imag) < 1e-12:
            return "%.6g" % value.real
        return "%.6g%+.6gi" % (value.real, value.imag)
    if isinstance(value, float):
        return "%.6g" % value
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return "%d/%d" % (value.numerator, value.denominator)


def _fmt_surd(value):
    a, b, d = value.a, value.b, value.d
    rad = radical(d)
    if a.denominator == b.denominator and a.denominator > 1:
        if a.numerator:
            body = "%d %s %s" % (
                a.numerator,
                "-" if b.numerator < 0 else "+",
                _int_term(abs(b.numerator), rad),
            )
        else:
            body = _int_term(b.numerator, rad)
        return "(%s)/%d" % (body, a.denominator)
    tail = rad if abs(b) == 1 else fmt(abs(b)) + rad
    if a == 0:
        return ("-" + tail) if b < 0 else tail
    return "%s %s %s" % (fmt(a), "-" if b < 0 else "+", tail)


def fmt_coeff(value):
    if value == 1:
        return ""
    if value == -1:
        return "-"
    text = fmt(value)
    return "(%s)" % text if " " in text else text


def matrix_lines(matrix, split_at=None, highlight=None, width_hint=None):
    rows = matrix.rows if hasattr(matrix, "rows") else matrix
    if not rows:
        return ["(empty)"]
    ncols = len(rows[0])
    cells = [[fmt(v) for v in row] for row in rows]
    widths = [max(len(cells[r][c]) for r in range(len(rows))) for c in range(ncols)]
    if width_hint:
        widths = [max(w, width_hint) for w in widths]
    if _ASCII:
        left, mid_l, right = "[", "[", "["
        rl, rm, rr = "]", "]", "]"
        divider = "|"
    else:
        left, mid_l, right = "⎡", "⎢", "⎣"
        rl, rm, rr = "⎤", "⎥", "⎦"
        divider = "│"
    out = []
    n = len(rows)
    for r in range(n):
        parts = []
        for c in range(ncols):
            text = cells[r][c].rjust(widths[c])
            if highlight and (r, c) in highlight:
                text = paint(text, C.YELLOW, C.BOLD)
            parts.append(text)
            if split_at is not None and c == split_at - 1:
                parts.append(divider)
        body = " " + "  ".join(parts) + " "
        if n == 1:
            out.append("[" + body + "]")
        elif r == 0:
            out.append(left + body + rl)
        elif r == n - 1:
            out.append(right + body + rr)
        else:
            out.append(mid_l + body + rm)
    return out


def matrix_str(matrix, split_at=None, indent=""):
    return "\n".join(indent + line for line in matrix_lines(matrix, split_at))


def side_by_side(blocks, gap=4, labels=None):
    grids = [b.splitlines() if isinstance(b, str) else list(b) for b in blocks]
    height = max((len(g) for g in grids), default=0)
    widths = [max((visible_len(l) for l in g), default=0) for g in grids]
    out = []
    if labels:
        out.append((" " * gap).join(pad(lb, w, "^") for lb, w in zip(labels, widths)))
    for i in range(height):
        parts = []
        for g, w in zip(grids, widths):
            parts.append(pad(g[i] if i < len(g) else "", w))
        out.append((" " * gap).join(parts).rstrip())
    return "\n".join(out)


def tuple_str(values):
    return "(" + ", ".join(fmt(v) for v in values) + ")"


def rule(width=64, char=None):
    return (char or glyph("─", "-")) * width


def heading(text, width=64):
    return paint(text.upper(), C.CYAN, C.BOLD) + "\n" + paint(rule(width), C.CYAN)


def subheading(text):
    return paint(text, C.BOLD)


def bullet(text):
    return "  " + glyph("•", "*") + " " + text


def table(rows, headers=None, indent="  "):
    if not rows:
        return ""
    body = [[str(c) for c in row] for row in rows]
    cols = max(len(r) for r in body)
    body = [r + [""] * (cols - len(r)) for r in body]
    head = list(headers) + [""] * (cols - len(headers)) if headers else None
    widths = [max(visible_len(r[c]) for r in body) for c in range(cols)]
    if head:
        widths = [max(widths[c], visible_len(head[c])) for c in range(cols)]
    out = []
    if head:
        out.append(indent + "  ".join(paint(pad(head[c], widths[c]), C.BOLD) for c in range(cols)))
        out.append(indent + "  ".join(glyph("─", "-") * widths[c] for c in range(cols)))
    for r in body:
        out.append(indent + "  ".join(pad(r[c], widths[c]) for c in range(cols)).rstrip())
    return "\n".join(out)


def lam():
    return glyph("λ", "L")


def arrow():
    return glyph("→", "->")


def leftrightarrow():
    return glyph("↔", "<->")


def dot():
    return glyph("·", ".")


def check():
    return glyph("✓", "OK")


def cross():
    return glyph("✗", "X")


