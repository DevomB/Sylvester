from __future__ import annotations

import os
import sys

UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"
ENTER = "enter"
ESC = "esc"
TAB = "tab"
SHIFT_TAB = "shift-tab"
BACKSPACE = "backspace"
DELETE = "delete"
HOME = "home"
END = "end"
PGUP = "pgup"
PGDN = "pgdn"
INTERRUPT = "interrupt"
UNKNOWN = "unknown"

_WIN_SPECIAL = {
    "H": UP, "P": DOWN, "K": LEFT, "M": RIGHT,
    "G": HOME, "O": END, "I": PGUP, "Q": PGDN, "S": DELETE,
}

_CSI = {
    "A": UP, "B": DOWN, "C": RIGHT, "D": LEFT,
    "H": HOME, "F": END, "Z": SHIFT_TAB,
    "1~": HOME, "4~": END, "5~": PGUP, "6~": PGDN, "3~": DELETE,
    "7~": HOME, "8~": END,
}


class Keyboard:
    def __init__(self):
        self.windows = os.name == "nt"
        self._fd = None
        self._saved = None

    def __enter__(self):
        if not self.windows:
            import termios
            import tty

            self._fd = sys.stdin.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
        return self

    def __exit__(self, *exc):
        if not self.windows and self._saved is not None:
            import termios

            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
        return False

    def read(self):
        return self._read_windows() if self.windows else self._read_posix()

    def _read_windows(self):
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            return _WIN_SPECIAL.get(msvcrt.getwch(), UNKNOWN)
        return _normalize(ch)

    def _read_posix(self):
        import select

        ch = sys.stdin.read(1)
        if ch != "\x1b":
            return _normalize(ch)
        if not select.select([sys.stdin], [], [], 0.05)[0]:
            return ESC
        second = sys.stdin.read(1)
        if second != "[":
            return _normalize(second)
        body = ""
        while True:
            if not select.select([sys.stdin], [], [], 0.05)[0]:
                break
            nxt = sys.stdin.read(1)
            body += nxt
            if nxt.isalpha() or nxt == "~":
                break
        return _CSI.get(body, _CSI.get(body[-1:], UNKNOWN))


def _normalize(ch):
    if ch in ("\r", "\n"):
        return ENTER
    if ch == "\x1b":
        return ESC
    if ch == "\t":
        return TAB
    if ch in ("\x7f", "\x08"):
        return BACKSPACE
    if ch == "\x03":
        return INTERRUPT
    if ch == "\x04":
        return ESC
    if len(ch) == 1 and ord(ch) < 32:
        return UNKNOWN
    return ch


def is_text(key):
    return len(key) == 1 and key.isprintable()
