from __future__ import annotations

import shutil
import sys

from ..render import enable_windows_ansi, visible_len

ALT_ON = "\033[?1049h"
ALT_OFF = "\033[?1049l"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"
HOME = "\033[H"
CLEAR = "\033[2J"
CLEAR_LINE = "\033[K"


class Screen:
    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.height = 0
        self.width = 0
        self._painted = 0

    def __enter__(self):
        enable_windows_ansi()
        self.write(ALT_ON + HIDE_CURSOR + CLEAR + HOME)
        return self

    def __exit__(self, *exc):
        self.write(SHOW_CURSOR + ALT_OFF)
        self.flush()
        return False

    def write(self, text):
        self.stream.write(text)

    def flush(self):
        self.stream.flush()

    def size(self):
        size = shutil.get_terminal_size((80, 24))
        self.width = max(40, size.columns)
        self.height = max(10, size.lines)
        return self.width, self.height

    def render(self, lines, cursor=None):
        width, height = self.size()
        body = []
        for line in lines[:height]:
            body.append(_fit(line, width) + CLEAR_LINE)
        blanks = max(0, min(self._painted, height) - len(body))
        body.extend([CLEAR_LINE] * blanks)
        self._painted = len(lines[:height])
        frame = HOME + "\r\n".join(body)
        if cursor:
            frame += "\033[%d;%dH%s" % (cursor[0] + 1, cursor[1] + 1, SHOW_CURSOR)
        else:
            frame += HIDE_CURSOR
        self.write(frame)
        self.flush()


def _fit(line, width):
    if visible_len(line) <= width:
        return line
    out = []
    used = 0
    i = 0
    n = len(line)
    while i < n and used < width:
        if line[i] == "\033":
            j = i
            while j < n and line[j] != "m":
                j += 1
            out.append(line[i:j + 1])
            i = j + 1
        else:
            out.append(line[i])
            used += 1
            i += 1
    return "".join(out) + "\033[0m"
