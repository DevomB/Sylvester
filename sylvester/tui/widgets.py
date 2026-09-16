from __future__ import annotations

from ..parse import ParseError, parse_value
from ..render import C, dim, fmt, glyph, pad, paint, slice_visible, visible_len
from . import keys as K


class Menu:
    def __init__(self, items, index=0):
        self.items = items
        self.index = min(index, max(0, len(items) - 1))

    @property
    def current(self):
        return self.items[self.index] if self.items else None

    def lines(self, width, selectable_hint=True):
        out = []
        marker = glyph("▸", ">")
        label_width = max((len(i[1]) for i in self.items), default=0)
        for n, item in enumerate(self.items):
            shortcut, label, hint = item[0], item[1], item[2] if len(item) > 2 else ""
            selected = n == self.index
            head = "%s %s" % (marker if selected else " ", paint(shortcut, C.CYAN) if shortcut else " ")
            body = pad(label, label_width)
            line = "  %s  %s" % (head, body)
            if hint:
                line += "   " + dim(hint)
            out.append(paint(line, C.BOLD) if selected else line)
        return out

    def handle(self, key):
        if key == K.UP:
            self.index = (self.index - 1) % len(self.items)
            return None
        if key == K.DOWN:
            self.index = (self.index + 1) % len(self.items)
            return None
        if key == K.HOME:
            self.index = 0
            return None
        if key == K.END:
            self.index = len(self.items) - 1
            return None
        if key == K.ENTER:
            return "select"
        if K.is_text(key):
            for n, item in enumerate(self.items):
                if item[0] and item[0].lower() == key.lower():
                    self.index = n
                    return "select"
        return None


class Pager:
    def __init__(self, content=""):
        self.lines = []
        self.offset = 0
        self.hoffset = 0
        self.set(content)

    def set(self, content):
        self.lines = content.splitlines() if isinstance(content, str) else list(content)
        self.offset = 0
        self.hoffset = 0

    @property
    def widest(self):
        return max((visible_len(l) for l in self.lines), default=0)

    def view(self, width, height):
        window = max(1, height)
        self.offset = max(0, min(self.offset, max(0, len(self.lines) - window)))
        rows = self.lines[self.offset:self.offset + window]
        if not self.hoffset:
            return rows
        return [slice_visible(l, self.hoffset, width) for l in rows]

    def scroll_hint(self, height, width=0):
        window = max(1, height)
        parts = []
        if len(self.lines) > window:
            last = min(len(self.lines), self.offset + window)
            parts.append("lines %d-%d of %d" % (self.offset + 1, last, len(self.lines)))
        if width and self.widest > width:
            parts.append("left/right to pan (column %d of %d)" % (self.hoffset + 1, self.widest))
        return "   ".join(parts)

    def handle(self, key, height, width=0):
        window = max(1, height)
        if key == K.UP:
            self.offset -= 1
        elif key == K.DOWN:
            self.offset += 1
        elif key == K.PGUP:
            self.offset -= window
        elif key == K.PGDN:
            self.offset += window
        elif key == K.HOME:
            self.offset = 0
            self.hoffset = 0
        elif key == K.END:
            self.offset = len(self.lines)
        elif key == K.LEFT:
            self.hoffset = max(0, self.hoffset - 8)
            return True
        elif key == K.RIGHT:
            self.hoffset = min(max(0, self.widest - 20), self.hoffset + 8)
            return True
        else:
            return False
        self.offset = max(0, min(self.offset, max(0, len(self.lines) - window)))
        return True


EDIT_CHARS = "0123456789-/."


class MatrixEditor:
    def __init__(self, matrix, name="A"):
        self.name = name
        self.cells = [[fmt(v) for v in row] for row in matrix.rows] or [["0"]]
        self.row = 0
        self.col = 0
        self.typing = False
        self.error = ""

    @property
    def nrows(self):
        return len(self.cells)

    @property
    def ncols(self):
        return len(self.cells[0])

    def matrix(self):
        from ..matrix import Matrix

        rows = []
        for r, row in enumerate(self.cells):
            parsed = []
            for c, cell in enumerate(row):
                try:
                    parsed.append(parse_value(cell) if cell.strip() else 0)
                except ParseError:
                    raise ParseError("cell (%d, %d) is not a number: %r" % (r + 1, c + 1, cell))
            rows.append(parsed)
        return Matrix(rows)

    def lines(self, split_at=None):
        widths = [max(len(self.cells[r][c]) for r in range(self.nrows)) for c in range(self.ncols)]
        widths = [max(3, w) for w in widths]
        out = []
        header = "     " + "  ".join(pad("c%d" % (c + 1), widths[c], "^") for c in range(self.ncols))
        out.append(dim(header))
        for r in range(self.nrows):
            cells = []
            for c in range(self.ncols):
                body = pad(self.cells[r][c], widths[c], ">")
                if r == self.row and c == self.col:
                    body = paint(body, C.ON_BLUE, C.BOLD)
                cells.append(body)
                if split_at is not None and c == split_at - 1:
                    cells.append(dim(glyph("│", "|")))
            out.append("%s  %s %s %s" % (dim("r%d" % (r + 1)), glyph("⎢", "["),
                                         "  ".join(cells), glyph("⎥", "]")))
        return out

    def handle(self, key):
        self.error = ""
        if key == K.LEFT:
            self.typing = False
            self.col = (self.col - 1) % self.ncols
        elif key == K.RIGHT:
            self.typing = False
            self.col = (self.col + 1) % self.ncols
        elif key == K.UP:
            self.typing = False
            self.row = (self.row - 1) % self.nrows
        elif key == K.DOWN:
            self.typing = False
            self.row = (self.row + 1) % self.nrows
        elif key in (K.ENTER, K.TAB):
            self.typing = False
            self._advance()
        elif key == K.BACKSPACE:
            current = self.cells[self.row][self.col]
            self.cells[self.row][self.col] = current[:-1] if self.typing and current else ""
            self.typing = True
        elif key == K.DELETE:
            self.cells[self.row][self.col] = "0"
            self.typing = False
        elif key == "+":
            self.cells.append(["0"] * self.ncols)
        elif key == "_":
            if self.nrows > 1:
                self.cells.pop()
                self.row = min(self.row, self.nrows - 1)
        elif key == "]":
            for row in self.cells:
                row.append("0")
        elif key == "[":
            if self.ncols > 1:
                for row in self.cells:
                    row.pop()
                self.col = min(self.col, self.ncols - 1)
        elif key in ("t", "T"):
            self.cells = [[self.cells[r][c] for r in range(self.nrows)] for c in range(self.ncols)]
            self.row, self.col = 0, 0
        elif key in ("i", "I"):
            n = self.nrows
            self.cells = [["1" if r == c else "0" for c in range(n)] for r in range(n)]
        elif key in ("z", "Z"):
            self.cells = [["0"] * self.ncols for _ in range(self.nrows)]
        elif K.is_text(key) and key in EDIT_CHARS:
            if not self.typing:
                self.cells[self.row][self.col] = ""
                self.typing = True
            self.cells[self.row][self.col] += key
        else:
            return False
        return True

    def _advance(self):
        self.col += 1
        if self.col >= self.ncols:
            self.col = 0
            self.row = (self.row + 1) % self.nrows


class Prompt:
    def __init__(self, label, value="", history=None):
        self.label = label
        self.value = value
        self.cursor = len(value)
        self.history = history if history is not None else []
        self.hindex = len(self.history)

    def line(self):
        return "%s %s%s" % (paint(self.label, C.CYAN, C.BOLD), self.value, paint(" ", C.ON_GREY))

    def handle(self, key):
        if key == K.BACKSPACE:
            if self.cursor:
                self.value = self.value[:self.cursor - 1] + self.value[self.cursor:]
                self.cursor -= 1
        elif key == K.DELETE:
            self.value = self.value[:self.cursor] + self.value[self.cursor + 1:]
        elif key == K.LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif key == K.RIGHT:
            self.cursor = min(len(self.value), self.cursor + 1)
        elif key == K.HOME:
            self.cursor = 0
        elif key == K.END:
            self.cursor = len(self.value)
        elif key == K.UP:
            if self.history and self.hindex > 0:
                self.hindex -= 1
                self.value = self.history[self.hindex]
                self.cursor = len(self.value)
        elif key == K.DOWN:
            if self.history and self.hindex < len(self.history) - 1:
                self.hindex += 1
                self.value = self.history[self.hindex]
            else:
                self.hindex = len(self.history)
                self.value = ""
            self.cursor = len(self.value)
        elif key == K.ENTER:
            return "submit"
        elif key == K.ESC:
            return "cancel"
        elif K.is_text(key):
            self.value = self.value[:self.cursor] + key + self.value[self.cursor:]
            self.cursor += 1
        return None

    def submit(self):
        text = self.value.strip()
        if text:
            self.history.append(text)
        self.hindex = len(self.history)
        self.value = ""
        self.cursor = 0
        return text


def frame(title, subtitle, body, footer, width, height):
    out = []
    bar = paint(pad(" %s" % title, width), C.ON_BLUE, C.BOLD)
    out.append(bar)
    if subtitle:
        out.append(dim(" " + subtitle))
        out.append("")
    out.extend(body)
    filler = height - len(out) - 1
    out.extend([""] * max(0, filler))
    out.append(paint(pad(" " + footer, width), C.ON_GREY))
    return out
