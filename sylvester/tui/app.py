from __future__ import annotations

from fractions import Fraction

from .. import proof, report
from ..expr import FUNCTION_HELP, evaluate
from ..matrix import Matrix
from ..parse import ParseError
from ..render import C, autodetect, dim, fmt, matrix_str, paint, set_color
from ..samples import CLO, by_clo
from ..reduce import HUMAN, MACHINE
from .. import vectors as V
from . import keys as K
from .keys import Keyboard
from .screen import Screen
from .widgets import MatrixEditor, Menu, Pager, Prompt, frame

TITLE = "SYLVESTER"
SUBTITLE = "an exact linear algebra workbench"

DEFAULTS = {
    "A": Matrix([[2, -1, 3], [1, 1, -2], [4, 1, -1]]),
    "B": Matrix.column([Fraction(-1), Fraction(1), Fraction(3)]),
    "U": Matrix.column([Fraction(1), Fraction(2), Fraction(2)]),
    "V": Matrix.column([Fraction(3), Fraction(0), Fraction(4)]),
}


class Screenlet:
    title = ""
    subtitle = ""
    footer = "esc back   q quit   ? help"

    def __init__(self, app):
        self.app = app

    def body(self, width, height):
        return []

    def handle(self, key):
        return None


class App:
    def __init__(self):
        self.registers = dict(DEFAULTS)
        self.stack = []
        self.status = ""
        self.running = True
        self.mode = HUMAN
        self.steps = True
        self.augmented = False
        self.history = []
        self.screen = None

    def push(self, screenlet):
        self.stack.append(screenlet)

    def pop(self):
        if len(self.stack) > 1:
            self.stack.pop()
        else:
            self.running = False

    @property
    def top(self):
        return self.stack[-1]

    def note(self, text):
        self.status = text

    def register(self, name):
        return self.registers.get(name)

    def set_register(self, name, value):
        self.registers[name] = value

    def matrices(self, predicate=None):
        return [k for k in sorted(self.registers) if predicate is None or predicate(self.registers[k])]

    def run(self, screen=None, keyboard=None):
        autodetect()
        set_color(True)
        self.push(HomeScreen(self))
        with (screen or Screen()) as screen, (keyboard or Keyboard()) as keyboard:
            self.screen = screen
            while self.running:
                width, height = screen.size()
                top = self.top
                body = top.body(width, height - 4)
                status = self.status or top.subtitle
                banner = "%s   %s" % (TITLE, top.title) if top.title else TITLE
                screen.render(frame(banner, status, body, top.footer, width, height))
                self.status = ""
                try:
                    key = keyboard.read()
                except (KeyboardInterrupt, EOFError):
                    break
                if key == K.INTERRUPT:
                    break
                self.dispatch(key)

    def dispatch(self, key):
        top = self.top
        result = top.handle(key)
        if result == "handled":
            return
        if key == K.ESC:
            self.pop()
        elif key in ("q", "Q") and not isinstance(top, (WorkbenchScreen, EditorScreen)):
            self.running = False
        elif key == "?":
            self.push(HelpScreen(self))


class HomeScreen(Screenlet):
    title = "home"
    subtitle = SUBTITLE
    footer = "up/down move   enter open   ? help   q quit"

    def __init__(self, app):
        super().__init__(app)
        self.menu = Menu([
            ("1", "Matrix registers", "edit A-Z, load samples"),
            ("2", "Row reduction", "CLO 2  REF, RREF, every operation"),
            ("3", "Linear systems", "CLO 1  solve, classify, general solution"),
            ("4", "Determinants", "CLO 3  row operations and cofactor expansion"),
            ("5", "Inverses and factorizations", "CLO 2  Gauss-Jordan, adjugate, elementary, LU"),
            ("6", "Rank, kernel and range", "CLO 7  the four fundamental subspaces"),
            ("7", "Vectors and orthogonality", "CLO 5  inner products, projections, Gram-Schmidt"),
            ("8", "Eigenvalues and eigenvectors", "CLO 8  spectrum, eigenspaces, diagonalization"),
            ("9", "Theorem lab", "CLO 1/4/5  verify a statement with a cited certificate"),
            ("0", "Workbench", "evaluate matrix expressions"),
            ("s", "Sample problems", "34 worked problems across all eight outcomes"),
            ("?", "Help and keys", ""),
        ])

    def body(self, width, height):
        out = []
        out.append("  " + paint("Registers", C.BOLD))
        out.append("  " + register_summary(self.app))
        out.append("")
        out.extend(self.menu.lines(width))
        out.append("")
        out.append(dim("  Exact rational arithmetic throughout. Nothing is rounded."))
        return out

    def handle(self, key):
        if self.menu.handle(key) == "select":
            open_area(self.app, self.menu.current[0])
            return "handled"
        return None


def register_summary(app):
    parts = []
    for name in sorted(app.registers):
        m = app.registers[name]
        parts.append("%s %s" % (paint(name, C.CYAN, C.BOLD), dim("%dx%d" % m.shape)))
    return "   ".join(parts) if parts else dim("none")


def open_area(app, shortcut):
    if shortcut == "1":
        app.push(RegistersScreen(app))
    elif shortcut == "2":
        app.push(ReduceScreen(app))
    elif shortcut == "3":
        app.push(SystemScreen(app))
    elif shortcut == "4":
        app.push(DeterminantScreen(app))
    elif shortcut == "5":
        app.push(InverseScreen(app))
    elif shortcut == "6":
        app.push(SubspaceScreen(app))
    elif shortcut == "7":
        app.push(VectorScreen(app))
    elif shortcut == "8":
        app.push(EigenScreen(app))
    elif shortcut == "9":
        app.push(TheoremCategoryScreen(app))
    elif shortcut == "0":
        app.push(WorkbenchScreen(app))
    elif shortcut == "s":
        app.push(SampleScreen(app))
    elif shortcut == "?":
        app.push(HelpScreen(app))


class RegistersScreen(Screenlet):
    title = "registers"
    subtitle = "matrix registers"
    footer = "enter edit   n new   d delete   c copy   esc back"

    def __init__(self, app):
        super().__init__(app)
        self.rebuild()

    def rebuild(self):
        items = []
        for name in sorted(self.app.registers):
            m = self.app.registers[name]
            items.append((name, "%s   %dx%d" % (name, m.nrows, m.ncols), preview(m)))
        if not items:
            items = [("", "no registers yet, press n", "")]
        self.menu = Menu(items, getattr(self, "menu", Menu([])).index if hasattr(self, "menu") else 0)

    def body(self, width, height):
        out = self.menu.lines(width)
        out.append("")
        current = self.current_matrix()
        if current is not None:
            out.append("  " + paint("Contents", C.BOLD))
            out.extend("  " + line for line in matrix_str(current).splitlines()[:height - len(out) - 2])
        return out

    def current_matrix(self):
        item = self.menu.current
        if not item or not item[0]:
            return None
        return self.app.registers.get(item[0])

    def handle(self, key):
        action = self.menu.handle(key)
        if action == "select" and self.menu.current[0]:
            self.app.push(EditorScreen(self.app, self.menu.current[0]))
            return "handled"
        if key in ("n", "N"):
            self.app.push(NewRegisterScreen(self.app, self))
            return "handled"
        if key in ("d", "D") and self.menu.current[0]:
            name = self.menu.current[0]
            if len(self.app.registers) > 1:
                del self.app.registers[name]
                self.rebuild()
                self.app.note("deleted register %s" % name)
            return "handled"
        if key in ("c", "C") and self.menu.current[0]:
            self.app.push(CopyRegisterScreen(self.app, self, self.menu.current[0]))
            return "handled"
        if action is not None:
            return "handled"
        return None


def preview(matrix):
    cells = []
    for row in matrix.rows[:1]:
        cells.extend(fmt(v) for v in row[:4])
    text = " ".join(cells)
    if matrix.nrows > 1 or matrix.ncols > 4:
        text += " ..."
    return text


class EditorScreen(Screenlet):
    title = "editor"
    footer = "arrows move   type value   + _ row   ] [ col   t transpose   i identity   enter save   esc cancel"

    def __init__(self, app, name):
        super().__init__(app)
        self.name = name
        self.editor = MatrixEditor(app.registers[name], name)
        self.subtitle = "editing register %s" % name
        self.error = ""

    def body(self, width, height):
        out = ["  " + paint("Register %s" % self.name, C.BOLD), ""]
        out.extend("  " + line for line in self.editor.lines())
        out.append("")
        out.append(dim("  %d x %d" % (self.editor.nrows, self.editor.ncols)))
        if self.error:
            out.append("  " + paint(self.error, C.RED, C.BOLD))
        out.append("")
        out.append(dim("  values may be integers, fractions like -2/3, or decimals like 1.5"))
        out.append(dim("  press w to write the register and go back"))
        return out

    def handle(self, key):
        if key in ("w", "W"):
            return self.save()
        if key == K.ESC:
            self.app.pop()
            return "handled"
        if self.editor.handle(key):
            return "handled"
        return None

    def save(self):
        try:
            self.app.set_register(self.name, self.editor.matrix())
        except ParseError as exc:
            self.error = str(exc)
            return "handled"
        self.app.note("saved register %s" % self.name)
        self.app.pop()
        return "handled"


class NewRegisterScreen(Screenlet):
    title = "new register"
    subtitle = "name the new register"
    footer = "type a letter   enter create   esc cancel"

    def __init__(self, app, parent):
        super().__init__(app)
        self.parent = parent
        self.prompt = Prompt("name (A-Z):")

    def body(self, width, height):
        return ["", "  " + self.prompt.line(), "",
                dim("  a new register starts as a 2x2 zero matrix")]

    def handle(self, key):
        action = self.prompt.handle(key)
        if action == "cancel":
            self.app.pop()
            return "handled"
        if action == "submit":
            name = self.prompt.submit().upper()[:1]
            if not name.isalpha():
                self.app.note("register names are single letters")
                return "handled"
            self.app.set_register(name, Matrix.zeros(2, 2))
            self.parent.rebuild()
            self.app.pop()
            self.app.push(EditorScreen(self.app, name))
            return "handled"
        return "handled"


class CopyRegisterScreen(Screenlet):
    title = "copy register"
    footer = "type a letter   enter copy   esc cancel"

    def __init__(self, app, parent, source):
        super().__init__(app)
        self.parent = parent
        self.source = source
        self.subtitle = "copy %s into which register?" % source
        self.prompt = Prompt("destination (A-Z):")

    def body(self, width, height):
        return ["", "  " + self.prompt.line()]

    def handle(self, key):
        action = self.prompt.handle(key)
        if action == "cancel":
            self.app.pop()
            return "handled"
        if action == "submit":
            name = self.prompt.submit().upper()[:1]
            if not name.isalpha():
                self.app.note("register names are single letters")
                return "handled"
            self.app.set_register(name, self.app.registers[self.source])
            self.parent.rebuild()
            self.app.note("copied %s to %s" % (self.source, name))
            self.app.pop()
            return "handled"
        return "handled"


class ReportScreen(Screenlet):
    extra_footer = ""

    def __init__(self, app):
        super().__init__(app)
        self.pager = Pager("")
        self.primary = self._first_register()
        self.refresh()

    def _first_register(self):
        return "A" if "A" in self.app.registers else sorted(self.app.registers)[0]

    def build(self):
        return ""

    def refresh(self):
        offset = self.pager.offset
        try:
            self.pager.set(self.build())
        except Exception as exc:
            self.pager.set(paint("  %s: %s" % (type(exc).__name__, exc), C.RED, C.BOLD))
        self.pager.offset = offset

    @property
    def matrix(self):
        return self.app.registers[self.primary]

    @property
    def footer(self):
        base = "r register(%s)   %s" % (self.primary, self.extra_footer)
        return base + "   up/down scroll   esc back"

    def body(self, width, height):
        lines = self.pager.view(width, height - 1)
        hint = self.pager.scroll_hint(height - 1, width)
        out = list(lines)
        if hint:
            out.append(dim("  " + hint))
        return out

    def handle(self, key):
        if self.pager.handle(key, self.app.screen.height - 5, self.app.screen.width):
            return "handled"
        if key in ("r", "R"):
            names = sorted(self.app.registers)
            self.primary = names[(names.index(self.primary) + 1) % len(names)]
            self.refresh()
            return "handled"
        if self.options(key):
            self.refresh()
            return "handled"
        return None

    def options(self, key):
        return False


class ReduceScreen(ReportScreen):
    title = "row reduction"
    subtitle = "CLO 2  elementary row operations, REF and RREF"
    extra_footer = "m mode   s steps   a augmented   c compare"

    def __init__(self, app):
        self.compare = False
        super().__init__(app)

    @property
    def footer(self):
        return ("r register(%s)   m %s   s steps:%s   a augmented:%s   c compare   up/down scroll   esc back"
                % (self.primary, self.app.mode, "on" if self.app.steps else "off",
                   "on" if self.app.augmented else "off"))

    def build(self):
        m = self.matrix
        split = m.ncols - 1 if self.app.augmented and m.ncols > 1 else None
        if self.compare:
            return report.compare_strategies(m, split, self.app.steps)
        return report.rref_report(m, self.app.mode, split, self.app.steps)

    def options(self, key):
        if key in ("m", "M"):
            self.app.mode = MACHINE if self.app.mode == HUMAN else HUMAN
            return True
        if key in ("s", "S"):
            self.app.steps = not self.app.steps
            return True
        if key in ("a", "A"):
            self.app.augmented = not self.app.augmented
            return True
        if key in ("c", "C"):
            self.compare = not self.compare
            return True
        return False


class SystemScreen(ReportScreen):
    title = "linear systems"
    subtitle = "CLO 1 and 6  solve Ax = b, classify, and read off the general solution"

    def __init__(self, app):
        self.secondary = "B"
        self.method = "reduce"
        super().__init__(app)

    @property
    def footer(self):
        return ("r A(%s)   b const(%s)   v %s   s steps:%s   up/down scroll   esc back"
                % (self.primary, self.secondary, self.method, "on" if self.app.steps else "off"))

    def build(self):
        a = self.matrix
        if self.method == "homogeneous":
            return report.homogeneous_text(a)
        b = self.app.registers.get(self.secondary)
        if b is None:
            return "  register %s is empty; press b to choose the constants vector" % self.secondary
        constants = V.as_vector(b)
        if len(constants) != a.nrows:
            return ("  A is %dx%d but the constants vector %s has %d entries.\n"
                    "  Press b to pick a different register, or r to pick a different A."
                    % (a.nrows, a.ncols, self.secondary, len(constants)))
        if self.method == "cramer":
            return report.cramer_text(a, constants)
        return report.system_report(a, constants, self.app.mode, self.app.steps)

    def options(self, key):
        if key in ("b",):
            names = sorted(self.app.registers)
            self.secondary = names[(names.index(self.secondary) % len(names) + 1) % len(names)] \
                if self.secondary in names else names[0]
            return True
        if key in ("v", "V"):
            order = ["reduce", "cramer", "homogeneous"]
            self.method = order[(order.index(self.method) + 1) % len(order)]
            return True
        if key in ("s", "S"):
            self.app.steps = not self.app.steps
            return True
        return False


class DeterminantScreen(ReportScreen):
    title = "determinants"
    subtitle = "CLO 3  row operations and cofactor expansion"

    def __init__(self, app):
        self.method = "both"
        super().__init__(app)

    @property
    def footer(self):
        return ("r register(%s)   v method:%s   s steps:%s   up/down scroll   esc back"
                % (self.primary, self.method, "on" if self.app.steps else "off"))

    def build(self):
        m = self.matrix
        if not m.is_square:
            return ("  The determinant needs a square matrix, but %s is %dx%d.\n"
                    "  Press r to choose another register." % (self.primary, m.nrows, m.ncols))
        return report.determinant_report(m, self.method, self.app.steps)

    def options(self, key):
        if key in ("v", "V"):
            order = ["both", "rowops", "cofactor"]
            self.method = order[(order.index(self.method) + 1) % len(order)]
            return True
        if key in ("s", "S"):
            self.app.steps = not self.app.steps
            return True
        return False


class InverseScreen(ReportScreen):
    title = "inverses and factorizations"
    subtitle = "CLO 2  Gauss-Jordan, adjugate, elementary matrices, LU"

    def __init__(self, app):
        self.method = "gauss"
        super().__init__(app)

    @property
    def footer(self):
        return ("r register(%s)   v method:%s   s steps:%s   up/down scroll   esc back"
                % (self.primary, self.method, "on" if self.app.steps else "off"))

    def build(self):
        m = self.matrix
        if not m.is_square and self.method != "lu":
            return ("  Only a square matrix has an inverse, but %s is %dx%d.\n"
                    "  Press r to choose another register, or v for the LU factorization."
                    % (self.primary, m.nrows, m.ncols))
        return report.inverse_report(m, self.method, self.app.steps)

    def options(self, key):
        if key in ("v", "V"):
            order = ["gauss", "adjugate", "elementary", "lu"]
            self.method = order[(order.index(self.method) + 1) % len(order)]
            return True
        if key in ("s", "S"):
            self.app.steps = not self.app.steps
            return True
        return False


class SubspaceScreen(ReportScreen):
    title = "rank, kernel and range"
    subtitle = "CLO 7  the four fundamental subspaces, rank and nullity"

    def __init__(self, app):
        self.method = "subspaces"
        super().__init__(app)

    @property
    def footer(self):
        return ("r register(%s)   v %s   up/down scroll   esc back" % (self.primary, self.method))

    def build(self):
        m = self.matrix
        if self.method == "subspaces":
            return report.subspaces_report(m)
        if self.method == "properties":
            return report.properties_report(m)
        if self.method == "independence":
            return report.independence_report(m.columns())
        return report.basis_report(m.columns())

    def options(self, key):
        if key in ("v", "V"):
            order = ["subspaces", "independence", "basis", "properties"]
            self.method = order[(order.index(self.method) + 1) % len(order)]
            return True
        return False


class VectorScreen(ReportScreen):
    title = "vectors and orthogonality"
    subtitle = "CLO 5  inner products, norms, projections, Gram-Schmidt"

    def __init__(self, app):
        self.method = "pair"
        self.secondary = "V"
        super().__init__(app)

    def _first_register(self):
        return "U" if "U" in self.app.registers else super()._first_register()

    @property
    def footer(self):
        return ("r u(%s)   b v(%s)   v %s   up/down scroll   esc back"
                % (self.primary, self.secondary, self.method))

    def build(self):
        first = self.app.registers.get(self.primary)
        second = self.app.registers.get(self.secondary)
        if self.method == "gramschmidt":
            return report.gram_schmidt_report(first.columns())
        if self.method == "span":
            if second is None:
                return "  pick a target vector with b"
            return report.span_report(V.as_vector(second), first.columns())
        if self.method == "projection":
            if second is None:
                return "  pick a vector to project with b"
            return report.projection_report(V.as_vector(second), first.columns())
        try:
            u = V.as_vector(first)
            w = V.as_vector(second) if second is not None else None
        except ValueError as exc:
            return ("  %s\n  Registers %s and %s must each be a single row or column."
                    % (exc, self.primary, self.secondary))
        if w is None:
            return "  pick a second vector with b"
        if len(u) != len(w):
            return "  u has %d entries but v has %d" % (len(u), len(w))
        return report.vector_report(u, w)

    def options(self, key):
        names = sorted(self.app.registers)
        if key in ("b",):
            self.secondary = names[(names.index(self.secondary) + 1) % len(names)] \
                if self.secondary in names else names[0]
            return True
        if key in ("v", "V"):
            order = ["pair", "gramschmidt", "projection", "span"]
            self.method = order[(order.index(self.method) + 1) % len(order)]
            return True
        return False


class EigenScreen(ReportScreen):
    title = "eigenvalues and eigenvectors"
    subtitle = "CLO 8  characteristic polynomial, eigenspaces, diagonalization"

    @property
    def footer(self):
        return ("r register(%s)   s steps:%s   up/down scroll   esc back"
                % (self.primary, "on" if self.app.steps else "off"))

    def build(self):
        m = self.matrix
        if not m.is_square:
            return ("  Eigenvalues need a square matrix, but %s is %dx%d.\n"
                    "  Press r to choose another register." % (self.primary, m.nrows, m.ncols))
        return report.eigen_report(m, self.app.steps)

    def options(self, key):
        if key in ("s", "S"):
            self.app.steps = not self.app.steps
            return True
        return False


class TheoremCategoryScreen(Screenlet):
    title = "theorem lab"
    subtitle = "CLO 1, 4 and 5  verify a statement and get a cited certificate"
    footer = "enter open   esc back"

    def __init__(self, app):
        super().__init__(app)
        groups = proof.by_category()
        self.groups = groups
        self.menu = Menu([
            (str(i + 1), name, "%d propositions" % len(items))
            for i, (name, items) in enumerate(groups.items())
        ])

    def body(self, width, height):
        out = self.menu.lines(width)
        out.append("")
        out.append(dim("  A proof certificate runs the statement on your registers and shows every"))
        out.append(dim("  step, the theorem it rests on, and whether it held."))
        return out

    def handle(self, key):
        if self.menu.handle(key) == "select":
            name = self.menu.current[1]
            self.app.push(TheoremListScreen(self.app, name, self.groups[name]))
            return "handled"
        return None


class TheoremListScreen(Screenlet):
    footer = "enter verify   esc back"

    def __init__(self, app, category, props):
        super().__init__(app)
        self.title = category
        self.subtitle = category
        self.props = props
        self.menu = Menu([
            (chr(ord("a") + i) if i < 26 else "", p.title, " ".join(p.operands))
            for i, p in enumerate(props)
        ])

    def body(self, width, height):
        out = self.menu.lines(width)
        out.append("")
        current = self.props[self.menu.index]
        out.append("  " + paint("Statement", C.BOLD))
        out.extend(wrap_lines(current.statement, width - 6, "    "))
        out.append("")
        out.append(dim("    needs: %s" % ", ".join(current.operands)))
        out.append(dim("    kind:  %s" % ("complete proof for the operands given"
                                          if current.kind == proof.PROOF
                                          else "verified instance of a general theorem")))
        return out

    def handle(self, key):
        if self.menu.handle(key) == "select":
            self.app.push(CertificateScreen(self.app, self.props[self.menu.index]))
            return "handled"
        return None


class CertificateScreen(ReportScreen):
    def __init__(self, app, prop):
        self.prop = prop
        self.operand_names = []
        super().__init__(app)
        self.title = prop.title
        self.subtitle = prop.title

    def _first_register(self):
        names = sorted(self.app.registers)
        self.operand_names = []
        for spec in self.prop.operands:
            if spec == proof.SCALAR:
                self.operand_names.append("2")
            elif spec == proof.VECTOR:
                self.operand_names.append(next((n for n in ("U", "V") if n in names), names[0]))
            else:
                self.operand_names.append(next((n for n in ("A", "B") if n in names and n not in self.operand_names), names[0]))
        return self.operand_names[0] if self.operand_names else names[0]

    @property
    def footer(self):
        slots = "  ".join("%d:%s" % (i + 1, n) for i, n in enumerate(self.operand_names))
        return "%s   1-4 cycle operand   up/down scroll   esc back" % slots

    def build(self):
        operands = []
        for spec, name in zip(self.prop.operands, self.operand_names):
            if spec == proof.SCALAR:
                operands.append(Fraction(name))
            elif spec == proof.VECTOR:
                operands.append(V.as_vector(self.app.registers[name]))
            else:
                operands.append(self.app.registers[name])
        return report.certificate_report(proof.check(self.prop.key, *operands))

    def handle(self, key):
        if self.pager.handle(key, self.app.screen.height - 5, self.app.screen.width):
            return "handled"
        if key in "1234" and key.isdigit():
            index = int(key) - 1
            if index < len(self.operand_names):
                self._cycle(index)
                self.refresh()
            return "handled"
        return None

    def _cycle(self, index):
        spec = self.prop.operands[index]
        if spec == proof.SCALAR:
            values = ["2", "3", "-1", "1/2"]
            current = self.operand_names[index]
            self.operand_names[index] = values[(values.index(current) + 1) % len(values)] \
                if current in values else values[0]
            return
        names = sorted(self.app.registers)
        current = self.operand_names[index]
        self.operand_names[index] = names[(names.index(current) + 1) % len(names)] \
            if current in names else names[0]


class WorkbenchScreen(Screenlet):
    title = "workbench"
    subtitle = "evaluate matrix expressions against your registers"
    footer = "enter evaluate   up/down history   = store result   esc back"

    def __init__(self, app):
        super().__init__(app)
        self.prompt = Prompt(">", history=app.history)
        self.output = []
        self.last = None

    def body(self, width, height):
        out = []
        out.append("  " + register_summary(self.app))
        out.append("")
        room = height - 8
        for entry in self.output[-room:]:
            out.extend(entry)
        out.append("")
        out.append("  " + self.prompt.line())
        out.append("")
        out.append(dim("  try:  A*B    det(A)    inv(A)*A    A^T    rref(A)    2*I(3) - A"))
        return out

    def handle(self, key):
        action = self.prompt.handle(key)
        if action == "cancel":
            self.app.pop()
            return "handled"
        if action == "submit":
            self.evaluate(self.prompt.submit())
            return "handled"
        if key == "=" and self.last is not None:
            self.app.push(StoreResultScreen(self.app, self.last))
            return "handled"
        return "handled"

    def evaluate(self, source):
        if not source:
            return
        entry = ["  " + paint("> " + source, C.CYAN)]
        try:
            value = evaluate(source, self.app.registers)
            self.last = value
            if isinstance(value, Matrix):
                entry.extend("    " + line for line in matrix_str(value).splitlines())
            else:
                entry.append("    " + paint(report.sval(value), C.BOLD))
        except Exception as exc:
            entry.append("    " + paint(str(exc), C.RED))
        entry.append("")
        self.output.append(entry)


class StoreResultScreen(Screenlet):
    title = "store result"
    subtitle = "store the last result in a register"
    footer = "type a letter   enter store   esc cancel"

    def __init__(self, app, value):
        super().__init__(app)
        self.value = value
        self.prompt = Prompt("register (A-Z):")

    def body(self, width, height):
        out = ["", "  " + self.prompt.line(), ""]
        if isinstance(self.value, Matrix):
            out.extend("  " + line for line in matrix_str(self.value).splitlines())
        else:
            out.append("  " + fmt(self.value))
        return out

    def handle(self, key):
        action = self.prompt.handle(key)
        if action == "cancel":
            self.app.pop()
            return "handled"
        if action == "submit":
            name = self.prompt.submit().upper()[:1]
            if not name.isalpha():
                self.app.note("register names are single letters")
                return "handled"
            value = self.value if isinstance(self.value, Matrix) else Matrix([[self.value]])
            self.app.set_register(name, value)
            self.app.note("stored in %s" % name)
            self.app.pop()
            return "handled"
        return "handled"


class SampleScreen(Screenlet):
    title = "sample problems"
    subtitle = "34 worked problems covering all eight course outcomes"
    footer = "enter load   esc back"

    def __init__(self, app):
        super().__init__(app)
        items = []
        self.lookup = []
        for clo in sorted(CLO):
            for s in by_clo().get(clo, []):
                items.append(("", "CLO %d  %s" % (clo, s.title), ""))
                self.lookup.append(s)
        self.menu = Menu(items)

    def body(self, width, height):
        room = height - 8
        start = max(0, min(self.menu.index - room // 2, len(self.menu.items) - room))
        start = max(0, start)
        all_lines = self.menu.lines(width)
        out = all_lines[start:start + room]
        current = self.lookup[self.menu.index]
        out.append("")
        out.append("  " + paint(CLO[current.clo], C.BOLD))
        out.extend(wrap_lines(current.note, width - 6, "    "))
        out.append("")
        out.append(dim("    loads registers: %s" % ", ".join(sorted(current.registers))))
        return out

    def handle(self, key):
        if self.menu.handle(key) == "select":
            sample = self.lookup[self.menu.index]
            for name, value in sample.registers.items():
                self.app.set_register(name, value)
            self.app.note("loaded %s" % sample.title)
            self.app.pop()
            open_sample_screen(self.app, sample)
            return "handled"
        return None


SAMPLE_SCREENS = {
    "system": "3", "homogeneous": "3", "cramer": "3",
    "reduce": "2", "inverse": "5", "elementary": "5",
    "determinant": "4", "proof": "9",
    "independence": "6", "subspaces": "6", "basis": "6",
    "vectors": "7", "gramschmidt": "7", "projection": "7",
    "eigen": "8",
}


def open_sample_screen(app, sample):
    shortcut = SAMPLE_SCREENS.get(sample.screen)
    if shortcut is None:
        return
    open_area(app, shortcut)
    top = app.top
    if sample.screen == "homogeneous" and isinstance(top, SystemScreen):
        top.method = "homogeneous"
    elif sample.screen == "cramer" and isinstance(top, SystemScreen):
        top.method = "cramer"
    elif sample.screen == "elementary" and isinstance(top, InverseScreen):
        top.method = "elementary"
    elif sample.screen == "independence" and isinstance(top, SubspaceScreen):
        top.method = "independence"
    elif sample.screen == "basis" and isinstance(top, SubspaceScreen):
        top.method = "basis"
    elif sample.screen == "gramschmidt" and isinstance(top, VectorScreen):
        top.method = "gramschmidt"
    elif sample.screen == "projection" and isinstance(top, VectorScreen):
        top.method = "projection"
    if isinstance(top, ReportScreen):
        top.refresh()


class HelpScreen(Screenlet):
    title = "help"
    subtitle = "keys and what this covers"
    footer = "up/down scroll   esc back"

    def __init__(self, app):
        super().__init__(app)
        self.pager = Pager(help_text())

    def body(self, width, height):
        return self.pager.view(width, height - 1)

    def handle(self, key):
        if self.pager.handle(key, self.app.screen.height - 5, self.app.screen.width):
            return "handled"
        return None


def wrap_lines(text, width, indent=""):
    words = text.split()
    out = []
    line = ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(indent + line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(indent + line)
    return out


def help_text():
    out = []
    out.append(paint("  NAVIGATION", C.BOLD))
    for k, v in [
        ("up / down", "move a menu selection or scroll a report"),
        ("page up / page down", "scroll a report a screen at a time"),
        ("home / end", "jump to the top or bottom"),
        ("enter", "open the highlighted item"),
        ("esc", "go back one screen"),
        ("q", "quit"),
        ("?", "this help"),
    ]:
        out.append("    %-22s %s" % (k, dim(v)))
    out.append("")
    out.append(paint("  ON A REPORT SCREEN", C.BOLD))
    for k, v in [
        ("r", "cycle which register the report uses"),
        ("b", "cycle the second operand, where there is one"),
        ("v", "cycle the method or view"),
        ("m", "swap the human and machine reduction strategies"),
        ("s", "show or hide the individual steps"),
        ("a", "treat the last column as constants"),
        ("c", "compare the two reduction strategies side by side"),
    ]:
        out.append("    %-22s %s" % (k, dim(v)))
    out.append("")
    out.append(paint("  IN THE MATRIX EDITOR", C.BOLD))
    for k, v in [
        ("arrows", "move between cells"),
        ("digits, -, /, .", "type a value; entering 1/3 keeps it exact"),
        ("+ / _", "add or remove a row"),
        ("] / [", "add or remove a column"),
        ("t", "transpose"),
        ("i", "make it an identity matrix"),
        ("z", "fill with zeros"),
        ("w", "write the register and go back"),
        ("esc", "cancel without saving"),
    ]:
        out.append("    %-22s %s" % (k, dim(v)))
    out.append("")
    out.append(paint("  WORKBENCH FUNCTIONS", C.BOLD))
    for name, desc in FUNCTION_HELP:
        out.append("    %-34s %s" % (name, dim(desc)))
    out.append("")
    out.append(paint("  COURSE OUTCOMES COVERED", C.BOLD))
    for clo in sorted(CLO):
        out.append("    %s" % paint("CLO %d" % clo, C.CYAN, C.BOLD))
        out.extend(wrap_lines(CLO[clo], 66, "      "))
    out.append("")
    out.append(paint("  EXACTNESS", C.BOLD))
    out.extend(wrap_lines(
        "Every number is an exact rational, or an exact element of a quadratic field "
        "when an eigenvalue needs one. 1/3 stays 1/3. An eigenvalue of (1 + sqrt 5)/2 "
        "stays that, and its eigenvector is computed in the same field. Only roots of "
        "irreducible polynomials of degree three or more fall back to numeric values, "
        "and those are labelled where they appear.", 66, "    "))
    return "\n".join(out)


def main():
    App().run()
    return 0
