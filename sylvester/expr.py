from __future__ import annotations

from fractions import Fraction

from .determinant import adjugate, determinant
from .exact import ONE, ZERO, Surd, sqrt_exact
from .inverse import inverse
from .matrix import Matrix
from .parse import ParseError, parse_value
from .reduce import ref, row_reduce, rref
from . import vectors as V

NUMBER = "number"
NAME = "name"
OP = "op"
END = "end"


class Token:
    __slots__ = ("kind", "text", "pos")

    def __init__(self, kind, text, pos):
        self.kind = kind
        self.text = text
        self.pos = pos


def tokenize(source):
    tokens = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and source[i + 1].isdigit()):
            j = i
            while j < n and (source[j].isdigit() or source[j] == "."):
                j += 1
            if j < n and source[j] in "eE" and j + 1 < n and (source[j + 1].isdigit() or source[j + 1] in "+-"):
                j += 2
                while j < n and source[j].isdigit():
                    j += 1
            tokens.append(Token(NUMBER, source[i:j], i))
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i
            while j < n and (source[j].isalnum() or source[j] == "_"):
                j += 1
            tokens.append(Token(NAME, source[i:j], i))
            i = j
            continue
        if ch in "+-*/^()[],;'":
            tokens.append(Token(OP, ch, i))
            i += 1
            continue
        raise ParseError("unexpected character %r at position %d" % (ch, i))
    tokens.append(Token(END, "", n))
    return tokens


class Evaluator:
    def __init__(self, registers=None):
        self.registers = registers if registers is not None else {}
        self.tokens = []
        self.pos = 0

    def evaluate(self, source):
        self.tokens = tokenize(source)
        self.pos = 0
        value = self._expr()
        if self._peek().kind != END:
            raise ParseError("unexpected %r at position %d" % (self._peek().text, self._peek().pos))
        return value

    def _peek(self, offset=0):
        index = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def _next(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _accept(self, text):
        if self._peek().kind == OP and self._peek().text == text:
            self.pos += 1
            return True
        return False

    def _expect(self, text):
        if not self._accept(text):
            raise ParseError("expected %r at position %d" % (text, self._peek().pos))

    def _expr(self):
        value = self._term()
        while True:
            if self._accept("+"):
                value = _add(value, self._term())
            elif self._accept("-"):
                value = _sub(value, self._term())
            else:
                return value

    def _term(self):
        value = self._unary()
        while True:
            if self._accept("*"):
                value = _mul(value, self._unary())
            elif self._accept("/"):
                value = _div(value, self._unary())
            elif self._starts_atom():
                value = _mul(value, self._unary())
            else:
                return value

    def _starts_atom(self):
        token = self._peek()
        if token.kind in (NUMBER, NAME):
            return True
        return token.kind == OP and token.text in "(["

    def _unary(self):
        if self._accept("-"):
            return _neg(self._unary())
        if self._accept("+"):
            return self._unary()
        return self._power()

    def _power(self):
        base = self._postfix()
        if self._accept("^"):
            token = self._peek()
            if token.kind == NAME and token.text in ("T", "t"):
                self._next()
                return self._finish_postfix(_transpose(base))
            exponent = self._unary()
            if isinstance(exponent, Matrix):
                raise ParseError("an exponent must be a number")
            return _power(base, exponent)
        return base

    def _postfix(self):
        return self._finish_postfix(self._atom())

    def _finish_postfix(self, value):
        while self._accept("'"):
            value = _transpose(value)
        return value

    def _atom(self):
        token = self._next()
        if token.kind == NUMBER:
            return parse_value(token.text)
        if token.kind == NAME:
            if self._peek().kind == OP and self._peek().text == "(":
                return self._call(token.text)
            return self._lookup(token.text)
        if token.kind == OP and token.text == "(":
            value = self._expr()
            self._expect(")")
            return value
        if token.kind == OP and token.text == "[":
            return self._literal_matrix()
        raise ParseError("unexpected %r at position %d" % (token.text, token.pos))

    def _literal_matrix(self):
        rows = [[]]
        while True:
            if self._accept("]"):
                break
            if self._accept(";"):
                rows.append([])
                continue
            if self._accept(","):
                continue
            rows[-1].append(self._expr())
        rows = [r for r in rows if r]
        if not rows:
            raise ParseError("empty matrix literal")
        return Matrix(rows)

    def _call(self, name):
        self._expect("(")
        args = []
        if not self._accept(")"):
            while True:
                args.append(self._expr())
                if self._accept(","):
                    continue
                self._expect(")")
                break
        return apply_function(name, args)

    def _lookup(self, name):
        key = name.upper()
        if key in self.registers:
            return self.registers[key]
        if name in CONSTANTS:
            return CONSTANTS[name]
        raise ParseError("nothing stored in %s" % key)


CONSTANTS = {}


def _is_matrix(value):
    return isinstance(value, Matrix)


def _add(a, b):
    if _is_matrix(a) != _is_matrix(b):
        raise ParseError("cannot add a matrix and a number")
    return a + b


def _sub(a, b):
    if _is_matrix(a) != _is_matrix(b):
        raise ParseError("cannot subtract a matrix and a number")
    return a - b


def _mul(a, b):
    if _is_matrix(a) and _is_matrix(b):
        return a.matmul(b)
    return a * b


def _div(a, b):
    if _is_matrix(b):
        if not _is_matrix(a):
            raise ParseError("cannot divide a number by a matrix; use inv(A)")
        return a.matmul(inverse(b))
    if not b:
        raise ParseError("division by zero")
    return a / b


def _neg(a):
    return -a


def _transpose(a):
    if not _is_matrix(a):
        return a
    return a.T


def _power(base, exponent):
    if isinstance(exponent, Surd) or Fraction(exponent).denominator != 1:
        raise ParseError("only whole-number powers are supported")
    return base ** int(exponent)


def _need_matrix(name, value):
    if not _is_matrix(value):
        raise ParseError("%s needs a matrix" % name)
    return value


def _need_int(name, value):
    if _is_matrix(value) or Fraction(value).denominator != 1:
        raise ParseError("%s needs a whole number" % name)
    return int(value)


def _as_vector(name, value):
    if _is_matrix(value):
        return V.as_vector(value)
    raise ParseError("%s needs a vector" % name)


def apply_function(name, args):
    key = name.lower()
    fn = FUNCTIONS.get(key)
    if fn is None:
        raise ParseError("no function called %s" % name)
    return fn(args)


def _one_matrix(name, fn):
    def run(args):
        if len(args) != 1:
            raise ParseError("%s takes one argument" % name)
        return fn(_need_matrix(name, args[0]))

    return run


def _identity(args):
    if len(args) != 1:
        raise ParseError("I takes one argument")
    return Matrix.identity(_need_int("I", args[0]))


def _zeros(args):
    if len(args) == 1:
        n = _need_int("zeros", args[0])
        return Matrix.zeros(n, n)
    if len(args) == 2:
        return Matrix.zeros(_need_int("zeros", args[0]), _need_int("zeros", args[1]))
    raise ParseError("zeros takes one or two arguments")


def _diag(args):
    if len(args) == 1 and _is_matrix(args[0]):
        m = args[0]
        return Matrix.diagonal([m[i, i] for i in range(min(m.shape))])
    return Matrix.diagonal(list(args))


def _dot(args):
    if len(args) != 2:
        raise ParseError("dot takes two vectors")
    return V.dot(_as_vector("dot", args[0]), _as_vector("dot", args[1]))


def _norm(args):
    if len(args) != 1:
        raise ParseError("norm takes one vector")
    return V.norm(_as_vector("norm", args[0]))


def _cross(args):
    if len(args) != 2:
        raise ParseError("cross takes two vectors in R^3")
    return Matrix.column(V.cross(_as_vector("cross", args[0]), _as_vector("cross", args[1])))


def _nullity(matrix):
    return Fraction(matrix.ncols - row_reduce(matrix).rank)


FUNCTIONS = {
    "det": _one_matrix("det", determinant),
    "trace": _one_matrix("trace", lambda m: m.trace()),
    "tr": _one_matrix("tr", lambda m: m.trace()),
    "rank": _one_matrix("rank", lambda m: Fraction(row_reduce(m).rank)),
    "nullity": _one_matrix("nullity", _nullity),
    "inv": _one_matrix("inv", inverse),
    "rref": _one_matrix("rref", rref),
    "ref": _one_matrix("ref", ref),
    "adj": _one_matrix("adj", adjugate),
    "adjugate": _one_matrix("adjugate", adjugate),
    "transpose": _one_matrix("transpose", lambda m: m.T),
    "t": _one_matrix("t", lambda m: m.T),
    "i": _identity,
    "eye": _identity,
    "zeros": _zeros,
    "diag": _diag,
    "dot": _dot,
    "norm": _norm,
    "cross": _cross,
    "sqrt": lambda args: sqrt_exact(args[0]),
}

FUNCTION_HELP = [
    ("det(A)", "determinant"),
    ("rank(A)", "rank"),
    ("nullity(A)", "nullity"),
    ("trace(A), tr(A)", "trace"),
    ("inv(A)", "inverse"),
    ("rref(A), ref(A)", "reduced and plain row echelon form"),
    ("adj(A)", "adjugate"),
    ("transpose(A), A', A^T", "transpose"),
    ("I(n), eye(n)", "identity matrix"),
    ("zeros(m, n)", "zero matrix"),
    ("diag(a, b, ...)", "diagonal matrix"),
    ("dot(u, v), norm(v), cross(u, v)", "vector operations"),
    ("sqrt(x)", "exact square root"),
]


def evaluate(source, registers=None):
    return Evaluator(registers).evaluate(source)
