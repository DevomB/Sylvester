from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd

from .exact import ONE, ZERO, Surd, content, denominators, weight
from .matrix import Matrix
from .render import arrow, fmt, fmt_coeff, leftrightarrow

HUMAN = "human"
MACHINE = "machine"


@dataclass
class Step:
    description: str
    matrix: Matrix
    note: str = ""
    kind: str = "op"
    op: tuple = ()

    @property
    def has_fraction(self):
        return any(d != 1 for v in self.matrix.entries() for d in denominators(v))


@dataclass
class Reduction:
    original: Matrix
    ref: Matrix
    rref: Matrix
    pivots: list
    steps: list = field(default_factory=list)
    factor: Fraction = ONE
    swaps: int = 0
    scalings: int = 0
    eliminations: int = 0
    pivot_columns_limit: int = 0

    @property
    def rank(self):
        return len(self.pivots)

    @property
    def pivot_columns(self):
        return [c for _, c in self.pivots]

    @property
    def free_columns(self):
        taken = set(self.pivot_columns)
        return [c for c in range(self.pivot_columns_limit) if c not in taken]

    @property
    def nullity(self):
        return self.pivot_columns_limit - self.rank

    @property
    def operation_count(self):
        return len(self.steps) - 1

    @property
    def fraction_steps(self):
        return sum(1 for s in self.steps if s.has_fraction)

    @property
    def worst_denominator(self):
        return max((d for s in self.steps for v in s.matrix.entries() for d in denominators(v)), default=1)

    @property
    def largest_entry(self):
        return max((weight(v) for s in self.steps for v in s.matrix.entries()), default=0)

    def elementary_ops(self):
        return [s.op for s in self.steps if s.op]


class _Work:
    __slots__ = ("m", "steps", "factor", "swaps", "scalings", "eliminations")

    def __init__(self, matrix):
        self.m = matrix.to_lists()
        self.steps = []
        self.factor = ONE
        self.swaps = 0
        self.scalings = 0
        self.eliminations = 0

    def record(self, description, note="", kind="op", op=()):
        self.steps.append(Step(description, Matrix(self.m), note, kind, op))

    def swap(self, i, j, note=""):
        if i == j:
            return
        self.m[i], self.m[j] = self.m[j], self.m[i]
        self.factor = -self.factor
        self.swaps += 1
        self.record("R%d %s R%d" % (i + 1, leftrightarrow(), j + 1), note, "swap", ("swap", i, j))

    def scale(self, i, k, note=""):
        self.m[i] = [k * v for v in self.m[i]]
        self.factor = self.factor * k
        self.scalings += 1
        self.record(
            "R%d %s %sR%d" % (i + 1, arrow(), fmt_coeff(k), i + 1), note, "scale", ("scale", i, k)
        )

    def combine(self, target, source, t_mult, s_mult, note=""):
        self.m[target] = [t_mult * a + s_mult * b for a, b in zip(self.m[target], self.m[source])]
        if t_mult != 1:
            self.factor = self.factor * t_mult
        self.eliminations += 1
        self.record(
            describe_combination(target, source, t_mult, s_mult),
            note,
            "add",
            ("combine", target, source, t_mult, s_mult),
        )


def describe_combination(target, source, t_mult, s_mult):
    left = fmt_coeff(t_mult) + "R" + str(target + 1)
    joiner = "-" if _negative(s_mult) else "+"
    mag = -s_mult if _negative(s_mult) else s_mult
    return "R%d %s %s %s %sR%d" % (target + 1, arrow(), left, joiner, fmt_coeff(mag), source + 1)


def _negative(value):
    if isinstance(value, Surd):
        return value.is_real and value.sign() < 0
    return isinstance(value, Fraction) and value < 0


def _is_rat(value):
    return isinstance(value, (int, Fraction))


def _orderable(value):
    return _is_rat(value) or (isinstance(value, Surd) and value.is_real)


def machine_pivot(m, r0, col):
    rows = [r for r in range(r0, len(m)) if m[r][col]]
    if not rows:
        return None
    if all(_orderable(m[r][col]) for r in rows):
        return max(rows, key=lambda r: abs(m[r][col]))
    if all(_is_rat(m[r][col]) or isinstance(m[r][col], Surd) for r in rows):
        return max(rows, key=lambda r: m[r][col].field_norm() if isinstance(m[r][col], Surd) else m[r][col] ** 2)
    return min(rows, key=lambda r: weight(m[r][col]))


def human_pivot(m, r0, col):
    candidates = [r for r in range(r0, len(m)) if m[r][col]]
    if not candidates:
        return None
    for want in (ONE, -ONE):
        for r in candidates:
            if m[r][col] == want:
                return r
    for r in candidates:
        p = m[r][col]
        if _is_rat(p) and p.denominator == 1:
            if all(_is_rat(m[o][col]) and (m[o][col] / p).denominator == 1 for o in candidates):
                return r
    return min(candidates, key=lambda r: (weight(m[r][col]), -sum(1 for v in m[r] if not v)))


def find_manufactured_one(m, r0, col):
    candidates = [r for r in range(r0, len(m)) if m[r][col]]
    if len(candidates) < 2 or not all(_is_rat(m[r][col]) for r in candidates):
        return None
    if any(abs(m[r][col]) == 1 for r in candidates):
        return None
    fallback = None
    for i in candidates:
        for j in candidates:
            if i == j:
                continue
            diff = m[i][col] - m[j][col]
            if diff == 1:
                return i, j
            if diff == -1 and fallback is None:
                fallback = (i, j)
    return fallback


def elimination_multipliers(entry, pivot, human):
    if not human or not (_is_rat(entry) and _is_rat(pivot)):
        return ONE, -entry / pivot
    ratio = entry / pivot
    if ratio.denominator == 1:
        return ONE, -ratio
    g = Fraction(
        gcd(entry.numerator * pivot.denominator, pivot.numerator * entry.denominator),
        entry.denominator * pivot.denominator,
    )
    if g == 0:
        return pivot, -entry
    t_mult, s_mult = pivot / g, -entry / g
    return (-t_mult, -s_mult) if t_mult < 0 else (t_mult, s_mult)


def _cancel(work, r, human):
    if not human:
        return
    g = content(work.m[r])
    if g != 1:
        work.scale(
            r,
            ONE / g,
            "every entry divides by %s, so cancel it out" % fmt(Fraction(g)),
        )


def _canonicalize_tail(work, pivots, limit):
    m = work.m
    ncols = len(m[0])
    row = len(pivots)
    for col in range(limit, ncols):
        if row >= len(m):
            return
        pick = next((r for r in range(row, len(m)) if m[r][col]), None)
        if pick is None:
            continue
        work.swap(pick, row, "bring the contradiction row up")
        if m[row][col] != 1:
            work.scale(row, ONE / m[row][col], "scale it so the row reads 0 = 1")
        for i in range(len(m)):
            if i != row and m[i][col]:
                work.combine(i, row, ONE, -m[i][col], "clear the constant against the contradiction row")
        pivots.append((row, col))
        row += 1


def row_reduce(matrix, pivot_limit=None, mode=HUMAN, reduced=True, canonical_tail=False):
    if matrix.nrows == 0 or matrix.ncols == 0:
        return Reduction(matrix, matrix, matrix, [], [Step("Initial matrix", matrix, kind="init")])
    limit = matrix.ncols if pivot_limit is None else pivot_limit
    human = mode == HUMAN
    work = _Work(matrix)
    work.record("Initial matrix", kind="init")
    m = work.m
    nrows = matrix.nrows
    pivots = []
    row = 0
    col = 0

    while row < nrows and col < limit:
        if human:
            made = find_manufactured_one(m, row, col)
            if made is not None:
                i, j = made
                work.combine(i, j, ONE, -ONE, "make a 1 in the pivot column so nothing below needs dividing")

        pick = human_pivot(m, row, col) if human else machine_pivot(m, row, col)
        if pick is None:
            col += 1
            continue

        work.swap(pick, row, "bring the friendliest pivot up" if human else "partial pivoting")

        if not human and m[row][col] != 1:
            work.scale(row, ONE / m[row][col], "scale the pivot to 1")

        pivot = m[row][col]
        for r in range(row + 1, nrows):
            if not m[r][col]:
                continue
            t_mult, s_mult = elimination_multipliers(m[r][col], pivot, human)
            work.combine(r, row, t_mult, s_mult)
            _cancel(work, r, human)

        pivots.append((row, col))
        row += 1
        col += 1

    ref = Matrix(m)

    if reduced:
        for r, c in reversed(pivots):
            pivot = m[r][c]
            for i in range(r):
                if not m[i][c]:
                    continue
                t_mult, s_mult = elimination_multipliers(m[i][c], pivot, human)
                work.combine(i, r, t_mult, s_mult, "clear above the pivot")
                _cancel(work, i, human)
        for r, c in pivots:
            if m[r][c] != 1:
                work.scale(
                    r,
                    ONE / m[r][c],
                    "only now divide, once, to land the leading 1" if human else "scale the pivot to 1",
                )
        if canonical_tail and limit < matrix.ncols:
            _canonicalize_tail(work, pivots, limit)
        _sink_zero_rows(work, pivots)

    return Reduction(
        original=matrix,
        ref=ref,
        rref=Matrix(work.m),
        pivots=pivots,
        steps=work.steps,
        factor=work.factor,
        swaps=work.swaps,
        scalings=work.scalings,
        eliminations=work.eliminations,
        pivot_columns_limit=limit,
    )


def _sink_zero_rows(work, pivots):
    m = work.m
    order = sorted(range(len(m)), key=lambda r: (all(not v for v in m[r]), r))
    if order == list(range(len(m))):
        return
    work.m[:] = [m[r] for r in order]
    if _permutation_parity(order) < 0:
        work.factor = -work.factor
    work.swaps += 1
    work.record("move empty rows to the bottom", "RREF keeps empty rows last", "swap")


def _permutation_parity(order):
    seen = [False] * len(order)
    sign = 1
    for i in range(len(order)):
        if seen[i]:
            continue
        length = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = order[j]
            length += 1
        if length % 2 == 0:
            sign = -sign
    return sign


def rref(matrix, pivot_limit=None, mode=HUMAN):
    return row_reduce(matrix, pivot_limit, mode).rref


def ref(matrix, pivot_limit=None, mode=HUMAN):
    return row_reduce(matrix, pivot_limit, mode, reduced=False).ref


def rank(matrix):
    return len(row_reduce(matrix, reduced=False).pivots)


def nullspace_basis(matrix):
    return nullspace_from(row_reduce(matrix))


def nullspace_from(reduction):
    width = reduction.original.ncols
    basis = []
    for free in reduction.free_columns:
        vec = [ZERO] * width
        vec[free] = ONE
        for r, c in reduction.pivots:
            vec[c] = -reduction.rref[r][free]
        basis.append(tuple(vec))
    return basis


def apply_op(matrix, op):
    rows = matrix.to_lists()
    if op[0] == "swap":
        _, i, j = op
        rows[i], rows[j] = rows[j], rows[i]
    elif op[0] == "scale":
        _, i, k = op
        rows[i] = [k * v for v in rows[i]]
    else:
        _, t, s, tm, sm = op
        rows[t] = [tm * a + sm * b for a, b in zip(rows[t], rows[s])]
    return Matrix(rows)


def op_to_elementary(op, n):
    from .matrix import elementary_scale, elementary_swap

    if op[0] == "swap":
        return elementary_swap(n, op[1], op[2])
    if op[0] == "scale":
        return elementary_scale(n, op[1], op[2])
    _, t, s, tm, sm = op
    e = Matrix.identity(n).to_lists()
    e[t][t] = tm
    e[t][s] = sm
    return Matrix(e)


def describe_op(op):
    if op[0] == "swap":
        return "R%d %s R%d" % (op[1] + 1, leftrightarrow(), op[2] + 1)
    if op[0] == "scale":
        return "R%d %s %sR%d" % (op[1] + 1, arrow(), fmt_coeff(op[2]), op[1] + 1)
    return describe_combination(op[1], op[2], op[3], op[4])
