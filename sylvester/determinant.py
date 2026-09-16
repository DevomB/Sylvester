from __future__ import annotations

from .exact import ONE, ZERO
from .matrix import Matrix
from .reduce import MACHINE, row_reduce


def _require_square(matrix, what="determinant"):
    if matrix.nrows != matrix.ncols:
        raise ValueError("the %s is only defined for a square matrix" % what)


def determinant(matrix):
    _require_square(matrix)
    n = matrix.nrows
    if n == 0:
        return ONE
    if n == 1:
        return matrix[0, 0]
    if n == 2:
        return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
    return bareiss(matrix)


def bareiss(matrix):
    n = matrix.nrows
    a = matrix.to_lists()
    sign = 1
    prev = ONE
    for k in range(n - 1):
        if not a[k][k]:
            pick = next((i for i in range(k + 1, n) if a[i][k]), None)
            if pick is None:
                return ZERO
            a[k], a[pick] = a[pick], a[k]
            sign = -sign
        pivot = a[k][k]
        for i in range(k + 1, n):
            row_i = a[i]
            aik = row_i[k]
            for j in range(k + 1, n):
                row_i[j] = (row_i[j] * pivot - aik * a[k][j]) / prev
            row_i[k] = ZERO
        prev = pivot
    return a[n - 1][n - 1] if sign > 0 else -a[n - 1][n - 1]


def minor(matrix, i, j):
    return determinant(matrix.minor_matrix(i, j))


def cofactor(matrix, i, j):
    m = minor(matrix, i, j)
    return m if (i + j) % 2 == 0 else -m


def cofactor_matrix(matrix):
    _require_square(matrix, "cofactor matrix")
    n = matrix.nrows
    return Matrix([[cofactor(matrix, i, j) for j in range(n)] for i in range(n)])


def adjugate(matrix):
    return cofactor_matrix(matrix).T


class Term:
    __slots__ = ("row", "col", "entry", "sign", "block", "value", "child")

    def __init__(self, row, col, entry, sign, block, value, child=None):
        self.row = row
        self.col = col
        self.entry = entry
        self.sign = sign
        self.block = block
        self.value = value
        self.child = child

    @property
    def contribution(self):
        return self.entry * self.value if self.sign > 0 else -(self.entry * self.value)


class Expansion:
    __slots__ = ("matrix", "axis", "index", "terms", "value", "reason")

    def __init__(self, matrix, axis, index, terms, value, reason=""):
        self.matrix = matrix
        self.axis = axis
        self.index = index
        self.terms = terms
        self.value = value
        self.reason = reason

    @property
    def label(self):
        return "%s %d" % ("row" if self.axis == "row" else "column", self.index + 1)


def best_axis(matrix):
    n = matrix.nrows
    best = ("row", 0, -1)
    for i in range(n):
        zeros = sum(1 for v in matrix.rows[i] if not v)
        if zeros > best[2]:
            best = ("row", i, zeros)
    for j in range(n):
        zeros = sum(1 for v in matrix.column_at(j) if not v)
        if zeros > best[2]:
            best = ("col", j, zeros)
    return best[0], best[1]


def expand_cofactors(matrix, axis=None, index=None, depth=0, max_depth=6):
    _require_square(matrix, "cofactor expansion")
    n = matrix.nrows
    if n == 1:
        return Expansion(matrix, "row", 0, [], matrix[0, 0], "a 1x1 determinant is its only entry")
    if axis is None or index is None:
        axis, index = best_axis(matrix)
    reason = ""
    zeros = sum(
        1 for v in (matrix.rows[index] if axis == "row" else matrix.column_at(index)) if not v
    )
    if zeros:
        reason = "chosen because it holds %d zero%s, so those terms vanish" % (
            zeros,
            "" if zeros == 1 else "s",
        )
    terms = []
    total = ZERO
    for k in range(n):
        i, j = (index, k) if axis == "row" else (k, index)
        entry = matrix[i, j]
        if not entry:
            continue
        block = matrix.minor_matrix(i, j)
        sign = 1 if (i + j) % 2 == 0 else -1
        if block.nrows > 2 and depth < max_depth:
            child = expand_cofactors(block, depth=depth + 1, max_depth=max_depth)
            value = child.value
        else:
            child = None
            value = determinant(block)
        term = Term(i, j, entry, sign, block, value, child)
        terms.append(term)
        total += term.contribution
    return Expansion(matrix, axis, index, terms, total, reason)


class RowOpDeterminant:
    __slots__ = ("matrix", "reduction", "value", "diagonal", "singular")

    def __init__(self, matrix, reduction, value, diagonal, singular):
        self.matrix = matrix
        self.reduction = reduction
        self.value = value
        self.diagonal = diagonal
        self.singular = singular


def det_by_row_reduction(matrix, mode=MACHINE):
    _require_square(matrix)
    red = row_reduce(matrix, mode=mode, reduced=False)
    n = matrix.nrows
    diagonal = [red.ref[i][i] for i in range(n)]
    product = ONE
    for v in diagonal:
        product = product * v
    value = product / red.factor
    return RowOpDeterminant(matrix, red, value, diagonal, not product)


def det_triangular(matrix):
    _require_square(matrix)
    product = ONE
    for i in range(matrix.nrows):
        product = product * matrix[i, i]
    return product


def is_invertible(matrix):
    return matrix.is_square and bool(determinant(matrix))


def cramer(matrix, constants):
    _require_square(matrix, "Cramer rule")
    d = determinant(matrix)
    if not d:
        return None, d, []
    numerators = []
    for j in range(matrix.ncols):
        replaced = matrix.with_column(j, constants)
        numerators.append((replaced, determinant(replaced)))
    return [num / d for _, num in numerators], d, numerators


def vandermonde(values):
    n = len(values)
    return Matrix([[values[i] ** j for j in range(n)] for i in range(n)])


def characteristic_via_cofactor(matrix):
    from .polynomial import Poly

    _require_square(matrix, "characteristic polynomial")
    n = matrix.nrows
    entries = [
        [(Poly([0, 1]) if i == j else Poly([])) - Poly([matrix[i, j]]) for j in range(n)]
        for i in range(n)
    ]
    return _poly_det(entries, n)


def _poly_det(entries, n):
    from .polynomial import Poly

    if n == 1:
        return entries[0][0]
    if n == 2:
        return entries[0][0] * entries[1][1] - entries[0][1] * entries[1][0]
    total = Poly([])
    for j in range(n):
        head = entries[0][j]
        if not head:
            continue
        block = [[entries[i][c] for c in range(n) if c != j] for i in range(1, n)]
        sub = _poly_det(block, n - 1)
        total = total + (head * sub if j % 2 == 0 else -(head * sub))
    return total
