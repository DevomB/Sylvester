from __future__ import annotations

from .determinant import adjugate, determinant
from .exact import ONE, ZERO
from .matrix import Matrix
from .reduce import MACHINE, op_to_elementary, row_reduce


class Singular(ValueError):
    pass


class GaussJordanInverse:
    __slots__ = ("matrix", "reduction", "inverse", "ok", "rank")

    def __init__(self, matrix, reduction, inverse, ok, rank):
        self.matrix = matrix
        self.reduction = reduction
        self.inverse = inverse
        self.ok = ok
        self.rank = rank


def inverse_gauss_jordan(matrix, mode=MACHINE):
    if not matrix.is_square:
        raise ValueError("only a square matrix can have an inverse")
    n = matrix.nrows
    red = row_reduce(matrix.augment(Matrix.identity(n)), n, mode)
    left = red.rref.left(n)
    ok = left.is_identity()
    return GaussJordanInverse(matrix, red, red.rref.right(n) if ok else None, ok, red.rank)


def inverse(matrix):
    result = inverse_gauss_jordan(matrix)
    if not result.ok:
        raise Singular("the matrix is singular, so it has no inverse (rank %d < %d)" % (result.rank, matrix.nrows))
    return result.inverse


class AdjugateInverse:
    __slots__ = ("matrix", "det", "adjugate", "inverse", "ok")

    def __init__(self, matrix, det, adj, inv, ok):
        self.matrix = matrix
        self.det = det
        self.adjugate = adj
        self.inverse = inv
        self.ok = ok


def inverse_adjugate(matrix):
    if not matrix.is_square:
        raise ValueError("only a square matrix can have an inverse")
    d = determinant(matrix)
    adj = adjugate(matrix)
    return AdjugateInverse(matrix, d, adj, adj * (ONE / d) if d else None, bool(d))


class ElementaryFactorization:
    __slots__ = ("matrix", "ops", "factors", "inverse_factors", "ok", "product")

    def __init__(self, matrix, ops, factors, inverse_factors, ok, product):
        self.matrix = matrix
        self.ops = ops
        self.factors = factors
        self.inverse_factors = inverse_factors
        self.ok = ok
        self.product = product


def elementary_factorization(matrix):
    if not matrix.is_square:
        raise ValueError("only a square matrix factors into elementary matrices")
    n = matrix.nrows
    red = row_reduce(matrix, mode=MACHINE)
    if not red.rref.is_identity():
        return ElementaryFactorization(matrix, [], [], [], False, None)
    ops = red.elementary_ops()
    factors = [op_to_elementary(op, n) for op in ops]
    inverse_factors = [invert_elementary(op, n) for op in ops]
    product = Matrix.identity(n)
    for e in inverse_factors:
        product = product.matmul(e)
    return ElementaryFactorization(matrix, ops, factors, inverse_factors, True, product)


def invert_elementary(op, n):
    from .matrix import elementary_scale, elementary_swap

    if op[0] == "swap":
        return elementary_swap(n, op[1], op[2])
    if op[0] == "scale":
        return elementary_scale(n, op[1], ONE / op[2])
    _, t, s, tm, sm = op
    e = Matrix.identity(n).to_lists()
    e[t][t] = ONE / tm
    e[t][s] = -sm / tm
    return Matrix(e)


class LU:
    __slots__ = ("matrix", "p", "l", "u", "permutation", "swaps", "ok")

    def __init__(self, matrix, p, l, u, permutation, swaps, ok):
        self.matrix = matrix
        self.p = p
        self.l = l
        self.u = u
        self.permutation = permutation
        self.swaps = swaps
        self.ok = ok

    def verify(self):
        return self.p.matmul(self.matrix) == self.l.matmul(self.u)


def lu_decomposition(matrix):
    n = matrix.nrows
    m = matrix.ncols
    u = matrix.to_lists()
    l = Matrix.identity(n).to_lists()
    perm = list(range(n))
    swaps = 0
    row = 0
    for col in range(m):
        if row >= n:
            break
        pick = next((r for r in range(row, n) if u[r][col]), None)
        if pick is None:
            continue
        if pick != row:
            u[row], u[pick] = u[pick], u[row]
            perm[row], perm[pick] = perm[pick], perm[row]
            for c in range(row):
                l[row][c], l[pick][c] = l[pick][c], l[row][c]
            swaps += 1
        pivot = u[row][col]
        for r in range(row + 1, n):
            if not u[r][col]:
                continue
            factor = u[r][col] / pivot
            l[r][row] = factor
            u[r] = [a - factor * b for a, b in zip(u[r], u[row])]
        row += 1
    p = Matrix([[ONE if perm[i] == j else ZERO for j in range(n)] for i in range(n)])
    result = LU(matrix, p, Matrix(l), Matrix(u), perm, swaps, True)
    result.ok = result.verify()
    return result


def solve_with_inverse(matrix, constants):
    inv = inverse(matrix)
    return inv.matmul(Matrix.column(constants)), inv
