from __future__ import annotations

from .determinant import cramer, determinant
from .exact import ONE, ZERO
from .matrix import Matrix
from .reduce import HUMAN, row_reduce

UNIQUE = "unique"
INFINITE = "infinite"
INCONSISTENT = "inconsistent"


class Solution:
    __slots__ = (
        "coefficients",
        "constants",
        "augmented",
        "reduction",
        "kind",
        "rank",
        "rank_augmented",
        "num_vars",
        "particular",
        "homogeneous_basis",
        "free_columns",
        "pivot_columns",
        "witness_row",
    )

    def __init__(self, coefficients, constants, augmented, reduction):
        self.coefficients = coefficients
        self.constants = tuple(constants)
        self.augmented = augmented
        self.reduction = reduction
        self.num_vars = coefficients.ncols
        self.pivot_columns = [c for _, c in reduction.pivots if c < self.num_vars]
        self.rank = len(self.pivot_columns)
        taken = set(self.pivot_columns)
        self.free_columns = [c for c in range(self.num_vars) if c not in taken]
        self.witness_row = _contradiction_row(reduction.rref, self.num_vars)
        if self.witness_row is not None:
            self.kind = INCONSISTENT
            self.rank_augmented = self.rank + 1
            self.particular = None
            self.homogeneous_basis = []
        else:
            self.rank_augmented = self.rank
            self.particular = _particular(reduction.rref, reduction.pivots, self.num_vars)
            self.homogeneous_basis = _homogeneous_basis(reduction.rref, reduction.pivots, self.num_vars, self.free_columns)
            self.kind = UNIQUE if not self.free_columns else INFINITE

    @property
    def nullity(self):
        return self.num_vars - self.rank

    @property
    def consistent(self):
        return self.kind != INCONSISTENT

    @property
    def parameters(self):
        return ["t%d" % (i + 1) for i in range(len(self.free_columns))]

    def values(self):
        if self.kind != UNIQUE:
            return None
        return dict(enumerate(self.particular))

    def verify(self):
        if not self.consistent:
            return True
        target = Matrix.column(self.constants)
        if self.coefficients.matmul(Matrix.column(self.particular)) != target:
            return False
        return all(self.coefficients.matmul(Matrix.column(v)).is_zero() for v in self.homogeneous_basis)

    def at(self, params):
        if not self.consistent:
            return None
        out = list(self.particular)
        for t, vec in zip(params, self.homogeneous_basis):
            out = [a + t * b for a, b in zip(out, vec)]
        return tuple(out)


def _contradiction_row(rref, num_vars):
    for r in range(rref.nrows):
        if all(not rref[r][c] for c in range(num_vars)) and any(
            rref[r][c] for c in range(num_vars, rref.ncols)
        ):
            return r
    return None


def _particular(rref, pivots, num_vars):
    out = [ZERO] * num_vars
    for r, c in pivots:
        if c < num_vars:
            out[c] = rref[r][num_vars]
    return tuple(out)


def _homogeneous_basis(rref, pivots, num_vars, free_columns):
    basis = []
    for free in free_columns:
        vec = [ZERO] * num_vars
        vec[free] = ONE
        for r, c in pivots:
            if c < num_vars:
                vec[c] = -rref[r][free]
        basis.append(tuple(vec))
    return basis


def solve(coefficients, constants, mode=HUMAN):
    if len(constants) != coefficients.nrows:
        raise ValueError(
            "expected %d constants for %d equations, got %d"
            % (coefficients.nrows, coefficients.nrows, len(constants))
        )
    augmented = coefficients.augment(list(constants))
    reduction = row_reduce(augmented, coefficients.ncols, mode, canonical_tail=True)
    return Solution(coefficients, constants, augmented, reduction)


def solve_augmented(augmented, mode=HUMAN):
    n = augmented.ncols - 1
    return solve(augmented.left(n), [row[n] for row in augmented.rows], mode)


def solve_homogeneous(coefficients, mode=HUMAN):
    return solve(coefficients, [ZERO] * coefficients.nrows, mode)


class HomogeneousReport:
    __slots__ = ("matrix", "solution", "det", "square", "nontrivial", "reason")

    def __init__(self, matrix, solution, det, square, nontrivial, reason):
        self.matrix = matrix
        self.solution = solution
        self.det = det
        self.square = square
        self.nontrivial = nontrivial
        self.reason = reason


def homogeneous_report(matrix, mode=HUMAN):
    sol = solve_homogeneous(matrix, mode)
    square = matrix.is_square
    det = determinant(matrix) if square else None
    nontrivial = sol.nullity > 0
    if square:
        reason = (
            "det(A) = 0, so the only square-system test says a nontrivial solution exists"
            if not det
            else "det(A) != 0, so A is invertible and x = 0 is the only solution"
        )
    elif matrix.ncols > matrix.nrows:
        reason = (
            "more unknowns (%d) than equations (%d), so a nontrivial solution always exists"
            % (matrix.ncols, matrix.nrows)
        )
    else:
        reason = "rank %d against %d unknowns leaves nullity %d" % (sol.rank, matrix.ncols, sol.nullity)
    return HomogeneousReport(matrix, sol, det, square, nontrivial, reason)


class CramerReport:
    __slots__ = ("matrix", "constants", "det", "numerators", "solution", "usable", "reason")

    def __init__(self, matrix, constants, det, numerators, solution, usable, reason):
        self.matrix = matrix
        self.constants = tuple(constants)
        self.det = det
        self.numerators = numerators
        self.solution = solution
        self.usable = usable
        self.reason = reason


def cramer_report(matrix, constants):
    if not matrix.is_square:
        return CramerReport(
            matrix, constants, None, [], None, False,
            "Cramer's rule needs a square system; this one is %dx%d" % matrix.shape,
        )
    solution, det, numerators = cramer(matrix, constants)
    if solution is None:
        return CramerReport(
            matrix, constants, det, numerators, None, False,
            "det(A) = 0, so Cramer's rule does not apply; row reduce instead",
        )
    return CramerReport(matrix, constants, det, numerators, solution, True, "")


