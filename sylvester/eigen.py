from __future__ import annotations

from .determinant import characteristic_via_cofactor, determinant
from .exact import ONE, ZERO, Surd, approx, unify_field
from .matrix import Matrix
from .polynomial import Poly, roots_of
from .reduce import nullspace_basis, row_reduce
from .vectors import gram_schmidt, is_orthogonal_set


def characteristic_polynomial(matrix):
    if not matrix.is_square:
        raise ValueError("the characteristic polynomial needs a square matrix")
    n = matrix.nrows
    if n <= 3:
        return characteristic_via_cofactor(matrix)
    return faddeev_leverrier(matrix)


def faddeev_leverrier(matrix):
    n = matrix.nrows
    identity = Matrix.identity(n)
    coeffs = [ZERO] * n + [ONE]
    m = Matrix.zeros(n)
    for k in range(1, n + 1):
        m = matrix.matmul(m) + identity * coeffs[n - k + 1]
        coeffs[n - k] = -(matrix.matmul(m)).trace() / k
    return Poly(coeffs)


def eigenvalues(matrix):
    return roots_of(characteristic_polynomial(matrix))


def eigenspace(matrix, value):
    n = matrix.nrows
    shifted = matrix - Matrix.identity(n) * value
    return nullspace_basis(shifted), shifted


class Eigenpair:
    __slots__ = ("value", "algebraic", "geometric", "basis", "shifted", "exact", "defect")

    def __init__(self, value, algebraic, geometric, basis, shifted, exact):
        self.value = value
        self.algebraic = algebraic
        self.geometric = geometric
        self.basis = basis
        self.shifted = shifted
        self.exact = exact
        self.defect = algebraic - geometric

    @property
    def defective(self):
        return self.defect > 0

    @property
    def is_real(self):
        if not self.exact:
            return abs(self.value.imag) < 1e-9
        return not isinstance(self.value, Surd) or self.value.is_real

    def verify(self, matrix):
        if not self.exact:
            return True
        for v in self.basis:
            left = matrix.matmul(Matrix.column(v))
            right = Matrix.column(v) * self.value
            if left != right:
                return False
        return bool(self.basis)


class Spectrum:
    __slots__ = ("matrix", "poly", "pairs", "exact", "trace_check", "det_check",
                 "diagonalizable", "reason", "field")

    def __init__(self, matrix, poly, pairs, exact, trace_check, det_check, diagonalizable, reason, field):
        self.matrix = matrix
        self.poly = poly
        self.pairs = pairs
        self.exact = exact
        self.trace_check = trace_check
        self.det_check = det_check
        self.diagonalizable = diagonalizable
        self.reason = reason
        self.field = field

    @property
    def total_geometric(self):
        return sum(p.geometric for p in self.pairs)

    @property
    def real_only(self):
        return all(p.is_real for p in self.pairs)

    def verify(self):
        return all(p.verify(self.matrix) for p in self.pairs)


def spectrum(matrix):
    if not matrix.is_square:
        raise ValueError("eigenvalues need a square matrix")
    n = matrix.nrows
    poly = characteristic_polynomial(matrix)
    roots, exact = roots_of(poly)
    pairs = []
    for root in roots:
        if root.exact:
            basis, shifted = eigenspace(matrix, root.value)
            pairs.append(Eigenpair(root.value, root.multiplicity, len(basis), basis, shifted, True))
        else:
            basis = numeric_eigenvectors(matrix, root.value)
            pairs.append(Eigenpair(root.value, root.multiplicity, len(basis), basis, None, False))
    trace_check = _safe_check(_sum_values, pairs, matrix.trace())
    det_check = _safe_check(_product_values, pairs, determinant(matrix))
    total = sum(p.geometric for p in pairs)
    diagonalizable = total == n
    if diagonalizable:
        reason = "the eigenvectors span R^%d, so A is diagonalizable" % n
    else:
        short = [p for p in pairs if p.defective]
        reason = "only %d independent eigenvectors for %d dimensions; %s falls short" % (
            total, n,
            ", ".join("lambda = %s" % _short(p.value) for p in short) or "an eigenvalue",
        )
    try:
        field = unify_field([p.value for p in pairs if p.exact])
    except ArithmeticError:
        field = None
    return Spectrum(matrix, poly, pairs, exact, trace_check, det_check, diagonalizable, reason, field)


def _short(value):
    from .render import fmt

    return fmt(value) if not isinstance(value, complex) else "%.4f%+.4fi" % (value.real, value.imag)


def _safe_check(fn, pairs, expected):
    try:
        return _close(fn(pairs), expected)
    except ArithmeticError:
        return None


def _sum_values(pairs):
    total = ZERO
    for p in pairs:
        total = total + p.value * p.algebraic
    return total


def _product_values(pairs):
    total = ONE
    for p in pairs:
        for _ in range(p.algebraic):
            total = total * p.value
    return total


def _close(a, b):
    if isinstance(a, complex) or isinstance(b, complex):
        return abs(complex(approx(a)) - complex(approx(b))) < 1e-7
    try:
        return a == b
    except ArithmeticError:
        return abs(approx(a) - approx(b)) < 1e-7


def numeric_eigenvectors(matrix, value, tol=1e-9):
    n = matrix.nrows
    a = [[complex(approx(matrix[i, j])) - (value if i == j else 0) for j in range(n)] for i in range(n)]
    pivots = []
    row = 0
    for col in range(n):
        if row >= n:
            break
        pick = max(range(row, n), key=lambda r: abs(a[r][col]))
        if abs(a[pick][col]) < tol:
            continue
        a[row], a[pick] = a[pick], a[row]
        head = a[row][col]
        a[row] = [x / head for x in a[row]]
        for r in range(n):
            if r != row and abs(a[r][col]) > tol:
                factor = a[r][col]
                a[r] = [x - factor * y for x, y in zip(a[r], a[row])]
        pivots.append((row, col))
        row += 1
    taken = {c for _, c in pivots}
    out = []
    for free in range(n):
        if free in taken:
            continue
        vec = [0j] * n
        vec[free] = 1 + 0j
        for r, c in pivots:
            vec[c] = -a[r][free]
        scale = max(abs(x) for x in vec)
        out.append(tuple(x / scale for x in vec))
    return out


class Diagonalization:
    __slots__ = ("matrix", "p", "d", "ok", "reason", "spectrum", "verified")

    def __init__(self, matrix, p, d, ok, reason, spec, verified):
        self.matrix = matrix
        self.p = p
        self.d = d
        self.ok = ok
        self.reason = reason
        self.spectrum = spec
        self.verified = verified


def diagonalize(matrix, spec=None):
    spec = spec or spectrum(matrix)
    n = matrix.nrows
    if not spec.diagonalizable:
        return Diagonalization(matrix, None, None, False, spec.reason, spec, False)
    if not all(p.exact for p in spec.pairs):
        return Diagonalization(
            matrix, None, None, False,
            "some eigenvalues are only known numerically, so no exact P is available",
            spec, False,
        )
    try:
        unify_field([p.value for p in spec.pairs])
    except ArithmeticError:
        return Diagonalization(
            matrix, None, None, False,
            "the eigenvalues live in different quadratic fields, so P has no single exact field",
            spec, False,
        )
    columns = []
    diag = []
    for pair in spec.pairs:
        for v in pair.basis:
            columns.append(v)
            diag.append(pair.value)
    p = Matrix.from_columns(columns)
    d = Matrix.diagonal(diag)
    verified = matrix.matmul(p) == p.matmul(d)
    return Diagonalization(matrix, p, d, True, "P^-1 A P = D", spec, verified)


class OrthogonalDiagonalization:
    __slots__ = ("matrix", "spectrum", "blocks", "orthogonal_basis", "ok", "reason", "q", "d")

    def __init__(self, matrix, spec, blocks, basis, ok, reason, q, d):
        self.matrix = matrix
        self.spectrum = spec
        self.blocks = blocks
        self.orthogonal_basis = basis
        self.ok = ok
        self.reason = reason
        self.q = q
        self.d = d


def orthogonally_diagonalize(matrix):
    if not matrix.is_symmetric():
        return OrthogonalDiagonalization(
            matrix, None, [], [], False,
            "the spectral theorem needs a symmetric matrix; this one is not symmetric",
            None, None,
        )
    spec = spectrum(matrix)
    if not spec.diagonalizable or not all(p.exact for p in spec.pairs):
        return OrthogonalDiagonalization(
            matrix, spec, [], [], False,
            "the eigenvalues are not all exact, so no exact orthogonal basis is available",
            None, None,
        )
    blocks = []
    basis = []
    diag = []
    for pair in spec.pairs:
        ortho = gram_schmidt(pair.basis)
        blocks.append((pair, ortho))
        for v in ortho.vectors:
            basis.append(v)
            diag.append(pair.value)
    try:
        unify_field([v for column in basis for v in column] + [p.value for p in spec.pairs])
    except ArithmeticError:
        return OrthogonalDiagonalization(
            matrix, spec, blocks, basis, False,
            "the eigenvalues span more than one quadratic field, so no single exact Q can be"
            " written here; each eigenspace above is already orthogonal, and the spectral theorem"
            " guarantees the eigenspaces are orthogonal to each other",
            None, None,
        )
    q = Matrix.from_columns(basis)
    d = Matrix.diagonal(diag)
    ok = is_orthogonal_set(basis) and matrix.matmul(q) == q.matmul(d)
    reason = (
        "eigenvectors from different eigenspaces of a symmetric matrix are automatically orthogonal;"
        " normalize each column to turn P into an orthogonal Q"
        if ok
        else "the basis did not come out orthogonal"
    )
    return OrthogonalDiagonalization(matrix, spec, blocks, basis, ok, reason, q, d)


def cayley_hamilton(matrix):
    poly = characteristic_polynomial(matrix)
    n = matrix.nrows
    acc = Matrix.zeros(n)
    terms = []
    for i, c in enumerate(poly.c):
        if not c:
            continue
        power = matrix ** i
        terms.append((i, c, power))
        acc = acc + power * c
    return poly, terms, acc, acc.is_zero()


def algebraic_multiplicity(matrix, value):
    poly = characteristic_polynomial(matrix)
    count = 0
    while poly.degree > 0 and poly.eval(value) == 0:
        poly = poly / Poly([-value, 1])
        count += 1
    return count


def similar(a, b, p):
    return p.matmul(b) == a.matmul(p)
