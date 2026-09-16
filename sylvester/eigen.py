from __future__ import annotations

from fractions import Fraction

from .algebraic import AlgebraicNumber, NumberField, ReducibleModulus
from .determinant import characteristic_via_cofactor, determinant
from .exact import ONE, ZERO, Surd, approx
from .factor import factor
from .matrix import Matrix
from .polynomial import Poly, approximate_roots, quadratic_roots
from .reduce import nullspace_from, row_reduce
from .vectors import dot, gram_schmidt


def characteristic_polynomial(matrix):
    if not matrix.is_square:
        raise ValueError("the characteristic polynomial needs a square matrix")
    if matrix.nrows <= 3:
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


def eigenspace(matrix, value):
    shifted = matrix - Matrix.identity(matrix.nrows) * value
    reduction = row_reduce(shifted)
    return nullspace_from(reduction), shifted, reduction


class Eigenpair:
    __slots__ = ("value", "algebraic", "basis", "shifted", "reduction", "minimal", "count",
                 "real_count", "roots", "symbolic")

    def __init__(self, value, algebraic, basis, shifted, reduction, minimal, count, real_count, roots,
                 symbolic=False):
        self.value = value
        self.algebraic = algebraic
        self.basis = basis
        self.shifted = shifted
        self.reduction = reduction
        self.minimal = minimal
        self.count = count
        self.real_count = real_count
        self.roots = roots
        self.symbolic = symbolic

    @property
    def geometric(self):
        return len(self.basis)

    @property
    def defect(self):
        return self.algebraic - self.geometric

    @property
    def defective(self):
        return self.defect > 0

    @property
    def is_real(self):
        return self.real_count == self.count

    def verify(self, matrix):
        return bool(self.basis) and all(is_eigenvector(matrix, v, self.value) for v in self.basis)


def is_eigenvector(matrix, vector, value):
    column = Matrix.column(vector)
    return not column.is_zero() and matrix.matmul(column) == column * value


class Spectrum:
    __slots__ = ("matrix", "poly", "factors", "pairs", "trace_check", "det_check", "diagonalizable",
                 "reason")

    def __init__(self, matrix, poly, factors, pairs, trace_check, det_check, diagonalizable, reason):
        self.matrix = matrix
        self.poly = poly
        self.factors = factors
        self.pairs = pairs
        self.trace_check = trace_check
        self.det_check = det_check
        self.diagonalizable = diagonalizable
        self.reason = reason

    @property
    def total_geometric(self):
        return sum(p.geometric * p.count for p in self.pairs)

    @property
    def real_only(self):
        return all(p.is_real for p in self.pairs)

    @property
    def symbolic(self):
        return any(p.symbolic for p in self.pairs)

    def verify(self):
        return all(p.verify(self.matrix) for p in self.pairs)


def spectrum(matrix):
    if not matrix.is_square:
        raise ValueError("eigenvalues need a square matrix")
    n = matrix.nrows
    poly = characteristic_polynomial(matrix)
    factors = [(q.monic(), m) for q, m in factor(poly)[1]]
    pairs = []
    for q, m in factors:
        pairs.extend(_pairs_for(matrix, q, m))
    pairs.sort(key=_order)
    trace_sum = ZERO
    det_product = ONE
    for q, m in factors:
        trace_sum -= q[q.degree - 1] * m
        det_product *= ((-1) ** q.degree * q[0]) ** m
    total = sum(p.geometric * p.count for p in pairs)
    real = all(p.is_real for p in pairs)
    if total == n:
        reason = "the eigenvectors span %s^%d, so A is diagonalizable%s" % (
            "R" if real else "C", n, "" if real else " over the complex numbers")
    else:
        reason = "only %d independent eigenvectors for %d dimensions; %s falls short" % (
            total, n, ", ".join(eigenvalue_label(p) for p in pairs if p.defective))
    return Spectrum(matrix, poly, factors, pairs, trace_sum == matrix.trace(),
                    det_product == determinant(matrix), total == n, reason)


def _pairs_for(matrix, q, multiplicity):
    k = q.degree
    if k == 1:
        value = -q[0]
        basis, shifted, reduction = eigenspace(matrix, value)
        return [Eigenpair(value, multiplicity, basis, shifted, reduction, q, 1, 1, [float(value)])]
    if k == 2:
        out = []
        for value in quadratic_roots(ONE, q[1], q[0]):
            basis, shifted, reduction = eigenspace(matrix, value)
            real = not isinstance(value, Surd) or value.is_real
            out.append(Eigenpair(value, multiplicity, basis, shifted, reduction, q, 1, int(real),
                                 [approx(value)]))
        return out
    field = NumberField(q.c, "lambda")
    real, complex_roots = approximate_roots(q)
    field.roots = tuple(real) + tuple(complex_roots)
    field.real_roots = len(real)
    try:
        basis, shifted, reduction = eigenspace(matrix, field.generator)
    except ReducibleModulus as exc:
        g = Poly(exc.factor)
        return _pairs_for(matrix, g, multiplicity) + _pairs_for(matrix, q / g, multiplicity)
    basis = [_integral(v) for v in basis]
    return [Eigenpair(field.generator, multiplicity, basis, shifted, reduction, q, k, len(real),
                      list(field.roots), True)]


def _integral(vector):
    den = 1
    for x in vector:
        for c in (x.c if isinstance(x, AlgebraicNumber) else (Fraction(x),)):
            den = den * c.denominator // _gcd(den, c.denominator)
    if den == 1:
        return vector
    return tuple(x * den for x in vector)


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _order(pair):
    values = [r.real if isinstance(r, complex) else r for r in pair.roots]
    return (0 if pair.is_real else 1, min(values))


def eigenvalue_label(pair):
    from .render import fmt, lam

    if pair.symbolic:
        return "each root of %s" % pair.minimal.shift_variable(lam())
    return "%s = %s" % (lam(), fmt(pair.value))


def eigenvalues(matrix):
    return spectrum(matrix).pairs


def root_values(pair):
    return list(pair.roots) if pair.symbolic else [pair.value]


def evaluate_at(entry, root):
    return entry.at(root) if isinstance(entry, AlgebraicNumber) else float(entry)


class Diagonalization:
    __slots__ = ("matrix", "spectrum", "ok", "reason", "columns", "p", "d", "verified")

    def __init__(self, matrix, spec, ok, reason, columns, p, d, verified):
        self.matrix = matrix
        self.spectrum = spec
        self.ok = ok
        self.reason = reason
        self.columns = columns
        self.p = p
        self.d = d
        self.verified = verified

    @property
    def symbolic(self):
        return self.spectrum is not None and self.spectrum.symbolic


def diagonalize(matrix, spec=None):
    spec = spec or spectrum(matrix)
    if not spec.diagonalizable:
        return Diagonalization(matrix, spec, False, spec.reason, [], None, None, False)
    columns = [(pair, v) for pair in spec.pairs for v in pair.basis]
    verified = spec.verify()
    p = d = None
    if not spec.symbolic:
        p = Matrix.from_columns([v for _, v in columns])
        d = Matrix.diagonal([pair.value for pair, _ in columns])
        verified = verified and matrix.matmul(p) == p.matmul(d)
    return Diagonalization(matrix, spec, True, spec.reason, columns, p, d, verified)


SAME = "same eigenvalue"
CONJUGATE = "conjugate eigenvalues"
DISTINCT = "different minimal polynomials"


class OrthogonalityCheck:
    __slots__ = ("first", "second", "method", "ok")

    def __init__(self, first, second, method, ok):
        self.first = first
        self.second = second
        self.method = method
        self.ok = ok


def polynomial_vector(pair, vector):
    k = pair.minimal.degree
    if pair.symbolic:
        return [list(x.c) if isinstance(x, AlgebraicNumber) else [Fraction(x)] + [ZERO] * (k - 1)
                for x in vector]
    if k == 1:
        return [[Fraction(x)] for x in vector]
    a0, b0 = pair.value.a, pair.value.b
    return [[x.a - x.b * a0 / b0, x.b / b0] if isinstance(x, Surd) else [Fraction(x), ZERO]
            for x in vector]


def _tensor_zero(u, w):
    for i in range(len(u[0])):
        for j in range(len(w[0])):
            total = ZERO
            for ur, wr in zip(u, w):
                if ur[i] and wr[j]:
                    total += ur[i] * wr[j]
            if total:
                return False
    return True


def _conjugates_zero(q, u, w):
    k = q.degree
    if k < 2:
        return True
    field = NumberField(q.c)
    alpha = field.generator
    coefficients = []
    for j in range(k):
        total = ZERO
        for ur, wr in zip(u, w):
            if wr[j]:
                total = total + field.element(ur) * wr[j]
        coefficients.append(total)
    cofactor = [ZERO] * k
    cofactor[k - 1] = ONE
    for i in range(k - 1, 0, -1):
        cofactor[i - 1] = q[i] + alpha * cofactor[i]
    top = coefficients[k - 1]
    return all(not (coefficients[i] - top * cofactor[i]) for i in range(k - 1))


def orthogonality_certificate(columns):
    polys = [polynomial_vector(pair, v) for pair, v in columns]
    checks = []
    for i, (pi, vi) in enumerate(columns):
        for j in range(i, len(columns)):
            pj, vj = columns[j]
            if pi is pj:
                if i != j:
                    checks.append(OrthogonalityCheck(i, j, SAME, not dot(vi, vj)))
                if pi.symbolic:
                    checks.append(OrthogonalityCheck(i, j, CONJUGATE,
                                                     _conjugates_zero(pi.minimal, polys[i], polys[j])))
            elif pi.minimal == pj.minimal:
                checks.append(OrthogonalityCheck(i, j, CONJUGATE,
                                                 _conjugates_zero(pi.minimal, polys[i], polys[j])))
            else:
                checks.append(OrthogonalityCheck(i, j, DISTINCT, _tensor_zero(polys[i], polys[j])))
    return checks


class OrthogonalDiagonalization:
    __slots__ = ("matrix", "spectrum", "blocks", "columns", "checks", "ok", "reason", "q", "d")

    def __init__(self, matrix, spec, blocks, columns, checks, ok, reason, q, d):
        self.matrix = matrix
        self.spectrum = spec
        self.blocks = blocks
        self.columns = columns
        self.checks = checks
        self.ok = ok
        self.reason = reason
        self.q = q
        self.d = d

    @property
    def orthogonal_basis(self):
        return [v for _, v in self.columns]


def orthogonally_diagonalize(matrix):
    if not matrix.is_symmetric():
        return OrthogonalDiagonalization(
            matrix, None, [], [], [], False,
            "the spectral theorem needs a symmetric matrix; this one is not symmetric", None, None)
    spec = spectrum(matrix)
    if not spec.real_only:
        return OrthogonalDiagonalization(
            matrix, spec, [], [], [], False,
            "a complex eigenvalue appeared, which a real symmetric matrix cannot have", None, None)
    blocks = []
    columns = []
    for pair in spec.pairs:
        ortho = gram_schmidt(pair.basis)
        blocks.append((pair, ortho))
        columns.extend((pair, v) for v in ortho.vectors)
    checks = orthogonality_certificate(columns)
    ok = (spec.diagonalizable and all(c.ok for c in checks)
          and all(is_eigenvector(matrix, v, pair.value) for pair, v in columns))
    q = d = None
    if not spec.symbolic:
        q = Matrix.from_columns([v for _, v in columns])
        d = Matrix.diagonal([pair.value for pair, _ in columns])
        ok = ok and matrix.matmul(q) == q.matmul(d)
    reason = (
        "every pair of columns is orthogonal and every column is an eigenvector, all checked exactly;"
        " normalizing the columns gives an orthogonal Q with Q^T A Q = D"
        if ok else "the eigenbasis did not verify as orthogonal"
    )
    return OrthogonalDiagonalization(matrix, spec, blocks, columns, checks, ok, reason, q, d)


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
