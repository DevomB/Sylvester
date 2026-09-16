from __future__ import annotations

import math

from .exact import ONE, ZERO, Surd, approx, content, sqrt_exact
from .matrix import Matrix


def as_vector(value):
    if isinstance(value, Matrix):
        if value.ncols == 1:
            return value.column_at(0)
        if value.nrows == 1:
            return value.rows[0]
        raise ValueError("expected a vector, got a %dx%d matrix" % value.shape)
    return tuple(value)


def _check(u, v):
    if len(u) != len(v):
        raise ValueError("vectors have different lengths: %d and %d" % (len(u), len(v)))


def add(u, v):
    _check(u, v)
    return tuple(a + b for a, b in zip(u, v))


def subtract(u, v):
    _check(u, v)
    return tuple(a - b for a, b in zip(u, v))


def scale(k, v):
    return tuple(k * a for a in v)


def negate(v):
    return tuple(-a for a in v)


def is_zero(v):
    return all(not a for a in v)


def dot(u, v):
    _check(u, v)
    total = ZERO
    for a, b in zip(u, v):
        if a and b:
            total += a * b
    return total


def norm_squared(v):
    return dot(v, v)


def norm(v):
    return sqrt_exact(norm_squared(v))


def distance(u, v):
    return norm(subtract(u, v))


def normalize(v):
    n = norm(v)
    if not n:
        raise ValueError("the zero vector has no direction to normalize")
    return tuple(a / n for a in v)


def are_orthogonal(u, v):
    return not dot(u, v)


def is_orthogonal_set(vectors):
    return all(
        not dot(vectors[i], vectors[j])
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    )


def cos_angle_squared(u, v):
    d = dot(u, v)
    denom = norm_squared(u) * norm_squared(v)
    if not denom:
        raise ValueError("the zero vector has no angle")
    return d * d / denom, d


def angle_degrees(u, v):
    d = approx(dot(u, v))
    nu = math.sqrt(approx(norm_squared(u)))
    nv = math.sqrt(approx(norm_squared(v)))
    if not nu or not nv:
        raise ValueError("the zero vector has no angle")
    ratio = max(-1.0, min(1.0, d / (nu * nv)))
    return math.degrees(math.acos(ratio))


def cross(u, v):
    if len(u) != 3 or len(v) != 3:
        raise ValueError("the cross product is only defined in R^3")
    return (
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    )


class Projection:
    __slots__ = ("vector", "onto", "coefficient", "parallel", "perpendicular", "is_subspace")

    def __init__(self, vector, onto, coefficient, parallel, perpendicular, is_subspace=False):
        self.vector = vector
        self.onto = onto
        self.coefficient = coefficient
        self.parallel = parallel
        self.perpendicular = perpendicular
        self.is_subspace = is_subspace

    def verify(self):
        recombined = add(self.parallel, self.perpendicular)
        if recombined != tuple(self.vector):
            return False
        if self.is_subspace:
            return all(not dot(self.perpendicular, b) for b in self.onto)
        return not dot(self.perpendicular, self.onto)


def project_onto_vector(v, onto):
    denom = norm_squared(onto)
    if not denom:
        raise ValueError("cannot project onto the zero vector")
    k = dot(v, onto) / denom
    parallel = scale(k, onto)
    return Projection(tuple(v), tuple(onto), k, parallel, subtract(v, parallel))


def project_onto_subspace(v, basis):
    if not basis:
        zero = tuple(ZERO for _ in v)
        return Projection(tuple(v), [], [], zero, tuple(v), True)
    ortho = gram_schmidt(basis).vectors
    parallel = tuple(ZERO for _ in v)
    coefficients = []
    for b in ortho:
        k = dot(v, b) / norm_squared(b)
        coefficients.append(k)
        parallel = add(parallel, scale(k, b))
    return Projection(tuple(v), list(basis), coefficients, parallel, subtract(v, parallel), True)


class GramSchmidt:
    __slots__ = ("input", "vectors", "steps", "dropped", "normalized")

    def __init__(self, source, vectors, steps, dropped):
        self.input = source
        self.vectors = vectors
        self.steps = steps
        self.dropped = dropped
        self.normalized = None

    def verify(self):
        return is_orthogonal_set(self.vectors)


class GramSchmidtStep:
    __slots__ = ("index", "source", "terms", "raw", "result", "cleared")

    def __init__(self, index, source, terms, raw, result, cleared):
        self.index = index
        self.source = source
        self.terms = terms
        self.raw = raw
        self.result = result
        self.cleared = cleared


def gram_schmidt(vectors, clear_fractions=True):
    out = []
    steps = []
    dropped = []
    for index, v in enumerate(vectors):
        terms = []
        current = tuple(v)
        for b in out:
            k = dot(v, b) / norm_squared(b)
            terms.append((k, b))
            if k:
                current = subtract(current, scale(k, b))
        raw = current
        cleared = None
        if is_zero(current):
            dropped.append(index)
            steps.append(GramSchmidtStep(index, tuple(v), terms, raw, None, None))
            continue
        if clear_fractions:
            current = _clear(current)
            if current != raw:
                cleared = current
        out.append(current)
        steps.append(GramSchmidtStep(index, tuple(v), terms, raw, current, cleared))
    return GramSchmidt(list(vectors), out, steps, dropped)


def _clear(v):
    if any(isinstance(a, Surd) for a in v):
        return v
    den = 1
    for a in v:
        den = den * a.denominator // math.gcd(den, a.denominator)
    scaled = tuple(a * den for a in v)
    g = content(scaled)
    if g != 1:
        scaled = tuple(a / g for a in scaled)
    for a in scaled:
        if a:
            return negate(scaled) if a < 0 else scaled
    return scaled


def orthonormalize(vectors):
    result = gram_schmidt(vectors)
    result.normalized = [normalize(v) for v in result.vectors]
    return result


def orthogonal_complement(vectors, dimension=None):
    from .reduce import nullspace_basis

    if not vectors:
        if dimension is None:
            raise ValueError("need the ambient dimension for an empty spanning set")
        return [tuple(ONE if i == j else ZERO for j in range(dimension)) for i in range(dimension)]
    return nullspace_basis(Matrix(list(vectors)))


class InnerProductLaws:
    __slots__ = ("u", "v", "dot", "norm_u2", "norm_v2", "cauchy_gap", "cauchy_holds",
                 "triangle_holds", "parallelogram_holds", "pythagoras_applies",
                 "pythagoras_holds", "orthogonal")

    def __init__(self, u, v):
        self.u = tuple(u)
        self.v = tuple(v)
        self.dot = dot(u, v)
        self.norm_u2 = norm_squared(u)
        self.norm_v2 = norm_squared(v)
        self.cauchy_gap = self.norm_u2 * self.norm_v2 - self.dot * self.dot
        self.cauchy_holds = self.cauchy_gap >= 0
        self.orthogonal = not self.dot
        self.triangle_holds = self.dot <= 0 or self.dot * self.dot <= self.norm_u2 * self.norm_v2
        self.parallelogram_holds = (
            norm_squared(add(u, v)) + norm_squared(subtract(u, v))
            == 2 * self.norm_u2 + 2 * self.norm_v2
        )
        self.pythagoras_applies = self.orthogonal
        self.pythagoras_holds = norm_squared(add(u, v)) == self.norm_u2 + self.norm_v2


def unit_axis(n, i):
    return tuple(ONE if j == i else ZERO for j in range(n))


def standard_basis(n):
    return [unit_axis(n, i) for i in range(n)]
