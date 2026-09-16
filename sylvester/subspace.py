from __future__ import annotations

from .exact import ONE, ZERO
from .matrix import Matrix
from .reduce import nullspace_basis, row_reduce
from .vectors import as_vector


class Independence:
    __slots__ = ("vectors", "matrix", "reduction", "independent", "rank", "pivot_columns",
                 "free_columns", "relation", "redundant")

    def __init__(self, vectors, matrix, reduction, relation, redundant):
        self.vectors = vectors
        self.matrix = matrix
        self.reduction = reduction
        self.rank = reduction.rank
        self.pivot_columns = reduction.pivot_columns
        self.free_columns = reduction.free_columns
        self.independent = relation is None
        self.relation = relation
        self.redundant = redundant

    def verify(self):
        if self.relation is None:
            return True
        combo = [ZERO] * len(self.vectors[0])
        for k, v in zip(self.relation, self.vectors):
            if k:
                combo = [a + k * b for a, b in zip(combo, v)]
        return all(not a for a in combo) and any(k for k in self.relation)


def independence(vectors):
    vectors = [as_vector(v) for v in vectors]
    if not vectors:
        return Independence([], Matrix([]), row_reduce(Matrix([])), None, [])
    matrix = Matrix.from_columns(vectors)
    red = row_reduce(matrix)
    if not red.free_columns:
        return Independence(vectors, matrix, red, None, [])
    free = red.free_columns[0]
    relation = [ZERO] * len(vectors)
    relation[free] = ONE
    for r, c in red.pivots:
        relation[c] = -red.rref[r][free]
    redundant = list(red.free_columns)
    return Independence(vectors, matrix, red, relation, redundant)


class SpanMembership:
    __slots__ = ("target", "vectors", "matrix", "solution", "inside", "coefficients", "unique")

    def __init__(self, target, vectors, matrix, solution):
        self.target = target
        self.vectors = vectors
        self.matrix = matrix
        self.solution = solution
        self.inside = solution.consistent
        self.coefficients = solution.particular if solution.consistent else None
        self.unique = solution.consistent and not solution.free_columns

    def verify(self):
        if not self.inside:
            return True
        combo = [ZERO] * len(self.target)
        for k, v in zip(self.coefficients, self.vectors):
            if k:
                combo = [a + k * b for a, b in zip(combo, v)]
        return tuple(combo) == tuple(self.target)


def in_span(target, vectors):
    from .solve import solve

    target = as_vector(target)
    vectors = [as_vector(v) for v in vectors]
    matrix = Matrix.from_columns(vectors) if vectors else Matrix.zeros(len(target), 0)
    if not vectors:
        matrix = Matrix([[] for _ in range(len(target))])
    return SpanMembership(target, vectors, matrix, solve(matrix, target))


def coordinates(target, basis):
    return in_span(target, basis)


class BasisExtraction:
    __slots__ = ("vectors", "matrix", "reduction", "kept", "basis", "dropped", "expressions")

    def __init__(self, vectors, matrix, reduction, kept, basis, dropped, expressions):
        self.vectors = vectors
        self.matrix = matrix
        self.reduction = reduction
        self.kept = kept
        self.basis = basis
        self.dropped = dropped
        self.expressions = expressions

    @property
    def dimension(self):
        return len(self.basis)


def basis_from_spanning_set(vectors):
    vectors = [as_vector(v) for v in vectors]
    if not vectors:
        return BasisExtraction([], Matrix([]), row_reduce(Matrix([])), [], [], [], {})
    matrix = Matrix.from_columns(vectors)
    red = row_reduce(matrix)
    kept = red.pivot_columns
    dropped = red.free_columns
    expressions = {}
    for free in dropped:
        expressions[free] = [(c, red.rref[r][free]) for r, c in red.pivots if red.rref[r][free]]
    return BasisExtraction(vectors, matrix, red, kept, [vectors[c] for c in kept], dropped, expressions)


def dimension_of_span(vectors):
    return basis_from_spanning_set(vectors).dimension


def extend_to_basis(vectors, dimension):
    from .vectors import standard_basis

    current = [as_vector(v) for v in vectors]
    added = []
    for candidate in standard_basis(dimension):
        probe = current + added + [candidate]
        if independence(probe).independent:
            added.append(candidate)
        if len(current) + len(added) == dimension:
            break
    return current + added, added


class FourSubspaces:
    __slots__ = ("matrix", "reduction", "column_space", "row_space", "null_space",
                 "left_null_space", "rank", "nullity", "left_nullity", "pivot_columns")

    def __init__(self, matrix, reduction, column_space, row_space, null_space, left_null_space):
        self.matrix = matrix
        self.reduction = reduction
        self.column_space = column_space
        self.row_space = row_space
        self.null_space = null_space
        self.left_null_space = left_null_space
        self.rank = reduction.rank
        self.pivot_columns = reduction.pivot_columns
        self.nullity = matrix.ncols - self.rank
        self.left_nullity = matrix.nrows - self.rank

    def verify(self):
        for v in self.null_space:
            if not self.matrix.matmul(Matrix.column(v)).is_zero():
                return False
        for v in self.left_null_space:
            if not Matrix.row_vector(v).matmul(self.matrix).is_zero():
                return False
        return (
            len(self.column_space) == len(self.row_space) == self.rank
            and len(self.null_space) == self.nullity
            and len(self.left_null_space) == self.left_nullity
        )


def four_subspaces(matrix):
    red = row_reduce(matrix)
    column_space = [matrix.column_at(c) for c in red.pivot_columns]
    row_space = [red.rref.rows[r] for r, _ in red.pivots]
    null_space = nullspace_basis(matrix)
    left_null_space = nullspace_basis(matrix.T)
    return FourSubspaces(matrix, red, column_space, row_space, null_space, left_null_space)


def rank_nullity(matrix):
    red = row_reduce(matrix)
    return red.rank, matrix.ncols - red.rank, matrix.ncols


class ChangeOfBasis:
    __slots__ = ("source", "target", "matrix", "ok", "coordinates")

    def __init__(self, source, target, matrix, ok, coords):
        self.source = source
        self.target = target
        self.matrix = matrix
        self.ok = ok
        self.coordinates = coords


def change_of_basis(source_basis, target_basis):
    from .inverse import inverse_gauss_jordan

    source = [as_vector(v) for v in source_basis]
    target = [as_vector(v) for v in target_basis]
    t = Matrix.from_columns(target)
    gj = inverse_gauss_jordan(t)
    if not gj.ok:
        return ChangeOfBasis(source, target, None, False, [])
    columns = []
    for v in source:
        columns.append(gj.inverse.matmul(Matrix.column(v)).column_at(0))
    return ChangeOfBasis(source, target, Matrix.from_columns(columns), True, columns)


def is_subspace_of(vectors, container):
    return all(in_span(v, container).inside for v in vectors)


def same_span(a, b):
    return is_subspace_of(a, b) and is_subspace_of(b, a)


class SubspaceAxioms:
    __slots__ = ("description", "zero_ok", "closed_add", "closed_scale", "counterexample")

    def __init__(self, description, zero_ok, closed_add, closed_scale, counterexample):
        self.description = description
        self.zero_ok = zero_ok
        self.closed_add = closed_add
        self.closed_scale = closed_scale
        self.counterexample = counterexample

    @property
    def is_subspace(self):
        return self.zero_ok and self.closed_add and self.closed_scale


def nullspace_is_subspace(matrix):
    basis = nullspace_basis(matrix)
    zero_ok = matrix.matmul(Matrix.zeros(matrix.ncols, 1)).is_zero()
    closed_add = True
    closed_scale = True
    for i, u in enumerate(basis):
        if not matrix.matmul(Matrix.column([3 * a for a in u])).is_zero():
            closed_scale = False
        for v in basis[i:]:
            if not matrix.matmul(Matrix.column([a + b for a, b in zip(u, v)])).is_zero():
                closed_add = False
    return SubspaceAxioms("null space of A", zero_ok, closed_add, closed_scale, None)
