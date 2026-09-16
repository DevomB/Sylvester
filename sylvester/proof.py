from __future__ import annotations

from fractions import Fraction

from .determinant import adjugate, cofactor_matrix, determinant, det_triangular
from .eigen import cayley_hamilton, characteristic_polynomial, spectrum
from .exact import ONE, ZERO, Surd, approx
from .inverse import Singular, inverse, inverse_gauss_jordan
from .matrix import Matrix, elementary_add, elementary_scale, elementary_swap
from .reduce import row_reduce
from .solve import solve, solve_homogeneous
from .subspace import (basis_from_spanning_set, four_subspaces, in_span, independence,
                       rank_nullity, same_span)
from . import vectors as V

PROOF = "proof"
INSTANCE = "instance"

MATRIX = "matrix"
SQUARE = "square"
VECTOR = "vector"
SCALAR = "scalar"


class Line:
    __slots__ = ("label", "value", "kind")

    def __init__(self, label, value, kind="text"):
        self.label = label
        self.value = value
        self.kind = kind


def text(label, value=""):
    return Line(label, value, "text")


def mat(label, value):
    return Line(label, value, "matrix")


def vec(label, value):
    return Line(label, value, "vector")


def num(label, value):
    return Line(label, value, "scalar")


class Certificate:
    __slots__ = ("key", "title", "statement", "holds", "lines", "conclusion", "citation", "kind")

    def __init__(self, key, title, statement, holds, lines, conclusion, citation, kind):
        self.key = key
        self.title = title
        self.statement = statement
        self.holds = holds
        self.lines = lines
        self.conclusion = conclusion
        self.citation = citation
        self.kind = kind


class Proposition:
    __slots__ = ("key", "title", "statement", "citation", "operands", "kind", "category", "run")

    def __init__(self, key, title, statement, citation, operands, kind, category, run):
        self.key = key
        self.title = title
        self.statement = statement
        self.citation = citation
        self.operands = operands
        self.kind = kind
        self.category = category
        self.run = run

    def certificate(self, holds, lines, conclusion):
        return Certificate(
            self.key, self.title, self.statement, holds, lines, conclusion, self.citation, self.kind
        )


REGISTRY = {}
ORDER = []


def proposition(key, title, statement, citation, operands, kind, category):
    def wrap(fn):
        prop = Proposition(key, title, statement, citation, operands, kind, category, fn)
        REGISTRY[key] = prop
        ORDER.append(key)
        return fn

    return wrap


def check(key, *operands):
    prop = REGISTRY[key]
    try:
        return prop.run(prop, *operands)
    except (ValueError, ArithmeticError, ZeroDivisionError) as exc:
        return Certificate(
            prop.key, prop.title, prop.statement, None,
            [text("cannot evaluate", str(exc))],
            "this proposition does not apply to the operands given",
            prop.citation, prop.kind,
        )


def by_category():
    groups = {}
    for key in ORDER:
        groups.setdefault(REGISTRY[key].category, []).append(REGISTRY[key])
    return groups


SYSTEMS = "Systems of linear equations"
MATRICES = "Matrices and determinants"
SPACES = "Vectors, subspaces and orthogonality"
EIGEN = "Eigenvalues and eigenvectors"


def _yes_no(flag):
    return "holds" if flag else "fails"


# ---------------------------------------------------------------- systems


@proposition(
    "consistency-rank",
    "Consistency test by rank",
    "Ax = b is consistent if and only if rank(A) = rank([A|b])",
    "Rouche-Capelli theorem",
    (MATRIX, VECTOR),
    PROOF,
    SYSTEMS,
)
def _consistency_rank(prop, a, b):
    sol = solve(a, b)
    holds = sol.consistent == (sol.rank == sol.rank_augmented)
    lines = [
        mat("A", a),
        vec("b", b),
        mat("RREF of [A|b]", sol.reduction.rref),
        num("rank(A)", sol.rank),
        num("rank([A|b])", sol.rank_augmented),
    ]
    if sol.consistent:
        lines.append(vec("a witness solution x", sol.particular))
        lines.append(text("check", "Ax = b verified exactly"))
        conclusion = "rank(A) = rank([A|b]) = %d, so the system is consistent" % sol.rank
    else:
        lines.append(text("contradiction row", "row %d of the RREF reads 0 = 1" % (sol.witness_row + 1)))
        conclusion = "rank(A) = %d but rank([A|b]) = %d, so the system is inconsistent" % (
            sol.rank, sol.rank_augmented,
        )
    return prop.certificate(holds, lines, conclusion)


@proposition(
    "homogeneous-nontrivial",
    "Wide homogeneous systems have nontrivial solutions",
    "If a homogeneous system has more unknowns than equations, it has a nontrivial solution",
    "rank(A) <= min(m, n); nullity = n - rank(A) > 0 when n > m",
    (MATRIX,),
    PROOF,
    SYSTEMS,
)
def _homogeneous_nontrivial(prop, a):
    sol = solve_homogeneous(a)
    wide = a.ncols > a.nrows
    holds = not wide or sol.nullity > 0
    lines = [
        mat("A", a),
        num("equations m", a.nrows),
        num("unknowns n", a.ncols),
        num("rank(A)", sol.rank),
        num("nullity = n - rank", sol.nullity),
    ]
    if sol.homogeneous_basis:
        lines.append(vec("nontrivial solution", sol.homogeneous_basis[0]))
        lines.append(text("check", "A x = 0 verified exactly"))
    if wide:
        conclusion = (
            "n = %d > m = %d forces rank(A) <= %d, so nullity >= %d and a nontrivial solution exists"
            % (a.ncols, a.nrows, a.nrows, a.ncols - a.nrows)
        )
    elif sol.nullity:
        conclusion = "the hypothesis n > m does not apply, but nullity = %d so one exists anyway" % sol.nullity
    else:
        conclusion = "the hypothesis n > m does not apply here, and x = 0 is the only solution"
    return prop.certificate(holds, lines, conclusion)


@proposition(
    "solution-structure",
    "General solution = particular + homogeneous",
    "Every solution of Ax = b has the form x_p + x_h with A x_h = 0",
    "the solution set is a coset of the null space",
    (MATRIX, VECTOR),
    PROOF,
    SYSTEMS,
)
def _solution_structure(prop, a, b):
    sol = solve(a, b)
    if not sol.consistent:
        return prop.certificate(
            True,
            [mat("A", a), vec("b", b), text("status", "inconsistent")],
            "the system has no solutions, so the statement is vacuously true",
        )
    lines = [mat("A", a), vec("b", b), vec("particular x_p", sol.particular)]
    for i, h in enumerate(sol.homogeneous_basis):
        lines.append(vec("null space basis v%d" % (i + 1), h))
    lines.append(text("check A x_p", "= b" if a.matmul(Matrix.column(sol.particular)) == Matrix.column(sol.constants) else "MISMATCH"))
    ok = sol.verify()
    probe = sol.at([Fraction(2)] * len(sol.homogeneous_basis))
    if sol.homogeneous_basis:
        lines.append(vec("sample x_p + 2(v1 + ...)", probe))
        lines.append(text("check", "A of that sample = b" if a.matmul(Matrix.column(probe)) == Matrix.column(sol.constants) else "MISMATCH"))
    conclusion = (
        "the solution set is x_p + span{%s}, of dimension %d"
        % (", ".join("v%d" % (i + 1) for i in range(len(sol.homogeneous_basis))) or "", sol.nullity)
        if sol.homogeneous_basis
        else "the null space is trivial, so x_p is the only solution"
    )
    return prop.certificate(ok, lines, conclusion)


@proposition(
    "superposition",
    "Solutions of Ax = 0 are closed under linear combination",
    "If Ax = 0 and Ay = 0 then A(sx + ty) = 0 for all scalars s, t",
    "linearity of the matrix product",
    (MATRIX,),
    PROOF,
    SYSTEMS,
)
def _superposition(prop, a):
    basis = solve_homogeneous(a).homogeneous_basis
    lines = [mat("A", a)]
    if len(basis) < 1:
        return prop.certificate(
            True, lines + [text("null space", "trivial")],
            "the null space is {0}, and A(s0 + t0) = 0 holds trivially",
        )
    x = basis[0]
    y = basis[1] if len(basis) > 1 else basis[0]
    s, t = Fraction(3), Fraction(-2)
    combo = tuple(s * p + t * q for p, q in zip(x, y))
    holds = a.matmul(Matrix.column(combo)).is_zero()
    lines += [
        vec("x", x), vec("y", y), num("s", s), num("t", t),
        vec("sx + ty", combo),
        vec("A(sx + ty)", a.matmul(Matrix.column(combo)).column_at(0)),
    ]
    return prop.certificate(
        holds, lines, "A(sx + ty) = sAx + tAy = s0 + t0 = 0, so the null space is closed"
    )


@proposition(
    "unique-iff-trivial-kernel",
    "Uniqueness is controlled by the null space",
    "Ax = b has at most one solution if and only if Ax = 0 has only the trivial solution",
    "if Ax = Ay = b then A(x - y) = 0",
    (MATRIX, VECTOR),
    PROOF,
    SYSTEMS,
)
def _unique_iff_trivial(prop, a, b):
    sol = solve(a, b)
    hom = solve_homogeneous(a)
    trivial = hom.nullity == 0
    at_most_one = sol.kind != "infinite"
    holds = trivial == at_most_one
    lines = [
        mat("A", a), vec("b", b),
        num("nullity of A", hom.nullity),
        text("null space", "trivial" if trivial else "nontrivial"),
        text("solution set", sol.kind),
    ]
    if not trivial and sol.consistent:
        other = tuple(p + q for p, q in zip(sol.particular, hom.homogeneous_basis[0]))
        lines.append(vec("solution 1", sol.particular))
        lines.append(vec("solution 2", other))
        lines.append(text("check", "both satisfy Ax = b, so uniqueness fails"))
    return prop.certificate(
        holds, lines,
        "nullity = 0 exactly when the solution (if any) is unique; here nullity = %d and the system is %s"
        % (hom.nullity, sol.kind),
    )


# ------------------------------------------------------- matrices and det


@proposition(
    "transpose-product",
    "Transpose of a product",
    "(AB)^T = B^T A^T",
    "definition of the transpose applied entrywise",
    (MATRIX, MATRIX),
    INSTANCE,
    MATRICES,
)
def _transpose_product(prop, a, b):
    left = a.matmul(b).T
    right = b.T.matmul(a.T)
    return prop.certificate(
        left == right,
        [mat("A", a), mat("B", b), mat("(AB)^T", left), mat("B^T A^T", right)],
        "(AB)^T and B^T A^T agree entry for entry",
    )


@proposition(
    "transpose-involution",
    "Transposing twice",
    "(A^T)^T = A and (A + B)^T = A^T + B^T",
    "the transpose is an involution and is additive",
    (MATRIX, MATRIX),
    INSTANCE,
    MATRICES,
)
def _transpose_involution(prop, a, b):
    holds = a.T.T == a and (a + b).T == a.T + b.T
    return prop.certificate(
        holds,
        [mat("A", a), mat("(A^T)^T", a.T.T), mat("(A + B)^T", (a + b).T), mat("A^T + B^T", a.T + b.T)],
        "double transpose returns A, and the transpose distributes over addition",
    )


@proposition(
    "inverse-product",
    "Inverse of a product",
    "(AB)^-1 = B^-1 A^-1",
    "socks-and-shoes: the inverse reverses the order",
    (SQUARE, SQUARE),
    INSTANCE,
    MATRICES,
)
def _inverse_product(prop, a, b):
    da, db = determinant(a), determinant(b)
    lines = [mat("A", a), mat("B", b), num("det(A)", da), num("det(B)", db)]
    if not da or not db:
        return prop.certificate(
            True, lines,
            "one factor is singular, so neither side is defined and the statement does not apply",
        )
    left = inverse(a.matmul(b))
    right = inverse(b).matmul(inverse(a))
    lines += [mat("(AB)^-1", left), mat("B^-1 A^-1", right)]
    return prop.certificate(left == right, lines, "(AB)(B^-1 A^-1) = A(BB^-1)A^-1 = I")


@proposition(
    "inverse-transpose",
    "Inverse of a transpose",
    "(A^T)^-1 = (A^-1)^T",
    "transpose both sides of A A^-1 = I",
    (SQUARE,),
    INSTANCE,
    MATRICES,
)
def _inverse_transpose(prop, a):
    d = determinant(a)
    lines = [mat("A", a), num("det(A)", d)]
    if not d:
        return prop.certificate(True, lines, "A is singular, so neither side is defined")
    left = inverse(a.T)
    right = inverse(a).T
    lines += [mat("(A^T)^-1", left), mat("(A^-1)^T", right)]
    return prop.certificate(left == right, lines, "transposing A A^-1 = I gives (A^-1)^T A^T = I")


@proposition(
    "det-product",
    "Determinant of a product",
    "det(AB) = det(A) det(B)",
    "multiplicativity of the determinant",
    (SQUARE, SQUARE),
    INSTANCE,
    MATRICES,
)
def _det_product(prop, a, b):
    da, db, dab = determinant(a), determinant(b), determinant(a.matmul(b))
    return prop.certificate(
        dab == da * db,
        [mat("A", a), mat("B", b), mat("AB", a.matmul(b)),
         num("det(A)", da), num("det(B)", db),
         num("det(A)det(B)", da * db), num("det(AB)", dab)],
        "det(AB) = det(A)det(B)",
    )


@proposition(
    "det-transpose",
    "Determinant of a transpose",
    "det(A^T) = det(A)",
    "cofactor expansion along a row of A is expansion along that column of A^T",
    (SQUARE,),
    INSTANCE,
    MATRICES,
)
def _det_transpose(prop, a):
    da, dt = determinant(a), determinant(a.T)
    return prop.certificate(
        da == dt,
        [mat("A", a), mat("A^T", a.T), num("det(A)", da), num("det(A^T)", dt)],
        "a matrix and its transpose have the same determinant",
    )


@proposition(
    "det-scalar",
    "Determinant scales by c^n",
    "det(cA) = c^n det(A) for an n x n matrix",
    "each of the n rows contributes one factor of c",
    (SQUARE, SCALAR),
    INSTANCE,
    MATRICES,
)
def _det_scalar(prop, a, c):
    n = a.nrows
    da = determinant(a)
    dca = determinant(a * c)
    expected = c ** n * da
    return prop.certificate(
        dca == expected,
        [mat("A", a), num("c", c), num("n", n), mat("cA", a * c),
         num("det(A)", da), num("c^n det(A)", expected), num("det(cA)", dca)],
        "pulling c out of each of the %d rows gives c^%d" % (n, n),
    )


@proposition(
    "det-inverse",
    "Determinant of an inverse",
    "det(A^-1) = 1 / det(A)",
    "apply det to A A^-1 = I",
    (SQUARE,),
    INSTANCE,
    MATRICES,
)
def _det_inverse(prop, a):
    d = determinant(a)
    lines = [mat("A", a), num("det(A)", d)]
    if not d:
        return prop.certificate(True, lines, "A is singular, so A^-1 does not exist")
    inv = inverse(a)
    di = determinant(inv)
    lines += [mat("A^-1", inv), num("det(A^-1)", di), num("1/det(A)", ONE / d)]
    return prop.certificate(di == ONE / d, lines, "det(A)det(A^-1) = det(I) = 1")


@proposition(
    "adjugate-identity",
    "The adjugate identity",
    "A adj(A) = adj(A) A = det(A) I",
    "cofactor expansion, with alien cofactor expansions giving zero",
    (SQUARE,),
    INSTANCE,
    MATRICES,
)
def _adjugate_identity(prop, a):
    adj = adjugate(a)
    d = determinant(a)
    target = Matrix.identity(a.nrows) * d
    left = a.matmul(adj)
    right = adj.matmul(a)
    return prop.certificate(
        left == target and right == target,
        [mat("A", a), mat("cofactor matrix C", cofactor_matrix(a)), mat("adj(A) = C^T", adj),
         num("det(A)", d), mat("A adj(A)", left), mat("adj(A) A", right), mat("det(A) I", target)],
        "the diagonal entries are cofactor expansions of det(A); the off-diagonal ones expand a matrix with a repeated row, so they vanish",
    )


@proposition(
    "row-op-determinant",
    "How row operations change the determinant",
    "A swap negates det, scaling a row by k multiplies det by k, adding a multiple of a row leaves det unchanged",
    "multilinearity and alternation of the determinant in the rows",
    (SQUARE,),
    PROOF,
    MATRICES,
)
def _row_op_determinant(prop, a):
    n = a.nrows
    d = determinant(a)
    lines = [mat("A", a), num("det(A)", d)]
    holds = True
    if n >= 2:
        swapped = elementary_swap(n, 0, 1).matmul(a)
        ds = determinant(swapped)
        holds = holds and ds == -d
        lines += [mat("after R1 <-> R2", swapped), num("det", ds), num("expected -det(A)", -d)]
        k = Fraction(3)
        scaled = elementary_scale(n, 0, k).matmul(a)
        dsc = determinant(scaled)
        holds = holds and dsc == k * d
        lines += [text("---"), mat("after R1 -> 3R1", scaled), num("det", dsc), num("expected 3 det(A)", k * d)]
        added = elementary_add(n, 0, 1, Fraction(5)).matmul(a)
        dad = determinant(added)
        holds = holds and dad == d
        lines += [text("---"), mat("after R1 -> R1 + 5R2", added), num("det", dad), num("expected det(A)", d)]
    return prop.certificate(holds, lines, "swap negates, scale multiplies, add leaves it alone")


@proposition(
    "det-triangular",
    "Determinant of a triangular matrix",
    "If A is triangular then det(A) is the product of its diagonal entries",
    "expand along the column (or row) that has a single nonzero entry, repeatedly",
    (SQUARE,),
    INSTANCE,
    MATRICES,
)
def _det_triangular(prop, a):
    triangular = a.is_triangular()
    d = determinant(a)
    prod = det_triangular(a)
    lines = [mat("A", a), text("triangular", "yes" if triangular else "no"),
             num("product of the diagonal", prod), num("det(A)", d)]
    if not triangular:
        return prop.certificate(
            True, lines, "A is not triangular, so the statement does not apply here"
        )
    return prop.certificate(d == prod, lines, "the determinant is the product of the diagonal")


@proposition(
    "invertible-matrix-theorem",
    "The Invertible Matrix Theorem",
    "For square A these are equivalent: det(A) != 0, rank(A) = n, RREF(A) = I, Ax = 0 only trivially, the columns are independent, the columns span R^n, A is a product of elementary matrices",
    "the standard chain of equivalences",
    (SQUARE,),
    PROOF,
    MATRICES,
)
def _invertible_matrix_theorem(prop, a):
    from .inverse import elementary_factorization

    n = a.nrows
    red = row_reduce(a)
    d = determinant(a)
    hom = solve_homogeneous(a)
    ind = independence(a.columns())
    gj = inverse_gauss_jordan(a)
    ef = elementary_factorization(a)
    conditions = [
        ("det(A) != 0", bool(d), "det(A) = %s" % _s(d)),
        ("rank(A) = n", red.rank == n, "rank = %d, n = %d" % (red.rank, n)),
        ("RREF(A) = I", red.rref.is_identity(), "the reduced form %s the identity" % ("is" if red.rref.is_identity() else "is not")),
        ("Ax = 0 only trivially", hom.nullity == 0, "nullity = %d" % hom.nullity),
        ("columns independent", ind.independent, "column rank = %d" % ind.rank),
        ("columns span R^n", ind.rank == n, "the columns span a %d-dimensional space" % ind.rank),
        ("A^-1 exists", gj.ok, "Gauss-Jordan %s" % ("succeeded" if gj.ok else "stalled")),
        ("A is a product of elementary matrices", ef.ok, "%d elementary factors" % len(ef.factors) if ef.ok else "no factorization"),
    ]
    values = [flag for _, flag, _ in conditions]
    holds = all(values) or not any(values)
    lines = [mat("A", a), mat("RREF(A)", red.rref)]
    for label, flag, detail in conditions:
        lines.append(text("  %s %s" % ("[x]" if flag else "[ ]", label), detail))
    return prop.certificate(
        holds, lines,
        "all eight conditions hold together" if all(values)
        else "all eight conditions fail together" if not any(values)
        else "the conditions disagree, which would contradict the theorem",
    )


@proposition(
    "trace-product",
    "Trace of a product commutes",
    "tr(AB) = tr(BA)",
    "both sides sum the same n^2 products a_ij b_ji",
    (SQUARE, SQUARE),
    INSTANCE,
    MATRICES,
)
def _trace_product(prop, a, b):
    ab, ba = a.matmul(b).trace(), b.matmul(a).trace()
    return prop.certificate(
        ab == ba,
        [mat("A", a), mat("B", b), num("tr(AB)", ab), num("tr(BA)", ba)],
        "tr(AB) = tr(BA) even though AB and BA usually differ",
    )


@proposition(
    "symmetric-decomposition",
    "Splitting a square matrix",
    "Every square A is S + K with S symmetric and K skew-symmetric",
    "S = (A + A^T)/2 and K = (A - A^T)/2",
    (SQUARE,),
    PROOF,
    MATRICES,
)
def _symmetric_decomposition(prop, a):
    half = Fraction(1, 2)
    s = (a + a.T) * half
    k = (a - a.T) * half
    holds = s.is_symmetric() and k.is_skew_symmetric() and s + k == a
    return prop.certificate(
        holds,
        [mat("A", a), mat("S = (A + A^T)/2", s), mat("K = (A - A^T)/2", k), mat("S + K", s + k)],
        "S is symmetric, K is skew-symmetric, and they add back to A",
    )


# ------------------------------------------------ vectors and subspaces


@proposition(
    "cauchy-schwarz",
    "Cauchy-Schwarz inequality",
    "<u, v>^2 <= ||u||^2 ||v||^2",
    "expand ||u - t v||^2 >= 0 and minimise over t",
    (VECTOR, VECTOR),
    PROOF,
    SPACES,
)
def _cauchy_schwarz(prop, u, v):
    laws = V.InnerProductLaws(u, v)
    equality = laws.cauchy_gap == 0
    return prop.certificate(
        laws.cauchy_holds,
        [vec("u", u), vec("v", v), num("<u, v>", laws.dot),
         num("<u, v>^2", laws.dot * laws.dot),
         num("||u||^2", laws.norm_u2), num("||v||^2", laws.norm_v2),
         num("||u||^2 ||v||^2", laws.norm_u2 * laws.norm_v2),
         num("slack", laws.cauchy_gap)],
        "equality holds, so u and v are parallel" if equality
        else "the slack is positive, so u and v are not parallel",
    )


@proposition(
    "triangle-inequality",
    "Triangle inequality",
    "||u + v|| <= ||u|| + ||v||",
    "square both sides and apply Cauchy-Schwarz",
    (VECTOR, VECTOR),
    PROOF,
    SPACES,
)
def _triangle(prop, u, v):
    laws = V.InnerProductLaws(u, v)
    s = V.add(u, v)
    return prop.certificate(
        laws.triangle_holds,
        [vec("u", u), vec("v", v), vec("u + v", s),
         num("||u + v||^2", V.norm_squared(s)),
         num("||u||^2", laws.norm_u2), num("||v||^2", laws.norm_v2),
         num("2<u, v>", 2 * laws.dot),
         text("||u + v||", "%.6f" % approx(V.norm(s))),
         text("||u|| + ||v||", "%.6f" % (approx(V.norm(u)) + approx(V.norm(v))))],
        "the inequality reduces to <u, v> <= ||u|| ||v||, which is Cauchy-Schwarz",
    )


@proposition(
    "parallelogram-law",
    "Parallelogram law",
    "||u + v||^2 + ||u - v||^2 = 2||u||^2 + 2||v||^2",
    "expand both norms through the inner product",
    (VECTOR, VECTOR),
    INSTANCE,
    SPACES,
)
def _parallelogram(prop, u, v):
    laws = V.InnerProductLaws(u, v)
    lhs = V.norm_squared(V.add(u, v)) + V.norm_squared(V.subtract(u, v))
    rhs = 2 * laws.norm_u2 + 2 * laws.norm_v2
    return prop.certificate(
        lhs == rhs,
        [vec("u", u), vec("v", v),
         num("||u + v||^2", V.norm_squared(V.add(u, v))),
         num("||u - v||^2", V.norm_squared(V.subtract(u, v))),
         num("left side", lhs), num("right side", rhs)],
        "the cross terms cancel, leaving twice each squared norm",
    )


@proposition(
    "pythagoras",
    "Pythagorean theorem",
    "u and v are orthogonal if and only if ||u + v||^2 = ||u||^2 + ||v||^2",
    "||u + v||^2 = ||u||^2 + 2<u, v> + ||v||^2",
    (VECTOR, VECTOR),
    PROOF,
    SPACES,
)
def _pythagoras(prop, u, v):
    laws = V.InnerProductLaws(u, v)
    holds = laws.orthogonal == laws.pythagoras_holds
    return prop.certificate(
        holds,
        [vec("u", u), vec("v", v), num("<u, v>", laws.dot),
         num("||u + v||^2", V.norm_squared(V.add(u, v))),
         num("||u||^2 + ||v||^2", laws.norm_u2 + laws.norm_v2)],
        "the two sides differ by exactly 2<u, v>, so they agree precisely when u and v are orthogonal",
    )


@proposition(
    "projection-orthogonality",
    "The projection residual is orthogonal",
    "v - proj_u(v) is orthogonal to u",
    "choose the coefficient so the residual has zero inner product with u",
    (VECTOR, VECTOR),
    PROOF,
    SPACES,
)
def _projection_orthogonality(prop, v, u):
    if V.is_zero(u):
        return prop.certificate(
            True, [vec("u", u)], "u is the zero vector, so there is nothing to project onto"
        )
    pr = V.project_onto_vector(v, u)
    return prop.certificate(
        pr.verify(),
        [vec("v", v), vec("u", u), num("<v, u>/<u, u>", pr.coefficient),
         vec("proj_u(v)", pr.parallel), vec("residual v - proj_u(v)", pr.perpendicular),
         num("<residual, u>", V.dot(pr.perpendicular, u))],
        "the residual meets u at a right angle, and v splits into parallel plus perpendicular parts",
    )


@proposition(
    "orthogonal-implies-independent",
    "Orthogonal nonzero vectors are independent",
    "A set of nonzero pairwise orthogonal vectors is linearly independent",
    "take the inner product of a dependency relation with each vector in turn",
    (VECTOR, VECTOR),
    PROOF,
    SPACES,
)
def _orthogonal_independent(prop, u, v):
    orthogonal = V.are_orthogonal(u, v) and not V.is_zero(u) and not V.is_zero(v)
    ind = independence([u, v])
    holds = not orthogonal or ind.independent
    return prop.certificate(
        holds,
        [vec("u", u), vec("v", v), num("<u, v>", V.dot(u, v)),
         text("pairwise orthogonal and nonzero", "yes" if orthogonal else "no"),
         text("independent", "yes" if ind.independent else "no")],
        "orthogonality forces every coefficient in a dependency relation to vanish"
        if orthogonal else "the hypothesis does not apply to this pair",
    )


@proposition(
    "linear-independence",
    "Linear independence of a set",
    "The given vectors are linearly independent, or a nontrivial dependency exists",
    "independence is exactly the absence of a free column in the RREF",
    (MATRIX,),
    PROOF,
    SPACES,
)
def _linear_independence(prop, a):
    cols = a.columns()
    ind = independence(cols)
    lines = [mat("vectors as the columns of A", a), mat("RREF(A)", ind.reduction.rref),
             num("rank", ind.rank), num("number of vectors", len(cols))]
    if ind.independent:
        return prop.certificate(
            True, lines,
            "every column is a pivot column, so the only dependency is the trivial one",
        )
    parts = " + ".join(
        "(%s)v%d" % (_s(k), i + 1) for i, k in enumerate(ind.relation) if k
    )
    lines.append(vec("dependency coefficients", ind.relation))
    lines.append(text("witness", "%s = 0" % parts))
    return prop.certificate(
        ind.verify(), lines,
        "column %d is free, giving an explicit nontrivial dependency"
        % (ind.free_columns[0] + 1),
    )


@proposition(
    "span-membership",
    "Membership in a span",
    "b lies in the span of the columns of A, or it does not",
    "b is in the column space exactly when Ax = b is consistent",
    (MATRIX, VECTOR),
    PROOF,
    SPACES,
)
def _span_membership(prop, a, b):
    m = in_span(b, a.columns())
    lines = [mat("A", a), vec("b", b), mat("RREF of [A|b]", m.solution.reduction.rref)]
    if m.inside:
        lines.append(vec("coefficients", m.coefficients))
        parts = " + ".join(
            "(%s)a%d" % (_s(k), i + 1) for i, k in enumerate(m.coefficients) if k
        ) or "0"
        lines.append(text("witness", "b = %s" % parts))
        return prop.certificate(
            m.verify(), lines,
            "b is in the span, and the combination above is %s"
            % ("unique" if m.unique else "one of infinitely many"),
        )
    lines.append(text("contradiction", "the RREF shows a row 0 = 1"))
    return prop.certificate(True, lines, "Ax = b is inconsistent, so b is not in the span")


@proposition(
    "subspace-test",
    "The null space is a subspace",
    "null(A) contains 0 and is closed under addition and scalar multiplication",
    "the three-part subspace test",
    (MATRIX,),
    PROOF,
    SPACES,
)
def _subspace_test(prop, a):
    from .subspace import nullspace_is_subspace

    axioms = nullspace_is_subspace(a)
    basis = solve_homogeneous(a).homogeneous_basis
    lines = [
        mat("A", a),
        text("contains 0", "yes" if axioms.zero_ok else "no"),
        text("closed under addition", "yes" if axioms.closed_add else "no"),
        text("closed under scaling", "yes" if axioms.closed_scale else "no"),
        num("dimension (nullity)", len(basis)),
    ]
    for i, v in enumerate(basis):
        lines.append(vec("basis v%d" % (i + 1), v))
    return prop.certificate(
        axioms.is_subspace, lines,
        "all three conditions hold, so null(A) is a subspace of R^%d" % a.ncols,
    )


@proposition(
    "rank-nullity",
    "Rank-nullity theorem",
    "rank(A) + nullity(A) = n, the number of columns",
    "pivot columns and free columns partition the columns",
    (MATRIX,),
    PROOF,
    SPACES,
)
def _rank_nullity(prop, a):
    r, nul, n = rank_nullity(a)
    f = four_subspaces(a)
    lines = [
        mat("A", a), mat("RREF(A)", f.reduction.rref),
        num("pivot columns (rank)", r), num("free columns (nullity)", nul),
        num("total columns n", n),
        text("check", "%d + %d = %d" % (r, nul, n)),
        num("dim col(A)", len(f.column_space)), num("dim row(A)", len(f.row_space)),
        num("dim null(A)", len(f.null_space)), num("dim left-null(A)", len(f.left_null_space)),
    ]
    return prop.certificate(
        r + nul == n and f.verify(), lines,
        "every column is either a pivot column or a free column, and the two counts add to n",
    )


@proposition(
    "rank-transpose",
    "Row rank equals column rank",
    "rank(A) = rank(A^T)",
    "both count the pivots of A",
    (MATRIX,),
    INSTANCE,
    SPACES,
)
def _rank_transpose(prop, a):
    ra = row_reduce(a).rank
    rt = row_reduce(a.T).rank
    return prop.certificate(
        ra == rt,
        [mat("A", a), mat("A^T", a.T), num("rank(A)", ra), num("rank(A^T)", rt)],
        "the row space and the column space always have the same dimension",
    )


@proposition(
    "gram-schmidt-span",
    "Gram-Schmidt preserves the span",
    "The orthogonal set produced by Gram-Schmidt spans the same subspace as the input",
    "each new vector differs from the old one by a combination of earlier ones",
    (MATRIX,),
    PROOF,
    SPACES,
)
def _gram_schmidt_span(prop, a):
    cols = a.columns()
    gs = V.gram_schmidt(cols)
    lines = [mat("input vectors as columns", a)]
    for i, v in enumerate(gs.vectors):
        lines.append(vec("w%d" % (i + 1), v))
    same = same_span(gs.vectors, cols) if gs.vectors else all(V.is_zero(c) for c in cols)
    lines += [
        text("pairwise orthogonal", "yes" if gs.verify() else "no"),
        text("same span", "yes" if same else "no"),
        num("dimension", len(gs.vectors)),
    ]
    return prop.certificate(
        gs.verify() and same, lines,
        "the outputs are pairwise orthogonal and span exactly what the inputs spanned",
    )


@proposition(
    "orthogonal-complement-dimension",
    "Dimension of an orthogonal complement",
    "dim W + dim W-perp = n for W the row space of A",
    "W-perp is the null space of A, so this is rank-nullity again",
    (MATRIX,),
    PROOF,
    SPACES,
)
def _orthogonal_complement_dimension(prop, a):
    f = four_subspaces(a)
    n = a.ncols
    lines = [mat("A", a), num("dim W = rank(A)", f.rank),
             num("dim W-perp = nullity(A)", f.nullity), num("n", n)]
    for i, v in enumerate(f.null_space):
        lines.append(vec("W-perp basis v%d" % (i + 1), v))
    orthogonal = all(V.dot(v, w) == 0 for v in f.null_space for w in f.row_space)
    lines.append(text("every complement vector meets every row space vector at a right angle",
                      "yes" if orthogonal else "no"))
    return prop.certificate(
        f.rank + f.nullity == n and orthogonal, lines,
        "the row space and the null space are orthogonal complements inside R^%d" % n,
    )


@proposition(
    "basis-dimension",
    "A basis extracted from a spanning set",
    "Any spanning set contains a basis, and every basis of the span has the same size",
    "pivot columns form a basis for the column space",
    (MATRIX,),
    PROOF,
    SPACES,
)
def _basis_dimension(prop, a):
    cols = a.columns()
    ex = basis_from_spanning_set(cols)
    lines = [mat("spanning vectors as columns", a), mat("RREF", ex.reduction.rref),
             num("dimension of the span", ex.dimension)]
    for c in ex.kept:
        lines.append(vec("basis vector from column %d" % (c + 1), cols[c]))
    for free, terms in ex.expressions.items():
        parts = " + ".join("(%s)v%d" % (_s(k), c + 1) for c, k in terms) or "0"
        lines.append(text("column %d is redundant" % (free + 1), "= %s" % parts))
    ok = ex.dimension == row_reduce(a).rank and (not ex.basis or same_span(ex.basis, cols))
    return prop.certificate(
        ok, lines,
        "the %d pivot columns form a basis; the other columns are combinations of them"
        % ex.dimension,
    )


# ------------------------------------------------------------- eigen


@proposition(
    "eigen-definition",
    "Eigenvector verification",
    "For each eigenvalue lambda and basis vector v of its eigenspace, Av = lambda v",
    "definition of an eigenvector",
    (SQUARE,),
    PROOF,
    EIGEN,
)
def _eigen_definition(prop, a):
    spec = spectrum(a)
    lines = [mat("A", a), text("characteristic polynomial", "p(L) = %s" % spec.poly.shift_variable("L"))]
    holds = True
    for pair in spec.pairs:
        lines.append(text("---"))
        lines.append(text("lambda", _s(pair.value)))
        lines.append(text("algebraic / geometric multiplicity", "%d / %d" % (pair.algebraic, pair.geometric)))
        for v in pair.basis:
            lines.append(vec("eigenvector v", v))
            if pair.exact:
                left = a.matmul(Matrix.column(v)).column_at(0)
                right = tuple(pair.value * x for x in v)
                lines.append(vec("Av", left))
                lines.append(vec("lambda v", right))
                holds = holds and left == right
    return prop.certificate(
        holds, lines,
        "every listed eigenvector satisfies Av = lambda v exactly" if spec.exact
        else "some eigenvalues are irrational of degree > 2 and are reported numerically",
    )


@proposition(
    "eigen-trace-det",
    "Eigenvalues determine trace and determinant",
    "The eigenvalues sum to tr(A) and multiply to det(A), counted with algebraic multiplicity",
    "compare coefficients of the characteristic polynomial",
    (SQUARE,),
    INSTANCE,
    EIGEN,
)
def _eigen_trace_det(prop, a):
    spec = spectrum(a)
    lines = [mat("A", a), text("characteristic polynomial", "p(L) = %s" % spec.poly.shift_variable("L"))]
    for pair in spec.pairs:
        lines.append(text("lambda (multiplicity %d)" % pair.algebraic, _s(pair.value)))
    lines += [
        num("tr(A)", a.trace()),
        num("det(A)", determinant(a)),
        text("sum of eigenvalues matches tr(A)", _flag(spec.trace_check)),
        text("product of eigenvalues matches det(A)", _flag(spec.det_check)),
    ]
    holds = spec.trace_check is not False and spec.det_check is not False
    return prop.certificate(
        holds, lines,
        "the second coefficient of p is -tr(A) and the constant term is (-1)^n det(A)",
    )


@proposition(
    "distinct-eigenvalues-independent",
    "Eigenvectors for distinct eigenvalues are independent",
    "Eigenvectors belonging to distinct eigenvalues are linearly independent",
    "apply (A - lambda I) to a dependency relation to kill one term at a time",
    (SQUARE,),
    PROOF,
    EIGEN,
)
def _distinct_eigenvalues_independent(prop, a):
    spec = spectrum(a)
    picked = [(p, p.basis[0]) for p in spec.pairs if p.exact and p.basis]
    lines = [mat("A", a)]
    for p, v in picked:
        lines.append(vec("eigenvector for lambda = %s" % _s(p.value), v))
    if len(picked) < 2:
        return prop.certificate(
            True, lines, "fewer than two distinct exact eigenvalues, so there is nothing to check"
        )
    try:
        ind = independence([v for _, v in picked])
    except ArithmeticError:
        return prop.certificate(
            True, lines,
            "the eigenvectors live in different quadratic fields, so they cannot share one matrix here",
        )
    lines.append(num("rank of the eigenvector set", ind.rank))
    return prop.certificate(
        ind.independent, lines,
        "the %d eigenvectors for distinct eigenvalues are independent" % len(picked),
    )


@proposition(
    "symmetric-real-eigenvalues",
    "Symmetric matrices have real eigenvalues",
    "A real symmetric matrix has only real eigenvalues and an orthogonal eigenbasis",
    "the spectral theorem",
    (SQUARE,),
    PROOF,
    EIGEN,
)
def _symmetric_real(prop, a):
    from .eigen import orthogonally_diagonalize

    if not a.is_symmetric():
        return prop.certificate(
            True, [mat("A", a), text("symmetric", "no")],
            "A is not symmetric, so the spectral theorem does not apply",
        )
    spec = spectrum(a)
    od = orthogonally_diagonalize(a)
    lines = [mat("A", a), text("symmetric", "yes"),
             text("characteristic polynomial", "p(L) = %s" % spec.poly.shift_variable("L"))]
    for pair in spec.pairs:
        lines.append(text("lambda", "%s  (real: %s)" % (_s(pair.value), "yes" if pair.is_real else "no")))
    for i, v in enumerate(od.orthogonal_basis):
        lines.append(vec("orthogonal eigenvector q%d" % (i + 1), v))
    lines.append(text("pairwise orthogonal", "yes" if od.ok else "no"))
    return prop.certificate(
        spec.real_only and (od.ok or not spec.exact), lines,
        "all eigenvalues are real and the eigenvectors can be chosen orthogonal",
    )


@proposition(
    "cayley-hamilton",
    "Cayley-Hamilton theorem",
    "A satisfies its own characteristic polynomial: p_A(A) = 0",
    "Cayley-Hamilton",
    (SQUARE,),
    INSTANCE,
    EIGEN,
)
def _cayley_hamilton(prop, a):
    poly, terms, acc, ok = cayley_hamilton(a)
    lines = [mat("A", a), text("p(L)", poly.shift_variable("L"))]
    for power, coeff, value in terms:
        lines.append(mat("(%s) A^%d" % (_s(coeff), power), value * coeff))
    lines.append(mat("p(A)", acc))
    return prop.certificate(ok, lines, "the powers of A combine to the zero matrix")


@proposition(
    "similar-same-charpoly",
    "Similar matrices share a characteristic polynomial",
    "If B = P^-1 A P then A and B have the same characteristic polynomial, trace and determinant",
    "det(L I - P^-1 A P) = det(P^-1 (L I - A) P) = det(L I - A)",
    (SQUARE, SQUARE),
    INSTANCE,
    EIGEN,
)
def _similar_same_charpoly(prop, a, p):
    d = determinant(p)
    lines = [mat("A", a), mat("P", p), num("det(P)", d)]
    if not d:
        return prop.certificate(True, lines, "P is singular, so it defines no similarity")
    b = inverse(p).matmul(a).matmul(p)
    pa = characteristic_polynomial(a)
    pb = characteristic_polynomial(b)
    lines += [mat("B = P^-1 A P", b),
              text("p_A(L)", pa.shift_variable("L")), text("p_B(L)", pb.shift_variable("L")),
              num("tr(A)", a.trace()), num("tr(B)", b.trace()),
              num("det(A)", determinant(a)), num("det(B)", determinant(b))]
    return prop.certificate(
        pa == pb, lines, "similarity preserves the characteristic polynomial, so also trace and determinant"
    )


def _s(value):
    from .render import fmt

    if isinstance(value, complex):
        return "%.4f%+.4fi" % (value.real, value.imag)
    return fmt(value)


def _flag(value):
    if value is None:
        return "not comparable in one field"
    return "yes" if value else "no"
