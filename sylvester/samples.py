from __future__ import annotations

from fractions import Fraction

from .matrix import Matrix

CLO = {
    1: "Prove elementary statements concerning systems of linear equations",
    2: "Perform elementary row operations on matrices, their inverses and transposes",
    3: "Calculate determinants using elementary row operations and cofactor expansions",
    4: "Prove elementary statements concerning the theory of matrices and determinants",
    5: "Prove algebraic statements about vectors, inner products, projections, norms, "
       "orthogonality, independence, spanning sets, subspaces, bases, dimension and rank",
    6: "Use determinants to solve homogeneous and non-homogeneous systems",
    7: "Use rank of A to check linear independence, kernel, rank, range and nullity",
    8: "Calculate eigenvalues, eigenvectors and eigenspaces",
}


class Sample:
    __slots__ = ("key", "title", "clo", "screen", "note", "registers")

    def __init__(self, key, title, clo, screen, note, registers):
        self.key = key
        self.title = title
        self.clo = clo
        self.screen = screen
        self.note = note
        self.registers = registers


def _m(rows):
    return Matrix(rows)


def _v(values):
    return Matrix.column([Fraction(x) for x in values])


SAMPLES = [
    Sample(
        "sys-inconsistent", "3x3 system with no solution", 1, "system",
        "The RREF ends with a row reading 0 = 1. rank(A) = 2 but rank([A|b]) = 3.",
        {"A": _m([[2, -1, 3], [1, 1, -2], [4, 1, -1]]), "B": _v([-1, 1, 3])},
    ),
    Sample(
        "sys-unique", "3x3 system with one solution", 1, "system",
        "rank(A) = rank([A|b]) = 3 = n, so exactly one solution.",
        {"A": _m([[1, -2, 4], [-1, 1, -2], [1, 5, 1]]), "B": _v([0, -1, 2])},
    ),
    Sample(
        "sys-infinite", "3x4 system with a line of solutions", 1, "system",
        "One free variable, so the solution set is a particular solution plus a line.",
        {"A": _m([[1, 2, 3, 4], [2, 4, 8, 10], [3, 6, 11, 14]]), "B": _v([5, 14, 19])},
    ),
    Sample(
        "sys-overdetermined", "4 equations, 3 unknowns, still consistent", 1, "system",
        "More equations than unknowns does not by itself make a system inconsistent.",
        {"A": _m([[1, 1, -1], [1, 1, 1], [1, 0, -1], [0, 1, -4]]), "B": _v([-2, 0, 1, -7])},
    ),
    Sample(
        "sys-wide-homogeneous", "Homogeneous system with more unknowns than equations", 1, "homogeneous",
        "n > m forces a nontrivial solution: rank is at most 2 against 4 unknowns.",
        {"A": _m([[1, 2, -1, 3], [2, 4, 1, 0]])},
    ),
    Sample(
        "rowops-ugly", "Ugly pivots, where the human route pays off", 2, "reduce",
        "Machine drops a fraction on step one. Human scales instead of dividing and never does.",
        {"A": _m([[6, 9, 15, 3], [4, 7, 11, 5], [8, 13, 21, 9]])},
    ),
    Sample(
        "rowops-manufactured", "Manufacturing a 1 to avoid fractions", 2, "reduce",
        "No entry is 1, but two rows differ by exactly 1 in the pivot column, so subtracting makes one.",
        {"A": _m([[3, 7, 2], [2, 5, 3], [4, 9, 1]])},
    ),
    Sample(
        "inverse-3x3", "Invert a 3x3 by Gauss-Jordan", 2, "inverse",
        "Reduce [A | I] until the left half is I; the right half is then the inverse.",
        {"A": _m([[1, 2, 3], [0, 1, 4], [5, 6, 0]])},
    ),
    Sample(
        "inverse-singular", "A singular matrix has no inverse", 2, "inverse",
        "Gauss-Jordan stalls: the left half can never become I because rank(A) = 2 < 3.",
        {"A": _m([[1, 2, 3], [4, 5, 6], [7, 8, 9]])},
    ),
    Sample(
        "elementary-factors", "Write a matrix as a product of elementary matrices", 2, "elementary",
        "Every invertible matrix is a product of elementary matrices, one per row operation.",
        {"A": _m([[2, 1], [1, 1]])},
    ),
    Sample(
        "det-cofactor", "Determinant by cofactor expansion", 3, "determinant",
        "Expanding along the row or column with the most zeros kills whole terms before you start.",
        {"A": _m([[1, 2, 0], [3, 0, 4], [5, 6, 0]])},
    ),
    Sample(
        "det-4x4", "A 4x4 determinant", 3, "determinant",
        "Row reduction is far cheaper than cofactor expansion once n reaches 4.",
        {"A": _m([[2, 1, -1, 3], [-3, -1, 2, 2], [8, 2, 1, 1], [4, 1, 4, 8]])},
    ),
    Sample(
        "det-triangular", "Determinant of a triangular matrix", 3, "determinant",
        "Triangular determinants are just the product of the diagonal.",
        {"A": _m([[3, 7, -2, 5], [0, -1, 4, 9], [0, 0, 2, 6], [0, 0, 0, 5]])},
    ),
    Sample(
        "det-rowops", "Watch row operations move the determinant", 3, "determinant",
        "Swaps negate it, scalings multiply it, adding a multiple of a row leaves it alone.",
        {"A": _m([[0, 2, 1], [3, -1, 2], [4, 0, 1]])},
    ),
    Sample(
        "adjugate", "The adjugate identity", 4, "proof",
        "A adj(A) = det(A) I, which is where the cofactor formula for the inverse comes from.",
        {"A": _m([[2, -1, 0], [1, 3, 4], [0, 2, 1]])},
    ),
    Sample(
        "imt", "The Invertible Matrix Theorem on one matrix", 4, "proof",
        "Eight equivalent conditions, all checked against each other on the same matrix.",
        {"A": _m([[1, 2, 3], [0, 1, 4], [5, 6, 0]])},
    ),
    Sample(
        "det-multiplicative", "det(AB) = det(A)det(B)", 4, "proof",
        "Two matrices whose product has a determinant you can predict before multiplying.",
        {"A": _m([[2, 1], [3, 4]]), "B": _m([[1, -2], [5, 0]])},
    ),
    Sample(
        "independence", "A dependent set of vectors", 5, "independence",
        "The third vector is 2v1 + v2, which the RREF exposes as a free column.",
        {"A": _m([[1, 3, 5], [2, 0, 4], [2, 4, 8]])},
    ),
    Sample(
        "orthogonality", "Inner products, angle and projection", 5, "vectors",
        "A 3-4-5 style pair: exact norms, an exact cosine, and a clean projection.",
        {"U": _v([1, 2, 2]), "V": _v([3, 0, 4])},
    ),
    Sample(
        "gram-schmidt", "Gram-Schmidt on three vectors", 5, "gramschmidt",
        "Orthogonalize without normalizing, so every intermediate vector stays exact.",
        {"A": _m([[1, 1, 1], [1, 0, 2], [0, 1, 1]])},
    ),
    Sample(
        "projection", "Project a vector onto a plane", 5, "projection",
        "The residual is the shortest distance from the point to the plane.",
        {"A": _m([[1, 1], [1, 0], [0, 1]]), "V": _v([2, 3, 5])},
    ),
    Sample(
        "subspace-basis", "Extract a basis from a spanning set", 5, "basis",
        "Five vectors spanning a 3-dimensional subspace; the pivot columns pick the basis.",
        {"A": _m([[1, 2, 0, 1, 3], [2, 4, 1, 3, 7], [1, 2, 1, 2, 4]])},
    ),
    Sample(
        "cramer", "Solve a system by Cramer's rule", 6, "cramer",
        "One determinant per unknown, each with a column replaced by the constants.",
        {"A": _m([[1, 2, 3], [0, 1, 4], [5, 6, 0]]), "B": _v([1, 2, 3])},
    ),
    Sample(
        "cramer-blocked", "Cramer's rule when det(A) = 0", 6, "cramer",
        "The rule divides by det(A), so a singular system needs row reduction instead.",
        {"A": _m([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), "B": _v([1, 2, 3])},
    ),
    Sample(
        "homogeneous-det", "Homogeneous system with a nontrivial solution", 6, "homogeneous",
        "det(A) = 0 is exactly the condition for a square homogeneous system to have more than x = 0.",
        {"A": _m([[1, 2, 3], [4, 5, 6], [7, 8, 9]])},
    ),
    Sample(
        "four-subspaces", "All four fundamental subspaces of one matrix", 7, "subspaces",
        "rank 2 in a 3x4 matrix: nullity 2, left nullity 1, and two orthogonality relations.",
        {"A": _m([[1, 2, 3, 4], [2, 4, 7, 10], [3, 6, 10, 14]])},
    ),
    Sample(
        "rank-nullity", "Rank-nullity on a wide matrix", 7, "subspaces",
        "Every column is either a pivot column or a free one, and the counts add to n.",
        {"A": _m([[1, -2, 1, 3, 0], [2, -4, 3, 5, 1], [1, -2, 2, 2, 1]])},
    ),
    Sample(
        "kernel-range", "Kernel and range of a linear map", 7, "subspaces",
        "The null space is the kernel, the column space is the range, and rank + nullity = n.",
        {"A": _m([[1, 0, -1], [2, 1, 0], [3, 1, -1]])},
    ),
    Sample(
        "eigen-distinct", "Distinct integer eigenvalues", 8, "eigen",
        "Three separate eigenvalues, so three independent eigenvectors and a clean diagonalization.",
        {"A": _m([[4, 0, 1], [-2, 1, 0], [-2, 0, 1]])},
    ),
    Sample(
        "eigen-symmetric", "A symmetric matrix and the spectral theorem", 8, "eigen",
        "Real eigenvalues, a repeated one with a full 2-dimensional eigenspace, orthogonal eigenvectors.",
        {"A": _m([[5, 4, 2], [4, 5, 2], [2, 2, 2]])},
    ),
    Sample(
        "eigen-defective", "A defective matrix", 8, "eigen",
        "Algebraic multiplicity 2 but geometric multiplicity 1, so it cannot be diagonalized.",
        {"A": _m([[1, 1], [0, 1]])},
    ),
    Sample(
        "eigen-irrational", "Irrational eigenvalues, exactly", 8, "eigen",
        "The Fibonacci matrix: eigenvalues are the golden ratio and its conjugate, kept exact.",
        {"A": _m([[1, 1], [1, 0]])},
    ),
    Sample(
        "eigen-complex", "Complex eigenvalues", 8, "eigen",
        "A quarter-turn rotation has no real eigenvector, and the eigenvalues are +i and -i.",
        {"A": _m([[0, -1], [1, 0]])},
    ),
    Sample(
        "eigen-3x3-repeated", "Repeated eigenvalue with a full eigenspace", 8, "eigen",
        "Algebraic multiplicity 2 matched by geometric multiplicity 2, so it still diagonalizes.",
        {"A": _m([[5, -1, -1], [-1, 5, -1], [-1, -1, 5]])},
    ),
]

BY_KEY = {s.key: s for s in SAMPLES}


def by_clo():
    groups = {}
    for s in SAMPLES:
        groups.setdefault(s.clo, []).append(s)
    return groups


def get(key):
    return BY_KEY.get(key)
