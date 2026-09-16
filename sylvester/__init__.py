from .determinant import (adjugate, cofactor, cofactor_matrix, determinant, expand_cofactors,
                          minor)
from .algebraic import AlgebraicNumber, NumberField
from .eigen import (characteristic_polynomial, diagonalize, eigenspace, eigenvalues,
                    orthogonally_diagonalize, spectrum)
from .exact import Surd, sqrt_exact, surd
from .expr import evaluate
from .factor import factor
from .inverse import (elementary_factorization, inverse, inverse_adjugate, inverse_gauss_jordan,
                      lu_decomposition)
from .matrix import Matrix, elementary_add, elementary_scale, elementary_swap
from .parse import parse_matrix, parse_vector
from .polynomial import Poly
from .proof import check as prove
from .reduce import nullspace_basis, rank, ref, row_reduce, rref
from .solve import solve, solve_homogeneous
from .subspace import (basis_from_spanning_set, four_subspaces, in_span, independence,
                       rank_nullity)

__version__ = "1.1.0"
__all__ = [
    "Matrix", "Poly", "Surd", "surd", "sqrt_exact", "evaluate", "prove",
    "factor", "NumberField", "AlgebraicNumber",
    "row_reduce", "rref", "ref", "rank", "nullspace_basis",
    "determinant", "minor", "cofactor", "cofactor_matrix", "adjugate", "expand_cofactors",
    "inverse", "inverse_gauss_jordan", "inverse_adjugate", "elementary_factorization",
    "lu_decomposition", "elementary_swap", "elementary_scale", "elementary_add",
    "solve", "solve_homogeneous",
    "independence", "in_span", "basis_from_spanning_set", "four_subspaces", "rank_nullity",
    "spectrum", "eigenvalues", "eigenspace", "characteristic_polynomial", "diagonalize",
    "orthogonally_diagonalize",
    "parse_matrix", "parse_vector",
]
