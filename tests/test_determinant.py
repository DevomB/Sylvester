import unittest
from fractions import Fraction

from sylvester.determinant import (adjugate, bareiss, cofactor, cofactor_matrix,
                                   characteristic_via_cofactor, cramer, det_by_row_reduction,
                                   det_triangular, determinant, expand_cofactors, is_invertible,
                                   minor)
from sylvester.inverse import (Singular, elementary_factorization, inverse, inverse_adjugate,
                               inverse_gauss_jordan, lu_decomposition, solve_with_inverse)
from sylvester.matrix import Matrix, elementary_add, elementary_scale, elementary_swap
from sylvester.reduce import HUMAN, MACHINE

from .support import naive_determinant, random_invertible, random_matrix, random_square, seeded


class DeterminantAgreementTest(unittest.TestCase):
    def test_every_algorithm_agrees(self):
        rng = seeded(31)
        for _ in range(500):
            n = rng.randint(1, 5)
            m = random_square(rng, n, -6, 6)
            expected = naive_determinant(m)
            self.assertEqual(determinant(m), expected, m.to_lists())
            self.assertEqual(bareiss(m), expected)
            self.assertEqual(expand_cofactors(m).value, expected)
            self.assertEqual(det_by_row_reduction(m, MACHINE).value, expected)
            self.assertEqual(det_by_row_reduction(m, HUMAN).value, expected)

    def test_cofactor_expansion_along_any_axis(self):
        rng = seeded(32)
        for _ in range(120):
            n = rng.randint(2, 4)
            m = random_square(rng, n)
            expected = determinant(m)
            for index in range(n):
                self.assertEqual(expand_cofactors(m, "row", index).value, expected)
                self.assertEqual(expand_cofactors(m, "col", index).value, expected)

    def test_rational_entries(self):
        rng = seeded(33)
        for _ in range(150):
            n = rng.randint(1, 4)
            m = Matrix([[Fraction(rng.randint(-6, 6), rng.randint(1, 4)) for _ in range(n)]
                        for _ in range(n)])
            self.assertEqual(determinant(m), naive_determinant(m))

    def test_empty_and_triangular(self):
        self.assertEqual(determinant(Matrix([])), 1)
        upper = Matrix([[3, 7, -2], [0, -1, 4], [0, 0, 2]])
        self.assertEqual(determinant(upper), -6)
        self.assertEqual(det_triangular(upper), -6)

    def test_non_square_is_rejected(self):
        with self.assertRaises(ValueError):
            determinant(Matrix([[1, 2, 3]]))


class DeterminantPropertiesTest(unittest.TestCase):
    def test_classical_identities(self):
        rng = seeded(34)
        for _ in range(300):
            n = rng.randint(1, 4)
            a, b = random_square(rng, n, -5, 5), random_square(rng, n, -5, 5)
            c = Fraction(rng.randint(-4, 4))
            self.assertEqual(determinant(a.matmul(b)), determinant(a) * determinant(b))
            self.assertEqual(determinant(a.T), determinant(a))
            self.assertEqual(determinant(a * c), c ** n * determinant(a))
            self.assertEqual(a.matmul(adjugate(a)), Matrix.identity(n) * determinant(a))
            self.assertEqual(adjugate(a).matmul(a), Matrix.identity(n) * determinant(a))

    def test_row_operations_move_the_determinant_predictably(self):
        rng = seeded(35)
        for _ in range(300):
            n = rng.randint(2, 4)
            a = random_square(rng, n)
            base = determinant(a)
            i, j = rng.sample(range(n), 2)
            k = Fraction(rng.randint(1, 5))
            self.assertEqual(determinant(elementary_swap(n, i, j).matmul(a)), -base)
            self.assertEqual(determinant(elementary_scale(n, i, k).matmul(a)), k * base)
            self.assertEqual(determinant(elementary_add(n, i, j, k).matmul(a)), base)

    def test_repeated_row_gives_zero(self):
        rng = seeded(36)
        for _ in range(100):
            n = rng.randint(2, 4)
            rows = random_square(rng, n).to_lists()
            rows[1] = list(rows[0])
            self.assertEqual(determinant(Matrix(rows)), 0)

    def test_minor_and_cofactor(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 10]])
        self.assertEqual(minor(m, 0, 0), 5 * 10 - 6 * 8)
        self.assertEqual(cofactor(m, 0, 1), -(4 * 10 - 6 * 7))
        self.assertEqual(cofactor_matrix(m).T, adjugate(m))

    def test_invertibility_matches_determinant(self):
        rng = seeded(37)
        for _ in range(200):
            m = random_square(rng, rng.randint(1, 4))
            self.assertEqual(is_invertible(m), bool(determinant(m)))

    def test_characteristic_polynomial_is_monic_of_degree_n(self):
        rng = seeded(38)
        for _ in range(120):
            n = rng.randint(1, 4)
            m = random_square(rng, n, -4, 4)
            p = characteristic_via_cofactor(m)
            self.assertEqual(p.degree, n)
            self.assertEqual(p.lead, 1)
            self.assertEqual(p.eval(0), (-1) ** n * determinant(m))


class InverseTest(unittest.TestCase):
    def test_methods_agree_and_actually_invert(self):
        rng = seeded(41)
        singular = 0
        for _ in range(400):
            n = rng.randint(1, 4)
            a = random_square(rng, n, -5, 5)
            gauss = inverse_gauss_jordan(a)
            adj = inverse_adjugate(a)
            self.assertEqual(gauss.ok, adj.ok)
            self.assertEqual(gauss.ok, bool(determinant(a)))
            if not gauss.ok:
                singular += 1
                continue
            self.assertEqual(gauss.inverse, adj.inverse)
            self.assertTrue(a.matmul(gauss.inverse).is_identity())
            self.assertTrue(gauss.inverse.matmul(a).is_identity())
            self.assertEqual(determinant(gauss.inverse) * determinant(a), 1)
        self.assertGreater(singular, 0)

    def test_inverse_of_product_and_transpose(self):
        rng = seeded(42)
        for _ in range(200):
            n = rng.randint(1, 4)
            a, b = random_invertible(rng, n), random_invertible(rng, n)
            self.assertEqual(inverse(a.matmul(b)), inverse(b).matmul(inverse(a)))
            self.assertEqual(inverse(a.T), inverse(a).T)
            self.assertEqual(inverse(inverse(a)), a)

    def test_singular_raises(self):
        with self.assertRaises(Singular):
            inverse(Matrix([[1, 2], [2, 4]]))

    def test_negative_powers(self):
        rng = seeded(43)
        for _ in range(60):
            a = random_invertible(rng, rng.randint(1, 3))
            self.assertEqual(a ** -2, inverse(a).matmul(inverse(a)))
            self.assertTrue((a ** -1).matmul(a).is_identity())

    def test_elementary_factorization_reconstructs_the_matrix(self):
        rng = seeded(44)
        for _ in range(250):
            n = rng.randint(1, 4)
            a = random_square(rng, n, -5, 5)
            result = elementary_factorization(a)
            self.assertEqual(result.ok, bool(determinant(a)))
            if not result.ok:
                continue
            self.assertEqual(result.product, a)
            self.assertEqual(len(result.factors), len(result.inverse_factors))
            for factor, undo in zip(result.factors, result.inverse_factors):
                self.assertTrue(factor.matmul(undo).is_identity())
            accumulated = Matrix.identity(n)
            for factor in result.factors:
                accumulated = factor.matmul(accumulated)
            self.assertTrue(accumulated.matmul(a).is_identity())

    def test_lu_factorization(self):
        rng = seeded(45)
        for _ in range(300):
            a = random_matrix(rng, rng.randint(1, 4), rng.randint(1, 4), -5, 5)
            result = lu_decomposition(a)
            self.assertTrue(result.ok, a.to_lists())
            self.assertEqual(result.p.matmul(a), result.l.matmul(result.u))
            self.assertTrue(result.l.is_lower_triangular())
            self.assertTrue(result.u.is_upper_triangular())
            for i in range(result.l.nrows):
                self.assertEqual(result.l[i, i], 1)

    def test_solving_with_the_inverse(self):
        rng = seeded(46)
        for _ in range(100):
            n = rng.randint(1, 4)
            a = random_invertible(rng, n)
            b = [Fraction(rng.randint(-5, 5)) for _ in range(n)]
            x, inv = solve_with_inverse(a, b)
            self.assertEqual(a.matmul(x), Matrix.column(b))


class CramerTest(unittest.TestCase):
    def test_cramer_returns_the_replaced_matrices(self):
        a = Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]])
        b = [Fraction(1), Fraction(2), Fraction(3)]
        solution, det, numerators = cramer(a, b)
        self.assertEqual(det, 1)
        self.assertEqual(len(numerators), 3)
        for j, (replaced, value) in enumerate(numerators):
            self.assertEqual(replaced.column_at(j), tuple(b))
            self.assertEqual(value, determinant(replaced))
        self.assertEqual(a.matmul(Matrix.column(solution)), Matrix.column(b))

    def test_cramer_returns_none_when_singular(self):
        solution, det, _ = cramer(Matrix([[1, 2], [2, 4]]), [Fraction(1), Fraction(2)])
        self.assertIsNone(solution)
        self.assertEqual(det, 0)


if __name__ == "__main__":
    unittest.main()
