import unittest
from fractions import Fraction

from sylvester.determinant import determinant
from sylvester.eigen import (algebraic_multiplicity, cayley_hamilton, characteristic_polynomial,
                             diagonalize, eigenspace, faddeev_leverrier,
                             orthogonally_diagonalize, similar, spectrum)
from sylvester.exact import Surd
from sylvester.matrix import Matrix
from sylvester import vectors as V

from .support import random_square, seeded


class CharacteristicPolynomialTest(unittest.TestCase):
    def test_monic_of_degree_n(self):
        rng = seeded(71)
        for _ in range(200):
            n = rng.randint(1, 5)
            a = random_square(rng, n, -4, 4)
            p = characteristic_polynomial(a)
            self.assertEqual(p.degree, n)
            self.assertEqual(p.lead, 1)

    def test_two_algorithms_agree(self):
        rng = seeded(72)
        for _ in range(120):
            n = rng.randint(1, 5)
            a = random_square(rng, n, -4, 4)
            self.assertEqual(characteristic_polynomial(a), faddeev_leverrier(a))

    def test_coefficients_encode_trace_and_determinant(self):
        rng = seeded(73)
        for _ in range(200):
            n = rng.randint(1, 4)
            a = random_square(rng, n, -4, 4)
            p = characteristic_polynomial(a)
            self.assertEqual(p[n - 1], -a.trace())
            self.assertEqual(p[0], (-1) ** n * determinant(a))

    def test_cayley_hamilton(self):
        rng = seeded(74)
        for _ in range(200):
            n = rng.randint(1, 4)
            a = random_square(rng, n, -4, 4)
            _, _, value, ok = cayley_hamilton(a)
            self.assertTrue(ok, a.to_lists())
            self.assertTrue(value.is_zero())

    def test_known_polynomials(self):
        self.assertEqual(str(characteristic_polynomial(Matrix([[2, 1], [1, 2]]))), "x^2 - 4x + 3")
        self.assertEqual(str(characteristic_polynomial(Matrix([[0, -1], [1, 0]]))), "x^2 + 1")
        self.assertEqual(str(characteristic_polynomial(Matrix([[1, 1], [1, 0]]))), "x^2 - x - 1")


class SpectrumTest(unittest.TestCase):
    def test_eigenpairs_satisfy_the_definition(self):
        rng = seeded(75)
        exact_cases = numeric_cases = 0
        for _ in range(400):
            n = rng.randint(1, 4)
            a = random_square(rng, n, -4, 4)
            spec = spectrum(a)
            self.assertTrue(spec.verify(), a.to_lists())
            self.assertEqual(sum(p.algebraic for p in spec.pairs), n)
            if spec.exact:
                exact_cases += 1
                self.assertIsNot(spec.trace_check, False)
                self.assertIsNot(spec.det_check, False)
                for pair in spec.pairs:
                    self.assertGreaterEqual(pair.geometric, 1)
                    self.assertLessEqual(pair.geometric, pair.algebraic)
                    for v in pair.basis:
                        left = a.matmul(Matrix.column(v))
                        right = Matrix.column(v) * pair.value
                        self.assertEqual(left, right)
            else:
                numeric_cases += 1
        self.assertGreater(exact_cases, 0)
        self.assertGreater(numeric_cases, 0)

    def test_eigenvalues_are_roots_of_the_characteristic_polynomial(self):
        rng = seeded(76)
        for _ in range(200):
            a = random_square(rng, rng.randint(1, 4), -4, 4)
            spec = spectrum(a)
            for pair in spec.pairs:
                if pair.exact:
                    self.assertEqual(spec.poly.eval(pair.value), 0)

    def test_triangular_eigenvalues_are_the_diagonal(self):
        a = Matrix([[2, 7, 1], [0, 3, 5], [0, 0, -4]])
        values = sorted(float(p.value) for p in spectrum(a).pairs)
        self.assertEqual(values, [-4.0, 2.0, 3.0])

    def test_defective_matrix(self):
        spec = spectrum(Matrix([[1, 1], [0, 1]]))
        self.assertEqual(len(spec.pairs), 1)
        pair = spec.pairs[0]
        self.assertEqual((pair.algebraic, pair.geometric), (2, 1))
        self.assertTrue(pair.defective)
        self.assertFalse(spec.diagonalizable)

    def test_repeated_eigenvalue_with_full_eigenspace(self):
        spec = spectrum(Matrix([[5, -1, -1], [-1, 5, -1], [-1, -1, 5]]))
        by_value = {float(p.value): p for p in spec.pairs}
        self.assertEqual((by_value[6.0].algebraic, by_value[6.0].geometric), (2, 2))
        self.assertTrue(spec.diagonalizable)

    def test_irrational_eigenvalues_stay_exact(self):
        spec = spectrum(Matrix([[1, 1], [1, 0]]))
        self.assertTrue(spec.exact)
        self.assertTrue(all(isinstance(p.value, Surd) for p in spec.pairs))
        golden = max(p.value.approx() for p in spec.pairs)
        self.assertAlmostEqual(golden, (1 + 5 ** 0.5) / 2)
        for pair in spec.pairs:
            for v in pair.basis:
                self.assertEqual(
                    Matrix([[1, 1], [1, 0]]).matmul(Matrix.column(v)),
                    Matrix.column(v) * pair.value,
                )

    def test_complex_eigenvalues(self):
        spec = spectrum(Matrix([[0, -1], [1, 0]]))
        self.assertTrue(spec.exact)
        self.assertFalse(spec.real_only)
        self.assertEqual(len(spec.pairs), 2)

    def test_eigenspace_helper(self):
        a = Matrix([[2, 0], [0, 3]])
        basis, shifted = eigenspace(a, Fraction(2))
        self.assertEqual(len(basis), 1)
        self.assertEqual(shifted, a - Matrix.identity(2) * 2)

    def test_algebraic_multiplicity_helper(self):
        self.assertEqual(algebraic_multiplicity(Matrix([[1, 1], [0, 1]]), Fraction(1)), 2)
        self.assertEqual(algebraic_multiplicity(Matrix([[2, 0], [0, 3]]), Fraction(2)), 1)

    def test_non_square_is_rejected(self):
        with self.assertRaises(ValueError):
            spectrum(Matrix([[1, 2, 3]]))


class DiagonalizationTest(unittest.TestCase):
    def test_diagonalization_reconstructs_the_matrix(self):
        rng = seeded(77)
        diagonalized = 0
        for _ in range(400):
            n = rng.randint(1, 4)
            a = random_square(rng, n, -4, 4)
            result = diagonalize(a)
            if not result.ok:
                continue
            diagonalized += 1
            self.assertTrue(result.verified)
            self.assertEqual(a.matmul(result.p), result.p.matmul(result.d))
            self.assertEqual(result.p.matmul(result.d).matmul(result.p.inverse()), a)
            self.assertTrue(result.d.is_diagonal())
        self.assertGreater(diagonalized, 0)

    def test_defective_matrices_are_refused_with_a_reason(self):
        result = diagonalize(Matrix([[1, 1], [0, 1]]))
        self.assertFalse(result.ok)
        self.assertIn("eigenvector", result.reason)

    def test_similar_matrices_share_a_characteristic_polynomial(self):
        rng = seeded(78)
        for _ in range(120):
            n = rng.randint(1, 3)
            a = random_square(rng, n, -4, 4)
            p = None
            for _ in range(40):
                candidate = random_square(rng, n, -3, 3)
                if determinant(candidate):
                    p = candidate
                    break
            if p is None:
                continue
            b = p.inverse().matmul(a).matmul(p)
            self.assertEqual(characteristic_polynomial(a), characteristic_polynomial(b))
            self.assertEqual(a.trace(), b.trace())
            self.assertEqual(determinant(a), determinant(b))
            self.assertTrue(similar(a, b, p))


class SpectralTheoremTest(unittest.TestCase):
    def test_symmetric_matrices_have_real_eigenvalues_and_orthogonal_eigenvectors(self):
        rng = seeded(79)
        single_field = multi_field = 0
        for _ in range(250):
            n = rng.randint(2, 4)
            b = random_square(rng, n, -3, 3)
            a = b + b.T
            spec = spectrum(a)
            if not spec.exact:
                continue
            self.assertTrue(spec.real_only, a.to_lists())
            self.assertTrue(spec.diagonalizable)
            result = orthogonally_diagonalize(a)
            if result.ok:
                single_field += 1
                self.assertTrue(V.is_orthogonal_set(result.orthogonal_basis))
                self.assertEqual(a.matmul(result.q), result.q.matmul(result.d))
            else:
                multi_field += 1
                self.assertIn("more than one quadratic field", result.reason)
                for pair, ortho in result.blocks:
                    self.assertTrue(ortho.verify())
        self.assertGreater(single_field, 50)

    def test_eigenvalues_in_two_quadratic_fields_are_reported_not_forced(self):
        a = Matrix([[-2, -2, -2, 3], [-2, -4, -1, -4], [-2, -1, 6, 0], [3, -4, 0, 0]])
        spec = spectrum(a)
        self.assertTrue(spec.exact)
        self.assertTrue(spec.real_only)
        self.assertIsNone(spec.field)
        result = orthogonally_diagonalize(a)
        self.assertFalse(result.ok)
        self.assertIn("more than one quadratic field", result.reason)
        self.assertFalse(diagonalize(a).ok)
        for pair, ortho in result.blocks:
            self.assertTrue(ortho.verify())

    def test_non_symmetric_is_refused(self):
        result = orthogonally_diagonalize(Matrix([[1, 2], [3, 4]]))
        self.assertFalse(result.ok)
        self.assertIn("symmetric", result.reason)

    def test_normalizing_the_basis_gives_an_orthogonal_matrix(self):
        a = Matrix([[2, 1], [1, 2]])
        result = orthogonally_diagonalize(a)
        self.assertTrue(result.ok)
        columns = [V.normalize(v) for v in result.orthogonal_basis]
        q = Matrix.from_columns(columns)
        self.assertTrue(q.is_orthogonal())
        self.assertEqual(q.T.matmul(a).matmul(q), result.d)


if __name__ == "__main__":
    unittest.main()
