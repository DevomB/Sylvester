import unittest
from fractions import Fraction

from sylvester.determinant import determinant
from sylvester.eigen import (algebraic_multiplicity, cayley_hamilton, characteristic_polynomial,
                             diagonalize, eigenspace, evaluate_at, faddeev_leverrier, is_eigenvector,
                             orthogonality_certificate, orthogonally_diagonalize, similar, spectrum)
from sylvester.exact import Surd
from sylvester.matrix import Matrix
from sylvester import vectors as V

from .support import random_square, seeded


class CharacteristicPolynomialTest(unittest.TestCase):
    def test_monic_of_degree_n(self):
        rng = seeded(71)
        for _ in range(200):
            n = rng.randint(1, 5)
            p = characteristic_polynomial(random_square(rng, n, -4, 4))
            self.assertEqual(p.degree, n)
            self.assertEqual(p.lead, 1)

    def test_two_algorithms_agree(self):
        rng = seeded(72)
        for _ in range(120):
            a = random_square(rng, rng.randint(1, 5), -4, 4)
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
            _, _, value, ok = cayley_hamilton(random_square(rng, rng.randint(1, 4), -4, 4))
            self.assertTrue(ok)
            self.assertTrue(value.is_zero())

    def test_known_polynomials(self):
        self.assertEqual(str(characteristic_polynomial(Matrix([[2, 1], [1, 2]]))), "x^2 - 4x + 3")
        self.assertEqual(str(characteristic_polynomial(Matrix([[0, -1], [1, 0]]))), "x^2 + 1")
        self.assertEqual(str(characteristic_polynomial(Matrix([[1, 1], [1, 0]]))), "x^2 - x - 1")


class SpectrumTest(unittest.TestCase):
    def test_every_spectrum_is_exact_and_verified(self):
        rng = seeded(75)
        kinds = {"rational": 0, "quadratic": 0, "family": 0}
        for _ in range(400):
            n = rng.randint(1, 4)
            a = random_square(rng, n, -4, 4)
            spec = spectrum(a)
            self.assertTrue(spec.verify(), a.to_lists())
            self.assertTrue(spec.trace_check, a.to_lists())
            self.assertTrue(spec.det_check, a.to_lists())
            self.assertEqual(sum(p.algebraic * p.count for p in spec.pairs), n)
            for pair in spec.pairs:
                self.assertGreaterEqual(pair.geometric, 1)
                self.assertLessEqual(pair.geometric, pair.algebraic)
                self.assertEqual(len(pair.roots), pair.count)
                if pair.symbolic:
                    kinds["family"] += 1
                    self.assertGreaterEqual(pair.minimal.degree, 3)
                elif pair.minimal.degree == 2:
                    kinds["quadratic"] += 1
                else:
                    kinds["rational"] += 1
        self.assertTrue(all(kinds.values()), kinds)

    def test_larger_matrices_stay_exact(self):
        rng = seeded(76)
        for _ in range(12):
            a = random_square(rng, rng.randint(5, 6), -5, 5)
            spec = spectrum(a)
            self.assertTrue(spec.verify())
            self.assertTrue(spec.trace_check and spec.det_check)

    def test_eigenvalues_are_roots_of_the_characteristic_polynomial(self):
        rng = seeded(77)
        for _ in range(200):
            spec = spectrum(random_square(rng, rng.randint(1, 4), -4, 4))
            for pair in spec.pairs:
                self.assertEqual(spec.poly.eval(pair.value), 0)
                self.assertEqual(pair.minimal.eval(pair.value), 0)

    def test_family_eigenvectors_hold_at_every_numeric_root(self):
        rng = seeded(78)
        checked = 0
        while checked < 40:
            a = random_square(rng, rng.randint(3, 4), -4, 4)
            for pair in spectrum(a).pairs:
                if not pair.symbolic:
                    continue
                checked += 1
                n = a.nrows
                for root in pair.roots:
                    for v in pair.basis:
                        values = [evaluate_at(x, root) for x in v]
                        image = [sum(float(a[i, j]) * values[j] for j in range(n)) for i in range(n)]
                        scale = max(abs(x) for x in values) * (1 + abs(root))
                        self.assertLessEqual(max(abs(image[i] - root * values[i]) for i in range(n)),
                                             1e-8 * scale)

    def test_irreducible_cubic_family(self):
        a = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]])
        spec = spectrum(a)
        self.assertEqual(len(spec.pairs), 1)
        pair = spec.pairs[0]
        self.assertTrue(pair.symbolic)
        self.assertEqual(pair.count, 3)
        self.assertEqual(pair.real_count, 3)
        self.assertEqual(str(pair.minimal), "x^3 - x^2 - 2x + 1")
        lam = pair.value
        self.assertEqual(pair.basis, [(lam ** 2 - 1, lam, Fraction(1))])
        self.assertTrue(spec.real_only)
        self.assertTrue(spec.diagonalizable)

    def test_complex_family(self):
        spec = spectrum(Matrix([[0, 0, 2], [1, 0, 0], [0, 1, 0]]))
        pair = spec.pairs[0]
        self.assertTrue(pair.symbolic)
        self.assertEqual((pair.count, pair.real_count), (3, 1))
        self.assertFalse(spec.real_only)
        lam = pair.value
        self.assertEqual(pair.basis, [(lam ** 2, lam, Fraction(1))])
        self.assertIn("complex", spec.reason)

    def test_a_wrong_eigenvector_is_rejected(self):
        a = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]])
        pair = spectrum(a).pairs[0]
        lam = pair.value
        self.assertTrue(is_eigenvector(a, (lam ** 2 - 1, lam, Fraction(1)), lam))
        self.assertFalse(is_eigenvector(a, (lam ** 2, lam, Fraction(1)), lam))
        self.assertFalse(is_eigenvector(a, (Fraction(0), Fraction(0), Fraction(0)), lam))

    def test_triangular_eigenvalues_are_the_diagonal(self):
        values = sorted(float(p.value) for p in spectrum(Matrix([[2, 7, 1], [0, 3, 5], [0, 0, -4]])).pairs)
        self.assertEqual(values, [-4.0, 2.0, 3.0])

    def test_defective_matrix(self):
        spec = spectrum(Matrix([[1, 1], [0, 1]]))
        pair = spec.pairs[0]
        self.assertEqual((pair.algebraic, pair.geometric), (2, 1))
        self.assertTrue(pair.defective)
        self.assertFalse(spec.diagonalizable)

    def test_repeated_family(self):
        block = [[0, 0, 2], [1, 0, 0], [0, 1, 0]]
        rows = [r + [0, 0, 0] for r in block] + [[0, 0, 0] + r for r in block]
        spec = spectrum(Matrix(rows))
        self.assertEqual(len(spec.pairs), 1)
        pair = spec.pairs[0]
        self.assertEqual((pair.algebraic, pair.geometric, pair.count), (2, 2, 3))
        self.assertTrue(spec.verify())
        self.assertTrue(spec.diagonalizable)

    def test_repeated_eigenvalue_with_full_eigenspace(self):
        spec = spectrum(Matrix([[5, -1, -1], [-1, 5, -1], [-1, -1, 5]]))
        by_value = {float(p.value): p for p in spec.pairs}
        self.assertEqual((by_value[6.0].algebraic, by_value[6.0].geometric), (2, 2))
        self.assertTrue(spec.diagonalizable)

    def test_irrational_quadratic_eigenvalues_stay_exact(self):
        a = Matrix([[1, 1], [1, 0]])
        spec = spectrum(a)
        self.assertTrue(all(isinstance(p.value, Surd) for p in spec.pairs))
        self.assertAlmostEqual(max(p.value.approx() for p in spec.pairs), (1 + 5 ** 0.5) / 2)
        self.assertTrue(spec.verify())

    def test_complex_quadratic_eigenvalues(self):
        spec = spectrum(Matrix([[0, -1], [1, 0]]))
        self.assertFalse(spec.real_only)
        self.assertEqual(len(spec.pairs), 2)
        self.assertTrue(spec.verify())

    def test_eigenspace_helper(self):
        a = Matrix([[2, 0], [0, 3]])
        basis, shifted, reduction = eigenspace(a, Fraction(2))
        self.assertEqual(len(basis), 1)
        self.assertEqual(shifted, a - Matrix.identity(2) * 2)
        self.assertEqual(reduction.rank, 1)

    def test_algebraic_multiplicity_helper(self):
        self.assertEqual(algebraic_multiplicity(Matrix([[1, 1], [0, 1]]), Fraction(1)), 2)
        self.assertEqual(algebraic_multiplicity(Matrix([[2, 0], [0, 3]]), Fraction(2)), 1)

    def test_non_square_is_rejected(self):
        with self.assertRaises(ValueError):
            spectrum(Matrix([[1, 2, 3]]))


class DiagonalizationTest(unittest.TestCase):
    def test_diagonalization_is_verified_whenever_it_exists(self):
        rng = seeded(79)
        concrete = families = 0
        for _ in range(400):
            a = random_square(rng, rng.randint(1, 4), -4, 4)
            result = diagonalize(a)
            if not result.ok:
                self.assertFalse(result.spectrum.diagonalizable)
                continue
            self.assertTrue(result.verified, a.to_lists())
            self.assertEqual(len(result.columns), sum(p.geometric for p in result.spectrum.pairs))
            if result.p is None:
                families += 1
                self.assertTrue(result.symbolic)
                continue
            concrete += 1
            self.assertEqual(a.matmul(result.p), result.p.matmul(result.d))
            self.assertTrue(result.d.is_diagonal())
        self.assertGreater(concrete, 0)
        self.assertGreater(families, 0)

    def test_single_field_diagonalization_inverts(self):
        a = Matrix([[1, 1], [1, 0]])
        result = diagonalize(a)
        self.assertEqual(result.p.matmul(result.d).matmul(result.p.inverse()), a)

    def test_two_quadratic_fields_diagonalize(self):
        a = Matrix([[5, 1, 3, -1], [1, 5, -1, 3], [3, -1, -3, 1], [-1, 3, 1, -3]])
        result = diagonalize(a)
        self.assertTrue(result.ok)
        self.assertTrue(result.verified)
        self.assertEqual({p.value.d for p in result.spectrum.pairs}, {2, 5})
        self.assertEqual(a.matmul(result.p), result.p.matmul(result.d))

    def test_defective_matrices_are_refused_with_a_reason(self):
        result = diagonalize(Matrix([[1, 1], [0, 1]]))
        self.assertFalse(result.ok)
        self.assertIn("eigenvector", result.reason)

    def test_similar_matrices_share_a_characteristic_polynomial(self):
        rng = seeded(80)
        for _ in range(120):
            n = rng.randint(1, 3)
            a = random_square(rng, n, -4, 4)
            p = next((c for c in (random_square(rng, n, -3, 3) for _ in range(40)) if determinant(c)), None)
            if p is None:
                continue
            b = p.inverse().matmul(a).matmul(p)
            self.assertEqual(characteristic_polynomial(a), characteristic_polynomial(b))
            self.assertTrue(similar(a, b, p))


class SpectralTheoremTest(unittest.TestCase):
    def test_every_symmetric_matrix_orthogonally_diagonalizes_exactly(self):
        rng = seeded(81)
        families = 0
        for _ in range(250):
            n = rng.randint(2, 4)
            b = random_square(rng, n, -3, 3)
            a = b + b.T
            result = orthogonally_diagonalize(a)
            self.assertTrue(result.spectrum.real_only, a.to_lists())
            self.assertTrue(result.ok, (a.to_lists(), result.reason))
            self.assertTrue(all(c.ok for c in result.checks))
            self.assertEqual(len(result.columns), n if not result.spectrum.symbolic
                             else sum(p.geometric for p in result.spectrum.pairs))
            families += result.spectrum.symbolic
            if result.q is not None:
                self.assertEqual(a.matmul(result.q), result.q.matmul(result.d))
        self.assertGreater(families, 20)

    def test_checks_cover_every_pair_of_columns(self):
        a = Matrix([[5, 1, 3, -1], [1, 5, -1, 3], [3, -1, -3, 1], [-1, 3, 1, -3]])
        result = orthogonally_diagonalize(a)
        self.assertTrue(result.ok)
        pairs = {(c.first, c.second) for c in result.checks}
        self.assertEqual(pairs, {(i, j) for i in range(4) for j in range(i + 1, 4)})
        self.assertEqual({c.method for c in result.checks},
                         {"conjugate eigenvalues", "different minimal polynomials"})

    def test_the_certificate_catches_a_broken_basis(self):
        rng = seeded(82)
        tested = 0
        for _ in range(200):
            n = rng.randint(2, 4)
            b = random_square(rng, n, -3, 3)
            result = orthogonally_diagonalize(b + b.T)
            if len(result.columns) < 2:
                continue
            pair, v = result.columns[0]
            other = result.columns[1][1]
            spot = next(i for i, x in enumerate(other) if x)
            broken = list(result.columns)
            broken[0] = (pair, tuple(x + 1 if i == spot else x for i, x in enumerate(v)))
            self.assertFalse(all(c.ok for c in orthogonality_certificate(broken)))
            tested += 1
        self.assertGreater(tested, 50)

    def test_a_family_is_orthogonal_across_its_own_conjugates(self):
        a = Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = orthogonally_diagonalize(a)
        self.assertTrue(result.ok)
        self.assertEqual([c.method for c in result.checks], ["conjugate eigenvalues"])
        pair, v = result.columns[0]
        lam = pair.value
        broken = [(pair, (lam ** 2, lam, Fraction(1)))]
        self.assertFalse(orthogonality_certificate(broken)[0].ok)

    def test_non_symmetric_is_refused(self):
        result = orthogonally_diagonalize(Matrix([[1, 2], [3, 4]]))
        self.assertFalse(result.ok)
        self.assertIn("symmetric", result.reason)

    def test_normalizing_a_rational_basis_gives_an_orthogonal_matrix(self):
        a = Matrix([[2, 1], [1, 2]])
        result = orthogonally_diagonalize(a)
        q = Matrix.from_columns([V.normalize(v) for v in result.orthogonal_basis])
        self.assertTrue(q.is_orthogonal())
        self.assertEqual(q.T.matmul(a).matmul(q), result.d)


if __name__ == "__main__":
    unittest.main()
