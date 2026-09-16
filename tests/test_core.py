import math
import unittest
from fractions import Fraction

from sylvester.exact import Surd, content, sqrt_exact, squarefree, surd, weight
from sylvester.matrix import Matrix, elementary_add, elementary_scale, elementary_swap
from sylvester.polynomial import Poly, from_roots, quadratic_roots, rational_roots
from sylvester.render import fmt

from .support import naive_determinant, random_matrix, random_square, seeded


class SquarefreeTest(unittest.TestCase):
    def test_extracts_square_part(self):
        for n in list(range(-60, 0)) + list(range(1, 60)):
            k, d = squarefree(n)
            self.assertEqual(k * k * d, n)
            self.assertTrue(all(d % (p * p) for p in range(2, abs(d) + 1)))

    def test_zero(self):
        self.assertEqual(squarefree(0), (0, 0))


class SurdTest(unittest.TestCase):
    def test_demotes_to_rational(self):
        self.assertIsInstance(surd(3, 0, 5), Fraction)
        self.assertIsInstance(surd(3, 2, 4), Fraction)
        self.assertEqual(surd(3, 2, 4), Fraction(7))

    def test_normalizes_radicand(self):
        value = surd(2, 3, 8)
        self.assertEqual(value.d, 2)
        self.assertEqual(value.b, 6)

    def test_fractional_radicand(self):
        value = surd(0, 1, Fraction(3, 4))
        self.assertAlmostEqual(value.approx(), math.sqrt(0.75))

    def test_arithmetic_matches_floats(self):
        rng = seeded(1)
        for _ in range(400):
            d = rng.choice([2, 3, 5, 7, -1, -2])
            a = Surd(Fraction(rng.randint(-6, 6)), Fraction(rng.randint(1, 6)), d)
            b = Surd(Fraction(rng.randint(-6, 6)), Fraction(rng.randint(1, 6)), d)
            for op in (lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y):
                exact = op(a, b)
                approx = op(complex(a.approx()), complex(b.approx()))
                self.assertAlmostEqual(complex(_approx(exact)), approx, places=9)
            quotient = a / b
            self.assertAlmostEqual(complex(_approx(quotient)), complex(a.approx()) / complex(b.approx()), places=9)

    def test_division_inverts(self):
        a = surd(3, 2, 7)
        self.assertEqual(a / a, 1)
        self.assertEqual(a * (1 / a), 1)

    def test_powers(self):
        a = surd(1, 2, 3)
        self.assertEqual(a ** 3, a * a * a)
        self.assertEqual(a ** -1 * a, 1)
        self.assertEqual(a ** 0, 1)

    def test_ordering(self):
        self.assertTrue(surd(0, 1, 2) < Fraction(3, 2))
        self.assertTrue(surd(0, 1, 2) > Fraction(1))
        self.assertTrue(surd(-1, 1, 2) > 0)
        self.assertTrue(surd(1, -1, 2) < 0)
        values = sorted([surd(0, 1, 2), Fraction(1), surd(-1, 1, 2), Fraction(-3)])
        self.assertEqual([float(v) for v in values], sorted(float(v) for v in values))

    def test_conjugate_and_norm(self):
        z = surd(2, 3, -1)
        self.assertEqual(z * z.conjugate(), z.field_norm())
        self.assertEqual(z.field_norm(), 13)

    def test_complex_has_no_sign(self):
        with self.assertRaises(TypeError):
            surd(1, 1, -1).sign()

    def test_cross_field_is_rejected(self):
        with self.assertRaises(ArithmeticError):
            surd(0, 1, 2) + surd(0, 1, 3)

    def test_sqrt_exact(self):
        self.assertEqual(sqrt_exact(Fraction(9, 4)), Fraction(3, 2))
        self.assertEqual(sqrt_exact(16), 4)
        self.assertIsInstance(sqrt_exact(2), Surd)
        self.assertFalse(sqrt_exact(-3).is_real)

    def test_helpers(self):
        self.assertEqual(content([Fraction(4), Fraction(6), Fraction(-8)]), 2)
        self.assertEqual(content([Fraction(1, 2), Fraction(3)]), 1)
        self.assertGreater(weight(Fraction(7, 3)), weight(Fraction(1)))


class SurdFormatTest(unittest.TestCase):
    def test_readable_forms(self):
        self.assertEqual(fmt(surd(0, 1, 5)), "sqrt5")
        self.assertEqual(fmt(surd(0, -1, 5)), "-sqrt5")
        self.assertEqual(fmt(surd(2, 3, 5)), "2 + 3sqrt5")
        self.assertEqual(fmt(surd(2, -3, 5)), "2 - 3sqrt5")
        self.assertEqual(fmt(surd(Fraction(1, 2), Fraction(1, 2), 5)), "(1 + sqrt5)/2")
        self.assertEqual(fmt(surd(0, 1, -1)), "i")
        self.assertEqual(fmt(surd(2, 3, -1)), "2 + 3i")
        self.assertEqual(fmt(surd(0, 2, -5)), "2isqrt5")


class PolynomialTest(unittest.TestCase):
    def test_arithmetic(self):
        a = Poly([1, 2, 3])
        b = Poly([0, 1])
        self.assertEqual((a * b).c, (0, 1, 2, 3))
        self.assertEqual((a + b).c, (1, 3, 3))
        self.assertEqual((a - a).c, ())
        self.assertEqual(a.degree, 2)

    def test_divmod_roundtrip(self):
        rng = seeded(2)
        for _ in range(200):
            a = Poly([rng.randint(-5, 5) for _ in range(rng.randint(1, 6))])
            b = Poly([rng.randint(-5, 5) for _ in range(rng.randint(1, 4))] + [rng.randint(1, 5)])
            q, r = a.divmod(b)
            self.assertEqual(q * b + r, a)
            self.assertTrue(not r or r.degree < b.degree)

    def test_evaluation_and_derivative(self):
        p = Poly([-6, 11, -6, 1])
        for x in (-2, 0, 1, 2, 3, 7):
            self.assertEqual(p.eval(x), x ** 3 - 6 * x ** 2 + 11 * x - 6)
        self.assertEqual(p.derivative().c, (11, -12, 3))

    def test_rational_roots_with_multiplicity(self):
        p = from_roots([Fraction(2), Fraction(2), Fraction(-1, 3)])
        found = dict(rational_roots(p))
        self.assertEqual(found[Fraction(2)], 2)
        self.assertEqual(found[Fraction(-1, 3)], 1)

    def test_quadratic_roots_are_exact(self):
        low, high = sorted(quadratic_roots(Fraction(1), Fraction(-1), Fraction(-1)), key=float)
        self.assertIsInstance(high, Surd)
        self.assertAlmostEqual(float(high), (1 + 5 ** 0.5) / 2)
        self.assertEqual(high * high - high - 1, 0)
        i, minus_i = quadratic_roots(Fraction(1), Fraction(0), Fraction(1))
        self.assertFalse(i.is_real)
        self.assertEqual(i * i, -1)
        self.assertEqual(quadratic_roots(Fraction(1), Fraction(-5), Fraction(6)), [3, 2])

    def test_formatting(self):
        self.assertEqual(str(Poly([-6, 11, -6, 1])), "x^3 - 6x^2 + 11x - 6")
        self.assertEqual(str(Poly([])), "0")
        self.assertEqual(Poly([0, 0, 1]).shift_variable("L"), "L^2")
        self.assertEqual(Poly([Fraction(1, 2), Fraction(-3, 2), 1]).shift_variable("L"), "L^2 - (3/2)L + 1/2")


class MatrixTest(unittest.TestCase):
    def test_rejects_ragged(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2], [3]])

    def test_shape_and_access(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(m.shape, (2, 3))
        self.assertEqual(m[1, 2], 6)
        self.assertEqual(m.column_at(1), (2, 5))
        self.assertFalse(m.is_square)

    def test_transpose_is_an_involution(self):
        rng = seeded(4)
        for _ in range(100):
            m = random_matrix(rng, rng.randint(1, 5), rng.randint(1, 5))
            self.assertEqual(m.T.T, m)
            self.assertEqual(m.T.shape, (m.ncols, m.nrows))

    def test_multiplication_is_associative_and_distributive(self):
        rng = seeded(5)
        for _ in range(120):
            n = rng.randint(1, 4)
            a, b, c = (random_square(rng, n) for _ in range(3))
            self.assertEqual((a.matmul(b)).matmul(c), a.matmul(b.matmul(c)))
            self.assertEqual(a.matmul(b + c), a.matmul(b) + a.matmul(c))
            self.assertEqual((a.matmul(b)).T, b.T.matmul(a.T))

    def test_identity_and_powers(self):
        rng = seeded(6)
        for _ in range(60):
            n = rng.randint(1, 4)
            a = random_square(rng, n)
            identity = Matrix.identity(n)
            self.assertEqual(a.matmul(identity), a)
            self.assertEqual(identity.matmul(a), a)
            self.assertEqual(a ** 0, identity)
            self.assertEqual(a ** 3, a.matmul(a).matmul(a))

    def test_dimension_mismatch_is_reported(self):
        with self.assertRaises(ValueError):
            Matrix([[1, 2]]).matmul(Matrix([[1, 2]]))
        with self.assertRaises(ValueError):
            Matrix([[1, 2]]) + Matrix([[1], [2]])

    def test_predicates(self):
        self.assertTrue(Matrix([[1, 2], [2, 1]]).is_symmetric())
        self.assertTrue(Matrix([[0, 2], [-2, 0]]).is_skew_symmetric())
        self.assertTrue(Matrix([[1, 5], [0, 1]]).is_upper_triangular())
        self.assertTrue(Matrix([[1, 0], [5, 1]]).is_lower_triangular())
        self.assertTrue(Matrix([[0, 1], [1, 0]]).is_orthogonal())
        self.assertTrue(Matrix([[0, 1], [0, 0]]).is_nilpotent())
        self.assertTrue(Matrix([[1, 0], [0, 1]]).is_involutory())
        self.assertTrue(Matrix([[1, 0], [0, 0]]).is_idempotent())

    def test_minor_and_submatrix(self):
        m = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertEqual(m.minor_matrix(0, 0), Matrix([[5, 6], [8, 9]]))
        self.assertEqual(m.minor_matrix(1, 1), Matrix([[1, 3], [7, 9]]))
        self.assertEqual(m.take_columns([0, 2]), Matrix([[1, 3], [4, 6], [7, 9]]))

    def test_augment_and_split(self):
        a = Matrix([[1, 2], [3, 4]])
        joined = a.augment(Matrix.identity(2))
        self.assertEqual(joined.shape, (2, 4))
        self.assertEqual(joined.left(2), a)
        self.assertEqual(joined.right(2), Matrix.identity(2))

    def test_elementary_matrices_act_as_row_operations(self):
        rng = seeded(7)
        for _ in range(120):
            n = rng.randint(2, 4)
            a = random_square(rng, n)
            i, j = rng.sample(range(n), 2)
            swapped = elementary_swap(n, i, j).matmul(a)
            expected = a.to_lists()
            expected[i], expected[j] = expected[j], expected[i]
            self.assertEqual(swapped, Matrix(expected))

            k = Fraction(rng.randint(1, 5))
            scaled = elementary_scale(n, i, k).matmul(a)
            expected = a.to_lists()
            expected[i] = [k * v for v in expected[i]]
            self.assertEqual(scaled, Matrix(expected))

            added = elementary_add(n, i, j, k).matmul(a)
            expected = a.to_lists()
            expected[i] = [x + k * y for x, y in zip(expected[i], expected[j])]
            self.assertEqual(added, Matrix(expected))

    def test_elementary_scale_rejects_zero(self):
        with self.assertRaises(ValueError):
            elementary_scale(3, 0, 0)

    def test_hashable(self):
        self.assertEqual(len({Matrix([[1]]), Matrix([[1]]), Matrix([[2]])}), 2)

    def test_naive_determinant_agrees_on_small_cases(self):
        self.assertEqual(naive_determinant(Matrix([[1, 2], [3, 4]])), -2)


def _approx(value):
    from sylvester.exact import approx

    return approx(value)


if __name__ == "__main__":
    unittest.main()
