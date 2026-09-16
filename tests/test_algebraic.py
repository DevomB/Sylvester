import unittest
from fractions import Fraction
from itertools import combinations

from sylvester.algebraic import AlgebraicNumber, NumberField, ReducibleModulus
from sylvester.factor import expand, factor, is_irreducible, squarefree_decomposition
from sylvester.polynomial import (Poly, approximate_roots, count_real_roots, from_roots,
                                  isolate_real_roots, numeric_roots)

from .support import seeded


def eisenstein(rng):
    p = rng.choice([2, 3, 5, 7])
    degree = rng.randint(2, 5)
    coeffs = [p * rng.choice([u for u in range(-4, 5) if u % p])]
    coeffs += [p * rng.randint(-2, 2) for _ in range(degree - 1)]
    coeffs.append(rng.choice([1, 1, 1, 2, 3]) if p > 3 else 1)
    return Poly(coeffs)


def has_proper_factor_numerically(poly):
    roots = numeric_roots(poly)
    lead = poly.lead
    n = poly.degree
    for size in range(1, n // 2 + 1):
        for subset in combinations(roots, size):
            coeffs = [1 + 0j]
            for r in subset:
                coeffs = [0j] + coeffs
                for i in range(len(coeffs) - 1):
                    coeffs[i] -= r * coeffs[i + 1]
            scaled = [c * complex(lead) for c in coeffs]
            if any(abs(c.imag) > 1e-6 or abs(c.real - round(c.real)) > 1e-6 for c in scaled):
                continue
            candidate = Poly([round(c.real) for c in scaled])
            if candidate.degree == size and not poly % candidate:
                return True
    return False


class FactorTest(unittest.TestCase):
    def test_classical_cases(self):
        cases = {
            (1, 0, 0, 0, 1): [(1, 0, 0, 0, 1)],
            (1, 0, -10, 0, 1): [(1, 0, -10, 0, 1)],
            (6, 0, -5, 0, 1): [(-3, 0, 1), (-2, 0, 1)],
            (4, 0, 0, 0, 1): [(2, -2, 1), (2, 2, 1)],
            (-1, 0, 0, 0, 0, 0, 1): [(-1, 1), (1, 1), (1, -1, 1), (1, 1, 1)],
            (-1, 0, 0, 0, 0, 0, 0, 0, 1): [(-1, 1), (1, 1), (1, 0, 1), (1, 0, 0, 0, 1)],
            (1, 0, 0, 0, 0, 0, 0, 0, 1): [(1, 0, 0, 0, 0, 0, 0, 0, 1)],
            (1, 1, 1, 1, 1): [(1, 1, 1, 1, 1)],
            (-2, 0, 0, 1): [(-2, 0, 0, 1)],
        }
        for coeffs, expected in cases.items():
            unit, factors = factor(Poly(coeffs))
            self.assertEqual(sorted(tuple(q.c) for q, _ in factors),
                             sorted(tuple(Fraction(c) for c in e) for e in expected), coeffs)
            self.assertEqual(expand(unit, factors), Poly(coeffs))

    def test_eisenstein_products_recover_their_factors(self):
        rng = seeded(101)
        checked = 0
        while checked < 150:
            parts = []
            for _ in range(rng.randint(1, 3)):
                candidate = eisenstein(rng)
                if all(candidate.primitive()[0] != q.primitive()[0] for q, _ in parts):
                    parts.append((candidate, rng.choice([1, 1, 1, 2])))
            if sum(q.degree * m for q, m in parts) > 12:
                continue
            product = Poly([Fraction(rng.choice([1, -3, 5]), rng.choice([1, 2, 7]))])
            for q, m in parts:
                product = product * q ** m
            unit, factors = factor(product)
            self.assertEqual(sorted((q.primitive()[0].c, m) for q, m in parts),
                             sorted((q.c, m) for q, m in factors))
            self.assertEqual(expand(unit, factors), product)
            checked += 1

    def test_random_polynomials_split_into_genuine_irreducibles(self):
        rng = seeded(102)
        for _ in range(250):
            poly = Poly([rng.randint(-9, 9) for _ in range(rng.randint(2, 7))] + [rng.randint(1, 4)])
            unit, factors = factor(poly)
            self.assertEqual(expand(unit, factors), poly)
            for q, _ in factors:
                self.assertTrue(all(c.denominator == 1 for c in q.c))
                self.assertGreater(q.lead, 0)
                if q.degree >= 2:
                    self.assertFalse(has_proper_factor_numerically(q), str(q))

    def test_characteristic_polynomials_factor_consistently(self):
        from sylvester.eigen import characteristic_polynomial
        from .support import random_square

        rng = seeded(103)
        for _ in range(120):
            poly = characteristic_polynomial(random_square(rng, rng.randint(1, 5), -4, 4))
            unit, factors = factor(poly)
            self.assertEqual(unit, 1)
            self.assertEqual(expand(unit, factors), poly)

    def test_squarefree_decomposition(self):
        x = Poly([0, 1])
        poly = x ** 2 * Poly([-1, 1]) ** 3 * Poly([1, 0, 1])
        parts = {m: q for q, m in squarefree_decomposition(poly)}
        self.assertEqual(parts[1], Poly([1, 0, 1]))
        self.assertEqual(parts[2], x)
        self.assertEqual(parts[3], Poly([-1, 1]))

    def test_rational_coefficients_and_constants(self):
        unit, factors = factor(Poly([Fraction(1, 2), Fraction(-3, 2), 1]))
        self.assertEqual(sorted(q.c for q, _ in factors), [(-1, 1), (-1, 2)])
        self.assertEqual(expand(unit, factors), Poly([Fraction(1, 2), Fraction(-3, 2), 1]))
        self.assertEqual(factor(Poly([7])), (7, []))
        self.assertTrue(is_irreducible(Poly([1, 0, 0, 0, 1])))
        self.assertFalse(is_irreducible(Poly([4, 0, 0, 0, 1])))


class NumberFieldTest(unittest.TestCase):
    def setUp(self):
        self.field = NumberField([1, -2, -1, 1])
        self.rng = seeded(104)
        self.roots = approximate_roots(Poly(self.field.modulus))[0]

    def element(self):
        return self.field.element([Fraction(self.rng.randint(-6, 6), self.rng.randint(1, 5)) for _ in range(3)])

    def test_field_axioms(self):
        for _ in range(300):
            a, b, c = self.element(), self.element(), self.element()
            self.assertEqual((a * b) * c, a * (b * c))
            self.assertEqual(a * (b + c), a * b + a * c)
            self.assertEqual(a + b - b, a)
            if a:
                self.assertEqual(a * (1 / a), 1)
            if b:
                self.assertEqual((a / b) * b, a)

    def test_arithmetic_agrees_with_every_embedding(self):
        def value(x, root):
            return x.at(root) if isinstance(x, AlgebraicNumber) else float(x)

        for _ in range(200):
            a, b = self.element(), self.element()
            for root in self.roots:
                self.assertAlmostEqual(value(a * b, root), value(a, root) * value(b, root), places=6)
                if b:
                    self.assertAlmostEqual(value(a / b, root), value(a, root) / value(b, root), places=5)

    def test_generator_satisfies_the_modulus(self):
        alpha = self.field.generator
        self.assertEqual(alpha ** 3 - alpha ** 2 - 2 * alpha + 1, 0)
        self.assertNotEqual(alpha ** 3 - 2 * alpha ** 2 - alpha + 1, 0)
        self.assertEqual(alpha ** -1 * alpha, 1)

    def test_constants_demote_to_rationals(self):
        alpha = self.field.generator
        self.assertIsInstance(alpha - alpha, Fraction)
        self.assertIsInstance(alpha * 0, Fraction)
        self.assertIsInstance(self.field.element([5]), Fraction)
        self.assertEqual(alpha + 1 - alpha, 1)

    def test_mixing_fields_is_refused(self):
        other = NumberField([-2, 0, 0, 1]).generator
        with self.assertRaises(ArithmeticError):
            self.field.generator + other

    def test_a_reducible_modulus_reports_its_factor(self):
        field = NumberField([6, 0, -5, 0, 1])
        with self.assertRaises(ReducibleModulus) as caught:
            (field.generator ** 2 - 2).inverse()
        self.assertEqual(caught.exception.factor, (Fraction(-2), Fraction(0), Fraction(1)))


class SturmTest(unittest.TestCase):
    def test_known_real_root_counts(self):
        for coeffs, real in [((1, -3, 0, 1), 3), ((-2, 0, 0, 1), 1), ((1, 0, 0, 0, 1), 0),
                             ((1, 0, -10, 0, 1), 4), ((-1, -1, 1), 2), ((1, 1, 1, 1, 1), 0)]:
            poly = Poly(coeffs)
            self.assertEqual(count_real_roots(poly), real)
            self.assertEqual(len(isolate_real_roots(poly)), real)

    def test_isolation_brackets_known_roots(self):
        rng = seeded(105)
        for _ in range(200):
            roots = list(dict.fromkeys(Fraction(rng.randint(-20, 20), rng.randint(1, 4))
                                       for _ in range(rng.randint(1, 5))))
            poly = from_roots(roots) * Poly([rng.randint(1, 5), 0, 1])
            intervals = isolate_real_roots(poly)
            self.assertEqual(len(intervals), len(roots))
            for (a, b), r in zip(intervals, sorted(roots)):
                self.assertTrue(a < r <= b or a == r == b)

    def test_approximations_are_roots(self):
        rng = seeded(106)
        for _ in range(150):
            poly = Poly([rng.randint(-9, 9) for _ in range(rng.randint(3, 7))] + [rng.randint(1, 3)])
            for q, _ in factor(poly)[1]:
                if q.degree < 2:
                    continue
                real, complex_roots = approximate_roots(q)
                self.assertEqual(len(real), count_real_roots(q))
                self.assertEqual(len(real) + len(complex_roots), q.degree)
                for r in real + complex_roots:
                    value = sum(complex(c) * r ** i for i, c in enumerate(q.c))
                    scale = sum(abs(complex(c)) * abs(r) ** i for i, c in enumerate(q.c))
                    self.assertLessEqual(abs(value), 1e-9 * scale)


if __name__ == "__main__":
    unittest.main()
