import math
import unittest
from fractions import Fraction

from sylvester.exact import Surd
from sylvester.matrix import Matrix
from sylvester.reduce import row_reduce
from sylvester.subspace import (basis_from_spanning_set, change_of_basis, coordinates,
                                dimension_of_span, extend_to_basis, four_subspaces, in_span,
                                independence, is_subspace_of, nullspace_is_subspace,
                                rank_nullity, same_span)
from sylvester import vectors as V

from .support import random_matrix, random_vector, seeded


class IndependenceTest(unittest.TestCase):
    def test_witness_is_a_genuine_dependency(self):
        rng = seeded(51)
        dependent = 0
        for _ in range(600):
            n, k = rng.randint(1, 4), rng.randint(1, 5)
            vectors = [random_vector(rng, n, -4, 4) for _ in range(k)]
            result = independence(vectors)
            self.assertTrue(result.verify(), vectors)
            self.assertEqual(result.independent, result.rank == k)
            if not result.independent:
                dependent += 1
                combo = [Fraction(0)] * n
                for coefficient, v in zip(result.relation, vectors):
                    combo = [a + coefficient * b for a, b in zip(combo, v)]
                self.assertTrue(all(x == 0 for x in combo))
                self.assertTrue(any(result.relation))
        self.assertGreater(dependent, 0)

    def test_more_vectors_than_dimensions_is_always_dependent(self):
        rng = seeded(52)
        for _ in range(200):
            n = rng.randint(1, 4)
            vectors = [random_vector(rng, n) for _ in range(n + rng.randint(1, 3))]
            self.assertFalse(independence(vectors).independent)

    def test_standard_basis_is_independent(self):
        for n in range(1, 6):
            self.assertTrue(independence(V.standard_basis(n)).independent)

    def test_a_set_containing_zero_is_dependent(self):
        zero = tuple(Fraction(0) for _ in range(3))
        self.assertFalse(independence([(Fraction(1), Fraction(0), Fraction(0)), zero]).independent)


class SpanTest(unittest.TestCase):
    def test_membership_and_coefficients(self):
        rng = seeded(53)
        inside = outside = 0
        for _ in range(600):
            n, k = rng.randint(1, 4), rng.randint(1, 4)
            vectors = [random_vector(rng, n, -4, 4) for _ in range(k)]
            target = random_vector(rng, n, -4, 4)
            result = in_span(target, vectors)
            self.assertTrue(result.verify())
            if result.inside:
                inside += 1
                combo = [Fraction(0)] * n
                for coefficient, v in zip(result.coefficients, vectors):
                    combo = [a + coefficient * b for a, b in zip(combo, v)]
                self.assertEqual(tuple(combo), target)
            else:
                outside += 1
        self.assertGreater(inside, 0)
        self.assertGreater(outside, 0)

    def test_every_spanning_vector_is_in_its_own_span(self):
        rng = seeded(54)
        for _ in range(200):
            vectors = [random_vector(rng, 3) for _ in range(3)]
            for v in vectors:
                self.assertTrue(in_span(v, vectors).inside)

    def test_coordinates_are_unique_against_a_basis(self):
        rng = seeded(55)
        for _ in range(200):
            n = rng.randint(1, 4)
            basis = None
            for _ in range(40):
                candidate = [random_vector(rng, n) for _ in range(n)]
                if independence(candidate).independent:
                    basis = candidate
                    break
            if basis is None:
                continue
            target = random_vector(rng, n)
            result = coordinates(target, basis)
            self.assertTrue(result.inside)
            self.assertTrue(result.unique)

    def test_empty_span_holds_only_zero(self):
        self.assertTrue(in_span((Fraction(0), Fraction(0)), []).inside)
        self.assertFalse(in_span((Fraction(1), Fraction(0)), []).inside)


class BasisTest(unittest.TestCase):
    def test_extraction_keeps_the_span_and_is_independent(self):
        rng = seeded(56)
        for _ in range(400):
            n, k = rng.randint(1, 4), rng.randint(1, 5)
            vectors = [random_vector(rng, n, -4, 4) for _ in range(k)]
            extraction = basis_from_spanning_set(vectors)
            self.assertEqual(extraction.dimension, dimension_of_span(vectors))
            if extraction.basis:
                self.assertTrue(independence(extraction.basis).independent)
                self.assertTrue(same_span(extraction.basis, vectors))
            for free, terms in extraction.expressions.items():
                combo = [Fraction(0)] * n
                for column, coefficient in terms:
                    combo = [a + coefficient * b for a, b in zip(combo, vectors[column])]
                self.assertEqual(tuple(combo), vectors[free])

    def test_extending_to_a_basis(self):
        rng = seeded(57)
        for _ in range(200):
            n = rng.randint(1, 4)
            start = [random_vector(rng, n) for _ in range(rng.randint(0, n))]
            if start and not independence(start).independent:
                continue
            full, added = extend_to_basis(start, n)
            self.assertEqual(len(full), n)
            self.assertTrue(independence(full).independent)
            self.assertEqual(full[:len(start)], start)

    def test_change_of_basis(self):
        rng = seeded(58)
        for _ in range(120):
            n = rng.randint(1, 3)
            source = target = None
            for _ in range(40):
                candidate = [random_vector(rng, n) for _ in range(n)]
                if independence(candidate).independent:
                    if source is None:
                        source = candidate
                    else:
                        target = candidate
                        break
            if source is None or target is None:
                continue
            result = change_of_basis(source, target)
            self.assertTrue(result.ok)
            for index, v in enumerate(source):
                rebuilt = [Fraction(0)] * n
                for coefficient, t in zip(result.coordinates[index], target):
                    rebuilt = [a + coefficient * b for a, b in zip(rebuilt, t)]
                self.assertEqual(tuple(rebuilt), v)


class FourSubspacesTest(unittest.TestCase):
    def test_dimensions_and_orthogonality(self):
        rng = seeded(59)
        for _ in range(500):
            nrows, ncols = rng.randint(1, 4), rng.randint(1, 4)
            a = random_matrix(rng, nrows, ncols, -5, 5)
            f = four_subspaces(a)
            self.assertTrue(f.verify(), a.to_lists())
            self.assertEqual(len(f.column_space), f.rank)
            self.assertEqual(len(f.row_space), f.rank)
            self.assertEqual(len(f.null_space), ncols - f.rank)
            self.assertEqual(len(f.left_null_space), nrows - f.rank)
            for v in f.null_space:
                for w in f.row_space:
                    self.assertEqual(V.dot(v, w), 0)
            for v in f.left_null_space:
                for w in f.column_space:
                    self.assertEqual(V.dot(v, w), 0)

    def test_rank_nullity(self):
        rng = seeded(60)
        for _ in range(400):
            ncols = rng.randint(1, 5)
            a = random_matrix(rng, rng.randint(1, 4), ncols)
            rank, nullity, n = rank_nullity(a)
            self.assertEqual(rank + nullity, n)
            self.assertEqual(n, ncols)

    def test_row_rank_equals_column_rank(self):
        rng = seeded(61)
        for _ in range(300):
            a = random_matrix(rng, rng.randint(1, 4), rng.randint(1, 4))
            self.assertEqual(row_reduce(a).rank, row_reduce(a.T).rank)

    def test_null_space_is_a_subspace(self):
        rng = seeded(62)
        for _ in range(200):
            a = random_matrix(rng, rng.randint(1, 3), rng.randint(1, 4))
            self.assertTrue(nullspace_is_subspace(a).is_subspace)

    def test_column_space_contains_every_column(self):
        rng = seeded(63)
        for _ in range(200):
            a = random_matrix(rng, rng.randint(1, 4), rng.randint(1, 4))
            f = four_subspaces(a)
            if not f.column_space:
                continue
            self.assertTrue(is_subspace_of(a.columns(), f.column_space))


class VectorTest(unittest.TestCase):
    def test_inner_product_is_bilinear_and_symmetric(self):
        rng = seeded(64)
        for _ in range(400):
            n = rng.randint(1, 5)
            u, v, w = (random_vector(rng, n) for _ in range(3))
            k = Fraction(rng.randint(-4, 4))
            self.assertEqual(V.dot(u, v), V.dot(v, u))
            self.assertEqual(V.dot(V.add(u, v), w), V.dot(u, w) + V.dot(v, w))
            self.assertEqual(V.dot(V.scale(k, u), v), k * V.dot(u, v))
            self.assertGreaterEqual(V.norm_squared(u), 0)

    def test_norm_is_exact_when_it_can_be(self):
        self.assertEqual(V.norm((Fraction(3), Fraction(4))), 5)
        self.assertEqual(V.norm((Fraction(1), Fraction(2), Fraction(2))), 3)
        self.assertIsInstance(V.norm((Fraction(1), Fraction(1))), Surd)
        self.assertAlmostEqual(float(V.norm((Fraction(1), Fraction(1)))), math.sqrt(2))

    def test_the_standard_inequalities(self):
        rng = seeded(65)
        orthogonal_seen = 0
        for _ in range(800):
            n = rng.randint(1, 5)
            u, v = random_vector(rng, n), random_vector(rng, n)
            laws = V.InnerProductLaws(u, v)
            self.assertTrue(laws.cauchy_holds)
            self.assertTrue(laws.triangle_holds)
            self.assertTrue(laws.parallelogram_holds)
            if laws.orthogonal:
                orthogonal_seen += 1
                self.assertTrue(laws.pythagoras_holds)
        self.assertGreater(orthogonal_seen, 0)

    def test_cauchy_schwarz_is_tight_for_parallel_vectors(self):
        u = (Fraction(1), Fraction(2), Fraction(3))
        v = V.scale(Fraction(5, 2), u)
        self.assertEqual(V.InnerProductLaws(u, v).cauchy_gap, 0)

    def test_projection_splits_the_vector(self):
        rng = seeded(66)
        for _ in range(400):
            n = rng.randint(2, 4)
            v = random_vector(rng, n)
            onto = random_vector(rng, n)
            if V.is_zero(onto):
                continue
            projection = V.project_onto_vector(v, onto)
            self.assertTrue(projection.verify())
            self.assertEqual(V.add(projection.parallel, projection.perpendicular), v)
            self.assertEqual(V.dot(projection.perpendicular, onto), 0)

    def test_gram_schmidt_orthogonalizes_and_keeps_the_span(self):
        rng = seeded(67)
        dropped_seen = 0
        for _ in range(500):
            n, k = rng.randint(2, 4), rng.randint(1, 4)
            vectors = [random_vector(rng, n, -4, 4) for _ in range(k)]
            result = V.gram_schmidt(vectors)
            self.assertTrue(result.verify())
            dropped_seen += bool(result.dropped)
            kept = [v for i, v in enumerate(vectors) if i not in result.dropped]
            if result.vectors:
                self.assertTrue(same_span(result.vectors, kept))
                self.assertTrue(independence(result.vectors).independent)
        self.assertGreater(dropped_seen, 0)

    def test_orthonormalize_produces_unit_vectors(self):
        result = V.orthonormalize([(Fraction(3), Fraction(0)), (Fraction(1), Fraction(2))])
        for q in result.normalized:
            self.assertEqual(V.norm_squared(q), 1)

    def test_projection_onto_a_subspace_minimizes_distance(self):
        rng = seeded(68)
        for _ in range(300):
            n = rng.randint(2, 4)
            basis = [random_vector(rng, n, -4, 4) for _ in range(rng.randint(1, n))]
            target = random_vector(rng, n, -4, 4)
            projection = V.project_onto_subspace(target, basis)
            self.assertTrue(projection.verify())
            best = V.norm_squared(projection.perpendicular)
            for _ in range(10):
                point = tuple(Fraction(0) for _ in range(n))
                for b in basis:
                    point = V.add(point, V.scale(Fraction(rng.randint(-3, 3)), b))
                self.assertLessEqual(best, V.norm_squared(V.subtract(target, point)))

    def test_orthogonal_complement(self):
        rng = seeded(69)
        for _ in range(200):
            n = rng.randint(2, 4)
            vectors = [random_vector(rng, n) for _ in range(rng.randint(1, 3))]
            complement = V.orthogonal_complement(vectors)
            for c in complement:
                for v in vectors:
                    self.assertEqual(V.dot(c, v), 0)

    def test_cross_product(self):
        rng = seeded(70)
        for _ in range(200):
            u, v = random_vector(rng, 3), random_vector(rng, 3)
            c = V.cross(u, v)
            self.assertEqual(V.dot(c, u), 0)
            self.assertEqual(V.dot(c, v), 0)
            self.assertEqual(V.cross(v, u), V.negate(c))
        with self.assertRaises(ValueError):
            V.cross((Fraction(1), Fraction(2)), (Fraction(1), Fraction(2)))

    def test_angle_of_known_pairs(self):
        self.assertAlmostEqual(
            V.angle_degrees((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1))), 90.0)
        self.assertAlmostEqual(
            V.angle_degrees((Fraction(1), Fraction(0)), (Fraction(1), Fraction(1))), 45.0)

    def test_zero_vector_guards(self):
        zero = (Fraction(0), Fraction(0))
        with self.assertRaises(ValueError):
            V.normalize(zero)
        with self.assertRaises(ValueError):
            V.project_onto_vector((Fraction(1), Fraction(1)), zero)
        with self.assertRaises(ValueError):
            V.angle_degrees(zero, (Fraction(1), Fraction(1)))

    def test_as_vector_accepts_rows_and_columns(self):
        self.assertEqual(V.as_vector(Matrix.column([1, 2, 3])), (1, 2, 3))
        self.assertEqual(V.as_vector(Matrix.row_vector([1, 2, 3])), (1, 2, 3))
        with self.assertRaises(ValueError):
            V.as_vector(Matrix([[1, 2], [3, 4]]))


if __name__ == "__main__":
    unittest.main()
