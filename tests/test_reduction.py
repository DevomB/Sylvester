import unittest
from fractions import Fraction

from sylvester.exact import surd
from sylvester.matrix import Matrix
from sylvester.reduce import (HUMAN, MACHINE, apply_op, nullspace_basis, op_to_elementary,
                              row_reduce)
from sylvester.solve import (INCONSISTENT, INFINITE, UNIQUE, cramer_report, homogeneous_report,
                             solve, solve_augmented, solve_homogeneous)
from sylvester.determinant import determinant

from .support import is_rref, naive_determinant, random_matrix, random_rational_matrix, random_square, seeded


class RowReductionTest(unittest.TestCase):
    def test_both_strategies_reach_the_same_rref(self):
        rng = seeded(11)
        for trial in range(1500):
            nrows, ncols = rng.randint(1, 4), rng.randint(1, 5)
            m = random_matrix(rng, nrows, ncols, -9, 9)
            limit = ncols - 1 if (ncols > 1 and trial % 2) else ncols
            human = row_reduce(m, limit, HUMAN, canonical_tail=True)
            machine = row_reduce(m, limit, MACHINE, canonical_tail=True)
            self.assertEqual(human.rref, machine.rref, m.to_lists())
            self.assertEqual(human.rank, machine.rank)

    def test_result_really_is_in_reduced_form(self):
        rng = seeded(12)
        for _ in range(600):
            m = random_matrix(rng, rng.randint(1, 4), rng.randint(1, 4))
            for mode in (HUMAN, MACHINE):
                self.assertTrue(is_rref(row_reduce(m, mode=mode).rref), m.to_lists())

    def test_rational_entries(self):
        rng = seeded(13)
        for _ in range(300):
            m = random_rational_matrix(rng, rng.randint(1, 3), rng.randint(1, 4))
            self.assertEqual(row_reduce(m, mode=HUMAN).rref, row_reduce(m, mode=MACHINE).rref)

    def test_rank_plus_nullity(self):
        rng = seeded(14)
        for _ in range(400):
            ncols = rng.randint(1, 5)
            m = random_matrix(rng, rng.randint(1, 4), ncols)
            reduction = row_reduce(m)
            self.assertEqual(reduction.rank + reduction.nullity, ncols)

    def test_steps_replay_to_the_final_matrix(self):
        rng = seeded(15)
        for _ in range(300):
            m = random_matrix(rng, rng.randint(1, 4), rng.randint(1, 4))
            for mode in (HUMAN, MACHINE):
                reduction = row_reduce(m, mode=mode)
                current = m
                for step in reduction.steps:
                    if step.op:
                        current = apply_op(current, step.op)
                    self.assertEqual(current, step.matrix)
                self.assertEqual(current, reduction.rref)

    def test_every_operation_is_an_invertible_matrix_acting_on_the_left(self):
        rng = seeded(16)
        for _ in range(200):
            n = rng.randint(1, 4)
            m = random_square(rng, n)
            reduction = row_reduce(m, mode=MACHINE)
            accumulated = Matrix.identity(n)
            for op in reduction.elementary_ops():
                accumulated = op_to_elementary(op, n).matmul(accumulated)
            self.assertEqual(accumulated.matmul(m), reduction.rref)

    def test_determinant_factor_is_tracked(self):
        rng = seeded(17)
        for _ in range(400):
            n = rng.randint(1, 4)
            m = random_square(rng, n)
            for mode in (HUMAN, MACHINE):
                reduction = row_reduce(m, mode=mode, reduced=False)
                product = Fraction(1)
                for i in range(n):
                    product *= reduction.ref[i][i]
                self.assertEqual(product, reduction.factor * naive_determinant(m))

    def test_human_route_meets_fewer_fractions(self):
        rng = seeded(18)
        human_hits = machine_hits = 0
        for _ in range(600):
            m = random_matrix(rng, 3, 4, -9, 9)
            human_hits += row_reduce(m, mode=HUMAN).fraction_steps > 0
            machine_hits += row_reduce(m, mode=MACHINE).fraction_steps > 0
        self.assertLess(human_hits, machine_hits)

    def test_null_space_basis_is_annihilated(self):
        rng = seeded(19)
        for _ in range(500):
            m = random_matrix(rng, rng.randint(1, 4), rng.randint(1, 5))
            basis = nullspace_basis(m)
            self.assertEqual(len(basis), m.ncols - row_reduce(m).rank)
            for v in basis:
                self.assertTrue(m.matmul(Matrix.column(v)).is_zero())

    def test_works_over_a_quadratic_field(self):
        m = Matrix([[surd(1, 1, 5), 2], [3, surd(0, 1, 5)]])
        reduction = row_reduce(m)
        self.assertEqual(reduction.rank, 2)
        self.assertTrue(reduction.rref.is_identity())

    def test_empty_and_degenerate(self):
        self.assertEqual(row_reduce(Matrix([])).rank, 0)
        self.assertEqual(row_reduce(Matrix([[0, 0], [0, 0]])).rank, 0)
        self.assertTrue(row_reduce(Matrix([[0, 0]])).rref.is_zero())


class SolveTest(unittest.TestCase):
    def test_classification_and_verification(self):
        rng = seeded(21)
        seen = set()
        for _ in range(1200):
            nrows, ncols = rng.randint(1, 4), rng.randint(1, 4)
            a = random_matrix(rng, nrows, ncols, -5, 5)
            b = tuple(Fraction(rng.randint(-5, 5)) for _ in range(nrows))
            solution = solve(a, b)
            seen.add(solution.kind)
            self.assertTrue(solution.verify())
            self.assertEqual(solution.rank + solution.nullity, ncols)
            if solution.consistent:
                self.assertEqual(solution.rank_augmented, solution.rank)
                self.assertEqual(len(solution.homogeneous_basis), solution.nullity)
            else:
                self.assertEqual(solution.rank_augmented, solution.rank + 1)
                self.assertIsNotNone(solution.witness_row)
        self.assertEqual(seen, {UNIQUE, INFINITE, INCONSISTENT})

    def test_general_solution_satisfies_the_system_for_any_parameters(self):
        rng = seeded(22)
        for _ in range(500):
            nrows, ncols = rng.randint(1, 4), rng.randint(1, 4)
            a = random_matrix(rng, nrows, ncols, -5, 5)
            b = tuple(Fraction(rng.randint(-5, 5)) for _ in range(nrows))
            solution = solve(a, b)
            if not solution.consistent:
                continue
            target = Matrix.column(b)
            for _ in range(4):
                params = [Fraction(rng.randint(-4, 4)) for _ in solution.homogeneous_basis]
                point = solution.at(params)
                self.assertEqual(a.matmul(Matrix.column(point)), target)

    def test_both_strategies_agree_on_the_solution(self):
        rng = seeded(23)
        for _ in range(400):
            n = rng.randint(1, 4)
            a = random_matrix(rng, n, n, -5, 5)
            b = tuple(Fraction(rng.randint(-5, 5)) for _ in range(n))
            human = solve(a, b, HUMAN)
            machine = solve(a, b, MACHINE)
            self.assertEqual(human.kind, machine.kind)
            self.assertEqual(human.particular, machine.particular)
            self.assertEqual(human.homogeneous_basis, machine.homogeneous_basis)

    def test_augmented_entry_point(self):
        solution = solve_augmented(Matrix([[1, 1, 2], [1, -1, 0]]))
        self.assertEqual(solution.kind, UNIQUE)
        self.assertEqual(solution.particular, (1, 1))

    def test_homogeneous_is_nontrivial_exactly_when_singular(self):
        rng = seeded(24)
        for _ in range(400):
            n = rng.randint(1, 4)
            a = random_square(rng, n, -4, 4)
            report = homogeneous_report(a)
            self.assertEqual(report.nontrivial, determinant(a) == 0)
            self.assertTrue(report.solution.verify())

    def test_wide_homogeneous_always_has_a_nontrivial_solution(self):
        rng = seeded(25)
        for _ in range(200):
            nrows = rng.randint(1, 3)
            a = random_matrix(rng, nrows, nrows + rng.randint(1, 3))
            self.assertGreater(solve_homogeneous(a).nullity, 0)

    def test_cramer_matches_row_reduction(self):
        rng = seeded(26)
        used = 0
        for _ in range(400):
            n = rng.randint(1, 4)
            a = random_square(rng, n, -4, 4)
            b = tuple(Fraction(rng.randint(-4, 4)) for _ in range(n))
            report = cramer_report(a, b)
            if not report.usable:
                self.assertFalse(determinant(a))
                continue
            used += 1
            solution = solve(a, b)
            self.assertEqual(solution.kind, UNIQUE)
            self.assertEqual(tuple(report.solution), solution.particular)
            self.assertEqual(len(report.numerators), n)
        self.assertGreater(used, 100)

    def test_cramer_refuses_non_square(self):
        self.assertFalse(cramer_report(Matrix([[1, 2, 3]]), (Fraction(1),)).usable)

    def test_mismatched_constants_are_rejected(self):
        with self.assertRaises(ValueError):
            solve(Matrix([[1, 2], [3, 4]]), [Fraction(1)])


if __name__ == "__main__":
    unittest.main()
