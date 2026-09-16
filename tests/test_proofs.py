import unittest
from fractions import Fraction

from sylvester import proof, report
from sylvester.matrix import Matrix
from sylvester.samples import CLO, SAMPLES, by_clo

from .support import random_matrix, random_square, random_vector, seeded


def operands_for(prop, rng):
    n = rng.randint(2, 4)
    out = []
    shape = None
    for spec in prop.operands:
        if spec == proof.SQUARE:
            out.append(random_square(rng, n, -4, 4))
        elif spec == proof.MATRIX:
            if shape is None:
                shape = (rng.randint(1, 4), rng.randint(1, 4))
                out.append(random_matrix(rng, shape[0], shape[1], -4, 4))
            else:
                out.append(random_matrix(rng, shape[0], shape[1], -4, 4))
        elif spec == proof.VECTOR:
            rows = out[0].nrows if out and isinstance(out[0], Matrix) else n
            out.append(random_vector(rng, rows, -4, 4))
        else:
            out.append(Fraction(rng.randint(-3, 3)))
    return out


class RegistryTest(unittest.TestCase):
    def test_registry_is_well_formed(self):
        self.assertEqual(len(proof.REGISTRY), len(proof.ORDER))
        self.assertGreaterEqual(len(proof.ORDER), 39)
        for key in proof.ORDER:
            prop = proof.REGISTRY[key]
            self.assertEqual(prop.key, key)
            self.assertTrue(prop.title)
            self.assertTrue(prop.statement)
            self.assertTrue(prop.citation)
            self.assertTrue(prop.operands)
            self.assertIn(prop.kind, (proof.PROOF, proof.INSTANCE))
            self.assertIn(prop.category, (proof.SYSTEMS, proof.MATRICES, proof.SPACES, proof.EIGEN))

    def test_every_category_is_populated(self):
        groups = proof.by_category()
        self.assertEqual(len(groups), 4)
        for name, props in groups.items():
            self.assertGreaterEqual(len(props), 4, name)


class PropositionTest(unittest.TestCase):
    def test_no_proposition_finds_a_counterexample(self):
        rng = seeded(81)
        failures = []
        verified = 0
        for key in proof.ORDER:
            prop = proof.REGISTRY[key]
            for _ in range(60):
                operands = operands_for(prop, rng)
                certificate = proof.check(key, *operands)
                if certificate.holds is False:
                    failures.append((key, [o.to_lists() if isinstance(o, Matrix) else o
                                           for o in operands]))
                elif certificate.holds:
                    verified += 1
        self.assertEqual(failures, [])
        self.assertGreater(verified, 1500)

    def test_certificates_carry_their_reasoning(self):
        rng = seeded(82)
        for key in proof.ORDER:
            prop = proof.REGISTRY[key]
            certificate = proof.check(key, *operands_for(prop, rng))
            self.assertEqual(certificate.key, key)
            self.assertTrue(certificate.conclusion)
            self.assertTrue(certificate.citation)
            self.assertTrue(certificate.lines)

    def test_bad_operands_do_not_crash(self):
        wide = Matrix([[1, 2, 3], [4, 5, 6]])
        for key in proof.ORDER:
            prop = proof.REGISTRY[key]
            operands = []
            for spec in prop.operands:
                if spec == proof.SCALAR:
                    operands.append(Fraction(2))
                elif spec == proof.VECTOR:
                    operands.append((Fraction(1),))
                else:
                    operands.append(wide)
            certificate = proof.check(key, *operands)
            self.assertIn(certificate.holds, (True, False, None))
            self.assertIsNot(certificate.holds, False)

    def test_known_outcomes(self):
        singular = Matrix([[1, 2], [2, 4]])
        certificate = proof.check("invertible-matrix-theorem", singular)
        self.assertTrue(certificate.holds)
        self.assertIn("fail together", certificate.conclusion)

        invertible = Matrix([[2, 1], [1, 1]])
        certificate = proof.check("invertible-matrix-theorem", invertible)
        self.assertIn("hold together", certificate.conclusion)

        dependent = Matrix([[1, 2], [2, 4]])
        certificate = proof.check("linear-independence", dependent)
        self.assertIn("free", certificate.conclusion)

        independent = Matrix([[1, 0], [0, 1]])
        certificate = proof.check("linear-independence", independent)
        self.assertIn("pivot", certificate.conclusion)

    def test_cauchy_schwarz_equality_is_detected(self):
        u = (Fraction(1), Fraction(2))
        certificate = proof.check("cauchy-schwarz", u, (Fraction(2), Fraction(4)))
        self.assertIn("parallel", certificate.conclusion)

    def test_every_certificate_renders(self):
        rng = seeded(83)
        for key in proof.ORDER:
            prop = proof.REGISTRY[key]
            text = report.certificate_report(proof.check(key, *operands_for(prop, rng)))
            self.assertTrue(text.strip())
            self.assertIn(prop.statement, text)


class SampleTest(unittest.TestCase):
    def test_every_outcome_has_samples(self):
        groups = by_clo()
        self.assertEqual(sorted(groups), sorted(CLO))
        for clo in CLO:
            self.assertGreaterEqual(len(groups[clo]), 3, clo)

    def test_sample_metadata(self):
        keys = set()
        for sample in SAMPLES:
            self.assertNotIn(sample.key, keys)
            keys.add(sample.key)
            self.assertIn(sample.clo, CLO)
            self.assertTrue(sample.title)
            self.assertTrue(sample.note)
            self.assertTrue(sample.registers)
            for name, value in sample.registers.items():
                self.assertTrue(len(name) == 1 and name.isupper(), name)
                self.assertIsInstance(value, Matrix)

    def test_samples_behave_as_described(self):
        from sylvester.determinant import determinant
        from sylvester.eigen import spectrum
        from sylvester.solve import INCONSISTENT, INFINITE, UNIQUE, solve
        from sylvester import vectors as V

        def registers(key):
            return {k: v for k, v in next(s for s in SAMPLES if s.key == key).registers.items()}

        r = registers("sys-inconsistent")
        self.assertEqual(solve(r["A"], V.as_vector(r["B"])).kind, INCONSISTENT)
        r = registers("sys-unique")
        self.assertEqual(solve(r["A"], V.as_vector(r["B"])).kind, UNIQUE)
        r = registers("sys-infinite")
        self.assertEqual(solve(r["A"], V.as_vector(r["B"])).kind, INFINITE)
        self.assertEqual(determinant(registers("inverse-singular")["A"]), 0)
        self.assertNotEqual(determinant(registers("inverse-3x3")["A"]), 0)
        self.assertTrue(registers("det-triangular")["A"].is_triangular())
        self.assertFalse(spectrum(registers("eigen-defective")["A"]).diagonalizable)
        self.assertTrue(spectrum(registers("eigen-3x3-repeated")["A"]).diagonalizable)
        self.assertFalse(spectrum(registers("eigen-complex")["A"]).real_only)
        self.assertTrue(registers("eigen-symmetric")["A"].is_symmetric())
        self.assertEqual(determinant(registers("cramer-blocked")["A"]), 0)


if __name__ == "__main__":
    unittest.main()
