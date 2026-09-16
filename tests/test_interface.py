import io
import re
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction

from sylvester import cli, report
from sylvester.expr import evaluate
from sylvester.matrix import Matrix
from sylvester.parse import ParseError, parse_augmented, parse_matrix, parse_value, parse_vector
from sylvester.render import slice_visible, visible_len
from sylvester.samples import SAMPLES
from sylvester.tui import keys as K
from sylvester.tui.app import App
from sylvester.tui.screen import _fit
from sylvester.tui.widgets import MatrixEditor, Menu, Pager, Prompt

from .support import random_matrix, random_square, random_vector, seeded

ANSI = re.compile(r"\x1b\[[0-9;]*m")


class ParseTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(parse_value("3"), 3)
        self.assertEqual(parse_value("-2/3"), Fraction(-2, 3))
        self.assertEqual(parse_value("1.5"), Fraction(3, 2))
        self.assertEqual(parse_value("2e1"), 20)
        self.assertEqual(parse_value("--3"), 3)
        self.assertAlmostEqual(float(parse_value("sqrt(8)")), 8 ** 0.5)

    def test_bad_values(self):
        for bad in ("", "abc", "1/0", "1/2/3"):
            with self.assertRaises(ParseError):
                parse_value(bad)

    def test_matrix_forms(self):
        expected = Matrix([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(parse_matrix("1 2 3; 4 5 6"), expected)
        self.assertEqual(parse_matrix("1,2,3;4,5,6"), expected)
        self.assertEqual(parse_matrix("[1 2 3][4 5 6]".replace("][", ";")), expected)
        self.assertEqual(parse_matrix("1 2 3\n4 5 6"), expected)

    def test_ragged_matrix_is_reported(self):
        with self.assertRaises(ParseError):
            parse_matrix("1 2 3; 4 5")

    def test_vectors_and_augmented(self):
        self.assertEqual(parse_vector("(1, 2, 3)"), (1, 2, 3))
        matrix, width = parse_augmented("1 2 | 3; 4 5 | 6")
        self.assertEqual(width, 2)
        self.assertEqual(matrix, Matrix([[1, 2, 3], [4, 5, 6]]))
        matrix, width = parse_augmented("1 2 3; 4 5 6")
        self.assertEqual(width, 2)

    def test_roundtrip_through_the_formatter(self):
        from sylvester.parse import format_matrix_input

        rng = seeded(91)
        for _ in range(200):
            m = Matrix([[Fraction(rng.randint(-9, 9), rng.randint(1, 5)) for _ in range(3)]
                        for _ in range(2)])
            self.assertEqual(parse_matrix(format_matrix_input(m)), m)


class ExpressionTest(unittest.TestCase):
    def setUp(self):
        self.registers = {
            "A": Matrix([[1, 2], [3, 4]]),
            "B": Matrix([[0, 1], [1, 0]]),
            "U": Matrix.column([1, 2, 2]),
            "W": Matrix.column([3, 0, 4]),
        }

    def evaluate(self, source):
        return evaluate(source, self.registers)

    def test_arithmetic(self):
        a, b = self.registers["A"], self.registers["B"]
        self.assertEqual(self.evaluate("A*B"), a.matmul(b))
        self.assertEqual(self.evaluate("A+B"), a + b)
        self.assertEqual(self.evaluate("A-B"), a - b)
        self.assertEqual(self.evaluate("2A"), a * 2)
        self.assertEqual(self.evaluate("A^2"), a.matmul(a))
        self.assertEqual(self.evaluate("A^0"), Matrix.identity(2))
        self.assertEqual(self.evaluate("-A"), -a)
        self.assertEqual(self.evaluate("A/2"), a * Fraction(1, 2))
        self.assertEqual(self.evaluate("3/4*A"), a * Fraction(3, 4))

    def test_transpose_and_inverse(self):
        a = self.registers["A"]
        self.assertEqual(self.evaluate("A^T"), a.T)
        self.assertEqual(self.evaluate("A'"), a.T)
        self.assertEqual(self.evaluate("transpose(A)"), a.T)
        self.assertTrue(self.evaluate("A^-1*A").is_identity())
        self.assertTrue(self.evaluate("inv(A)*A").is_identity())
        self.assertEqual(self.evaluate("A/B"), a.matmul(b_inv(self.registers["B"])))

    def test_functions(self):
        self.assertEqual(self.evaluate("det(A)"), -2)
        self.assertEqual(self.evaluate("rank(A)"), 2)
        self.assertEqual(self.evaluate("tr(A)"), 5)
        self.assertEqual(self.evaluate("nullity([1,2;2,4])"), 1)
        self.assertEqual(self.evaluate("I(3)"), Matrix.identity(3))
        self.assertEqual(self.evaluate("zeros(2,3)"), Matrix.zeros(2, 3))
        self.assertEqual(self.evaluate("diag(1,2,3)"), Matrix.diagonal([1, 2, 3]))
        self.assertEqual(self.evaluate("dot(U,W)"), 11)
        self.assertEqual(self.evaluate("norm(U)"), 3)
        self.assertEqual(self.evaluate("rref(A)"), Matrix.identity(2))
        self.assertAlmostEqual(float(self.evaluate("sqrt(8)")), 8 ** 0.5)

    def test_literals_and_grouping(self):
        self.assertEqual(self.evaluate("[1,2;3,4]"), Matrix([[1, 2], [3, 4]]))
        self.assertEqual(self.evaluate("[1,2;3,4]*[1;1]"), Matrix([[3], [7]]))
        self.assertEqual(self.evaluate("(A+B)*(A-B)"),
                         (self.registers["A"] + self.registers["B"]).matmul(
                             self.registers["A"] - self.registers["B"]))

    def test_identity_holds_through_the_evaluator(self):
        self.assertEqual(self.evaluate("det(A*B) - det(A)*det(B)"), 0)
        self.assertEqual(self.evaluate("(A*B)^T - B^T*A^T"), Matrix.zeros(2, 2))

    def test_errors_are_reported(self):
        for bad in ("A+1", "Q", "A*", "foo(A)", "A^(1/2)", "1/0", "A^A", "[", "det()"):
            with self.assertRaises(Exception, msg=bad):
                self.evaluate(bad)


def b_inv(m):
    from sylvester.inverse import inverse

    return inverse(m)


class ReportTest(unittest.TestCase):
    def test_every_report_renders_for_many_shapes(self):
        rng = seeded(92)
        for _ in range(40):
            nrows, ncols = rng.randint(1, 4), rng.randint(1, 4)
            a = random_matrix(rng, nrows, ncols, -5, 5)
            square = random_square(rng, rng.randint(1, 4), -5, 5)
            b = random_vector(rng, nrows, -5, 5)
            u = random_vector(rng, 3, -5, 5)
            v = random_vector(rng, 3, -5, 5)
            texts = [
                report.rref_report(a, show_steps=True),
                report.compare_strategies(a),
                report.system_report(a, b),
                report.determinant_report(square),
                report.inverse_report(square, "gauss"),
                report.inverse_report(square, "adjugate"),
                report.inverse_report(square, "elementary"),
                report.inverse_report(a, "lu"),
                report.subspaces_report(a),
                report.properties_report(square),
                report.independence_report(a.columns()),
                report.basis_report(a.columns()),
                report.span_report(b, a.columns()),
                report.vector_report(u, v),
                report.gram_schmidt_report([u, v]),
                report.projection_report(u, [v]),
                report.eigen_report(square),
                report.homogeneous_text(a),
                report.cramer_text(square, random_vector(rng, square.nrows, -5, 5)),
            ]
            for text in texts:
                self.assertTrue(text.strip())

    def test_reports_state_the_right_conclusion(self):
        text = report.system_report(Matrix([[1, 1], [1, 1]]), (Fraction(1), Fraction(2)))
        self.assertIn("inconsistent", ANSI.sub("", text))
        text = report.system_report(Matrix([[1, 0], [0, 1]]), (Fraction(1), Fraction(2)))
        self.assertIn("Exactly one solution", ANSI.sub("", text))
        text = report.system_report(Matrix([[1, 1]]), (Fraction(2),))
        self.assertIn("Infinitely many", ANSI.sub("", text))
        text = report.determinant_report(Matrix([[1, 2], [2, 4]]))
        self.assertIn("singular", ANSI.sub("", text))
        text = report.inverse_report(Matrix([[1, 2], [2, 4]]), "gauss")
        self.assertIn("no inverse", ANSI.sub("", text))

    def test_eigen_report_covers_the_interesting_cases(self):
        for matrix, expected in [
            (Matrix([[1, 1], [0, 1]]), ["not diagonalizable"]),
            (Matrix([[2, 1], [1, 2]]), ["diagonalizable", "Spectral theorem"]),
            (Matrix([[1, 1], [1, 0]]), ["sqrt5", "Spectral theorem"]),
            (Matrix([[0, -1], [1, 0]]), ["i", "over the complex numbers"]),
            (Matrix([[1, 1, 0], [1, 0, 1], [0, 1, 0]]),
             ["each root of L^3 - L^2 - 2L + 1", "(irreducible)", "(L^2 - 1, L, 1)", "all 3 real",
              "Sturm", "conjugate eigenvalues"]),
            (Matrix([[0, 0, 2], [1, 0, 0], [0, 1, 0]]),
             ["1 real, 2 complex", "(L^2, L, 1)", "over the complex numbers", "w1(L1)"]),
            (Matrix([[5, 1, 3, -1], [1, 5, -1, 3], [3, -1, -3, 1], [-1, 3, 1, -3]]),
             ["4sqrt2", "2 + 2sqrt5", "different minimal polynomials", "AP = PD"]),
        ]:
            text = ANSI.sub("", report.eigen_report(matrix, show_steps=True))
            for fragment in expected:
                self.assertIn(fragment, text, (matrix.to_lists(), fragment))
            for fallback in ("known numerically", "reported numerically", "not comparable", "degree above 2"):
                self.assertNotIn(fallback, text)

    def test_parametric_entry_and_vector_form(self):
        from sylvester.solve import solve

        solution = solve(Matrix([[1, 1, 1]]), (Fraction(3),))
        text = ANSI.sub("", report.system_report(Matrix([[1, 1, 1]]), (Fraction(3),)))
        self.assertIn("t1", text)
        self.assertIn("x_p", text)
        self.assertEqual(report.parametric_entry(solution, 1), "t1")


class CliTest(unittest.TestCase):
    def run_cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = cli.main(argv + ["--no-color", "--ascii"])
        return code, ANSI.sub("", out.getvalue()), err.getvalue()

    def test_commands_succeed(self):
        matrix = "1 2 3; 0 1 4; 5 6 0"
        cases = [
            (["rref", "-m", matrix], "Reduced row echelon form"),
            (["rref", "-m", matrix, "--compare"], "human"),
            (["rref", "-m", matrix, "--mode", "machine", "--no-steps"], "RREF"),
            (["solve", "-m", matrix, "-b", "1 2 3"], "solution"),
            (["solve", "-m", "1 2 | 3; 4 5 | 6"], "solution"),
            (["solve", "-m", matrix, "--method", "cramer"], "CRAMER"),
            (["solve", "-m", matrix, "--method", "homogeneous"], "HOMOGENEOUS"),
            (["det", "-m", matrix], "det(A)"),
            (["det", "-m", matrix, "--method", "cofactor"], "expand along"),
            (["inverse", "-m", matrix], "A inverse"),
            (["inverse", "-m", matrix, "--method", "adjugate"], "adj(A)"),
            (["inverse", "-m", matrix, "--method", "elementary"], "elementary"),
            (["inverse", "-m", matrix, "--method", "lu"], "PA = LU"),
            (["subspaces", "-m", matrix], "FUNDAMENTAL"),
            (["properties", "-m", matrix], "determinant"),
            (["eigen", "-m", "2 1; 1 2"], "Eigenvalues"),
            (["independence", "-m", "1 2; 2 4"], "dependent"),
            (["basis", "-m", "1 2; 2 4"], "basis"),
            (["span", "-m", "1 0; 0 1", "-b", "2 3"], "in the span"),
            (["vectors", "-u", "1 2 2", "-v", "3 0 4"], "Cauchy"),
            (["gram-schmidt", "-m", "1 1; 1 0"], "orthogonal"),
            (["project", "-m", "1 0; 0 1; 0 0", "-v", "1 2 3"], "proj"),
            (["prove", "--list"], "det-product"),
            (["prove", "det-product", "1 2; 3 4", "0 1; 1 0"], "VERIFIED"),
            (["prove", "cauchy-schwarz", "1 2", "3 4"], "VERIFIED"),
            (["eval", "det([1,2;3,4])"], "-2"),
            (["eval", "A*A", "-m", "A=1 2; 3 4"], "7"),
            (["samples"], "CLO 1"),
            (["samples", "--clo", "8"], "eigen"),
            (["outcomes"], "CLO 8"),
        ]
        for argv, expected in cases:
            code, out, err = self.run_cli(argv)
            self.assertEqual(code, 0, (argv, err))
            self.assertIn(expected, out, argv)

    def test_every_sample_runs_through_the_cli(self):
        for sample in SAMPLES:
            code, out, err = self.run_cli(["samples", sample.key])
            self.assertEqual(code, 0, (sample.key, err))
            self.assertIn(sample.title, out)
            self.assertGreater(len(out.splitlines()), 8, sample.key)

    def test_bad_input_is_reported_not_raised(self):
        code, out, err = self.run_cli(["det", "-m", "1 2; 3"])
        self.assertEqual(code, 2)
        self.assertIn("could not read", err)
        code, out, err = self.run_cli(["det", "-m", "1 2 3"])
        self.assertEqual(code, 1)
        self.assertIn("square", err)

    def test_unknown_proposition(self):
        code, out, _ = self.run_cli(["prove", "nope", "1 2; 3 4"])
        self.assertEqual(code, 0)
        self.assertIn("no proposition", out)


class WidgetTest(unittest.TestCase):
    def test_menu_navigation_and_shortcuts(self):
        menu = Menu([("a", "Alpha", ""), ("b", "Beta", ""), ("c", "Gamma", "")])
        self.assertEqual(menu.index, 0)
        menu.handle(K.DOWN)
        self.assertEqual(menu.index, 1)
        menu.handle(K.UP)
        menu.handle(K.UP)
        self.assertEqual(menu.index, 2)
        menu.handle(K.HOME)
        self.assertEqual(menu.index, 0)
        self.assertEqual(menu.handle("c"), "select")
        self.assertEqual(menu.index, 2)
        self.assertEqual(menu.handle(K.ENTER), "select")

    def test_pager_scrolls_within_bounds(self):
        pager = Pager("\n".join("line %d" % i for i in range(100)))
        pager.handle(K.PGDN, 10)
        self.assertEqual(pager.offset, 10)
        pager.handle(K.END, 10)
        self.assertEqual(pager.offset, 90)
        pager.handle(K.DOWN, 10)
        self.assertEqual(pager.offset, 90)
        pager.handle(K.HOME, 10)
        self.assertEqual(pager.offset, 0)
        pager.handle(K.UP, 10)
        self.assertEqual(pager.offset, 0)
        self.assertEqual(len(pager.view(80, 10)), 10)

    def test_pager_pans_horizontally(self):
        pager = Pager("x" * 200)
        pager.handle(K.RIGHT, 10, 80)
        self.assertEqual(pager.hoffset, 8)
        pager.handle(K.LEFT, 10, 80)
        self.assertEqual(pager.hoffset, 0)
        pager.handle(K.LEFT, 10, 80)
        self.assertEqual(pager.hoffset, 0)

    def test_matrix_editor_edits_and_parses(self):
        editor = MatrixEditor(Matrix([[1, 2], [3, 4]]))
        editor.handle("5")
        self.assertEqual(editor.matrix()[0, 0], 5)
        editor.handle(K.RIGHT)
        editor.handle("-")
        editor.handle("1")
        editor.handle("/")
        editor.handle("2")
        self.assertEqual(editor.matrix()[0, 1], Fraction(-1, 2))
        editor.handle("+")
        self.assertEqual(editor.nrows, 3)
        editor.handle("]")
        self.assertEqual(editor.ncols, 3)
        editor.handle("[")
        editor.handle("_")
        self.assertEqual((editor.nrows, editor.ncols), (2, 2))
        editor.handle("i")
        self.assertTrue(editor.matrix().is_identity())
        editor.handle("z")
        self.assertTrue(editor.matrix().is_zero())

    def test_matrix_editor_reports_a_bad_cell(self):
        editor = MatrixEditor(Matrix([[1]]))
        editor.cells[0][0] = "zzz"
        with self.assertRaises(ParseError):
            editor.matrix()

    def test_prompt_editing(self):
        prompt = Prompt(">")
        for ch in "abc":
            prompt.handle(ch)
        self.assertEqual(prompt.value, "abc")
        prompt.handle(K.BACKSPACE)
        self.assertEqual(prompt.value, "ab")
        prompt.handle(K.LEFT)
        prompt.handle("X")
        self.assertEqual(prompt.value, "aXb")
        self.assertEqual(prompt.handle(K.ENTER), "submit")
        self.assertEqual(prompt.submit(), "aXb")
        self.assertEqual(prompt.value, "")

    def test_ansi_aware_measurement(self):
        painted = "\033[91mred\033[0m"
        self.assertEqual(visible_len(painted), 3)
        self.assertEqual(visible_len(slice_visible(painted, 1, 2)), 2)
        self.assertEqual(visible_len(_fit("abcdef", 3)), 3)


class FakeScreen:
    def __init__(self, width=100, height=32):
        self.width, self.height, self.frames = width, height, []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def size(self):
        return self.width, self.height

    def render(self, lines, cursor=None):
        self.frames.append([_fit(line, self.width) for line in lines[:self.height]])


class Stop(Exception):
    pass


class FakeKeyboard:
    def __init__(self, script):
        self.script = list(script)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        if not self.script:
            raise Stop()
        return self.script.pop(0)


def drive(script):
    app = App()
    screen = FakeScreen()
    try:
        app.run(screen, FakeKeyboard(script))
    except Stop:
        pass
    return app, screen


class TuiTest(unittest.TestCase):
    def assertRenders(self, script, expected=()):
        app, screen = drive(script)
        self.assertTrue(screen.frames)
        last = ANSI.sub("", "\n".join(screen.frames[-1]))
        for text in expected:
            self.assertIn(text, last, script)
        for frame in screen.frames:
            for line in frame:
                self.assertLessEqual(visible_len(line), screen.width)
        return last

    def test_home(self):
        self.assertRenders([], ["SYLVESTER", "Row reduction", "Theorem lab"])

    def test_every_area_opens(self):
        for shortcut, expected in [
            ("1", "Contents"),
            ("2", "ROW REDUCTION"),
            ("3", "SOLVING A LINEAR SYSTEM"),
            ("4", "DETERMINANT"),
            ("5", "INVERSE"),
            ("6", "FUNDAMENTAL SUBSPACES"),
            ("7", "VECTORS"),
            ("8", "EIGENVALUES"),
            ("9", "Systems of linear equations"),
            ("0", "workbench"),
            ("s", "CLO 1"),
            ("?", "NAVIGATION"),
        ]:
            self.assertRenders([shortcut], [expected])

    def test_report_toggles_do_not_crash(self):
        self.assertRenders(["2", "m", "s", "a", "c", "r"])
        self.assertRenders(["3", "v", "v", "v", "s", "b"])
        self.assertRenders(["4", "v", "v", "v", "s"])
        self.assertRenders(["5", "v", "v", "v", "v", "s"])
        self.assertRenders(["6", "v", "v", "v", "v"])
        self.assertRenders(["7", "v", "v", "v", "v", "b"])
        self.assertRenders(["8", "s", "r"])

    def test_scrolling(self):
        self.assertRenders(["2", K.PGDN, K.PGDN, K.END, K.HOME, K.DOWN, K.UP, K.RIGHT, K.LEFT])

    def test_editor_round_trip(self):
        app, _ = drive(["1", K.ENTER, "9", K.ENTER, "w"])
        self.assertEqual(app.registers["A"][0, 0], 9)

    def test_editor_cancel_keeps_the_old_value(self):
        app, _ = drive(["1", K.ENTER, "9", K.ESC])
        self.assertEqual(app.registers["A"][0, 0], 2)

    def test_new_and_copy_registers(self):
        app, _ = drive(["1", "n", "Z", K.ENTER, "w"])
        self.assertIn("Z", app.registers)
        app, _ = drive(["1", "c", "Y", K.ENTER])
        self.assertEqual(app.registers["Y"], app.registers["A"])

    def test_theorem_lab_produces_a_certificate(self):
        self.assertRenders(["9", K.ENTER, K.ENTER], ["Statement", "VERIFIED"])

    def test_certificate_operand_cycling(self):
        self.assertRenders(["9", K.ENTER, K.ENTER, "1", "2"])

    def test_workbench_evaluates_and_reports_errors(self):
        last = self.assertRenders(["0"] + list("det(A)") + [K.ENTER])
        self.assertIn("det(A)", last)
        last = self.assertRenders(["0"] + list("Q+1") + [K.ENTER])
        self.assertIn("nothing stored", last)

    def test_workbench_stores_a_result(self):
        app, _ = drive(["0"] + list("A*A") + [K.ENTER, "=", "M", K.ENTER])
        self.assertIn("M", app.registers)
        self.assertEqual(app.registers["M"], app.registers["A"].matmul(app.registers["A"]))

    def test_loading_every_sample_from_the_browser(self):
        for index in range(len(SAMPLES)):
            script = ["s"] + [K.DOWN] * index + [K.ENTER]
            self.assertRenders(script)

    def test_quit_and_back(self):
        app, _ = drive(["2", K.ESC, "q"])
        self.assertFalse(app.running)
        app, _ = drive(["q"])
        self.assertFalse(app.running)

    def test_q_is_text_inside_the_workbench(self):
        app, _ = drive(["0", "q"])
        self.assertTrue(app.running)

    def test_deep_navigation(self):
        self.assertRenders(
            ["9", K.DOWN, K.ENTER, K.DOWN, K.ENTER, K.ESC, K.ESC, K.ESC,
             "8", "r", K.ESC, "7", K.ESC, "s", K.ESC, "?"])


if __name__ == "__main__":
    unittest.main()
