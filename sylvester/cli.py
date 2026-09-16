from __future__ import annotations

import argparse
import sys

from . import proof, report
from .expr import evaluate
from .parse import ParseError, parse_augmented, parse_matrix, parse_vector
from .render import autodetect, matrix_str, set_ascii, set_color
from .reduce import HUMAN, MACHINE
from .samples import CLO, by_clo, get
from . import vectors as V

PROGRAM = "sylvester"
TAGLINE = "an exact linear algebra workbench for the terminal"


def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--no-color", action="store_true", help="disable ANSI color")
    common.add_argument("--ascii", action="store_true", help="use ASCII instead of box drawing")
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=TAGLINE,
        epilog="run with no arguments to open the terminal interface",
        parents=[common],
    )
    sub_raw = parser.add_subparsers(dest="command")

    class sub:
        @staticmethod
        def add_parser(name, **kwargs):
            kwargs["parents"] = [common]
            return sub_raw.add_parser(name, **kwargs)

    def matrix_arg(p, flag="-m", name="matrix", required=True, help_text="rows separated by ';'"):
        p.add_argument(flag, "--" + name, required=required, help=help_text)

    p = sub.add_parser("rref", help="row reduce to REF and RREF, showing every operation")
    matrix_arg(p)
    p.add_argument("--mode", choices=[HUMAN, MACHINE], default=HUMAN)
    p.add_argument("--augmented", action="store_true", help="treat the last column as constants")
    p.add_argument("--no-steps", action="store_true")
    p.add_argument("--compare", action="store_true", help="contrast the human and machine routes")

    p = sub.add_parser("solve", help="solve a linear system and classify the solution set")
    matrix_arg(p)
    p.add_argument("-b", "--constants", help="constants vector; omit if -m is the augmented matrix")
    p.add_argument("--mode", choices=[HUMAN, MACHINE], default=HUMAN)
    p.add_argument("--method", choices=["reduce", "cramer", "homogeneous"], default="reduce")
    p.add_argument("--no-steps", action="store_true")

    p = sub.add_parser("det", help="determinant by row operations and cofactor expansion")
    matrix_arg(p)
    p.add_argument("--method", choices=["both", "rowops", "cofactor"], default="both")
    p.add_argument("--no-steps", action="store_true")

    p = sub.add_parser("inverse", help="inverse by Gauss-Jordan, adjugate, elementary matrices, or LU")
    matrix_arg(p)
    p.add_argument("--method", choices=["gauss", "adjugate", "elementary", "lu"], default="gauss")
    p.add_argument("--no-steps", action="store_true")

    p = sub.add_parser("subspaces", help="the four fundamental subspaces, rank and nullity")
    matrix_arg(p)

    p = sub.add_parser("properties", help="a summary of the matrix properties")
    matrix_arg(p)

    p = sub.add_parser("eigen", help="eigenvalues, eigenvectors, eigenspaces and diagonalization")
    matrix_arg(p)
    p.add_argument("--steps", action="store_true", help="show the reduction inside each eigenspace")

    p = sub.add_parser("independence", help="decide independence of the columns, with a witness")
    matrix_arg(p)

    p = sub.add_parser("basis", help="extract a basis from the columns as a spanning set")
    matrix_arg(p)

    p = sub.add_parser("span", help="decide whether b is in the span of the columns")
    matrix_arg(p)
    p.add_argument("-b", "--target", required=True, help="the target vector")

    p = sub.add_parser("vectors", help="inner product, norms, angle, projection and the inequalities")
    p.add_argument("-u", required=True, help="first vector")
    p.add_argument("-v", required=True, help="second vector")

    p = sub.add_parser("gram-schmidt", help="orthogonalize the columns")
    matrix_arg(p)

    p = sub.add_parser("project", help="project a vector onto the span of the columns")
    matrix_arg(p)
    p.add_argument("-v", "--vector", required=True, help="the vector to project")

    p = sub.add_parser("prove", help="verify a proposition and print a cited certificate")
    p.add_argument("proposition", nargs="?", help="proposition key; omit to list them")
    p.add_argument("operands", nargs="*", help="matrices, vectors or scalars, in order")
    p.add_argument("--list", action="store_true", help="list every proposition")

    p = sub.add_parser("eval", help="evaluate a matrix expression")
    p.add_argument("expression")
    p.add_argument("-m", "--matrix", action="append", default=[],
                   help="define a register, as NAME=1 2; 3 4 (repeatable)")

    p = sub.add_parser("samples", help="list or run the built-in problems")
    p.add_argument("key", nargs="?", help="sample key to run")
    p.add_argument("--clo", type=int, help="only show samples for one course outcome")

    sub.add_parser("outcomes", help="show the course outcomes and what covers each")
    sub.add_parser("tui", help="open the terminal interface")
    return parser


def _matrix(text):
    return parse_matrix(text)


def _vector(text):
    return parse_vector(text)


def run(args):
    command = args.command
    if command == "rref":
        m = _matrix(args.matrix)
        split = m.ncols - 1 if args.augmented and m.ncols > 1 else None
        if args.compare:
            return report.compare_strategies(m, split, not args.no_steps)
        return report.rref_report(m, args.mode, split, not args.no_steps)

    if command == "solve":
        if args.constants:
            a, b = _matrix(args.matrix), _vector(args.constants)
        else:
            full, width = parse_augmented(args.matrix)
            a, b = full.left(width), full.column_at(width)
        if args.method == "homogeneous":
            return report.homogeneous_text(a)
        if args.method == "cramer":
            return report.cramer_text(a, b)
        return report.system_report(a, b, args.mode, not args.no_steps)

    if command == "det":
        return report.determinant_report(_matrix(args.matrix), args.method, not args.no_steps)

    if command == "inverse":
        return report.inverse_report(_matrix(args.matrix), args.method, not args.no_steps)

    if command == "subspaces":
        return report.subspaces_report(_matrix(args.matrix))

    if command == "properties":
        return report.properties_report(_matrix(args.matrix))

    if command == "eigen":
        return report.eigen_report(_matrix(args.matrix), args.steps)

    if command == "independence":
        return report.independence_report(_matrix(args.matrix).columns())

    if command == "basis":
        return report.basis_report(_matrix(args.matrix).columns())

    if command == "span":
        return report.span_report(_vector(args.target), _matrix(args.matrix).columns())

    if command == "vectors":
        return report.vector_report(_vector(args.u), _vector(args.v))

    if command == "gram-schmidt":
        return report.gram_schmidt_report(_matrix(args.matrix).columns())

    if command == "project":
        return report.projection_report(_vector(args.vector), _matrix(args.matrix).columns())

    if command == "prove":
        return run_prove(args)

    if command == "eval":
        return run_eval(args)

    if command == "samples":
        return run_samples(args)

    if command == "outcomes":
        return outcomes_text()

    return None


def run_prove(args):
    if args.list or not args.proposition:
        out = []
        for category, props in proof.by_category().items():
            out.append(category)
            for p in props:
                out.append("  %-34s %s" % (p.key, p.statement))
                out.append("  %-34s %s" % ("", "needs: " + ", ".join(p.operands)))
            out.append("")
        return "\n".join(out)
    prop = proof.REGISTRY.get(args.proposition)
    if prop is None:
        return "no proposition called %r; run 'prove --list' to see them all" % args.proposition
    if len(args.operands) != len(prop.operands):
        return ("%s needs %d operand(s): %s"
                % (prop.key, len(prop.operands), ", ".join(prop.operands)))
    operands = []
    for spec, text in zip(prop.operands, args.operands):
        if spec == proof.VECTOR:
            operands.append(_vector(text))
        elif spec == proof.SCALAR:
            from .parse import parse_value

            operands.append(parse_value(text))
        else:
            operands.append(_matrix(text))
    return report.certificate_report(proof.check(prop.key, *operands))


def run_eval(args):
    registers = {}
    for item in args.matrix:
        name, _, body = item.partition("=")
        if not body:
            raise ParseError("use NAME=1 2; 3 4 to define a register")
        registers[name.strip().upper()] = parse_matrix(body)
    value = evaluate(args.expression, registers)
    from .matrix import Matrix

    if isinstance(value, Matrix):
        return matrix_str(value)
    return report.sval(value)


def run_samples(args):
    if args.key:
        sample = get(args.key)
        if sample is None:
            return "no sample called %r; run 'samples' to list them" % args.key
        return sample_text(sample)
    out = []
    groups = by_clo()
    for clo in sorted(CLO):
        if args.clo and clo != args.clo:
            continue
        out.append("CLO %d  %s" % (clo, CLO[clo]))
        for s in groups.get(clo, []):
            out.append("   %-22s %s" % (s.key, s.title))
        out.append("")
    return "\n".join(out)


def sample_text(sample):
    out = ["CLO %d  %s" % (sample.clo, CLO[sample.clo]), "", sample.title, sample.note, ""]
    a = sample.registers.get("A")
    b = sample.registers.get("B")
    u = sample.registers.get("U")
    v = sample.registers.get("V")
    screen = sample.screen
    if screen in ("system",) and a is not None and b is not None:
        out.append(report.system_report(a, V.as_vector(b), HUMAN, True))
    elif screen == "homogeneous" and a is not None:
        out.append(report.homogeneous_text(a))
    elif screen == "cramer" and a is not None and b is not None:
        out.append(report.cramer_text(a, V.as_vector(b)))
    elif screen == "reduce" and a is not None:
        out.append(report.compare_strategies(a, None, True))
    elif screen == "determinant" and a is not None:
        out.append(report.determinant_report(a, "both", True))
    elif screen == "inverse" and a is not None:
        out.append(report.inverse_report(a, "gauss", True))
    elif screen == "elementary" and a is not None:
        out.append(report.inverse_report(a, "elementary", True))
    elif screen == "subspaces" and a is not None:
        out.append(report.subspaces_report(a))
    elif screen == "independence" and a is not None:
        out.append(report.independence_report(a.columns()))
    elif screen == "basis" and a is not None:
        out.append(report.basis_report(a.columns()))
    elif screen == "vectors" and u is not None and v is not None:
        out.append(report.vector_report(V.as_vector(u), V.as_vector(v)))
    elif screen == "gramschmidt" and a is not None:
        out.append(report.gram_schmidt_report(a.columns()))
    elif screen == "projection" and a is not None and v is not None:
        out.append(report.projection_report(V.as_vector(v), a.columns()))
    elif screen == "eigen" and a is not None:
        out.append(report.eigen_report(a, False))
    elif screen == "proof":
        key = {"adjugate": "adjugate-identity", "imt": "invertible-matrix-theorem",
               "det-multiplicative": "det-product"}[sample.key]
        prop = proof.REGISTRY[key]
        operands = [sample.registers[n] for n in ("A", "B")[:len(prop.operands)]]
        out.append(report.certificate_report(proof.check(key, *operands)))
    return "\n".join(out)


def outcomes_text():
    from .render import heading

    out = [heading("SJSU linear algebra course outcomes", 72), ""]
    coverage = {
        1: "solve, outcomes in the Theorem lab, rank versus rank of the augmented matrix",
        2: "rref, inverse (gauss / adjugate / elementary / lu)",
        3: "det --method rowops and det --method cofactor",
        4: "prove: determinant and matrix propositions, the Invertible Matrix Theorem",
        5: "vectors, gram-schmidt, project, span, basis, independence",
        6: "solve --method cramer and solve --method homogeneous",
        7: "subspaces: column space, row space, null space, left null space, rank-nullity",
        8: "eigen: characteristic polynomial, eigenspaces, diagonalization, spectral theorem",
    }
    groups = by_clo()
    for clo in sorted(CLO):
        out.append("CLO %d" % clo)
        out.append("  %s" % CLO[clo])
        out.append("  covered by: %s" % coverage[clo])
        out.append("  samples:    %s" % ", ".join(s.key for s in groups.get(clo, [])))
        out.append("")
    return "\n".join(out)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    autodetect()
    if args.no_color:
        set_color(False)
    if args.ascii:
        set_ascii(True)
    if args.command in (None, "tui"):
        from .tui.app import main as tui_main

        return tui_main()
    try:
        output = run(args)
    except ParseError as exc:
        print("could not read that input: %s" % exc, file=sys.stderr)
        return 2
    except (ValueError, ArithmeticError, ZeroDivisionError) as exc:
        print("%s" % exc, file=sys.stderr)
        return 1
    if output is None:
        parser.print_help()
        return 1
    print(output)
    return 0
