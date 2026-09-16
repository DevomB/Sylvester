from __future__ import annotations

from fractions import Fraction

from .determinant import (cofactor_matrix, det_by_row_reduction, determinant,
                          expand_cofactors)
from .eigen import diagonalize, orthogonally_diagonalize, spectrum
from .exact import ONE, Surd
from .inverse import (elementary_factorization, inverse_adjugate, inverse_gauss_jordan,
                      lu_decomposition)
from .matrix import Matrix
from .reduce import HUMAN, MACHINE, row_reduce
from .render import (C, bullet, check, cross, dim, fmt, fmt_coeff, heading, lam, matrix_str,
                     paint, rule, side_by_side, subheading, table, tuple_str)
from .solve import INCONSISTENT, UNIQUE, cramer_report, homogeneous_report, solve
from .subspace import basis_from_spanning_set, four_subspaces, in_span, independence
from . import vectors as V

WIDTH = 72


def sval(value):
    body = fmt(value) if not isinstance(value, complex) else "%.6f%+.6fi" % (value.real, value.imag)
    if isinstance(value, Surd):
        a = value.approx()
        body += dim("  ~ %s" % ("%.6f" % a if not isinstance(a, complex) else "%.4f%+.4fi" % (a.real, a.imag)))
    return body


def block(label, matrix, split_at=None, indent="  "):
    out = [subheading(label)] if label else []
    out.append(matrix_str(matrix, split_at, indent))
    return "\n".join(out)


def vec_line(label, values):
    return "%s = %s" % (label, tuple_str(values))


def steps_block(reduction, split_at=None, show_notes=True):
    out = []
    for i, step in enumerate(reduction.steps):
        head = "Step %d:  %s" % (i, step.description) if i else step.description
        out.append(paint(head, C.BLUE, C.BOLD))
        if show_notes and step.note:
            out.append(dim("   " + step.note))
        out.append(matrix_str(step.matrix, split_at, "   "))
        out.append("")
    return "\n".join(out)


def work_summary(reduction):
    return table(
        [
            ["row operations", reduction.operation_count],
            ["swaps / scalings / eliminations",
             "%d / %d / %d" % (reduction.swaps, reduction.scalings, reduction.eliminations)],
            ["steps containing a fraction", "%d of %d" % (reduction.fraction_steps, len(reduction.steps))],
            ["worst denominator reached", reduction.worst_denominator],
            ["largest number seen", reduction.largest_entry],
        ]
    )


def rref_report(matrix, mode=HUMAN, split_at=None, show_steps=True):
    red = row_reduce(matrix, split_at, mode, canonical_tail=split_at is not None)
    out = [heading("row reduction  (%s strategy)" % mode, WIDTH), ""]
    if show_steps:
        out.append(steps_block(red, split_at))
    out.append(block("Row echelon form (REF)", red.ref, split_at))
    out.append("")
    out.append(block("Reduced row echelon form (RREF)", red.rref, split_at))
    out.append("")
    out.append(subheading("Reading it off"))
    out.append(table([
        ["rank", red.rank],
        ["nullity", red.nullity],
        ["pivot columns", ", ".join(str(c + 1) for c in red.pivot_columns) or "none"],
        ["free columns", ", ".join(str(c + 1) for c in red.free_columns) or "none"],
    ]))
    out.append("")
    out.append(subheading("Work done"))
    out.append(work_summary(red))
    return "\n".join(out)


def compare_strategies(matrix, split_at=None, show_steps=False):
    human = row_reduce(matrix, split_at, HUMAN, canonical_tail=split_at is not None)
    machine = row_reduce(matrix, split_at, MACHINE, canonical_tail=split_at is not None)
    out = [heading("human route versus machine route", WIDTH), ""]
    out.append(
        "Machine takes the largest pivot and scales it to 1 straight away, which is right\n"
        "in floating point. Working exactly, that first division drops a fraction into the\n"
        "matrix and every later step drags it along. Human hunts for a pivot of 1, refuses\n"
        "to divide until the end, and cancels common factors as they appear."
    )
    out.append("")
    if show_steps:
        out.append(paint("HUMAN", C.GREEN, C.BOLD))
        out.append(steps_block(human, split_at))
        out.append(paint("MACHINE", C.YELLOW, C.BOLD))
        out.append(steps_block(machine, split_at))
    out.append(side_by_side(
        [matrix_str(human.rref, split_at), matrix_str(machine.rref, split_at)],
        gap=6, labels=["human RREF", "machine RREF"],
    ))
    out.append("")
    rows = [
        ["row operations", human.operation_count, machine.operation_count],
        ["steps with a fraction", human.fraction_steps, machine.fraction_steps],
        ["worst denominator", human.worst_denominator, machine.worst_denominator],
        ["largest number", human.largest_entry, machine.largest_entry],
    ]
    out.append(table(rows, headers=["", "human", "machine"]))
    out.append("")
    same = human.rref == machine.rref
    out.append("%s both routes reach the same RREF, as they must: it is unique." % (check() if same else cross()))
    if human.fraction_steps < machine.fraction_steps:
        out.append("The human route stayed on whole numbers longer, which is the whole point of it.")
    elif human.operation_count > machine.operation_count:
        out.append("The machine route was shorter here; human traded steps for easier arithmetic.")
    else:
        out.append("Both routes cost about the same on this matrix.")
    return "\n".join(out)


def system_report(coefficients, constants, mode=HUMAN, show_steps=True):
    sol = solve(coefficients, constants, mode)
    n = sol.num_vars
    out = [heading("solving a linear system", WIDTH), ""]
    out.append(subheading("The system"))
    out.append(equations_text(coefficients, constants))
    out.append("")
    out.append(block("Augmented matrix [A | b]", sol.augmented, n))
    out.append("")
    if show_steps:
        out.append(steps_block(sol.reduction, n))
    out.append(block("RREF", sol.reduction.rref, n))
    out.append("")
    out.append(subheading("Consistency"))
    out.append(table([
        ["rank(A)", sol.rank],
        ["rank([A|b])", sol.rank_augmented],
        ["unknowns n", n],
        ["nullity", sol.nullity],
    ]))
    out.append("")
    if sol.kind == INCONSISTENT:
        out.append(paint("No solution: the system is inconsistent.", C.RED, C.BOLD))
        out.append("Row %d of the RREF reads 0 = 1, which no choice of x can satisfy." % (sol.witness_row + 1))
        out.append("rank(A) = %d but rank([A|b]) = %d, so by Rouche-Capelli the system is inconsistent."
                   % (sol.rank, sol.rank_augmented))
        return "\n".join(out)
    if sol.kind == UNIQUE:
        out.append(paint("Exactly one solution.", C.GREEN, C.BOLD))
        out.append(table([["x%d" % (i + 1), sval(v)] for i, v in enumerate(sol.particular)]))
        out.append("")
        out.append("rank(A) = rank([A|b]) = n = %d, so the solution is unique." % n)
        out.append("%s checked: A x = b exactly." % check())
        return "\n".join(out)
    out.append(paint("Infinitely many solutions.", C.GREEN, C.BOLD))
    params = sol.parameters
    out.append("Free variables: " + ", ".join(
        "x%d = %s" % (c + 1, p) for c, p in zip(sol.free_columns, params)))
    out.append("")
    out.append(subheading("General solution, one variable at a time"))
    out.append(table([[ "x%d" % (i + 1), parametric_entry(sol, i)] for i in range(n)]))
    out.append("")
    out.append(subheading("General solution, in vector form"))
    out.append(vector_form(sol))
    out.append("")
    out.append("The solution set is a %d-dimensional %s of R^%d: one particular solution"
               % (sol.nullity, "plane" if sol.nullity == 2 else "line" if sol.nullity == 1 else "flat", n))
    out.append("plus the whole null space of A.")
    out.append("%s checked: A x_p = b, and A v = 0 for every basis vector v." % check())
    return "\n".join(out)


def parametric_entry(sol, index):
    if index in sol.free_columns:
        return sol.parameters[sol.free_columns.index(index)]
    terms = []
    base = sol.particular[index]
    if base or not any(v[index] for v in sol.homogeneous_basis):
        terms.append(fmt(base))
    for p, vecs in zip(sol.parameters, sol.homogeneous_basis):
        k = vecs[index]
        if not k:
            continue
        mag = abs(k)
        body = p if mag == 1 else "%s%s" % (fmt(mag), p)
        if not terms:
            terms.append(body if k > 0 else "-" + body)
        else:
            terms.append((" + " if k > 0 else " - ") + body)
    return "".join(terms) or "0"


def vector_form(sol):
    blocks = [matrix_str([[v] for v in sol.particular])]
    labels = ["x_p"]
    for p, vecs in zip(sol.parameters, sol.homogeneous_basis):
        blocks.append(matrix_str([[v] for v in vecs]))
        labels.append("%s %s" % (p, dim("x")))
    body = side_by_side(blocks, gap=3, labels=labels)
    return "x  =  " + "\n      ".join(body.splitlines())


def equations_text(coefficients, constants):
    lines = []
    for r in range(coefficients.nrows):
        terms = []
        for c in range(coefficients.ncols):
            k = coefficients[r, c]
            if not k:
                continue
            body = "x%d" % (c + 1) if abs(k) == 1 else "%sx%d" % (fmt(abs(k)), c + 1)
            if not terms:
                terms.append(body if k > 0 else "-" + body)
            else:
                terms.append((" + " if k > 0 else " - ") + body)
        lines.append("  " + ("".join(terms) or "0") + " = " + fmt(constants[r]))
    return "\n".join(lines)


def determinant_report(matrix, method="both", show_steps=True):
    out = [heading("determinant", WIDTH), "", block("A", matrix), ""]
    value = determinant(matrix)
    if matrix.is_triangular():
        diag = [matrix[i, i] for i in range(matrix.nrows)]
        out.append("A is triangular, so det(A) is just the product of the diagonal:")
        out.append("  " + (" %s " % dim("x")).join(fmt(v) for v in diag) + "  =  " + sval(value))
        out.append("")
    if method in ("rowops", "both"):
        out.append(subheading("By elementary row operations"))
        out.append(row_op_determinant_text(matrix, show_steps))
        out.append("")
    if method in ("cofactor", "both"):
        out.append(subheading("By cofactor expansion"))
        out.append(cofactor_text(expand_cofactors(matrix)))
        out.append("")
    out.append(paint("det(A) = %s" % sval(value), C.BOLD))
    out.append("")
    out.append("%s A is %s." % (
        check() if value else cross(),
        "invertible, since det(A) is nonzero" if value else "singular, since det(A) = 0",
    ))
    return "\n".join(out)


def row_op_determinant_text(matrix, show_steps=True):
    result = det_by_row_reduction(matrix)
    red = result.reduction
    out = []
    if show_steps:
        out.append(steps_block(red))
    out.append("Reaching the row echelon form used:")
    out.append(table([
        ["row swaps", "%d, each multiplying det by -1" % red.swaps],
        ["row scalings", "%d, contributing the factors below" % red.scalings],
        ["row additions", "%d, which never change det" % red.eliminations],
    ]))
    out.append("")
    out.append("The row echelon form is triangular, so its determinant is the diagonal product:")
    out.append("  " + ("  %s  " % dim("x")).join(fmt(v) for v in result.diagonal)
               + "  =  " + fmt(_product(result.diagonal)))
    out.append("Those operations multiplied the determinant by %s along the way," % fmt(red.factor))
    out.append("so det(A) = %s / %s = %s" % (
        fmt(_product(result.diagonal)), fmt(red.factor), sval(result.value)))
    return "\n".join(out)


def _product(values):
    total = ONE
    for v in values:
        total = total * v
    return total


def cofactor_text(expansion, indent="  ", depth=0):
    out = []
    pad = indent * (depth + 1)
    if not expansion.terms:
        return pad + "det = %s" % fmt(expansion.value)
    out.append(pad + "expand along %s%s" % (
        expansion.label, dim(" - " + expansion.reason) if expansion.reason else ""))
    pieces = []
    for t in expansion.terms:
        sign = "-" if t.sign < 0 else "+"
        pieces.append("%s %s%sM%d%d" % (sign, fmt_coeff(abs(t.entry)) or "", "", t.row + 1, t.col + 1))
    out.append(pad + "det = " + " ".join(pieces).lstrip("+ "))
    for t in expansion.terms:
        out.append("")
        out.append(pad + "M%d%d  (delete row %d, column %d)" % (t.row + 1, t.col + 1, t.row + 1, t.col + 1))
        out.append(matrix_str(t.block, None, pad + "  "))
        if t.child is not None:
            out.append(cofactor_text(t.child, indent, depth + 1))
        out.append(pad + "  det M%d%d = %s" % (t.row + 1, t.col + 1, fmt(t.value)))
        out.append(pad + "  term = %s(%s)(%s) = %s" % (
            "+" if t.sign > 0 else "-", fmt(t.entry), fmt(t.value), fmt(t.contribution)))
    out.append("")
    out.append(pad + "sum = %s" % fmt(expansion.value))
    return "\n".join(out)


def inverse_report(matrix, method="gauss", show_steps=True):
    out = [heading("inverse", WIDTH), "", block("A", matrix), ""]
    if method == "gauss":
        result = inverse_gauss_jordan(matrix)
        n = matrix.nrows
        out.append(subheading("Gauss-Jordan on [A | I]"))
        out.append("Row reduce until the left half is the identity; the right half is then A inverse.")
        out.append("")
        if show_steps:
            out.append(steps_block(result.reduction, n))
        out.append(block("Final form", result.reduction.rref, n))
        out.append("")
        if not result.ok:
            out.append(paint("A is singular, so it has no inverse.", C.RED, C.BOLD))
            out.append("The left half never became the identity: rank(A) = %d < %d."
                       % (result.rank, matrix.nrows))
            return "\n".join(out)
        out.append(block("A inverse", result.inverse))
        out.append("")
        out.append("%s checked: A A^-1 = I and A^-1 A = I." % check())
        return "\n".join(out)
    if method == "adjugate":
        result = inverse_adjugate(matrix)
        out.append(subheading("By the adjugate formula"))
        out.append("A^-1 = adj(A) / det(A), where adj(A) is the transpose of the cofactor matrix.")
        out.append("")
        out.append(block("cofactor matrix C", cofactor_matrix(matrix)))
        out.append("")
        out.append(block("adj(A) = C transpose", result.adjugate))
        out.append("")
        out.append("det(A) = %s" % sval(result.det))
        out.append("")
        if not result.ok:
            out.append(paint("det(A) = 0, so A has no inverse.", C.RED, C.BOLD))
            return "\n".join(out)
        out.append(block("A inverse = adj(A) / det(A)", result.inverse))
        out.append("")
        out.append("%s checked: A adj(A) = det(A) I." % check())
        return "\n".join(out)
    if method == "elementary":
        result = elementary_factorization(matrix)
        out.append(subheading("As a product of elementary matrices"))
        if not result.ok:
            out.append(paint("A is singular, so it is not a product of elementary matrices.", C.RED, C.BOLD))
            return "\n".join(out)
        from .reduce import describe_op

        out.append("Row reducing A to I used these operations, each an elementary matrix:")
        out.append("")
        for i, (op, e, ei) in enumerate(zip(result.ops, result.factors, result.inverse_factors)):
            out.append("  E%d:  %s" % (i + 1, describe_op(op)))
            out.append(side_by_side([matrix_str(e, None, "   "), matrix_str(ei, None, "   ")],
                                    gap=4, labels=["E%d" % (i + 1), "E%d inverse" % (i + 1)]))
            out.append("")
        out.append("E%d ... E1 A = I, so A = E1^-1 E2^-1 ... E%d^-1:"
                   % (len(result.factors), len(result.factors)))
        out.append("")
        out.append(block("product of the inverses", result.product))
        out.append("")
        out.append("%s checked: the product reproduces A exactly."
                   % (check() if result.product == matrix else cross()))
        return "\n".join(out)
    result = lu_decomposition(matrix)
    out.append(subheading("LU factorization with partial pivoting"))
    out.append("PA = LU, with L unit lower triangular and U upper triangular.")
    out.append("")
    out.append(side_by_side(
        [matrix_str(result.p), matrix_str(result.l), matrix_str(result.u)],
        gap=4, labels=["P", "L", "U"],
    ))
    out.append("")
    out.append("%s checked: PA = LU." % (check() if result.ok else cross()))
    if matrix.is_square:
        out.append("det(A) = (-1)^%d %s (product of the diagonal of U) = %s"
                   % (result.swaps, dim("x"), fmt(determinant(matrix))))
    return "\n".join(out)


def subspaces_report(matrix):
    f = four_subspaces(matrix)
    m, n = matrix.shape
    out = [heading("the four fundamental subspaces", WIDTH), "", block("A", matrix), ""]
    out.append(block("RREF(A)", f.reduction.rref))
    out.append("")
    out.append(subheading("Rank and nullity"))
    out.append(table([
        ["rank(A)", "%d   (pivot columns %s)" % (f.rank, ", ".join(str(c + 1) for c in f.pivot_columns) or "none")],
        ["nullity(A)", "%d   (free columns %s)" % (f.nullity, ", ".join(str(c + 1) for c in f.reduction.free_columns) or "none")],
        ["rank + nullity", "%d + %d = %d = n" % (f.rank, f.nullity, n)],
        ["rank(A) = rank(A^T)", "%d, so row rank equals column rank" % f.rank],
    ]))
    out.append("")
    for title, vectors, where, note in (
        ("Column space  col(A)", f.column_space, "R^%d" % m, "the pivot columns of A, untouched by row reduction"),
        ("Row space  row(A)", f.row_space, "R^%d" % n, "the nonzero rows of the RREF"),
        ("Null space  null(A)", f.null_space, "R^%d" % n, "solutions of Ax = 0, one per free column"),
        ("Left null space  null(A^T)", f.left_null_space, "R^%d" % m, "solutions of A^T y = 0"),
    ):
        out.append(subheading("%s  in %s" % (title, where)))
        out.append(dim("  " + note))
        if not vectors:
            out.append("  the zero subspace, dimension 0")
        else:
            for i, v in enumerate(vectors):
                out.append("  " + vec_line("b%d" % (i + 1), v))
            out.append("  dimension %d" % len(vectors))
        out.append("")
    out.append(subheading("Orthogonality"))
    out.append(bullet("row(A) and null(A) are orthogonal complements in R^%d: %d + %d = %d"
                      % (n, f.rank, f.nullity, n)))
    out.append(bullet("col(A) and null(A^T) are orthogonal complements in R^%d: %d + %d = %d"
                      % (m, f.rank, f.left_nullity, m)))
    out.append("")
    out.append("%s checked: every null space vector is orthogonal to every row space vector."
               % (check() if f.verify() else cross()))
    return "\n".join(out)


def independence_report(vectors):
    ind = independence(vectors)
    out = [heading("linear independence", WIDTH), ""]
    for i, v in enumerate(vectors):
        out.append("  " + vec_line("v%d" % (i + 1), v))
    out.append("")
    out.append(block("placed as the columns of A", ind.matrix))
    out.append("")
    out.append(block("RREF(A)", ind.reduction.rref))
    out.append("")
    out.append(table([
        ["vectors", len(vectors)],
        ["rank", ind.rank],
        ["pivot columns", ", ".join(str(c + 1) for c in ind.pivot_columns) or "none"],
        ["free columns", ", ".join(str(c + 1) for c in ind.free_columns) or "none"],
    ]))
    out.append("")
    if ind.independent:
        out.append(paint("Linearly independent.", C.GREEN, C.BOLD))
        out.append("Every column is a pivot column, so c1v1 + ... = 0 forces every ci = 0.")
        out.append("They therefore form a basis for the %d-dimensional space they span." % ind.rank)
    else:
        out.append(paint("Linearly dependent.", C.RED, C.BOLD))
        parts = " + ".join("(%s)v%d" % (fmt(k), i + 1) for i, k in enumerate(ind.relation) if k)
        out.append("Column %d is free, which produces an explicit dependency:"
                   % (ind.free_columns[0] + 1))
        out.append("  " + paint("%s  =  0" % parts, C.BOLD))
        out.append("Not every coefficient is zero, so the set is dependent.")
        out.append("%s checked: the combination really is the zero vector." % check())
    return "\n".join(out)


def span_report(target, vectors):
    m = in_span(target, vectors)
    out = [heading("membership in a span", WIDTH), ""]
    for i, v in enumerate(vectors):
        out.append("  " + vec_line("v%d" % (i + 1), v))
    out.append("  " + vec_line("b ", target))
    out.append("")
    out.append(block("solving [v1 ... vk | b]", m.solution.augmented, len(vectors)))
    out.append("")
    out.append(block("RREF", m.solution.reduction.rref, len(vectors)))
    out.append("")
    if not m.inside:
        out.append(paint("b is NOT in the span.", C.RED, C.BOLD))
        out.append("The system is inconsistent: rank of the coefficients is %d but rank with b is %d."
                   % (m.solution.rank, m.solution.rank_augmented))
        return "\n".join(out)
    out.append(paint("b IS in the span.", C.GREEN, C.BOLD))
    parts = " + ".join("(%s)v%d" % (fmt(k), i + 1) for i, k in enumerate(m.coefficients) if k) or "0"
    out.append("  " + paint("b = %s" % parts, C.BOLD))
    out.append("")
    out.append("The coefficients are %s." % (
        "unique, so these vectors are independent and b has one coordinate vector"
        if m.unique else "not unique, since the spanning set is dependent"))
    out.append("%s checked: the combination reproduces b exactly." % check())
    return "\n".join(out)


def basis_report(vectors):
    ex = basis_from_spanning_set(vectors)
    out = [heading("extracting a basis", WIDTH), ""]
    for i, v in enumerate(vectors):
        out.append("  " + vec_line("v%d" % (i + 1), v))
    out.append("")
    out.append(block("RREF of the matrix with these columns", ex.reduction.rref))
    out.append("")
    out.append(subheading("Basis for the span"))
    if not ex.basis:
        out.append("  the span is the zero subspace, whose basis is empty")
    for c in ex.kept:
        out.append("  " + vec_line("v%d" % (c + 1), vectors[c]))
    out.append("")
    out.append("dimension of the span = %d" % ex.dimension)
    if ex.dropped:
        out.append("")
        out.append(subheading("Redundant vectors, written in terms of the basis"))
        for free, terms in ex.expressions.items():
            parts = " + ".join("(%s)v%d" % (fmt(k), c + 1) for c, k in terms) or "0"
            out.append("  v%d = %s" % (free + 1, parts))
    out.append("")
    out.append("Pivot columns of the RREF mark a maximal independent subset, and every")
    out.append("other vector is a combination of them, so they form a basis of the span.")
    return "\n".join(out)


def vector_report(u, w):
    laws = V.InnerProductLaws(u, w)
    out = [heading("vectors, inner product and geometry", WIDTH), ""]
    out.append("  " + vec_line("u", u))
    out.append("  " + vec_line("v", w))
    out.append("")
    out.append(subheading("Inner product and norms"))
    out.append(table([
        ["<u, v>", sval(laws.dot)],
        ["||u||^2", fmt(laws.norm_u2)],
        ["||u||", sval(V.norm(u))],
        ["||v||^2", fmt(laws.norm_v2)],
        ["||v||", sval(V.norm(w))],
        ["distance ||u - v||", sval(V.distance(u, w))],
    ]))
    out.append("")
    if not V.is_zero(u) and not V.is_zero(w):
        out.append(subheading("Angle"))
        cos2, _ = V.cos_angle_squared(u, w)
        out.append(table([
            ["cos^2 theta = <u,v>^2 / (||u||^2 ||v||^2)", fmt(cos2)],
            ["theta", "%.4f degrees" % V.angle_degrees(u, w)],
            ["orthogonal", "yes" if laws.orthogonal else "no"],
        ]))
        out.append("")
        out.append(subheading("Projection of u onto v"))
        pr = V.project_onto_vector(u, w)
        out.append(table([
            ["<u,v>/<v,v>", fmt(pr.coefficient)],
            ["proj_v(u)", tuple_str(pr.parallel)],
            ["residual u - proj_v(u)", tuple_str(pr.perpendicular)],
            ["<residual, v>", fmt(V.dot(pr.perpendicular, w))],
        ]))
        out.append("  %s the residual is orthogonal to v, so u splits into parallel + perpendicular."
                   % check())
        out.append("")
    out.append(subheading("The standard inequalities"))
    out.append(table([
        ["Cauchy-Schwarz  <u,v>^2 <= ||u||^2||v||^2",
         "%s   slack %s" % (check() if laws.cauchy_holds else cross(), fmt(laws.cauchy_gap))],
        ["triangle  ||u+v|| <= ||u||+||v||", check() if laws.triangle_holds else cross()],
        ["parallelogram law", check() if laws.parallelogram_holds else cross()],
        ["Pythagoras (needs orthogonality)",
         (check() if laws.pythagoras_holds else cross()) if laws.orthogonal else dim("not applicable")],
    ]))
    if len(u) == 3 and len(w) == 3:
        out.append("")
        out.append(subheading("Cross product (R^3 only)"))
        cr = V.cross(u, w)
        out.append("  " + vec_line("u x v", cr))
        out.append("  <u x v, u> = %s,  <u x v, v> = %s" % (fmt(V.dot(cr, u)), fmt(V.dot(cr, w))))
    return "\n".join(out)


def gram_schmidt_report(vectors):
    gs = V.gram_schmidt(vectors)
    out = [heading("gram-schmidt orthogonalization", WIDTH), ""]
    for i, v in enumerate(vectors):
        out.append("  " + vec_line("v%d" % (i + 1), v))
    out.append("")
    for step in gs.steps:
        out.append(subheading("w%d from v%d" % (step.index + 1, step.index + 1)))
        if not step.terms:
            out.append("  w1 = v1, nothing to subtract yet")
        else:
            parts = " - ".join(
                "(%s)w%d" % (fmt(k), j + 1) for j, (k, _) in enumerate(step.terms))
            out.append("  w%d = v%d - %s" % (step.index + 1, step.index + 1, parts))
            for j, (k, b) in enumerate(step.terms):
                out.append(dim("     <v%d, w%d>/<w%d, w%d> = %s" % (
                    step.index + 1, j + 1, j + 1, j + 1, fmt(k))))
        if step.result is None:
            out.append("  " + paint("this vector was already in the span of the earlier ones, so it drops out", C.YELLOW))
        else:
            out.append("  " + vec_line("w%d" % (step.index + 1), step.raw))
            if step.cleared is not None:
                out.append(dim("     cleared of fractions (same direction): %s" % tuple_str(step.cleared)))
        out.append("")
    out.append(subheading("Orthogonal basis"))
    for i, v in enumerate(gs.vectors):
        out.append("  " + vec_line("w%d" % (i + 1), v))
    out.append("")
    if gs.vectors:
        out.append(subheading("Normalized (orthonormal) basis"))
        for i, v in enumerate(gs.vectors):
            out.append("  q%d = (1/%s) %s" % (i + 1, sval(V.norm(v)), tuple_str(v)))
        out.append("")
    out.append(subheading("Checks"))
    grid = [["<w%d, w%d>" % (i + 1, j + 1), fmt(V.dot(a, b))]
            for i, a in enumerate(gs.vectors) for j, b in enumerate(gs.vectors) if i < j]
    out.append(table(grid) if grid else "  only one vector, nothing to compare")
    out.append("")
    out.append("%s pairwise orthogonal, and the span is unchanged." % (check() if gs.verify() else cross()))
    return "\n".join(out)


def projection_report(target, basis):
    pr = V.project_onto_subspace(target, basis)
    gs = V.gram_schmidt(basis)
    out = [heading("projection onto a subspace", WIDTH), ""]
    out.append("  " + vec_line("v", target))
    for i, b in enumerate(basis):
        out.append("  " + vec_line("spanning u%d" % (i + 1), b))
    out.append("")
    out.append("First make the spanning set orthogonal, so the projection splits into independent pieces.")
    out.append("")
    for i, w in enumerate(gs.vectors):
        out.append("  " + vec_line("w%d" % (i + 1), w))
    out.append("")
    out.append(subheading("Projection"))
    for i, (k, w) in enumerate(zip(pr.coefficient, gs.vectors)):
        out.append("  <v, w%d>/<w%d, w%d> = %s" % (i + 1, i + 1, i + 1, fmt(k)))
    out.append("")
    out.append("  " + vec_line("proj_W(v)", pr.parallel))
    out.append("  " + vec_line("v - proj_W(v)", pr.perpendicular))
    out.append("")
    out.append(subheading("Checks"))
    rows = [["proj + residual = v", check() if V.add(pr.parallel, pr.perpendicular) == tuple(target) else cross()]]
    for i, w in enumerate(gs.vectors):
        rows.append(["<residual, w%d>" % (i + 1), fmt(V.dot(pr.perpendicular, w))])
    rows.append(["distance from v to W", sval(V.norm(pr.perpendicular))])
    out.append(table(rows))
    out.append("")
    out.append("The residual is orthogonal to every basis vector, so proj_W(v) is the")
    out.append("closest point of W to v: that minimum distance is the number above.")
    return "\n".join(out)


def eigen_report(matrix, show_steps=False):
    spec = spectrum(matrix)
    n = matrix.nrows
    out = [heading("eigenvalues, eigenvectors and eigenspaces", WIDTH), "", block("A", matrix), ""]
    out.append(subheading("Characteristic polynomial"))
    out.append("  det(%sI - A) = %s" % (lam(), spec.poly.shift_variable(lam())))
    out.append("")
    if not spec.exact:
        out.append(paint(
            "Some roots are irrational of degree above 2; those are reported numerically.", C.YELLOW))
        out.append("")
    out.append(subheading("Eigenvalues"))
    rows = []
    for p in spec.pairs:
        rows.append([
            "%s = %s" % (lam(), sval(p.value)),
            "algebraic %d" % p.algebraic,
            "geometric %d" % p.geometric,
            "defective" if p.defective else "",
            "" if p.is_real else "complex",
        ])
    out.append(table(rows))
    out.append("")
    out.append(table([
        ["sum of eigenvalues = tr(A) = %s" % fmt(matrix.trace()), _tick(spec.trace_check)],
        ["product of eigenvalues = det(A) = %s" % fmt(determinant(matrix)), _tick(spec.det_check)],
    ]))
    out.append("")
    for p in spec.pairs:
        out.append(subheading("Eigenspace for %s = %s" % (lam(), sval(p.value))))
        if not p.exact:
            out.append("  eigenvalue known numerically; eigenvectors below are numeric too")
            for v in p.basis:
                out.append("  (" + ", ".join("%.6f%+.6fi" % (x.real, x.imag) for x in v) + ")")
            out.append("")
            continue
        out.append("  solve (A - %sI)x = 0:" % lam())
        out.append(matrix_str(p.shifted, None, "   "))
        red = row_reduce(p.shifted)
        if show_steps:
            out.append("")
            out.append(steps_block(red))
        out.append("")
        out.append("  RREF:")
        out.append(matrix_str(red.rref, None, "   "))
        out.append("")
        if not p.basis:
            out.append("  no nonzero solutions found")
        for i, v in enumerate(p.basis):
            out.append("  " + vec_line("basis vector v%d" % (i + 1), v))
        out.append("  dim E(%s) = %d, algebraic multiplicity %d%s"
                   % (lam(), p.geometric, p.algebraic,
                      ", so this eigenvalue is defective" if p.defective else ""))
        if p.basis:
            v = p.basis[0]
            out.append("  %s check: Av = %s and %sv = %s"
                       % (check(), tuple_str(matrix.matmul(Matrix.column(v)).column_at(0)),
                          lam(), tuple_str(tuple(p.value * x for x in v))))
        out.append("")
    out.append(subheading("Diagonalization"))
    d = diagonalize(matrix, spec)
    if not d.ok:
        out.append(paint("A is not diagonalizable.", C.RED, C.BOLD))
        out.append("  " + d.reason)
    else:
        out.append(paint("A is diagonalizable.", C.GREEN, C.BOLD))
        out.append("  " + spec.reason)
        out.append("")
        out.append(side_by_side([matrix_str(d.p), matrix_str(d.d)], gap=4, labels=["P", "D"]))
        out.append("")
        out.append("  %s checked: AP = PD, so P^-1 A P = D." % (check() if d.verified else cross()))
    if matrix.is_symmetric():
        out.append("")
        out.append(subheading("Spectral theorem"))
        od = orthogonally_diagonalize(matrix)
        out.append("  A is symmetric, so its eigenvalues are real and its eigenvectors")
        out.append("  can be chosen orthogonal.")
        for i, v in enumerate(od.orthogonal_basis):
            out.append("  " + vec_line("q%d" % (i + 1), v))
        if od.ok:
            out.append("  %s the basis is pairwise orthogonal; normalize each column to get an orthogonal Q."
                       % check())
        else:
            out.extend("  " + line for line in _wrap(od.reason, 68))
    return "\n".join(out)


def _wrap(text, width):
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        out.append(line)
    return out


def _tick(flag):
    if flag is None:
        return dim("not comparable in a single field")
    return check() if flag else cross()


def properties_report(matrix):
    out = [heading("matrix properties", WIDTH), "", block("A", matrix), ""]
    m, n = matrix.shape
    red = row_reduce(matrix)
    rows = [
        ["shape", "%d x %d" % (m, n)],
        ["rank", red.rank],
        ["nullity", n - red.rank],
    ]
    if matrix.is_square:
        d = determinant(matrix)
        rows += [
            ["determinant", sval(d)],
            ["trace", sval(matrix.trace())],
            ["invertible", "yes" if d else "no"],
            ["symmetric", "yes" if matrix.is_symmetric() else "no"],
            ["skew-symmetric", "yes" if matrix.is_skew_symmetric() else "no"],
            ["diagonal", "yes" if matrix.is_diagonal() else "no"],
            ["upper triangular", "yes" if matrix.is_upper_triangular() else "no"],
            ["lower triangular", "yes" if matrix.is_lower_triangular() else "no"],
            ["orthogonal (A^T A = I)", "yes" if matrix.is_orthogonal() else "no"],
            ["idempotent (A^2 = A)", "yes" if matrix.is_idempotent() else "no"],
            ["involutory (A^2 = I)", "yes" if matrix.is_involutory() else "no"],
            ["nilpotent", "yes" if matrix.is_nilpotent() else "no"],
        ]
    out.append(table(rows))
    out.append("")
    out.append(block("RREF(A)", red.rref))
    out.append("")
    out.append(block("A transpose", matrix.T))
    return "\n".join(out)


def homogeneous_text(matrix):
    rep = homogeneous_report(matrix)
    sol = rep.solution
    out = [heading("homogeneous system  Ax = 0", WIDTH), "", block("A", matrix), ""]
    out.append(block("RREF", sol.reduction.rref, matrix.ncols))
    out.append("")
    rows = [["rank(A)", sol.rank], ["unknowns", matrix.ncols], ["nullity", sol.nullity]]
    if rep.square:
        rows.append(["det(A)", sval(rep.det)])
    out.append(table(rows))
    out.append("")
    out.append("  " + rep.reason)
    out.append("")
    if not rep.nontrivial:
        out.append(paint("Only the trivial solution x = 0.", C.GREEN, C.BOLD))
        return "\n".join(out)
    out.append(paint("Nontrivial solutions exist.", C.GREEN, C.BOLD))
    out.append("The solution space is the null space of A, of dimension %d:" % sol.nullity)
    out.append("")
    for i, v in enumerate(sol.homogeneous_basis):
        out.append("  " + vec_line("v%d" % (i + 1), v))
    out.append("")
    out.append("  x = " + " + ".join("%s v%d" % (p, i + 1)
                                     for i, p in enumerate(sol.parameters)))
    out.append("")
    out.append("%s checked: Av = 0 for every basis vector." % check())
    return "\n".join(out)


def cramer_text(matrix, constants):
    rep = cramer_report(matrix, constants)
    out = [heading("cramer's rule", WIDTH), ""]
    out.append(equations_text(matrix, constants))
    out.append("")
    out.append(block("A", matrix))
    out.append("")
    if not rep.usable:
        out.append(paint(rep.reason, C.RED, C.BOLD))
        if rep.det is not None:
            out.append("")
            out.append("det(A) = %s" % sval(rep.det))
            out.append("Cramer's rule divides by det(A), so it needs a nonzero determinant.")
            out.append("Row reduce the augmented matrix instead to classify the solution set.")
        return "\n".join(out)
    out.append("det(A) = %s" % paint(sval(rep.det), C.BOLD))
    out.append("")
    for j, (replaced, value) in enumerate(rep.numerators):
        out.append(subheading("A%d: column %d replaced by b" % (j + 1, j + 1)))
        out.append(matrix_str(replaced, None, "   "))
        out.append("   det(A%d) = %s" % (j + 1, sval(value)))
        out.append("   x%d = det(A%d)/det(A) = %s" % (j + 1, j + 1, sval(rep.solution[j])))
        out.append("")
    out.append(subheading("Solution"))
    out.append(table([["x%d" % (i + 1), sval(v)] for i, v in enumerate(rep.solution)]))
    out.append("")
    ok = matrix.matmul(Matrix.column(rep.solution)) == Matrix.column(constants)
    out.append("%s checked: A x = b exactly." % (check() if ok else cross()))
    return "\n".join(out)


def certificate_report(cert):
    out = [heading(cert.title, WIDTH), ""]
    out.append(paint("Statement", C.BOLD))
    out.append("  " + cert.statement)
    out.append("")
    out.append(dim("  justification: " + cert.citation))
    out.append("")
    if cert.holds is None:
        out.append(paint("Cannot evaluate on these operands.", C.YELLOW, C.BOLD))
    for line in cert.lines:
        if line.kind == "matrix":
            out.append(subheading("  " + line.label))
            out.append(matrix_str(line.value, None, "    "))
        elif line.kind == "vector":
            out.append("  %s = %s" % (line.label, tuple_str(line.value)))
        elif line.kind == "scalar":
            out.append("  %s = %s" % (line.label, sval(line.value)))
        elif line.label == "---":
            out.append("  " + dim(rule(40)))
        else:
            out.append("  %s%s" % (line.label, (": " + str(line.value)) if line.value != "" else ""))
    out.append("")
    if cert.holds is True:
        tag = paint("%s VERIFIED" % check(), C.GREEN, C.BOLD)
    elif cert.holds is False:
        tag = paint("%s FAILED" % cross(), C.RED, C.BOLD)
    else:
        tag = paint("NOT APPLICABLE", C.YELLOW, C.BOLD)
    out.append("%s  %s" % (tag, cert.conclusion))
    out.append("")
    out.append(dim(
        "This is a complete proof for the operands shown."
        if cert.kind == "proof"
        else "This verifies the identity on the operands shown; the cited theorem gives the general case."
    ))
    return "\n".join(out)
