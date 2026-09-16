from __future__ import annotations

from fractions import Fraction

from .exact import Surd, sqrt_exact
from .matrix import Matrix


class ParseError(ValueError):
    pass


def parse_value(token):
    token = token.strip().replace(" ", "")
    if not token:
        raise ParseError("empty value")
    negative = False
    while token[:1] in "+-":
        negative ^= token[0] == "-"
        token = token[1:]
    value = _magnitude(token)
    return -value if negative else value


def _magnitude(token):
    if token.startswith("sqrt(") and token.endswith(")"):
        return sqrt_exact(parse_value(token[5:-1]))
    if "/" in token:
        num, _, den = token.partition("/")
        top, bottom = parse_value(num), parse_value(den)
        if isinstance(top, Surd) or isinstance(bottom, Surd):
            return top / bottom
        if bottom == 0:
            raise ParseError("division by zero in %s" % token)
        return Fraction(top) / Fraction(bottom)
    try:
        if "." in token or "e" in token.lower():
            return Fraction(token)
        return Fraction(int(token))
    except (ValueError, ZeroDivisionError):
        raise ParseError("cannot read %r as a number" % token)


def split_rows(text):
    cleaned = text.replace("\r", "")
    for ch in "[]{}":
        cleaned = cleaned.replace(ch, "")
    parts = []
    for chunk in cleaned.replace("\n", ";").split(";"):
        if chunk.strip():
            parts.append(chunk)
    return parts


def parse_row(text):
    cells = text.replace(",", " ").split()
    if not cells:
        raise ParseError("empty row")
    return [parse_value(c) for c in cells]


def parse_matrix(text):
    rows = [parse_row(chunk) for chunk in split_rows(text)]
    if not rows:
        raise ParseError("no rows found")
    width = len(rows[0])
    for i, row in enumerate(rows):
        if len(row) != width:
            raise ParseError(
                "row %d has %d entries but row 1 has %d" % (i + 1, len(row), width)
            )
    return Matrix(rows)


def parse_vector(text):
    cleaned = text.replace("(", " ").replace(")", " ")
    return tuple(parse_row(cleaned))


def parse_vectors(text):
    return [tuple(parse_row(chunk)) for chunk in split_rows(text)]


def parse_augmented(text):
    if "|" in text:
        rows = []
        width = None
        for chunk in split_rows(text):
            left, _, right = chunk.partition("|")
            row = parse_row(left) + parse_row(right)
            if width is None:
                width = len(parse_row(left))
            rows.append(row)
        return Matrix(rows), width
    matrix = parse_matrix(text)
    return matrix, matrix.ncols - 1


def format_matrix_input(matrix):
    return "; ".join(
        " ".join(_cell(v) for v in row) for row in matrix.rows
    )


def _cell(value):
    from .render import fmt

    return fmt(value).replace(" ", "")
