import random
from fractions import Fraction

from sylvester.matrix import Matrix
from sylvester.render import set_ascii, set_color

set_ascii(True)
set_color(False)


def seeded(seed):
    return random.Random(seed)


def random_matrix(rng, nrows, ncols, lo=-6, hi=6):
    return Matrix([[rng.randint(lo, hi) for _ in range(ncols)] for _ in range(nrows)])


def random_square(rng, n, lo=-6, hi=6):
    return random_matrix(rng, n, n, lo, hi)


def random_invertible(rng, n, tries=64):
    from sylvester.determinant import determinant

    for _ in range(tries):
        m = random_square(rng, n)
        if determinant(m):
            return m
    return Matrix.identity(n)


def random_vector(rng, n, lo=-6, hi=6):
    return tuple(Fraction(rng.randint(lo, hi)) for _ in range(n))


def random_rational_matrix(rng, nrows, ncols):
    return Matrix([
        [Fraction(rng.randint(-8, 8), rng.randint(1, 6)) for _ in range(ncols)]
        for _ in range(nrows)
    ])


def naive_determinant(m):
    n = m.nrows
    if n == 0:
        return Fraction(1)
    if n == 1:
        return m[0, 0]
    total = Fraction(0)
    for j in range(n):
        if m[0, j]:
            term = m[0, j] * naive_determinant(m.minor_matrix(0, j))
            total += term if j % 2 == 0 else -term
    return total


def is_rref(m, limit=None):
    limit = m.ncols if limit is None else limit
    last_pivot = -1
    seen_zero_row = False
    for r in range(m.nrows):
        pivot = next((c for c in range(limit) if m[r][c]), None)
        if pivot is None:
            seen_zero_row = True
            continue
        if seen_zero_row:
            return False
        if pivot <= last_pivot:
            return False
        if m[r][pivot] != 1:
            return False
        for other in range(m.nrows):
            if other != r and m[other][pivot]:
                return False
        last_pivot = pivot
    return True
