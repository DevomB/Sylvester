from __future__ import annotations

from fractions import Fraction

from .algebraic import AlgebraicNumber
from .exact import ONE, ZERO, Surd

FIELD_TYPES = (Surd, AlgebraicNumber)


def scalar(value):
    return value if isinstance(value, FIELD_TYPES) else Fraction(value)


class Matrix:
    __slots__ = ("rows", "nrows", "ncols")

    def __init__(self, rows):
        data = tuple(tuple(scalar(v) for v in row) for row in rows)
        if data:
            width = len(data[0])
            if any(len(r) != width for r in data):
                raise ValueError("every row must have the same number of entries")
        self.rows = data
        self.nrows = len(data)
        self.ncols = len(data[0]) if data else 0

    @classmethod
    def identity(cls, n):
        return cls([[ONE if i == j else ZERO for j in range(n)] for i in range(n)])

    @classmethod
    def zeros(cls, nrows, ncols=None):
        ncols = nrows if ncols is None else ncols
        return cls([[ZERO] * ncols for _ in range(nrows)])

    @classmethod
    def diagonal(cls, values):
        n = len(values)
        return cls([[values[i] if i == j else ZERO for j in range(n)] for i in range(n)])

    @classmethod
    def from_columns(cls, columns):
        if not columns:
            return cls([])
        return cls([[col[i] for col in columns] for i in range(len(columns[0]))])

    @classmethod
    def column(cls, values):
        return cls([[v] for v in values])

    @classmethod
    def row_vector(cls, values):
        return cls([list(values)])

    @property
    def shape(self):
        return self.nrows, self.ncols

    @property
    def is_square(self):
        return self.nrows == self.ncols and self.nrows > 0

    @property
    def T(self):
        return Matrix([[self.rows[i][j] for i in range(self.nrows)] for j in range(self.ncols)])

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self.rows[key[0]][key[1]]
        return self.rows[key]

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return self.nrows

    def entries(self):
        for row in self.rows:
            for v in row:
                yield v

    def column_at(self, j):
        return tuple(row[j] for row in self.rows)

    def columns(self):
        return [self.column_at(j) for j in range(self.ncols)]

    def to_lists(self):
        return [list(row) for row in self.rows]

    def map(self, fn):
        return Matrix([[fn(v) for v in row] for row in self.rows])

    def __eq__(self, other):
        return isinstance(other, Matrix) and self.rows == other.rows

    def __hash__(self):
        return hash(self.rows)

    def __add__(self, other):
        self._same_shape(other)
        return Matrix([[a + b for a, b in zip(x, y)] for x, y in zip(self.rows, other.rows)])

    def __sub__(self, other):
        self._same_shape(other)
        return Matrix([[a - b for a, b in zip(x, y)] for x, y in zip(self.rows, other.rows)])

    def __neg__(self):
        return Matrix([[-v for v in row] for row in self.rows])

    def __pos__(self):
        return self

    def __mul__(self, other):
        if isinstance(other, Matrix):
            return self.matmul(other)
        k = scalar(other)
        return Matrix([[k * v for v in row] for row in self.rows])

    def __rmul__(self, other):
        k = scalar(other)
        return Matrix([[k * v for v in row] for row in self.rows])

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            return self.matmul(other.inverse())
        k = scalar(other)
        return Matrix([[v / k for v in row] for row in self.rows])

    def __matmul__(self, other):
        return self.matmul(other)

    def matmul(self, other):
        if self.ncols != other.nrows:
            raise ValueError(
                "cannot multiply %dx%d by %dx%d: inner dimensions differ"
                % (self.nrows, self.ncols, other.nrows, other.ncols)
            )
        width = other.ncols
        brows = other.rows
        out = []
        for arow in self.rows:
            acc = [ZERO] * width
            for aij, brow in zip(arow, brows):
                if aij:
                    for j in range(width):
                        if brow[j]:
                            acc[j] += aij * brow[j]
            out.append(acc)
        return Matrix(out)

    def __pow__(self, n):
        if not self.is_square:
            raise ValueError("only a square matrix can be raised to a power")
        if n < 0:
            return self.inverse() ** -n
        result, base = Matrix.identity(self.nrows), self
        while n:
            if n & 1:
                result = result.matmul(base)
            base = base.matmul(base)
            n >>= 1
        return result

    def inverse(self):
        from .inverse import inverse

        return inverse(self)

    def det(self):
        from .determinant import determinant

        return determinant(self)

    def rank(self):
        from .reduce import row_reduce

        return len(row_reduce(self).pivots)

    def rref(self):
        from .reduce import row_reduce

        return row_reduce(self).rref

    def _same_shape(self, other):
        if not isinstance(other, Matrix) or self.shape != other.shape:
            raise ValueError("shapes do not match: %s and %s" % (self.shape, getattr(other, "shape", None)))

    def augment(self, other):
        if isinstance(other, Matrix):
            if other.nrows != self.nrows:
                raise ValueError("row counts differ")
            extra = other.rows
        else:
            if len(other) != self.nrows:
                raise ValueError("row counts differ")
            extra = [[v] for v in other]
        return Matrix([list(a) + list(b) for a, b in zip(self.rows, extra)])

    def stack(self, other):
        if self.ncols != other.ncols:
            raise ValueError("column counts differ")
        return Matrix(self.to_lists() + other.to_lists())

    def submatrix(self, rows, cols):
        return Matrix([[self.rows[i][j] for j in cols] for i in rows])

    def take_columns(self, cols):
        return Matrix([[row[j] for j in cols] for row in self.rows])

    def take_rows(self, rows):
        return Matrix([list(self.rows[i]) for i in rows])

    def minor_matrix(self, i, j):
        return Matrix(
            [[v for c, v in enumerate(row) if c != j] for r, row in enumerate(self.rows) if r != i]
        )

    def left(self, k):
        return Matrix([list(row[:k]) for row in self.rows])

    def right(self, k):
        return Matrix([list(row[-k:]) for row in self.rows])

    def trace(self):
        if not self.is_square:
            raise ValueError("trace needs a square matrix")
        total = ZERO
        for i in range(self.nrows):
            total += self.rows[i][i]
        return total

    def is_zero(self):
        return all(not v for v in self.entries())

    def is_identity(self):
        return self.is_square and self == Matrix.identity(self.nrows)

    def is_symmetric(self):
        return self.is_square and self == self.T

    def is_skew_symmetric(self):
        return self.is_square and self == -self.T

    def is_diagonal(self):
        return self.is_square and all(
            not self.rows[i][j] for i in range(self.nrows) for j in range(self.ncols) if i != j
        )

    def is_upper_triangular(self):
        return all(
            not self.rows[i][j] for i in range(self.nrows) for j in range(min(i, self.ncols))
        )

    def is_lower_triangular(self):
        return all(
            not self.rows[i][j]
            for i in range(self.nrows)
            for j in range(i + 1, self.ncols)
        )

    def is_triangular(self):
        return self.is_upper_triangular() or self.is_lower_triangular()

    def is_orthogonal(self):
        return self.is_square and self.T.matmul(self) == Matrix.identity(self.nrows)

    def is_involutory(self):
        return self.is_square and self.matmul(self).is_identity()

    def is_idempotent(self):
        return self.is_square and self.matmul(self) == self

    def is_nilpotent(self):
        if not self.is_square:
            return False
        return self.__pow__(self.nrows).is_zero()

    def is_rational(self):
        return not any(isinstance(v, FIELD_TYPES) for v in self.entries())

    def with_entry(self, i, j, value):
        rows = self.to_lists()
        rows[i][j] = scalar(value)
        return Matrix(rows)

    def with_column(self, j, values):
        rows = self.to_lists()
        for i, v in enumerate(values):
            rows[i][j] = scalar(v)
        return Matrix(rows)

    def resized(self, nrows, ncols):
        return Matrix(
            [
                [self.rows[i][j] if i < self.nrows and j < self.ncols else ZERO for j in range(ncols)]
                for i in range(nrows)
            ]
        )

    def __repr__(self):
        return "Matrix(%dx%d)" % (self.nrows, self.ncols)


def elementary_swap(n, i, j):
    rows = Matrix.identity(n).to_lists()
    rows[i], rows[j] = rows[j], rows[i]
    return Matrix(rows)


def elementary_scale(n, i, k):
    k = scalar(k)
    if not k:
        raise ValueError("an elementary scaling must use a nonzero factor")
    rows = Matrix.identity(n).to_lists()
    rows[i][i] = k
    return Matrix(rows)


def elementary_add(n, target, source, k):
    if target == source:
        raise ValueError("an elementary addition must use two different rows")
    rows = Matrix.identity(n).to_lists()
    rows[target][source] = scalar(k)
    return Matrix(rows)


