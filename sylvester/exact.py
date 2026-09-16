from __future__ import annotations

import math
from fractions import Fraction

Q = Fraction
ZERO = Fraction(0)
ONE = Fraction(1)


def squarefree(n):
    if n == 0:
        return 0, 0
    sign = -1 if n < 0 else 1
    n = abs(n)
    k = 1
    d = 2
    while d * d <= n:
        sq = d * d
        while n % sq == 0:
            n //= sq
            k *= d
        d += 1 if d == 2 else 2
    return k, sign * n


class Surd:
    __slots__ = ("a", "b", "d")

    def __init__(self, a, b, d):
        self.a = a
        self.b = b
        self.d = d

    @property
    def is_real(self):
        return self.d > 0

    def conjugate(self):
        return surd(self.a, -self.b, self.d)

    def field_norm(self):
        return self.a * self.a - self.b * self.b * self.d

    def _coerce(self, other):
        if isinstance(other, Surd):
            if other.d != self.d:
                raise ArithmeticError("cannot mix Q(sqrt %d) with Q(sqrt %d)" % (self.d, other.d))
            return other.a, other.b
        if isinstance(other, (int, Fraction)):
            return Fraction(other), ZERO
        return None

    def __add__(self, other):
        parts = self._coerce(other)
        if parts is None:
            return NotImplemented
        return surd(self.a + parts[0], self.b + parts[1], self.d)

    __radd__ = __add__

    def __sub__(self, other):
        parts = self._coerce(other)
        if parts is None:
            return NotImplemented
        return surd(self.a - parts[0], self.b - parts[1], self.d)

    def __rsub__(self, other):
        parts = self._coerce(other)
        if parts is None:
            return NotImplemented
        return surd(parts[0] - self.a, parts[1] - self.b, self.d)

    def __mul__(self, other):
        parts = self._coerce(other)
        if parts is None:
            return NotImplemented
        c, e = parts
        return surd(self.a * c + self.b * e * self.d, self.a * e + self.b * c, self.d)

    __rmul__ = __mul__

    def __truediv__(self, other):
        parts = self._coerce(other)
        if parts is None:
            return NotImplemented
        c, e = parts
        if e == 0:
            if c == 0:
                raise ZeroDivisionError("division by zero")
            return surd(self.a / c, self.b / c, self.d)
        n = c * c - e * e * self.d
        if n == 0:
            raise ZeroDivisionError("division by zero")
        return surd((self.a * c - self.b * e * self.d) / n, (self.b * c - self.a * e) / n, self.d)

    def __rtruediv__(self, other):
        parts = self._coerce(other)
        if parts is None:
            return NotImplemented
        n = self.field_norm()
        if n == 0:
            raise ZeroDivisionError("division by zero")
        c, e = parts
        return surd((c * self.a - e * self.b * self.d) / n, (e * self.a - c * self.b) / n, self.d)

    def __pow__(self, n):
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            return ONE / self ** -n
        result, base = ONE, self
        while n:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    def __neg__(self):
        return surd(-self.a, -self.b, self.d)

    def __pos__(self):
        return self

    def __eq__(self, other):
        if isinstance(other, Surd):
            return self.d == other.d and self.a == other.a and self.b == other.b
        if isinstance(other, (int, Fraction)):
            return False
        return NotImplemented

    def __hash__(self):
        return hash((self.a, self.b, self.d))

    def __bool__(self):
        return True

    def sign(self):
        if not self.is_real:
            raise TypeError("a complex value has no sign")
        a, b, d = self.a, self.b, self.d
        if a == 0:
            return 1 if b > 0 else -1
        if (a > 0) == (b > 0):
            return 1 if a > 0 else -1
        t = a * a - b * b * d
        s = (t > 0) - (t < 0)
        return s if a > 0 else -s

    def __abs__(self):
        return self if self.sign() >= 0 else -self

    def __lt__(self, other):
        return _cmp_real(self, other) < 0

    def __le__(self, other):
        return _cmp_real(self, other) <= 0

    def __gt__(self, other):
        return _cmp_real(self, other) > 0

    def __ge__(self, other):
        return _cmp_real(self, other) >= 0

    def approx(self):
        if self.is_real:
            return float(self.a) + float(self.b) * math.sqrt(self.d)
        return complex(float(self.a), float(self.b) * math.sqrt(-self.d))

    def __float__(self):
        if not self.is_real:
            raise TypeError("cannot convert a complex value to float")
        return self.approx()

    def __complex__(self):
        return complex(self.approx())

    def __repr__(self):
        return "Surd(%s, %s, %d)" % (self.a, self.b, self.d)


def _cmp_real(x, y):
    diff = x - y
    if isinstance(diff, Surd):
        return diff.sign()
    return (diff > 0) - (diff < 0)


def surd(a, b, d):
    a = Fraction(a)
    b = Fraction(b)
    if b == 0 or d == 0:
        return a
    d = Fraction(d)
    if d.denominator != 1:
        b /= d.denominator
        d = d.numerator * d.denominator
    else:
        d = d.numerator
    k, d = squarefree(d)
    b *= k
    if d == 1:
        return a + b
    return Surd(a, b, d)


def sqrt_exact(value):
    value = Fraction(value)
    if value >= 0:
        num = math.isqrt(value.numerator)
        den = math.isqrt(value.denominator)
        if num * num == value.numerator and den * den == value.denominator:
            return Fraction(num, den)
    return surd(0, 1, value)


def is_exact_square(value):
    value = Fraction(value)
    if value < 0:
        return False
    num = math.isqrt(value.numerator)
    den = math.isqrt(value.denominator)
    return num * num == value.numerator and den * den == value.denominator


def is_real(value):
    return not isinstance(value, Surd) or value.is_real


def is_rational(value):
    return isinstance(value, (int, Fraction))


def radicand_of(value):
    return value.d if isinstance(value, Surd) else 1


def unify_field(values):
    field = 1
    for v in values:
        if isinstance(v, Surd):
            if field != 1 and field != v.d:
                raise ArithmeticError("values span more than one quadratic field")
            field = v.d
    return field


def approx(value):
    if isinstance(value, Surd):
        return value.approx()
    if isinstance(value, complex):
        return value
    return float(value)


def weight(value):
    if isinstance(value, Surd):
        return max(weight(value.a), weight(value.b)) * (1 + abs(value.d))
    value = Fraction(value)
    return max(abs(value.numerator), value.denominator)


def common_denominator(values):
    den = 1
    for v in values:
        if isinstance(v, Surd):
            den = den * v.a.denominator // math.gcd(den, v.a.denominator)
            den = den * v.b.denominator // math.gcd(den, v.b.denominator)
        else:
            f = Fraction(v)
            den = den * f.denominator // math.gcd(den, f.denominator)
    return den


def content(values):
    g = 0
    for v in values:
        if isinstance(v, Surd):
            if v.a.denominator != 1 or v.b.denominator != 1:
                return 1
            g = math.gcd(g, abs(v.a.numerator), abs(v.b.numerator))
        else:
            f = Fraction(v)
            if f.denominator != 1:
                return 1
            g = math.gcd(g, abs(f.numerator))
    return g if g > 1 else 1
