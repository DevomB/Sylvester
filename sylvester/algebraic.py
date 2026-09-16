from __future__ import annotations

from fractions import Fraction

ZERO = Fraction(0)
ONE = Fraction(1)
RATIONAL = (int, Fraction)


class ReducibleModulus(ArithmeticError):
    def __init__(self, factor):
        super().__init__("the defining polynomial of this field is reducible")
        self.factor = factor


def trim(c):
    while c and not c[-1]:
        c.pop()
    return c


def poly_mul(a, b):
    if not a or not b:
        return []
    out = [ZERO] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                if y:
                    out[i + j] += x * y
    return trim(out)


def poly_sub(a, b):
    out = list(a) + [ZERO] * max(0, len(b) - len(a))
    for i, y in enumerate(b):
        out[i] -= y
    return trim(out)


def poly_divmod(a, b):
    rem = list(a)
    db = len(b) - 1
    lead = b[-1]
    quot = [ZERO] * max(0, len(rem) - db)
    for i in range(len(rem) - 1, db - 1, -1):
        k = rem[i] / lead
        if k:
            quot[i - db] = k
            for j, y in enumerate(b):
                if y:
                    rem[i - db + j] -= k * y
    return trim(quot), trim(rem[:db])


class NumberField:
    __slots__ = ("modulus", "degree", "symbol", "table", "roots", "real_roots")

    def __init__(self, modulus, symbol="alpha"):
        coeffs = trim([Fraction(c) for c in modulus])
        if len(coeffs) < 2:
            raise ValueError("a number field needs a defining polynomial of degree at least 1")
        lead = coeffs[-1]
        coeffs = [c / lead for c in coeffs]
        n = len(coeffs) - 1
        self.modulus = tuple(coeffs)
        self.degree = n
        self.symbol = symbol
        power = [-c for c in coeffs[:n]]
        table = []
        for _ in range(n - 1):
            table.append(tuple(power))
            carry = power[-1]
            power = [ZERO] + power[:-1]
            if carry:
                for i in range(n):
                    power[i] -= carry * coeffs[i]
        self.table = tuple(table)
        self.roots = None
        self.real_roots = None

    def __eq__(self, other):
        return isinstance(other, NumberField) and self.modulus == other.modulus

    def __hash__(self):
        return hash(self.modulus)

    def make(self, coeffs):
        for c in coeffs[1:]:
            if c:
                return AlgebraicNumber(self, tuple(coeffs))
        return coeffs[0] if coeffs else ZERO

    def element(self, coeffs):
        c = trim([Fraction(x) for x in coeffs])
        if len(c) > self.degree:
            c = poly_divmod(c, list(self.modulus))[1]
        return self.make(c + [ZERO] * (self.degree - len(c)))

    @property
    def generator(self):
        return self.element([ZERO, ONE])

    def __repr__(self):
        return "NumberField(%s)" % (self.modulus,)


class AlgebraicNumber:
    __slots__ = ("field", "c")

    def __init__(self, field, coeffs):
        self.field = field
        self.c = coeffs

    def _same(self, other):
        if other.field is not self.field and other.field.modulus != self.field.modulus:
            raise ArithmeticError("cannot mix elements of two different number fields")

    def __add__(self, other):
        if isinstance(other, RATIONAL):
            c = list(self.c)
            c[0] += other
            return AlgebraicNumber(self.field, tuple(c))
        if isinstance(other, AlgebraicNumber):
            self._same(other)
            return self.field.make([a + b for a, b in zip(self.c, other.c)])
        return NotImplemented

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, RATIONAL):
            c = list(self.c)
            c[0] -= other
            return AlgebraicNumber(self.field, tuple(c))
        if isinstance(other, AlgebraicNumber):
            self._same(other)
            return self.field.make([a - b for a, b in zip(self.c, other.c)])
        return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, RATIONAL):
            c = [-a for a in self.c]
            c[0] += other
            return AlgebraicNumber(self.field, tuple(c))
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, RATIONAL):
            if not other:
                return ZERO
            return AlgebraicNumber(self.field, tuple(a * other for a in self.c))
        if isinstance(other, AlgebraicNumber):
            self._same(other)
            field = self.field
            n = field.degree
            raw = [ZERO] * (2 * n - 1)
            for i, a in enumerate(self.c):
                if a:
                    for j, b in enumerate(other.c):
                        if b:
                            raw[i + j] += a * b
            out = raw[:n]
            for e in range(n, 2 * n - 1):
                k = raw[e]
                if k:
                    for i, r in enumerate(field.table[e - n]):
                        if r:
                            out[i] += k * r
            return field.make(out)
        return NotImplemented

    __rmul__ = __mul__

    def inverse(self):
        modulus = list(self.field.modulus)
        r0, r1 = modulus, trim(list(self.c))
        s0, s1 = [], [ONE]
        while len(r1) > 1:
            quot, rem = poly_divmod(r0, r1)
            r0, r1 = r1, rem
            s0, s1 = s1, poly_sub(s0, poly_mul(quot, s1))
        if not r1:
            lead = r0[-1]
            raise ReducibleModulus(tuple(x / lead for x in r0))
        k = ONE / r1[0]
        s = [x * k for x in s1]
        if len(s) > self.field.degree:
            s = poly_divmod(s, modulus)[1]
        return self.field.make(s + [ZERO] * (self.field.degree - len(s)))

    def __truediv__(self, other):
        if isinstance(other, RATIONAL):
            if not other:
                raise ZeroDivisionError("division by zero")
            return AlgebraicNumber(self.field, tuple(a / other for a in self.c))
        if isinstance(other, AlgebraicNumber):
            self._same(other)
            return self * other.inverse()
        return NotImplemented

    def __rtruediv__(self, other):
        if isinstance(other, RATIONAL):
            return self.inverse() * other
        return NotImplemented

    def __pow__(self, exponent):
        if not isinstance(exponent, int):
            return NotImplemented
        if exponent < 0:
            return (ONE / self) ** -exponent
        result, base = ONE, self
        while exponent:
            if exponent & 1:
                result = result * base
            base = base * base
            exponent >>= 1
        return result

    def __neg__(self):
        return AlgebraicNumber(self.field, tuple(-a for a in self.c))

    def __pos__(self):
        return self

    def __eq__(self, other):
        if isinstance(other, AlgebraicNumber):
            return self.c == other.c and self.field == other.field
        if isinstance(other, RATIONAL):
            return False
        return NotImplemented

    def __hash__(self):
        return hash((self.field.modulus, self.c))

    def __bool__(self):
        return True

    def at(self, point):
        total = 0
        for a in reversed(self.c):
            total = total * point + float(a)
        return total

    def conjugates(self):
        return [self.at(r) for r in self.field.roots or ()]

    def __repr__(self):
        return "AlgebraicNumber(%s mod %s)" % (self.c, self.field.modulus)
