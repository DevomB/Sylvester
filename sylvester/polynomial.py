from __future__ import annotations

import cmath
import math
from fractions import Fraction

from .exact import ONE, ZERO, Surd, sqrt_exact


class Poly:
    __slots__ = ("c",)

    def __init__(self, coeffs):
        c = [Fraction(x) for x in coeffs]
        while c and c[-1] == 0:
            c.pop()
        self.c = tuple(c)

    @classmethod
    def constant(cls, value):
        return cls([value])

    @classmethod
    def x(cls, power=1):
        return cls([0] * power + [1])

    @property
    def degree(self):
        return len(self.c) - 1

    @property
    def lead(self):
        return self.c[-1] if self.c else ZERO

    def __bool__(self):
        return bool(self.c)

    def __eq__(self, other):
        return isinstance(other, Poly) and self.c == other.c

    def __hash__(self):
        return hash(self.c)

    def __getitem__(self, i):
        return self.c[i] if 0 <= i < len(self.c) else ZERO

    def __add__(self, other):
        other = _as_poly(other)
        n = max(len(self.c), len(other.c))
        return Poly([self[i] + other[i] for i in range(n)])

    __radd__ = __add__

    def __neg__(self):
        return Poly([-x for x in self.c])

    def __sub__(self, other):
        return self + (-_as_poly(other))

    def __rsub__(self, other):
        return _as_poly(other) + (-self)

    def __mul__(self, other):
        other = _as_poly(other)
        if not self.c or not other.c:
            return Poly([])
        out = [ZERO] * (len(self.c) + len(other.c) - 1)
        for i, a in enumerate(self.c):
            if a:
                for j, b in enumerate(other.c):
                    if b:
                        out[i + j] += a * b
        return Poly(out)

    __rmul__ = __mul__

    def __pow__(self, n):
        result, base = Poly([1]), self
        while n:
            if n & 1:
                result = result * base
            base = base * base
            n >>= 1
        return result

    def divmod(self, other):
        other = _as_poly(other)
        if not other.c:
            raise ZeroDivisionError("polynomial division by zero")
        rem = list(self.c)
        dl = other.lead
        dd = other.degree
        quot = [ZERO] * max(0, len(rem) - dd)
        for i in range(len(rem) - 1, dd - 1, -1):
            factor = rem[i] / dl
            if factor:
                quot[i - dd] = factor
                for j, b in enumerate(other.c):
                    rem[i - dd + j] -= factor * b
        return Poly(quot), Poly(rem)

    def __truediv__(self, other):
        if isinstance(other, (int, Fraction)):
            return Poly([x / Fraction(other) for x in self.c])
        return self.divmod(other)[0]

    def __mod__(self, other):
        return self.divmod(other)[1]

    def __call__(self, value):
        return self.eval(value)

    def eval(self, value):
        total = ZERO
        for coeff in reversed(self.c):
            total = total * value + coeff
        return total

    def derivative(self):
        return Poly([self.c[i] * i for i in range(1, len(self.c))])

    def monic(self):
        return self / self.lead if self.c else self

    def content(self):
        den = 1
        for x in self.c:
            den = den * x.denominator // math.gcd(den, x.denominator)
        num = 0
        for x in self.c:
            num = math.gcd(num, abs(x.numerator) * (den // x.denominator))
        return Fraction(num, den)

    def primitive(self):
        if not self.c:
            return self, ONE
        g = self.content()
        p = Poly([x / g for x in self.c])
        if p.lead < 0:
            p, g = -p, -g
        return p, g

    def gcd(self, other):
        a, b = self, _as_poly(other)
        while b:
            a, b = b, a % b
        return a.monic() if a else a

    def squarefree_part(self):
        if self.degree < 1:
            return self
        return (self / self.gcd(self.derivative())).monic()

    def shift_variable(self, name):
        return format_poly(self.c, name)

    def __str__(self):
        return format_poly(self.c)

    def __repr__(self):
        return "Poly(%s)" % (self.c,)


def _as_poly(value):
    return value if isinstance(value, Poly) else Poly([value])


def format_poly(coeffs, name="x"):
    terms = []
    for i in range(len(coeffs) - 1, -1, -1):
        a = coeffs[i]
        if a == 0:
            continue
        mag = abs(a)
        if i == 0:
            body = _num(mag)
        else:
            power = name if i == 1 else "%s^%d" % (name, i)
            body = power if mag == 1 else "%s%s" % (_num(mag), power)
        if not terms:
            terms.append(("-" if a < 0 else "") + body)
        else:
            terms.append((" - " if a < 0 else " + ") + body)
    return "".join(terms) if terms else "0"


def _num(value):
    return str(value.numerator) if value.denominator == 1 else "%d/%d" % (value.numerator, value.denominator)


def divisors(n):
    n = abs(n)
    if n == 0:
        return []
    small, large = [], []
    i = 1
    while i * i <= n:
        if n % i == 0:
            small.append(i)
            if i * i != n:
                large.append(n // i)
        i += 1
    large.reverse()
    return small + large


def rational_roots(poly):
    if poly.degree < 1:
        return []
    ints, _ = poly.primitive()
    coeffs = [int(x) for x in ints.c]
    roots = []
    zeros = 0
    while coeffs and coeffs[0] == 0:
        coeffs.pop(0)
        zeros += 1
    if zeros:
        roots.append((ZERO, zeros))
    if len(coeffs) < 2:
        return roots
    candidates = []
    for p in divisors(coeffs[0]):
        for q in divisors(coeffs[-1]):
            if math.gcd(p, q) == 1:
                candidates.append(Fraction(p, q))
                candidates.append(Fraction(-p, q))
    for cand in candidates:
        mult = 0
        while len(coeffs) > 1:
            quot, rem = _synthetic(coeffs, cand)
            if rem != 0:
                break
            coeffs = quot
            mult += 1
        if mult:
            roots.append((cand, mult))
        if len(coeffs) < 2:
            break
    return roots


def _synthetic(coeffs, root):
    out = [ZERO] * (len(coeffs) - 1)
    carry = Fraction(coeffs[-1])
    for i in range(len(coeffs) - 2, -1, -1):
        out[i] = carry
        carry = coeffs[i] + carry * root
    return out, carry


def deflate(poly, root, times=1):
    coeffs = list(poly.c)
    for _ in range(times):
        coeffs, _ = _synthetic(coeffs, root)
    return Poly(coeffs)


def discriminant(a, b, c):
    return b * b - 4 * a * c


def quadratic_roots(a, b, c):
    r = sqrt_exact(discriminant(a, b, c))
    return [(-b + r) / (2 * a), (-b - r) / (2 * a)]


def _biquadratic_roots(poly):
    if poly.degree != 4 or poly[1] != 0 or poly[3] != 0:
        return None
    out = []
    for value in quadratic_roots(poly[4], poly[2], poly[0]):
        if isinstance(value, Surd):
            return None
        r = sqrt_exact(value)
        out.extend([r, -r] if r != 0 else [r, r])
    return out


def numeric_roots(poly, iterations=500, tol=1e-14):
    coeffs = [complex(x) for x in poly.c]
    n = len(coeffs) - 1
    if n < 1:
        return []
    lead = coeffs[-1]
    monic = [c / lead for c in coeffs]
    guesses = [(0.4 + 0.9j) ** k for k in range(n)]
    for _ in range(iterations):
        delta = 0.0
        for i in range(n):
            num = _horner(monic, guesses[i])
            den = 1 + 0j
            for j in range(n):
                if i != j:
                    den *= guesses[i] - guesses[j]
            if den == 0:
                continue
            step = num / den
            guesses[i] -= step
            delta = max(delta, abs(step))
        if delta < tol:
            break
    return guesses


def _horner(coeffs, z):
    total = 0j
    for a in reversed(coeffs):
        total = total * z + a
    return total


class Root:
    __slots__ = ("value", "multiplicity", "exact")

    def __init__(self, value, multiplicity, exact=True):
        self.value = value
        self.multiplicity = multiplicity
        self.exact = exact

    @property
    def is_real(self):
        if self.exact:
            return not isinstance(self.value, Surd) or self.value.is_real
        return abs(self.value.imag) < 1e-9

    def approx(self):
        if not self.exact:
            return self.value
        if isinstance(self.value, Surd):
            return self.value.approx()
        return float(self.value)

    def __repr__(self):
        return "Root(%r, x%d, exact=%s)" % (self.value, self.multiplicity, self.exact)


def roots_of(poly):
    if poly.degree < 1:
        return [], True
    found = []
    remaining = poly
    for value, mult in rational_roots(poly):
        found.append(Root(value, mult))
        remaining = deflate(remaining, value, mult)
    exact = True
    if remaining.degree == 1:
        found.append(Root(-remaining[0] / remaining[1], 1))
    elif remaining.degree == 2:
        for r in quadratic_roots(remaining[2], remaining[1], remaining[0]):
            _merge(found, r)
    elif remaining.degree > 2:
        bi = _biquadratic_roots(remaining)
        if bi is not None:
            for r in bi:
                _merge(found, r)
        else:
            exact = False
            for z in numeric_roots(remaining):
                _merge_numeric(found, z)
    found.sort(key=_root_order)
    return found, exact


def _merge(found, value):
    for r in found:
        if r.exact and r.value == value:
            r.multiplicity += 1
            return
    found.append(Root(value, 1))


def _merge_numeric(found, z, tol=1e-7):
    if abs(z.imag) < tol:
        z = complex(z.real, 0.0)
    for r in found:
        if not r.exact and abs(r.value - z) < tol:
            r.multiplicity += 1
            return
    found.append(Root(z, 1, exact=False))


def _root_order(root):
    value = root.approx()
    if isinstance(value, complex):
        return (1, value.real, value.imag)
    return (0, value, 0.0)


def from_roots(roots):
    p = Poly([1])
    for r in roots:
        p = p * Poly([-r, 1])
    return p
