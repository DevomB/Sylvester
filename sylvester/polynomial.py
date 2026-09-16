from __future__ import annotations

import cmath
import math
from fractions import Fraction

from .exact import ONE, ZERO, sqrt_exact


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
            if mag == 1:
                body = power
            elif mag.denominator == 1:
                body = "%s%s" % (_num(mag), power)
            else:
                body = "(%s)%s" % (_num(mag), power)
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


def discriminant(a, b, c):
    return b * b - 4 * a * c


def quadratic_roots(a, b, c):
    r = sqrt_exact(discriminant(a, b, c))
    return [(-b + r) / (2 * a), (-b - r) / (2 * a)]


def root_bound(poly):
    lead = abs(poly.lead)
    return 1 + max((abs(c) / lead for c in poly.c[:-1]), default=ZERO)


def numeric_roots(poly, iterations=800, tol=1e-15):
    coeffs = [complex(x) for x in poly.c]
    n = len(coeffs) - 1
    if n < 1:
        return []
    lead = coeffs[-1]
    monic = [c / lead for c in coeffs]
    radius = float(root_bound(poly))
    guesses = [radius * cmath.exp(1j * (2 * cmath.pi * k / n + 0.4)) for k in range(n)]
    for _ in range(iterations):
        delta = 0.0
        for i in range(n):
            den = 1 + 0j
            for j in range(n):
                if i != j:
                    den *= guesses[i] - guesses[j]
            if den == 0:
                continue
            step = _horner(monic, guesses[i]) / den
            guesses[i] -= step
            delta = max(delta, abs(step))
        if delta < tol:
            break
    return [_polish(poly, z) for z in guesses]


def _horner(coeffs, z):
    total = 0j
    for a in reversed(coeffs):
        total = total * z + a
    return total


def _polish(poly, z, steps=40):
    coeffs = [complex(c) for c in poly.c]
    slope = [complex(c * i) for i, c in enumerate(poly.c)][1:]
    for _ in range(steps):
        d = _horner(slope, z)
        if not d:
            break
        step = _horner(coeffs, z) / d
        z -= step
        if abs(step) <= 1e-17 * max(1.0, abs(z)):
            break
    return z


def sturm_chain(poly):
    chain = [poly, poly.derivative()]
    while chain[-1].degree > 0:
        remainder = chain[-2] % chain[-1]
        if not remainder:
            break
        chain.append(-remainder)
    return chain


def _variations(values):
    count = 0
    previous = 0
    for v in values:
        if v:
            sign = 1 if v > 0 else -1
            if previous and sign != previous:
                count += 1
            previous = sign
    return count


def _variations_at(chain, x):
    return _variations([q.eval(x) for q in chain])


def _variations_at_infinity(chain, sign):
    return _variations([q.lead if sign > 0 or q.degree % 2 == 0 else -q.lead for q in chain])


def count_real_roots(poly):
    chain = sturm_chain(poly)
    return _variations_at_infinity(chain, -1) - _variations_at_infinity(chain, 1)


def isolate_real_roots(poly):
    chain = sturm_chain(poly)
    bound = root_bound(poly) + 1
    pending = [(-bound, bound, _variations_at(chain, -bound), _variations_at(chain, bound))]
    out = []
    while pending:
        a, b, va, vb = pending.pop()
        count = va - vb
        if count == 1:
            out.append((a, b))
        elif count > 1:
            mid = (a + b) / 2
            while not poly.eval(mid):
                mid = (mid + b) / 2
            vm = _variations_at(chain, mid)
            pending.append((a, mid, va, vm))
            pending.append((mid, b, vm, vb))
    out.sort()
    return out


def refine_real_root(poly, a, b, bits=64):
    if not poly.eval(b):
        return float(b)
    rising = poly.eval(a) < 0
    width = Fraction(1, 1 << bits)
    while b - a > width:
        mid = (a + b) / 2
        value = poly.eval(mid)
        if not value:
            return float(mid)
        if (value < 0) == rising:
            a = mid
        else:
            b = mid
    return float((a + b) / 2)


def approximate_roots(poly):
    real = [refine_real_root(poly, a, b) for a, b in isolate_real_roots(poly)]
    missing = poly.degree - len(real)
    if not missing:
        return real, []
    candidates = sorted(numeric_roots(poly), key=lambda z: -abs(z.imag))[:missing]
    upper = sorted((z for z in candidates if z.imag > 0), key=lambda z: (z.real, z.imag))
    paired = [w for z in upper for w in (z, z.conjugate())]
    if len(paired) != missing:
        paired = sorted(candidates, key=lambda z: (z.real, -z.imag))
    return real, paired


def from_roots(roots):
    p = Poly([1])
    for r in roots:
        p = p * Poly([-r, 1])
    return p
