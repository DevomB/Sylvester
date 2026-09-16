from __future__ import annotations

import math
import random
from fractions import Fraction
from itertools import combinations

from .polynomial import Poly, rational_roots

SEED = 20260916


def _odd_primes(limit):
    flags = bytearray([1]) * (limit + 1)
    flags[0] = flags[1] = 0
    for i in range(2, math.isqrt(limit) + 1):
        if flags[i]:
            flags[i * i::i] = bytearray(len(range(i * i, limit + 1, i)))
    return [i for i in range(3, limit + 1) if flags[i]]


PRIMES = _odd_primes(4000)


def _trim(a):
    while a and not a[-1]:
        a.pop()
    return a


def _reduce(a, p):
    return _trim([x % p for x in a])


def _add(a, b, p):
    if len(a) < len(b):
        a, b = b, a
    out = list(a)
    for i, y in enumerate(b):
        out[i] = (out[i] + y) % p
    return _trim(out)


def _sub(a, b, p):
    n = max(len(a), len(b))
    return _trim([((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % p for i in range(n)])


def _mul(a, b, p):
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return _trim([x % p for x in out])


def _mul_exact(a, b):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] += x * y
    return out


def _divmod(a, b, p):
    rem = list(a)
    db = len(b) - 1
    inv = pow(b[-1], -1, p)
    quot = [0] * max(0, len(rem) - db)
    for i in range(len(rem) - 1, db - 1, -1):
        k = rem[i] * inv % p
        if k:
            quot[i - db] = k
            for j, y in enumerate(b):
                rem[i - db + j] -= k * y
    return _trim(quot), _reduce(rem[:db], p)


def _monic(a, p):
    inv = pow(a[-1], -1, p)
    return [x * inv % p for x in a]


def _gcd(a, b, p):
    while b:
        a, b = b, _divmod(a, b, p)[1]
    return _monic(a, p) if a else a


def _derivative(a, p):
    return _trim([i * a[i] % p for i in range(1, len(a))])


def _powmod(base, exponent, modulus, p):
    result = [1]
    base = _divmod(base, modulus, p)[1]
    while exponent:
        if exponent & 1:
            result = _divmod(_mul(result, base, p), modulus, p)[1]
        base = _divmod(_mul(base, base, p), modulus, p)[1]
        exponent >>= 1
    return result


def _distinct_degree(f, p):
    out = []
    h = [0, 1]
    i = 0
    while len(f) - 1 >= 2 * (i + 1):
        i += 1
        h = _powmod(h, p, f, p)
        g = _gcd(f, _sub(h, [0, 1], p), p)
        if len(g) > 1:
            out.append((g, i))
            f = _divmod(f, g, p)[0]
            h = _divmod(h, f, p)[1]
    if len(f) > 1:
        out.append((f, len(f) - 1))
    return out


def _equal_degree(g, d, p, rng):
    n = len(g) - 1
    if n == d:
        return [g]
    exponent = (p ** d - 1) // 2
    while True:
        a = _trim([rng.randrange(p) for _ in range(n)])
        if len(a) < 2:
            continue
        h = _gcd(g, _sub(_powmod(a, exponent, g, p), [1], p), p)
        if 1 < len(h) < len(g):
            return _equal_degree(h, d, p, rng) + _equal_degree(_divmod(g, h, p)[0], d, p, rng)


def _factor_mod_p(f, p, rng):
    out = []
    for g, d in _distinct_degree(_monic(f, p), p):
        out.extend(_equal_degree(g, d, p, rng))
    return out


def _bezout(a, b, p):
    r0, r1 = a, b
    s0, s1 = [1], []
    t0, t1 = [], [1]
    while r1:
        q, r = _divmod(r0, r1, p)
        r0, r1 = r1, r
        s0, s1 = s1, _sub(s0, _mul(q, s1, p), p)
        t0, t1 = t1, _sub(t0, _mul(q, t1, p), p)
    inv = pow(r0[0], -1, p)
    return [x * inv % p for x in s0], [x * inv % p for x in t0]


def _lift(target, g, h, p, k):
    s, t = _bezout(g, h, p)
    g0, h0 = g, h
    m = p
    for _ in range(k - 1):
        nxt = m * p
        gh = _mul_exact(g, h)
        size = max(len(target), len(gh))
        e = _trim([
            ((target[i] if i < len(target) else 0) - (gh[i] if i < len(gh) else 0)) % nxt // m
            for i in range(size)
        ])
        if e:
            quot, dg = _divmod(_mul(t, e, p), g0, p)
            dh = _add(_mul(s, e, p), _mul(quot, h0, p), p)
            g = _trim([((g[i] if i < len(g) else 0) + m * (dg[i] if i < len(dg) else 0)) % nxt
                       for i in range(max(len(g), len(dg)))])
            h = _trim([((h[i] if i < len(h) else 0) + m * (dh[i] if i < len(dh) else 0)) % nxt
                       for i in range(max(len(h), len(dh)))])
        m = nxt
    return g, h


def _hensel(f, factors, p, k):
    modulus = p ** k
    target = [c % modulus for c in f]
    lifted = []
    for index in range(len(factors) - 1):
        h = [target[-1] % p]
        for other in factors[index + 1:]:
            h = _mul(h, other, p)
        g, target = _lift(target, factors[index], h, p, k)
        lifted.append(g)
    inv = pow(target[-1], -1, modulus)
    lifted.append([c * inv % modulus for c in target])
    return lifted


def _primitive(a):
    g = 0
    for c in a:
        g = math.gcd(g, c)
    a = [c // g for c in a] if g > 1 else list(a)
    return [-c for c in a] if a[-1] < 0 else a


def _exact_div(f, g):
    rem = list(f)
    dg = len(g) - 1
    lead = g[-1]
    quot = [0] * (len(rem) - dg)
    for i in range(len(rem) - 1, dg - 1, -1):
        c = rem[i]
        if c % lead:
            return None
        c //= lead
        quot[i - dg] = c
        if c:
            for j, y in enumerate(g):
                rem[i - dg + j] -= c * y
    return None if any(rem[:dg]) else quot


def _recombine(f, lifted, modulus, allowed):
    half = modulus // 2
    found = []
    remaining = list(range(len(lifted)))
    size = 1
    while 2 * size <= len(remaining):
        hit = None
        for subset in combinations(remaining, size):
            if not allowed >> sum(len(lifted[i]) - 1 for i in subset) & 1:
                continue
            g = [f[-1] % modulus]
            for i in subset:
                g = _mul(g, lifted[i], modulus)
            g = _primitive([c - modulus if c > half else c for c in g])
            if not g[0] or f[0] % g[0]:
                continue
            quotient = _exact_div(f, g)
            if quotient is not None:
                hit = subset
                found.append(g)
                f = quotient
                break
        if hit is None:
            size += 1
        else:
            remaining = [i for i in remaining if i not in hit]
    if len(f) > 1:
        found.append(_primitive(f))
    return found


def _zassenhaus(f):
    n = len(f) - 1
    lead = f[-1]
    rng = random.Random(SEED)
    allowed = (1 << (n + 1)) - 1
    proper = ((1 << (n - 1)) - 1) << 1
    best = None
    good = 0
    for p in PRIMES:
        if lead % p == 0:
            continue
        fp = _reduce(f, p)
        if len(_gcd(fp, _derivative(fp, p), p)) != 1:
            continue
        modular = _factor_mod_p(fp, p, rng)
        mask = 1
        for g in modular:
            mask |= mask << (len(g) - 1)
        allowed &= mask
        if not allowed & proper:
            return [f]
        if best is None or len(modular) < len(best[1]):
            best = (p, modular)
        good += 1
        if good == 5:
            break
    p, modular = best
    bound = 2 * lead * (1 << n) * (math.isqrt(sum(c * c for c in f)) + 1)
    k = 1
    modulus = p
    while modulus <= bound:
        modulus *= p
        k += 1
    return _recombine(f, _hensel(f, modular, p, k), modulus, allowed)


def _factor_squarefree(f):
    out = []
    if not f[0]:
        out.append([0, 1])
        f = f[1:]
    if len(f) < 2:
        return out
    for root, _ in rational_roots(Poly(f)):
        linear = [-root.numerator, root.denominator]
        f = _exact_div(f, linear)
        out.append(linear)
    if len(f) < 2:
        return out
    if len(f) <= 4:
        out.append(f)
        return out
    out.extend(_zassenhaus(f))
    return out


def squarefree_decomposition(poly):
    out = []
    g = poly.gcd(poly.derivative())
    w = poly / g
    multiplicity = 1
    while w.degree > 0:
        y = w.gcd(g)
        z = w / y
        if z.degree > 0:
            out.append((z, multiplicity))
        multiplicity += 1
        w = y
        g = g / y
    return out


def factor(poly):
    if not isinstance(poly, Poly):
        poly = Poly(poly)
    if poly.degree < 1:
        return (poly.lead if poly else Fraction(0)), []
    factors = []
    for part, multiplicity in squarefree_decomposition(poly):
        primitive, _ = part.primitive()
        for irreducible in _factor_squarefree([int(c) for c in primitive.c]):
            factors.append((Poly(irreducible), multiplicity))
    factors.sort(key=lambda item: (item[0].degree, item[0].c, item[1]))
    product = Poly([1])
    for q, m in factors:
        product = product * q ** m
    return poly.lead / product.lead, factors


def expand(unit, factors):
    product = Poly([unit])
    for q, m in factors:
        product = product * q ** m
    return product


def is_irreducible(poly):
    _, factors = factor(poly)
    return len(factors) == 1 and factors[0][1] == 1
