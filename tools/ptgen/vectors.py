"""Symbolic wavevector algebra of the one-loop kernels (port of the prototype's kernels.py).

Every wavevector is an integer combination ``a k + b p + c r`` of the basis vectors ``k``
(external), ``p`` (loop momentum) and ``r`` (a third, generic vector used only by the tests).
Scalar products are expressed through the magnitudes ``k``, ``p``, ``r`` and the cosines
``x = khat . phat``, ``y = khat . rhat``, ``z = phat . rhat``; projections onto the line of
sight ``n`` through ``k mu`` (``mu = khat . n``), ``np`` (= ``n . p``) and ``nr`` (= ``n . r``).
Because the wavevectors are integer combinations, a vanishing sum such as ``p + (-p)`` is
detected exactly, which the vanishing-momentum rule of :mod:`ptgen.kernels` needs.  The
tables use ``k`` and ``p`` only (``c = 0``), for which every expression is the prototype's.
"""
from functools import lru_cache

import sympy as sp

k, p, q, x, mu = sp.symbols('k p q x mu', positive=True)
np_ = sp.Symbol('np')          # n . p (line-of-sight projection of p)
r, y, z = sp.symbols('r y z', positive=True)
nr_ = sp.Symbol('nr')          # n . r (tests only)


class Vec(tuple):
    """Integer combination ``(a, b, c)`` of the basis vectors ``(k, p, r)``."""

    def __new__(cls, a, b, c=0):
        return super().__new__(cls, (int(a), int(b), int(c)))

    def __add__(self, other):
        return Vec(self[0] + other[0], self[1] + other[1], self[2] + other[2])

    def __neg__(self):
        return Vec(-self[0], -self[1], -self[2])

    def __sub__(self, other):
        return self + (-other)

    def is_zero(self):
        return self[0] == 0 and self[1] == 0 and self[2] == 0


K_VEC = Vec(1, 0)
P_VEC = Vec(0, 1)
R_VEC = Vec(0, 0, 1)


def vsum(vs):
    """Sum of the wavevectors ``vs``."""
    out = Vec(0, 0)
    for v in vs:
        out = out + v
    return out


def dot(u, v):
    """Scalar product ``u . v``."""
    return (u[0] * v[0] * k**2 + u[1] * v[1] * p**2 + u[2] * v[2] * r**2
            + (u[0] * v[1] + u[1] * v[0]) * k * p * x
            + (u[0] * v[2] + u[2] * v[0]) * k * r * y
            + (u[1] * v[2] + u[2] * v[1]) * p * r * z)


def norm2(u):
    """``|u|^2``."""
    return dot(u, u)


def ndot(u):
    """``n . u``."""
    return u[0] * k * mu + u[1] * np_ + u[2] * nr_


# ---------------------------------------------------------------------------
# line-of-sight average

@lru_cache(maxsize=None)
def los_moment(j):
    """Average of ``(n . phat)^j`` over the azimuth of ``n`` about ``khat``.

    ``n . phat = x mu + sqrt(1-x^2) sqrt(1-mu^2) cos(phi)``; ``<cos^{2i}> = binom(2i, i) / 4^i``.
    """
    out = sp.Integer(0)
    for i in range(j // 2 + 1):
        out += (sp.binomial(j, 2 * i) * (x * mu)**(j - 2 * i)
                * ((1 - x**2) * (1 - mu**2))**i * sp.binomial(2 * i, i) / 4**i)
    return sp.expand(out)


def los_average(expr):
    """Replace the powers of ``np = n . p`` by their azimuthal average (times ``p^j``).

    ``expr`` must be polynomial in ``np`` once put over a common denominator (true for all
    kernels here: ``np`` only appears in numerators).
    """
    num, den = sp.fraction(sp.together(expr))
    if den.has(np_):
        raise ValueError('np appears in a denominator')
    poly = sp.Poly(sp.expand(num), np_)
    out = sp.Integer(0)
    for (j,), c in poly.terms():
        out += c * p**j * los_moment(j)
    return out / den


def monomial_coefficients(expr, gens):
    """``{exponent tuple: coefficient}`` of a polynomial in ``gens``."""
    poly = sp.Poly(sp.expand(expr), *gens)
    return dict(poly.terms())
