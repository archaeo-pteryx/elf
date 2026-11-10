"""The Legendre representation 'leg' of the angular integrals of the LPT final integral (sympy, exact), angular.dat.

The final integral needs, per point (k, mu) and per q, the average over the directions of q of a moment's angular
polynomial P_j = sum_t coef_t mu_K^a mu_x^b (lpt_model.DENSITY_MOMENTS, degree n = a + b) against the plane wave and
the damping, in the frame of moments.dat (mu_K = Khat.qhat, mu_x = e_x.qhat, X = khat.qhat = c mu_K + s mu_x,
rho = k q, B = -(1/2) K^2 Y_lt(q) <= 0).  The plane-wave expansion
e^{i rho X} = sum_L (2L + 1) i^L j_L(rho) P_L(X) gives

    (-i)^(n mod 2) <P_j e^{i rho X} e^{B mu_K^2}> = sum_L j_L(rho) C_L^j,
    C_L^j = (2L + 1) i^L (-i)^(n mod 2) <P_L(X) P_j e^{B mu_K^2}>,

with <.> = int d mu_K / 2 int d phi / (2 pi).  The parity q -> -q leaves L = n mod 2 only, where the phase is
phi_L = i^L (-i)^(n mod 2) = (-1)^floor(L/2).  With mu_x = sqrt(1 - mu_K^2) cos phi, the addition theorem
P_L(X) = sum_r eps_r (L - r)!/(L + r)! P_L^r(c) P_L^r(mu_K) cos(r phi) (P_L^r(x) = (1 - x^2)^(r/2) d^r P_L/dx^r, so
P_L^r(c) = s^r P_L^(r)(c), eps_0 = 1 and eps_r = 2 for r > 0) and the azimuthal average
<cos(r phi) cos^b(phi)> = 2^-b binom(b, (b - r)/2) (r = b mod 2, r <= b) split C_L into the per-point factors
s^r P_L^(r)(c) and an exact table,

    C_L^j = sum_t coef_t sum_r s^r P_L^(r)(c) sum_i T^(a_t,b_t,r)_(L,i) J_i(B),
    J_i(B) = int_{-1}^{1} du P_i(u) e^{B u^2},
    T^(a,b,r)_(L,i) = phi_L ((2L + 1)/2) kappa_(b,r) (L - r)!/(L + r)! R^(a,b,r)_(L,i),
    kappa_(b,r) = eps_r 2^-b binom(b, (b - r)/2),

P_L^(r) the r-th derivative of P_L, r in {b, b - 2, ..} >= 0 and R^(a,b,r)_(L,i) the coefficient of P_i in the
polynomial u^a (1 - u^2)^((b + r)/2) P_L^(r)(u).  R is banded, |i - L| <= a + b with i even.  The r integrations by
parts make (1 - u^2)^r P_L^(r) orthogonal to the P_i with |i - L| > r, the remaining factor u^a (1 - u^2)^((b - r)/2)
has degree a + b - r, and the polynomial is even for L = n mod 2.

The checks are ``check_addition_theorem`` (the azimuthal average behind T, a polynomial identity in c, s, u),
``check_band`` (the band of a table), ``check_against_quadrature`` (C_L from a table with J_i by scipy quad against a
direct 2-D quadrature of its definition) and ``check_series`` (sum_L j_L C_L against the direct angular average, which
fixes the phases).  ``pt_autogen.lpt_verify.angular`` runs them on the written angular.dat.
"""
import math
import os
from fractions import Fraction

import numpy as np
import sympy as sp

from . import emit
from . import lpt_model as model

# Convention.  A type (a, b) is the monomial mu_K^a mu_x^b of the declarations, the paper's (m, n) of I_{m,n}, whose
# azimuth phi has mu_x = -sqrt(1 - mu_K^2) cos phi.  The derivation writes mu_x = sqrt(1 - mu_K^2) cos phi'
# (phi' = phi + pi), so X = c mu_K + s mu_x and the factor (-1)^r of the paper's form of the addition theorem does
# not enter T.
u, c, s = sp.symbols('u c s', real=True)

L_TABLE = 40
"""The highest order L of angular.dat (a runtime lmax up to L_TABLE)."""

FILE = emit.LPT_DIR + '/angular.dat'
"""The table's path relative to the table directory."""

QUADRATURE_LMAX = 8
"""The orders L of the quadrature check (and of the addition theorem)."""

SERIES_LMAX = 24
"""The truncation of the plane-wave series in the series check."""


# ------------------------------------------------------------------ the exact table
def types_of(declaration):
    """The types (a, b, r) of a declaration {moment: ((coef, a, b, b_y), ..)}, r in {b, b - 2, ..} >= 0, ordered by
    (a + b, b, a, r)."""
    out = set()
    for terms in declaration.values():
        for _, a, b, by in terms:
            assert by == 0, 'the table has no out-of-plane monomials mu_y'
            out.update((a, b, r) for r in range(b % 2, b + 1, 2))
    return sorted(out, key=lambda t: (t[0] + t[1], t[1], t[0], t[2]))


def kappa(b, r):
    """kappa_(b,r) = eps_r 2^-b binom(b, (b - r)/2) = eps_r <cos(r phi) cos^b(phi)>_phi (eps_0 = 1, eps_r = 2)."""
    return sp.Rational(1 if r == 0 else 2, 2**b) * sp.binomial(b, (b - r) // 2)


def phase(L):
    """phi_L = i^L (-i)^(n mod 2) = (-1)^floor(L/2), for L = n mod 2."""
    return (-1)**(L // 2)


def legendre(L):
    """P_L(u) as an exact sympy Poly."""
    return sp.legendre_poly(L, u, polys=True)


def legendre_coefficients(poly):
    """{i: R_i} with poly(u) = sum_i R_i P_i(u) (a sympy Poly in u), exact, by
    u^p = sum_l (2l + 1) p! / (2^h h! (p + l + 1)!!) P_l(u), h = (p - l)/2, asserted to rebuild poly."""
    out = {}
    for (p,), cp in poly.terms():
        for l in range(p % 2, p + 1, 2):
            h = (p - l) // 2
            out[l] = out.get(l, 0) + cp * sp.Rational((2 * l + 1) * math.factorial(p),
                                                      2**h * math.factorial(h) * int(sp.factorial2(p + l + 1)))
    out = {i: sp.Rational(v) for i, v in out.items() if v != 0}
    assert sum((v * legendre(i) for i, v in out.items()), sp.Poly(0, u)) == poly, 'Legendre expansion'
    return out


def R_row(a, b, r, L):
    """{i: R^(a,b,r)_(L,i)}, the P_i coefficients of u^a (1 - u^2)^((b + r)/2) P_L^(r)(u) (empty for L < r), asserted
    banded."""
    if L < r:
        return {}
    poly = sp.Poly(u**a * (1 - u**2)**((b + r) // 2), u) * (legendre(L).diff((u, r)) if r else legendre(L))
    R = legendre_coefficients(poly)
    assert all(abs(i - L) <= a + b and (i - L - a - b) % 2 == 0 for i in R), ('band', a, b, r, L, sorted(R))
    return R


def T_row(a, b, r, L):
    """{i: T^(a,b,r)_(L,i)} as Fractions (empty for L != a + b mod 2 or L < r)."""
    if (L - a - b) % 2 or L < r:
        return {}
    pref = (phase(L) * sp.Rational(2 * L + 1, 2) * kappa(b, r)
            * sp.Rational(math.factorial(L - r), math.factorial(L + r)))
    return {i: emit.to_fraction(pref * R) for i, R in R_row(a, b, r, L).items()}


def table(declaration, L_table=L_TABLE):
    """{(a, b, r): {(L, i): T^(a,b,r)_(L,i)}} (Fractions, L <= L_table, by L and then i) of the types of
    ``declaration``."""
    return {t: {(L, i): v for L in range(L_table + 1) for i, v in sorted(T_row(*t, L).items())}
            for t in types_of(declaration)}


def read_table(table_dir):
    """The table of ``<table_dir>/lpt/angular.dat`` as {(a, b, r): {(L, i): Fraction}}."""
    out = {}
    for a, b, r, L, i, coef in emit.read_text_table(os.path.join(table_dir, FILE)):
        out.setdefault((int(a), int(b), int(r)), {})[(int(L), int(i))] = Fraction(coef)
    return out


# ------------------------------------------------------------------ symbolic checks
def check_addition_theorem(a, b, Lmax=QUADRATURE_LMAX):
    """Assert the azimuthal average behind T, for w = sqrt(1 - u^2) cos phi, s^2 = 1 - c^2 and L <= Lmax,
        <P_L(c u + s w) u^a w^b>_phi
            = sum_r kappa_(b,r) (L - r)!/(L + r)! s^r P_L^(r)(c) u^a (1 - u^2)^((b + r)/2) P_L^(r)(u)
    (a polynomial identity in c, s, u), and return the number of orders L checked."""
    cp, sq = sp.symbols('cp sq', real=True)              # cos(phi) and sqrt(1 - u^2)
    for L in range(Lmax + 1):
        lhs = sp.expand(sp.legendre(L, c * u + s * sq * cp) * u**a * (sq * cp)**b)
        # the azimuthal average of cos^k(phi) is (k - 1)!!/k!! for even k and 0 for odd k
        avg = sp.expand(sum(coef * sp.factorial2(k - 1) / sp.factorial2(k)
                            for (k,), coef in sp.Poly(lhs, cp).terms() if k % 2 == 0))
        # after the average sq has even powers only (k + b even)
        terms = sp.Poly(avg, sq).terms()
        assert all(k % 2 == 0 for (k,), _ in terms), (a, b, L)
        avg_u = sum(coef * (1 - u**2)**(k // 2) for (k,), coef in terms)
        rhs = sum(kappa(b, r) * sp.factorial(L - r) / sp.factorial(L + r) * s**r * sp.diff(sp.legendre(L, c), c, r)
                  * u**a * (1 - u**2)**((b + r) // 2) * sp.diff(sp.legendre(L, u), u, r)
                  for r in range(b % 2, min(b, L) + 1, 2))
        # reduced modulo s^2 = 1 - c^2, s^k -> s^(k mod 2) (1 - c^2)^(k div 2)
        diff = sp.Poly(sp.expand(avg_u - rhs), s).terms()
        reduced = sp.expand(sum(coef * s**(k % 2) * (1 - c**2)**(k // 2) for (k,), coef in diff))
        assert reduced == 0, ('addition theorem', a, b, L, reduced)
    return Lmax + 1


def check_band(T):
    """Assert that every entry of the table ``T`` ({(a, b, r): {(L, i): value}}) is a nonzero value with
    L = a + b mod 2, L >= r, i even and |i - L| <= a + b, and return the number of entries."""
    n = 0
    for (a, b, r), entries in T.items():
        for (L, i), v in entries.items():
            assert v != 0 and (L - a - b) % 2 == 0 and L >= r and i % 2 == 0 and abs(i - L) <= a + b, \
                ('band', a, b, r, L, i)
            n += 1
    return n


# ------------------------------------------------------------------ numeric checks (numpy, scipy)
def geometry(f, mu):
    """(c, s, A_mu, B_mu) of the frame of lpt._rsd_geometry at the growth rate f and mu."""
    Kfac, sin = math.sqrt(1 + f * (2 + f) * mu**2), math.sqrt(1 - mu**2)
    return (1 + f * mu**2) / Kfac, f * mu * sin / Kfac, (1 + f) * mu / Kfac, sin / Kfac


QUADRATURE_POINTS = (geometry(0.8, 0.6) + (-1.3,), (0.3, math.sqrt(0.91), 0.7, 0.4, -0.05),
                     (0.999, math.sqrt(1 - 0.999**2), 0.2, 0.98, -6.0), (1.0, 0.0, 0.6, 0.8, -12.0))
"""Points (c, s, A_mu, B_mu, B) of the quadrature check, a physical geometry, a generic angle, a nearly parallel K and
k, and K parallel to k (s = 0).  A_mu, B_mu are free coefficients there (the identity is linear in them), chosen so that
no moment vanishes identically."""

SERIES_POINTS = tuple(g + (rho, B) for g in (geometry(0.8, 0.6), (0.3, math.sqrt(0.91), 0.7, 0.4))
                      for B in (-0.1, -3.0) for rho in (0.3, 2.0, 6.0))
"""Points (c, s, A_mu, B_mu, rho, B) of the series check, two geometries of QUADRATURE_POINTS, rho in {0.3, 2, 6} and
B in {-0.1, -3}."""


def exp_moments(B, imax):
    """[J_i(B) for i <= imax], J_i = int_{-1}^{1} du P_i(u) e^{B u^2} by scipy.integrate.quad (zero for odd i)."""
    from scipy.integrate import quad
    from scipy.special import eval_legendre
    J = np.zeros(imax + 1)
    for i in range(0, imax + 1, 2):
        J[i] = quad(lambda x: eval_legendre(i, x) * math.exp(B * x * x), -1, 1, epsabs=1e-14, epsrel=1e-13,
                    limit=200)[0]
    return J


def moment_value(coef, Am, Bm):
    """The float value of a declared coefficient (a number or a polynomial in lpt_model.A_mu, B_mu)."""
    return float(sp.sympify(coef).subs({model.A_mu: Am, model.B_mu: Bm}))


def coefficients(T, terms, point, Lmax):
    """C_L^j for L <= Lmax of a moment (its declared terms) at ``point`` = (c, s, A_mu, B_mu, B), from the table ``T``
    with J_i by ``exp_moments`` (float64)."""
    c_, s_, Am, Bm, B = point
    J = exp_moments(B, Lmax + max(a + b for _, a, b, _ in terms))
    rmax = max(b for _, _, b, _ in terms)
    dP = [[np.polynomial.legendre.Legendre.basis(L).deriv(r)(c_) for r in range(rmax + 1)] for L in range(Lmax + 1)]
    C = np.zeros(Lmax + 1)
    for coef, a, b, _ in terms:
        k = moment_value(coef, Am, Bm)
        for r in range(b % 2, b + 1, 2):
            for (L, i), v in T[(a, b, r)].items():
                if L <= Lmax:
                    C[L] += k * s_**r * dP[L][r] * float(v) * J[i]
    return C


def directions(n_u=96, nphi=64):
    """(mu_K, mu_x, weight) arrays (n_u, nphi) of the direct angular quadrature, with Gauss-Legendre nodes in mu_K, the
    trapezoid in phi (exact for trigonometric polynomials of degree < nphi), mu_x = -sqrt(1 - mu_K^2) cos phi (the
    paper's azimuth, X = c mu_K + s mu_x) and the weights of <.> = int d mu_K / 2 int d phi / (2 pi)."""
    x, w = np.polynomial.legendre.leggauss(n_u)
    phi = 2 * np.pi * np.arange(nphi) / nphi
    muK = np.repeat(x[:, None], nphi, axis=1)
    return muK, -np.sqrt(1 - muK**2) * np.cos(phi), np.outer(w / 2, np.full(nphi, 1 / nphi))


def _polynomial(terms, muK, mux, Am, Bm):
    """The moment polynomial sum_t coef_t mu_K^a mu_x^b on the quadrature nodes, and its degree mod 2."""
    assert len({(a + b) % 2 for _, a, b, _ in terms}) == 1, 'a moment has one parity'
    return sum(moment_value(coef, Am, Bm) * muK**a * mux**b for coef, a, b, _ in terms), (terms[0][1] + terms[0][2]) % 2


def check_against_quadrature(declaration, points, T=None):
    """Assert that C_L^j of the table ``T`` (default: derived here for L <= QUADRATURE_LMAX) with J_i by scipy quad
    equals the direct 2-D quadrature of (2L + 1) i^L (-i)^(n mod 2) <P_L(X) P_j e^{B mu_K^2}> for L <= QUADRATURE_LMAX,
    every moment j of ``declaration`` and the ``points`` (c, s, A_mu, B_mu, B), to 1e-12 of max_L |C_L^j|, and return
    the worst relative error."""
    T = table(declaration, QUADRATURE_LMAX) if T is None else T
    muK, mux, w = directions()
    worst = 0.0
    for point in points:
        c_, s_, Am, Bm, B = point
        X = c_ * muK + s_ * mux
        PL = [np.polynomial.legendre.legval(X, [0] * L + [1]) for L in range(QUADRATURE_LMAX + 1)]
        for terms in declaration.values():
            poly, odd = _polynomial(terms, muK, mux, Am, Bm)
            direct = np.array([(2 * L + 1) * 1j**L * (-1j)**odd * np.sum(w * PL[L] * poly * np.exp(B * muK**2))
                               for L in range(QUADRATURE_LMAX + 1)])
            err = np.max(np.abs(coefficients(T, terms, point, QUADRATURE_LMAX) - direct)) / np.max(np.abs(direct))
            worst = max(worst, float(err))
    assert worst < 1e-12, f'C_L of the table vs the direct quadrature: {worst:.2e}'
    return worst


def check_series(declaration, points, T=None):
    """Assert that sum_{L <= SERIES_LMAX} j_L(rho) C_L^j, with C_L^j of the table ``T`` (default: derived here) as in
    ``check_against_quadrature``, equals the direct average (-i)^(n mod 2) <P_j e^{i rho X} e^{B mu_K^2}> (complex, so
    the phases are fixed) for every moment j and the ``points`` (c, s, A_mu, B_mu, rho, B), to 1e-10 of the largest
    term max_L |j_L(rho) C_L^j|, and return the worst relative error."""
    from scipy.special import spherical_jn
    T = table(declaration, SERIES_LMAX) if T is None else T
    muK, mux, w = directions()
    jL = np.arange(SERIES_LMAX + 1)
    worst = 0.0
    for c_, s_, Am, Bm, rho, B in points:
        X = c_ * muK + s_ * mux
        for terms in declaration.values():
            poly, odd = _polynomial(terms, muK, mux, Am, Bm)
            direct = (-1j)**odd * np.sum(w * poly * np.exp(1j * rho * X + B * muK**2))
            series = spherical_jn(jL, rho) * coefficients(T, terms, (c_, s_, Am, Bm, B), SERIES_LMAX)
            worst = max(worst, float(abs(series.sum() - direct) / np.max(np.abs(series))))
    assert worst < 1e-10, f'the plane-wave series of the table vs the direct average: {worst:.2e}'
    return worst
