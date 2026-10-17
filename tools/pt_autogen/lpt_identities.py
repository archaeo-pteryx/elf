"""Identities behind the LPT row recipes, proved or derived by sympy.

* ``named_sources`` and ``decompose``: the declared source names of lpt_model.SOURCE_NAMES reduced like the derived
  sources, and a source as their unique combination.
* ``full_integral_zero``, ``full_integral_ratio``: identities of full-k integrals M_n[S_a] = c M_n[S_b], proved by
  symbolic angular integration.  M_{-2}[S] = int_{p1} int_{p2} f(p1, p2, m) P(p1) P(p2), and int_{-1}^1 dm of the
  p1 <-> p2 symmetrised kernel is computed in closed form (rational part + coefficient of log((p1 + p2)/|p1 - p2|)).
* ``bessel_identities``: the recurrences of j_l behind H_l^{n-1} = q/(2l+1)(H_{l-1}^n + H_{l+1}^n), the ladder
  H_{l+-1}^{n+1} and the finite-range kernels, on sympy's closed forms of j_l.
* ``ladder`` and ``phi_form``: every xi_l^n of the xi_2^{-2} family as a polynomial in phi = xi_2^{-2} and its
  derivatives.  ``total_derivative`` finds G with G' = target in the ansatz of lpt_model.TOTAL_DERIVATIVE_ANSATZ.
* ``finite_range_xi_form``: the q-space closed forms of the rows of the Gauss-law kind (H_1^-1 and H_2^-2 of a 22
  source) in the product variables of lpt_model.PRODUCT_VARIABLES, and ``finite_range_check`` the four finite-range
  formulas on a Gaussian at 30 digits (mpmath).
"""
from collections import defaultdict
from functools import lru_cache

import sympy as sp

from . import lpt_model as model
from . import reduction as RD
from .lpt_reduce import QS, _clean, add_sources, xi, xi_index
from .vectors import x

# ---------------------------------------------------------------------------------------------------------------
# the named sources

def normalised_22(F):
    """``(degree, source)`` of a 22 integrand F(k, p, x), its blocks ('22', l, n) normalised to degree 0."""
    blocks = RD.reduce22(sp.cancel(F))
    degs = {m + 2 * n for (l, n, m) in blocks}
    assert len(degs) == 1, degs
    return degs.pop(), _clean({('22', l, n): c for (l, n, m), c in blocks.items()})


def normalised_13(F):
    """``(degree, source)`` of a 13 integrand F(k, p, x), its canonical blocks ('13', l, n) normalised to degree 0."""
    F = sp.cancel((F + F.subs(x, -x)) / 2)
    blocks, zl, _ = RD.reduce13(F)
    assert not zl, zl
    blocks = RD.canonical13(blocks)
    degs = {m + n - 2 for (l, n, m) in blocks}
    assert len(degs) == 1, degs
    return degs.pop(), _clean({('13', l, n): c for (l, n, m), c in blocks.items()})


def normalised_source(kind, F):
    return normalised_22(F) if kind == '22' else normalised_13(F)


@lru_cache(maxsize=None)
def named_sources():
    """{name: (degree, normalised source)} of lpt_model.SOURCE_NAMES."""
    return {name: normalised_source(kind, F) for name, (kind, F) in model.SOURCE_NAMES.items()}


def decompose(src, names=None):
    """{name: coefficient} with src = sum coefficient * source(name), or None if src is outside their span."""
    named = named_sources()
    names = tuple(names or named)
    a = sp.symbols(f'a0:{len(names)}')
    keys = set(src)
    for nm in names:
        keys |= set(named[nm][1])
    eqs = [sum(ai * named[nm][1].get(key, 0) for ai, nm in zip(a, names)) - src.get(key, 0) for key in keys]
    sol = sp.solve(eqs, a, dict=True)
    if not sol:
        return None
    sol = sol[0]
    assert set(sol) == set(a), ('not unique', sol)
    return {nm: sol[ai] for ai, nm in zip(a, names) if sol[ai] != 0}


def named_to_blocks(named):
    """The block source of a combination {name: coefficient} of the named sources."""
    return add_sources(*[(c, named_sources()[s][1]) for s, c in named.items()])


# ---------------------------------------------------------------------------------------------------------------
# full-k integrals by symbolic angular integration

p1, p2, m = sp.symbols('p1 p2 m', positive=True)


def full_integral_kernel(src, n=-2):
    """f(p1, p2, m) with M_n[S] = int_{p1} int_{p2} f P(p1) P(p2), m the cosine between p1 and p2.

    A 22 block c k^{-2nn} B22_{l,nn} (B22 = (-1)^l int_p p^nn |k-p|^nn L_l P P with the (-1)^l inside c) gives, with
    p1 = p and p2 = k - p, f = c K^{n - 2 nn} (p1 p2)^nn L_l(m) and K^2 = p1^2 + p2^2 + 2 p1 p2 m.  A 13 block
    c k^{2-nn} B13_{l,nn} = c k^{2-nn} P(k) int_p p^nn L_l(x) / |k-p|^2 P(p) gives, with p1 = k, p2 = p and m = x,
    f = c p1^{n + 2 - nn} p2^nn L_l(m) / (p1^2 + p2^2 - 2 p1 p2 m)."""
    f = sp.Integer(0)
    K2 = p1**2 + p2**2 + 2 * p1 * p2 * m
    for (kind, l, nn), c in src.items():
        Ll = sp.legendre(l, m)
        if kind == '22':
            e = n - 2 * nn
            assert e % 2 == 0
            f += c * K2**(e // 2) * (p1 * p2)**nn * Ll
        elif kind == '13':
            f += c * p1**(n + 2 - nn) * p2**nn * Ll / (p1**2 + p2**2 - 2 * p1 * p2 * m)
        else:
            raise ValueError(kind)
    return f


def angular_integral(f):
    """int_{-1}^1 dm f = rational + coefficient * Lambda, Lambda = log((p1 + p2)/|p1 - p2|), for f with denominators
    p1^2 + p2^2 +- 2 p1 p2 m (power 1)."""
    rat, lam = sp.Integer(0), sp.Integer(0)
    for term in sp.Add.make_args(sp.apart(sp.together(f), m)):
        num, den = sp.fraction(sp.factor(term))
        if not den.has(m):
            rat += sp.integrate(sp.expand(term), (m, -1, 1))
            continue
        b, a = sp.Poly(den, m).all_coeffs()
        assert not num.has(m), term
        # int_{-1}^1 dm / (a + b m) = (1/b) log((a + b)/(a - b)), with a = s (p1^2 + p2^2) and b = +-2 s p1 p2 this is
        # Lambda / (s p1 p2) for both signs
        ratio = sp.cancel(a / b)
        assert sp.cancel(ratio**2 - ((p1**2 + p2**2) / (2 * p1 * p2))**2) == 0, (a, b)
        sgn = sp.sign(sp.cancel(ratio))
        assert sgn in (1, -1), ratio
        lam += 2 * num * sgn / b
    return sp.cancel(rat), sp.cancel(lam)


def symmetrise(expr):
    return sp.cancel((expr + expr.subs({p1: p2, p2: p1}, simultaneous=True)) / 2)


def full_integral_zero(src, n=-2):
    """``(zero, (rational part, Lambda coefficient))`` of int dm of the symmetrised kernel of M_n[src]."""
    rat, lam = angular_integral(full_integral_kernel(src, n))
    rat, lam = symmetrise(rat), symmetrise(lam)
    return (rat == 0 and lam == 0), (rat, lam)


def full_integral_ratio(named, ref_named):
    """c with M_{-2}[S] = c M_{-2}[S_ref] for the full integrals of two combinations of the named sources."""
    ra, la = angular_integral(full_integral_kernel(named_to_blocks(named)))
    rb, lb = angular_integral(full_integral_kernel(named_to_blocks(ref_named)))
    ra, la, rb, lb = (symmetrise(v) for v in (ra, la, rb, lb))
    c = sp.cancel(la / lb) if lb != 0 else sp.cancel(ra / rb)
    assert sp.cancel(ra - c * rb) == 0 and sp.cancel(la - c * lb) == 0, (named, ref_named)
    assert not c.free_symbols
    return c


# ---------------------------------------------------------------------------------------------------------------
# identities of j_l (sympy's closed forms)

def bessel_identities(lmax=5):
    """{name: True/False} of the j_l identities the recipes use, checked on sympy's closed forms of j_l."""
    t, kk, qq, rr = sp.symbols('t kk qq rr', positive=True)

    def j(l, arg):
        return sp.expand_func(sp.jn(l, arg))
    out = {}
    for l in range(1, lmax):
        out[f'recurrence l={l}: j_(l-1) + j_(l+1) = (2l+1) j_l / t'] = (
            sp.simplify(j(l - 1, t) + j(l + 1, t) - (2 * l + 1) * j(l, t) / t) == 0)
    for l in range(0, lmax):
        out[f'raise l={l}: d/dq[q^-l j_l(kq)] = -k q^-l j_(l+1)(kq)'] = (
            sp.simplify(sp.diff(qq**(-l) * j(l, kk * qq), qq) + kk * qq**(-l) * j(l + 1, kk * qq)) == 0)
    for l in range(1, lmax):
        out[f'lower l={l}: d/dq[q^(l+1) j_l(kq)] = k q^(l+1) j_(l-1)(kq)'] = (
            sp.simplify(sp.diff(qq**(l + 1) * j(l, kk * qq), qq) - kk * qq**(l + 1) * j(l - 1, kk * qq)) == 0)
    out['finite range: int_0^q r^2 j_0(kr) dr = q^2 j_1(kq)/k'] = sp.simplify(
        sp.integrate(rr**2 * j(0, kk * rr), (rr, 0, qq)) - qq**2 * j(1, kk * qq) / kk) == 0
    out['finite range: int_0^q r^2 (q^2 - r^2) j_0(kr) dr = 2 q^3 j_2(kq)/k^2'] = sp.simplify(
        sp.integrate(rr**2 * (qq**2 - rr**2) * j(0, kk * rr), (rr, 0, qq)) - 2 * qq**3 * j(2, kk * qq) / kk**2) == 0
    return out


# ---------------------------------------------------------------------------------------------------------------
# the xi ladder and phi = xi_2^{-2}

PHI = sp.symbols('phi0:5')        # phi, phi', phi'', ...


def d_dq(expr):
    """d/dq of a polynomial in PHI with coefficients rational in q."""
    out = sp.diff(expr, QS)
    for j in range(len(PHI) - 1):
        out += sp.diff(expr, PHI[j]) * PHI[j + 1]
    return sp.expand(out)


@lru_cache(maxsize=None)
def ladder(l, n):
    """xi_l^n as a polynomial in phi = xi_2^{-2} and its derivatives (the paper's Eq. xi_ladder, from the j_l
    relations H_{l+1}^{n+1} = -(H_l^n)' + (l/q) H_l^n and H_{l-1}^{n+1} = (H_l^n)' + ((l+1)/q) H_l^n)."""
    if (l, n) == (2, -2):
        return PHI[0]
    if n <= -2 or (l + n) % 2 or l < 0:
        raise KeyError(f'xi_{l}^{n} is not in the xi_2^-2 family')
    try:
        if l >= 1:
            prev = ladder(l - 1, n - 1)            # raise from (l-1, n-1)
            return sp.expand(-d_dq(prev) + sp.Rational(l - 1) / QS * prev)
    except KeyError:
        pass
    prev = ladder(l + 1, n - 1)                    # lower from (l+1, n-1)
    return sp.expand(d_dq(prev) + sp.Rational(l + 2) / QS * prev)


def phi_form(expr):
    """Substitute every xi(l, n) symbol of ``expr`` by its ladder form."""
    subs = {s: ladder(*xi_index(s)) for s in expr.free_symbols if s.name.startswith('xi_')}
    return sp.expand(expr.subs(subs))


def total_derivative(target):
    """G = sum c q^e m (lpt_model.TOTAL_DERIVATIVE_ANSATZ, m quadratic in phi, phi') with dG/dq = target, or None."""
    monomials = (PHI[0]**2, PHI[0] * PHI[1], PHI[1]**2)
    cs, G = [], sp.Integer(0)
    for e in model.TOTAL_DERIVATIVE_ANSATZ['exponents']:
        for mono in monomials:
            c = sp.Symbol(f'c_{len(cs)}')
            cs.append(c)
            G += c * QS**e * mono
    diff = sp.expand((d_dq(G) - target) * QS**12)
    sol = sp.solve(sp.Poly(diff, *PHI, QS).coeffs(), cs, dict=True)
    if not sol:
        return None
    return sp.expand(G.subs(sol[0]).subs({c: 0 for c in cs}))


def small_q_order(G):
    """The lowest power of q of G for phi ~ q^2 (xi_2^{-2} = O(q^2) at small q).  G(0) = 0 iff it is > 0."""
    s = sp.Symbol('s', positive=True)
    Gs = sp.expand(G.subs({PHI[0]: s**2, PHI[1]: 2 * s, PHI[2]: 2}).subs(QS, s))
    return min(sp.Poly(sp.expand(Gs * s**20), s).monoms()[-1][0] - 20, 99)


# ---------------------------------------------------------------------------------------------------------------
# finite-range closed forms of the Gauss-law rows, in the product variables

XI1, XI3 = xi(1, -1), xi(3, -1)
PHI_TO_XI = {PHI[0]: QS * (XI1 + XI3) / 5, PHI[1]: (2 * XI1 - 3 * XI3) / 5}
"""phi = xi_2^{-2} = q (xi_1^-1 + xi_3^-1)/5 and phi' = (2 xi_1^-1 - 3 xi_3^-1)/5 (lpt_model.PRODUCT_VARIABLES), the
inverse of the ladder on (phi, phi'), so a quadratic form in (phi, phi') is a unique quadratic form in (xi_1^-1,
xi_3^-1)."""
assert model.PRODUCT_VARIABLES == ('xi_1^-1', 'xi_3^-1'), 'PHI_TO_XI is written for these variables'


def d_dq_xi(expr, rule=None):
    """d/dq of a polynomial in the xi symbols, with lpt_model.DERIVATIVE_RULE for d xi_l^n/dq."""
    rule = rule or model.DERIVATIVE_RULE
    out = sp.diff(expr, QS)
    for s in expr.free_symbols:
        if s.name.startswith('xi_'):
            l, n = xi_index(s)
            if rule == 'raise':
                ds = sp.Rational(l) / QS * s - xi(l + 1, n + 1)
            else:
                ds = xi(l - 1, n + 1) - sp.Rational(l + 1) / QS * s
            out += sp.diff(expr, s) * ds
    return sp.expand(out)


def source_as_products(src):
    """``{j: F_j}`` with a 22 source sum_j k^{2j} FT0[F_j], F_j = sum c (xi_l^{-j})^2 (the coefficient carries the
    (-1)^l of the block), or None if the source has 13 blocks."""
    out = defaultdict(lambda: sp.Integer(0))
    for key, c in src.items():
        if key[0] != '22':
            return None
        _, l, n = key
        out[-n] += c * xi(l, n)**2
    return dict(out)


def finite_range_xi_form(l, n, src):
    """The q-space closed form of H_l^n[src] for a 22 source and (l, n) in {(1, -1), (2, -2)}, or None.

        H_1^{-1}[FT0 F] = q^-2 int_0^q r^2 F,    H_1^{-1}[k^2 FT0 F] = -F',
        H_2^{-2}[FT0 F] = (2 q^3)^-1 int_0^q r^2 (q^2 - r^2) F,   H_2^{-2}[k^2 FT0 F] = -F + 3 q^-3 int_0^q r^2 F
    (the finite-range kernels of ``bessel_identities``).  The integrals come from ``total_derivative`` (with G(0) = 0
    checked) through PHI_TO_XI, the local terms -F' and -F of the k^2 parts are written in xi with d_dq_xi."""
    parts = source_as_products(src)
    if parts is None or (l, n) not in ((1, -1), (2, -2)) or any(j > 1 for j in parts):
        return None
    out = sp.Integer(0)
    for j, F in parts.items():
        Fp = phi_form(F)
        G2 = total_derivative(sp.expand(QS**2 * Fp))
        if G2 is None or small_q_order(G2) <= 0:
            return None
        G2x = G2.subs(PHI_TO_XI)
        if (l, n) == (1, -1):
            out += G2x / QS**2 if j == 0 else -d_dq_xi(F)
        elif j == 0:
            G4 = total_derivative(sp.expand(QS**4 * Fp))
            if G4 is None or small_q_order(G4) <= 0:
                return None
            out += (QS**2 * G2x - G4.subs(PHI_TO_XI)) / (2 * QS**3)
        else:
            out += -F + 3 * G2x / QS**3
    return sp.expand(out)


def same_function(a, b):
    """Equality of two q-space forms as functions (through the ladder)."""
    return sp.expand(phi_form(a) - phi_form(b)) == 0


def same_expression(a, b):
    """Identity of two xi forms as expressions (the same arithmetic up to sympy's normal form)."""
    return sp.expand(a - b) == 0


def finite_range_check(dps=30):
    """The four finite-range formulas of ``finite_range_xi_form`` on the Gaussian F = exp(-r^2) (FT0 F = pi^{3/2}
    exp(-k^2/4)) at three separations by mpmath quadrature at ``dps`` digits, the largest |lhs - rhs|.  The two k^2
    formulas are integrations by parts (k^2 FT0 F = FT0[-laplacian F]), which this checks."""
    import mpmath as mp
    mp.mp.dps = dps

    def j(l, t):
        return mp.sqrt(mp.pi / (2 * t)) * mp.besselj(l + mp.mpf(1) / 2, t)

    def H(l, n, src, q):
        return mp.quad(lambda kk: kk**(n + 2) * src(kk) * j(l, kk * q), [0, 5, 10, 20, mp.inf]) / (2 * mp.pi**2)
    S = lambda kk: mp.pi**1.5 * mp.exp(-kk**2 / 4)            # noqa: E731
    S2 = lambda kk: kk**2 * S(kk)                              # noqa: E731
    F = lambda r: mp.exp(-r**2)                               # noqa: E731
    worst = mp.mpf(0)
    for q in (mp.mpf('0.3'), mp.mpf('1.1'), mp.mpf('2.7')):
        I2 = mp.quad(lambda r: r**2 * F(r), [0, q])
        I4 = mp.quad(lambda r: r**4 * F(r), [0, q])
        for lhs, rhs in ((H(1, -1, S, q), I2 / q**2), (H(1, -1, S2, q), 2 * q * F(q)),
                         (H(2, -2, S, q), (q**2 * I2 - I4) / (2 * q**3)), (H(2, -2, S2, q), -F(q) + 3 * I2 / q**3)):
            worst = max(worst, abs(lhs - rhs))
    return float(worst)
