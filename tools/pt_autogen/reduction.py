"""Reduction of one-loop integrands to Hankel blocks (port of the prototype's reduction.py).

Conventions (identical to src/elf/ept.py):

* 22-type  int_p F(k, p, x) P(p) P(|k-p|)
      = sum c * k^m * 4 pi int r^2 dr j0(k r) [xi_l^n(r)]^2           (block (l, n, m))
  with  4 pi int r^2 j0 [xi_l^n]^2 = (-1)^l int_p p^n q^n L_l(phat.qhat) P P,
  i.e. the stored coefficient c already contains the factor (-1)^l.
  Only n = -m/2 <= 0 occurs; the representation is unique.

* 13-type  int_p F(k, p, x) P(p)
      = sum c * k^m * I^{(l,n)}(k)  +  sum z * k^m * int_p p^n P(p)
  with  I^{(l,n)}(k) = int_p p^n L_l(x) / |k-p|^2 P(p) = int r dr j_l(kr) xi_l^n(r).
  The representation is brought to -2 <= n <= 0 by the cosine rule; within that
  range it is unique only modulo the null vectors
      N_l = k^2 p^-2 L_l + L_l - 2 k p^-1 x L_l   (times 1/D-, l >= 1),
  see ``canonical13``.

x = khat.phat, q = |k - p|, D- = q^2 = k^2 + p^2 - 2 k p x.
"""
from collections import defaultdict

import sympy as sp

from .vectors import k, p, q, x

K_, m12, u = sp.symbols('K m12 u', positive=True)
_tt = sp.Symbol('tt')
Dm = k**2 + p**2 - 2 * k * p * x
Dp = k**2 + p**2 + 2 * k * p * x


def legendre_expand(poly_expr, var):
    """``{l: coeff}`` with ``poly_expr = sum_l coeff_l L_l(var)``."""
    rem = sp.expand(poly_expr)
    if rem == 0:
        return {}
    deg = sp.Poly(rem, var).degree()
    out = {}
    for l in range(deg, -1, -1):
        c = sp.Poly(rem, var).coeff_monomial(var**l)
        if c == 0:
            continue
        lead = sp.Poly(sp.legendre(l, var), var).coeff_monomial(var**l)
        a = sp.cancel(c / lead)
        out[l] = a
        rem = sp.expand(rem - a * sp.legendre(l, var))
    assert rem == 0
    return out


def split_monomials(expr, gens):
    """Collect ``expr`` (polynomial in ``gens``) into ``{exponents: coefficient}``."""
    expr = sp.expand(expr)
    out = defaultdict(lambda: sp.Integer(0))
    for term in sp.Add.make_args(expr):
        powers = term.as_powers_dict()
        exps = tuple(int(powers.get(g, 0)) for g in gens)
        coeff = term
        for g, e in zip(gens, exps):
            coeff = coeff / g**e
        out[exps] += coeff
    return dict(out)


def laurent_sym_to_t(expr_u, t):
    """Symmetric Laurent polynomial in u (u <-> 1/u) as a polynomial in t = u + 1/u.

    Uses u^n + u^-n = 2 T_n(t/2) (Chebyshev polynomial of the first kind).
    """
    num, den = sp.fraction(sp.together(sp.expand(expr_u)))
    num_p = sp.Poly(sp.expand(num), u)
    den_p = sp.Poly(den, u)
    assert len(den_p.terms()) == 1, den
    (dexp,), dcoef = den_p.terms()[0]
    lp = {e - dexp: sp.cancel(c / dcoef) for (e,), c in num_p.terms()}
    out = sp.Integer(0)
    for e, c in lp.items():
        if e < 0:
            assert sp.cancel(lp.get(-e, 0) - c) == 0, ('not symmetric', e, lp)
            continue
        if e == 0:
            out += c
        else:
            out += c * sp.expand(2 * sp.chebyshevt(e, t / 2))
    return sp.expand(out)


def homogeneity_degree(F, gens):
    """Degree d of the homogeneous ``F``: ``F(s gens) = s^d F(gens)``."""
    s = sp.Symbol('s_hom', positive=True)
    ratio = sp.cancel(F.subs({g: s * g for g in gens}, simultaneous=True) / F)
    d = sp.Poly(sp.numer(ratio), s).degree() - sp.Poly(sp.denom(ratio), s).degree()
    assert sp.cancel(ratio - s**d) == 0, ('not homogeneous', ratio)
    return d


def reduce22(F, in_q=False):
    """Blocks ``{(l, n, m): c}`` of ``int_p F P(p) P(|k-p|)``.

    F is rational in (k, p, x) (or in (k, p, q) if ``in_q``); its coefficients may contain
    further symbols (treated as constants).  F must be homogeneous in (k, p); a factor k^d
    is pulled out of the integral.
    """
    if not in_q:
        F = sp.cancel(F.subs(x, (k**2 + p**2 - q**2) / (2 * k * p)))
    if F == 0:
        return {}
    d = homogeneity_degree(F, (k, p, q))
    F0 = sp.cancel(F / k**d)
    Fs = sp.cancel((F0 + F0.subs({p: q, q: p}, simultaneous=True)) / 2)
    numF, denF = sp.fraction(Fs)
    for poly in (numF, denF):
        degs = [m_[0] for m_ in sp.Poly(poly, k).monoms()]
        assert all(dd % 2 == 0 for dd in degs), 'odd power of k'
    G = sp.cancel(Fs.subs(k, sp.sqrt(K_ * p * q)).subs({p: u, q: 1}))
    numG, denG = sp.fraction(G)
    assert not denG.has(K_), ('1/k^2 survives', G)
    blocks = defaultdict(lambda: sp.Integer(0))
    for (jK,), cK in sp.Poly(sp.expand(numG), K_).terms():
        tpoly = laurent_sym_to_t(cK / denG, _tt)
        full = sp.expand(tpoly.subs(_tt, K_ - 2 * m12) * K_**jK)
        for (jK2,), c2 in sp.Poly(full, K_).terms():
            for l, cl in legendre_expand(c2, m12).items():
                blocks[(l, -jK2, 2 * jK2 + d)] += (-1)**l * cl
    return {key: sp.expand(v) for key, v in blocks.items() if sp.expand(v) != 0}


def numerator_over_Dm(F):
    """Write ``int_p F P(p)`` as ``int_p N(k,p,x)/D- P(p)``: partial fractions in x, flip
    p -> -p (x -> -x) in the 1/D+ part, multiply the polynomial part by D-/D-."""
    N = sp.Integer(0)
    for term in sp.Add.make_args(sp.apart(sp.cancel(F), x)):
        tn, td = sp.fraction(sp.factor(term))
        if td.has(x):
            if not sp.cancel(td / Dm).has(x):
                N += sp.cancel(term * Dm)
            elif not sp.cancel(td / Dp).has(x):
                N += sp.cancel((term * Dp).subs(x, -x))
            else:
                raise ValueError(('unexpected denominator', td))
        else:
            N += term * Dm
    return sp.expand(sp.cancel(N))


def lower_n(blocks13, n_min=-2):
    """Bring every block to ``0 >= n >= n_min`` with the cosine rule p^2 = D- - k^2 + 2 k p x.

    Lowering (n > 0):
        p^n L_l / D- = p^{n-2} L_l - k^2 p^{n-2} L_l / D- + 2 k p^{n-1} x L_l / D-
    Raising (n < n_min), the same identity read the other way:
        k^2 p^n L_l / D- = p^n L_l - p^{n+2} L_l / D- + 2 k p^{n+1} x L_l / D-
    x L_l = (l+1)/(2l+1) L_{l+1} + l/(2l+1) L_{l-1}.  The pieces without 1/D- (zero lag)
    integrate to zero for l > 0; for l = 0 they are collected in ``zero_lag`` as
    ``{(n_zl, m): coeff}`` meaning ``coeff k^m int_p p^{n_zl} P``.
    blocks13: ``{(l, n, m): coeff}``.
    """
    blocks = defaultdict(lambda: sp.Integer(0), blocks13)
    zero_lag = defaultdict(lambda: sp.Integer(0))

    def add_x(l, n, m, c):
        blocks[(l + 1, n, m)] += c * sp.Rational(l + 1, 2 * l + 1)
        if l > 0:
            blocks[(l - 1, n, m)] += c * sp.Rational(l, 2 * l + 1)

    while True:
        live = [key for key, v in blocks.items() if sp.expand(v) != 0]
        pos = [key for key in live if key[1] > 0]
        neg = [key for key in live if key[1] < n_min]
        if pos:
            key = max(pos, key=lambda t_: t_[1])
            l, n, m = key
            c = blocks.pop(key)
            if l == 0:
                zero_lag[(n - 2, m)] += c
            blocks[(l, n - 2, m + 2)] += -c
            add_x(l, n - 1, m + 1, 2 * c)
        elif neg:
            key = min(neg, key=lambda t_: t_[1])
            l, n, m = key
            c = blocks.pop(key)
            if l == 0:
                zero_lag[(n, m - 2)] += c
            blocks[(l, n + 2, m - 2)] += -c
            add_x(l, n + 1, m - 1, 2 * c)
        else:
            break
    blocks = {kk: sp.expand(v) for kk, v in blocks.items() if sp.expand(v) != 0}
    zero_lag = {kk: sp.expand(v) for kk, v in zero_lag.items() if sp.expand(v) != 0}
    return blocks, zero_lag


def raw13(N):
    """``{(l, n, m): c}`` of N(k, p, x) (Laurent in k, p, polynomial in x)."""
    raw = defaultdict(lambda: sp.Integer(0))
    SH = 40
    Nx = sp.Poly(sp.expand(N * k**SH * p**SH), x)
    for (jx,), cx in Nx.terms():
        for l, cl in legendre_expand(x**jx, x).items():
            kp = sp.Poly(sp.expand(cx * cl), k, p)
            for (ek, ep), ckp in kp.terms():
                raw[(l, ep - SH, ek - SH)] += ckp
    return {kk: sp.expand(v) for kk, v in raw.items() if sp.expand(v) != 0}


def reduce13(F, n_min=-2):
    """``(blocks, zero_lag, raw)`` of ``int_p F(k, p, x) P(p)``; see the module docstring."""
    if F == 0:
        return {}, {}, {}
    raw = raw13(numerator_over_Dm(F))
    blocks, zl = lower_n(raw, n_min=n_min)
    return blocks, zl, raw


def canonical13(blocks):
    """Remove the (l >= 1, n = -2) blocks with the null vectors N_l.

    N_l / D-:  k^{m} p^-2 L_l  =  - k^{m-2} L_l + 2 k^{m-1} p^-1 x L_l   (m >= 2, l >= 1)
    after integration (the zero-lag piece of an l >= 1 Legendre integrates to 0).
    Two 13-type representations are equal as integrals iff their canonical forms
    are equal (within the -2 <= n <= 0 family).
    """
    b = defaultdict(lambda: sp.Integer(0), blocks)
    changed = True
    while changed:
        changed = False
        for key in list(b):
            l, n, m = key
            if n == -2 and l >= 1 and sp.expand(b[key]) != 0:
                c = b.pop(key)
                b[(l, 0, m - 2)] += -c
                b[(l + 1, -1, m - 1)] += 2 * c * sp.Rational(l + 1, 2 * l + 1)
                b[(l - 1, -1, m - 1)] += 2 * c * sp.Rational(l, 2 * l + 1)
                changed = True
    return {kk: sp.expand(v) for kk, v in b.items() if sp.expand(v) != 0}
