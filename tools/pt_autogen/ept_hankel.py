"""EPT Hankel-method blocks (the ``pt_coeff`` tables) from the generated kernels (port of the
prototype's ept_hankel_tables.py).

P22: 2 Z2(p, k-p)^2  --LOS average-->  S(k, p, q; mu)  --symmetrise p<->q-->
     polynomial in K = k^2/(p q) and mu12 = phat.qhat  --Legendre in mu12-->
     coefficient of k^{2j} (pq)^{-j} L_l(mu12)  ->  block (l, n=-j, m=2j),
     table entry = (-1)^l * coefficient  (the code evaluates
     4 pi int r^2 j0(kr) [xi_l^n]^2 = (-1)^l int_p p^n q^n L_l P P).

P13: 6 Z1(k) Z3(k, p, -p)  --LOS average-->  S(k, p, x; mu), x = khat.phat
     --partial fractions in x--> a/D- + a'/D+ + poly(x)  (D-+ = |k -+ p|^2)
     --flip p -> -p in the D+ part-->  N / D-, N = a + a'(x->-x) + poly(x) D-
     --Legendre in x-->  k^m p^n L_l(x) / D-  --lower n > 0 with
     p^2 = D- - k^2 + 2 k p x-->  blocks (l, n <= 0, m); the zero-lag pieces
     p^{n-2} L_l (no 1/D-) are collected separately and dropped (model.RENORMALISED_OPERATORS).
     table entry = coefficient (the code evaluates int r dr j_l(kr) xi_l^n
     = int_p p^n L_l(x) / D- P(p)).

Table conventions (``src/elf/tables/ept_hankel_<22|13>.npz``, read by ``src/elf/pt_coeff.py``):
block names ``'22_l_n_m'`` and ``'13_l_n_m'``, sorted as strings; the P13 blocks have
-2 <= n <= 0 (``reduction.lower_n``, the cosine rule; no null-vector canonicalisation, so the
(2, -2, 4) block stays); the dropped zero-lag part of P13 holds only the b2, b3 and bdG2 terms
(asserted) and is scaleless in the matrix method.

The blocks are exact: ``{(l, n, m): polynomial in (b1, b2, bG2, bGamma3, f, mu)}``.
"""
from collections import defaultdict

import sympy as sp

from . import kernels, model
from .reduction import reduce13, reduce22, split_monomials
from .vectors import K_VEC, P_VEC, los_average, mu

f = kernels.f
BIAS = tuple(kernels.BIAS[name] for name in model.BIAS_NAMES)
RENORMALISED = tuple(kernels.BIAS[name] for name in model.RENORMALISED_NAMES)


def _clean(blocks):
    return {key: sp.expand(v) for key, v in blocks.items() if sp.expand(v) != 0}


def _pair_products(factor, monomials):
    """``factor x (sum_a c_a m_a)^2`` as ``{exponents: coefficient}``."""
    out = defaultdict(lambda: sp.Integer(0))
    keys = list(monomials)
    for i, a in enumerate(keys):
        for b in keys[i:]:
            mult = 1 if a == b else 2
            expo = tuple(ea + eb for ea, eb in zip(a, b))
            out[expo] += factor * mult * monomials[a] * monomials[b]
    return out


def p22_integrand_monomials():
    """``(gens, {exponents of gens: coefficient})`` of ``2 Z2(p, k-p)^2``."""
    factor, orders = model.LOOP_TERMS['22']
    assert orders == (2, 2)
    gens = BIAS + (f,)
    Z2 = kernels.Z(P_VEC, K_VEC - P_VEC)
    return gens, _pair_products(factor, split_monomials(sp.expand(Z2), gens))


def p22_blocks():
    """``{(l, n, m): polynomial}`` of P22."""
    gens, integrand = p22_integrand_monomials()
    blocks = defaultdict(lambda: sp.Integer(0))
    for expo, c in integrand.items():
        mono = sp.Mul(*[g**e for g, e in zip(gens, expo)])
        num, den = sp.fraction(sp.together(los_average(c)))
        for (jm,), cm in sp.Poly(sp.expand(num), mu).terms():
            for key, v in reduce22(sp.cancel(cm / den)).items():
                blocks[key] += v * mono * mu**jm
    return _clean(blocks)


def p13_integrand():
    """``(gens, 6 Z1(k) Z3(k, p, -p))`` with the renormalised operators included."""
    factor, orders = model.LOOP_TERMS['13']
    assert orders == (1, 3)
    gens = BIAS + RENORMALISED + (f,)
    S = factor * kernels.Z(K_VEC) * kernels.Z(K_VEC, P_VEC, -P_VEC, renormalised=True)
    return gens, S


def p13_reduction():
    """``(blocks, zero_lag, raw)`` of P13 before the renormalisation convention."""
    gens, S = p13_integrand()
    Sm = split_monomials(sp.expand(S), gens)
    raw = defaultdict(lambda: sp.Integer(0))
    blocks = defaultdict(lambda: sp.Integer(0))
    zero_lag = defaultdict(lambda: sp.Integer(0))
    for expo, c in Sm.items():
        mono = sp.Mul(*[g**e for g, e in zip(gens, expo)])
        num, den = sp.fraction(sp.together(los_average(c)))
        for (jm,), cm in sp.Poly(sp.expand(num), mu).terms():
            bl, zl, rw = reduce13(sp.cancel(cm / den))
            for tgt, src in ((blocks, bl), (zero_lag, zl), (raw, rw)):
                for key, v in src.items():
                    tgt[key] += v * mono * mu**jm
    return _clean(blocks), _clean(zero_lag), _clean(raw)


def dropped_symbols():
    """The symbols whose P13 terms the renormalisation convention removes: b2 and the
    renormalised operators."""
    return (kernels.BIAS['b2'],) + RENORMALISED


def p13_blocks():
    """``(blocks, zero_lag)`` of P13 after the renormalisation convention.

    Asserts that the blocks contain no b2 or renormalised-operator term and that the
    zero-lag part contains nothing else, so dropping the zero-lag part is the convention.
    """
    blocks, zero_lag, _ = p13_reduction()
    zero = {s: 0 for s in dropped_symbols()}
    for key, v in blocks.items():
        if sp.expand(v - v.subs(zero)) != 0:
            raise AssertionError(f'P13 block {key} carries a renormalised term')
    for key, v in zero_lag.items():
        if sp.expand(v.subs(zero)) != 0:
            raise AssertionError(f'P13 zero-lag part {key} carries a kept term')
    return blocks, zero_lag
