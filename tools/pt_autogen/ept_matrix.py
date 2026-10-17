"""EPT matrix-method tables (the ``pt_matrix`` blocks and the UV term) from the generated
kernels (port of the prototype's ept_matrix_tables.py).

Master integral (dimensionless, analytic continuation in nu):
    int_p p^{-2 nu1} |k-p|^{-2 nu2} = k^{3 - 2 nu12} I(nu1, nu2),
    I = Gamma(3/2-nu1) Gamma(3/2-nu2) Gamma(nu12-3/2)
        / (8 pi^{3/2} Gamma(nu1) Gamma(nu2) Gamma(3-nu12)).

P22 block = sum_terms c I(nu1 - B, nu2 - C) / I(nu1, nu2)   for each term
            c k^{2A} p^{2B} q^{2C} of 2 Z2^2 (LOS averaged, q = |k-p|),
            so that P22 = k^3 sum c_m1 c_m2 k^{-2 nu12} block(nu1, nu2) I(nu1, nu2).
P13 block = sum_terms c I(nu1 - B, 1 - C)                   for each term
            c k^{2A} p^{2B} q^{2C} / q^2 of 6 Z1 Z3 written as N / D-
            (same N as the Hankel path); I(., <= 0) = 0 (scaleless).
            I(nu1 - B, 1) = tan(pi nu1) / (8 pi (nu1 - B - 1)), so the block is
            R(nu1) tan(pi nu1) / pi with R rational; the tables store R.

UV term: ``Z3_UV = 6 lim_{p -> inf} (p/k)^2 <Z3(k, p, -p)>`` over the directions of p, for the
matter group (``model.MATRIX_UV_TERM``); the table holds the monomials of ``Z1 Z3_UV``.

Table conventions (``src/elf/tables/ept_matrix.npz``, read by ``src/elf/pt_matrix.py``): one block
per (bias, f, mu) monomial, named ``'22=b1-i_b2-j_bG2-k_f-l_mu-m'`` and
``'13=b1-i_bG2-j_bGamma3-k_f-l_mu-m'``, sorted as strings; the runtime multiplies a P22 block by
I(nu1, nu2) and a P13 block by tan(pi nu1) / pi; R is stored as a constant times a product of
integer polynomial factors (positive powers in the numerator, negative in the denominator).
"""
import sympy as sp

from . import kernels, model
from .emit import to_fraction
from .ept_hankel import BIAS, RENORMALISED, dropped_symbols, p22_integrand_monomials
from .reduction import numerator_over_Dm, split_monomials
from .vectors import K_VEC, P_VEC, k, los_average, mu, p, q, x

f = kernels.f
nu1, nu2 = sp.symbols('nu1 nu2')
BIAS_GROUP = tuple(kernels.BIAS[name] for name in ('b2', 'bG2', 'bGamma3'))


def gamma_ratio(a, n):
    """``Gamma(a + n) / Gamma(a)`` for integer n."""
    n = int(n)
    if n >= 0:
        return sp.Mul(*[a + i for i in range(n)])
    return 1 / sp.Mul(*[a + n + i for i in range(-n)])


def I_ratio(B, C):
    """``I(nu1 - B, nu2 - C) / I(nu1, nu2)`` as a rational function."""
    s = nu1 + nu2
    return (gamma_ratio(sp.Rational(3, 2) - nu1, B) * gamma_ratio(sp.Rational(3, 2) - nu2, C)
            * gamma_ratio(s - sp.Rational(3, 2), -(B + C))
            * gamma_ratio(nu1 - B, B) * gamma_ratio(nu2 - C, C)
            / gamma_ratio(3 - s, B + C))


def laurent_terms(expr, gens):
    """``{exponents: coeff}`` of a Laurent polynomial in ``gens``."""
    num, den = sp.fraction(sp.together(expr))
    dp = sp.Poly(den, *gens)
    assert len(dp.terms()) == 1, den
    dexp, dc = dp.terms()[0]
    out = {}
    for e, c in sp.Poly(sp.expand(num), *gens).terms():
        out[tuple(a - b for a, b in zip(e, dexp))] = c / dc
    return out


def _name(prefix, columns, powers):
    """Block name; asserts the total bias degree (``model.MAX_BIAS_DEGREE``)."""
    if sum(powers.get(name, 0) for name in model.BIAS_NAMES) > model.MAX_BIAS_DEGREE:
        raise ValueError(f'bias degree > {model.MAX_BIAS_DEGREE}: {powers}')
    return prefix + '=' + '_'.join('%s-%d' % (c, powers[c]) for c in columns)


def p22_matrix():
    """``{name: R(nu1, nu2)}`` of the P22 blocks (one monomial in (bias, f, mu) per block)."""
    gens, integrand = p22_integrand_monomials()
    xsub = (k**2 + p**2 - q**2) / (2 * k * p)
    out = {}
    for expo, c in integrand.items():
        powers = {str(g): e for g, e in zip(gens, expo)}
        if powers['bGamma3']:
            raise AssertionError('bGamma3 in P22')
        avg = sp.together(los_average(c).subs(x, xsub))
        num, den = sp.fraction(avg)
        for (jm,), cm in sp.Poly(sp.expand(num), mu).terms():
            F = sp.cancel(cm / den)
            M = sp.Integer(0)
            for (ek, ep, eq), c_ in laurent_terms(F, (k, p, q)).items():
                assert ek % 2 == 0 and ep % 2 == 0 and eq % 2 == 0 and ek + ep + eq == 0
                M += c_ * I_ratio(ep // 2, eq // 2)
            powers['mu'] = jm
            out[_name('22', ('b1', 'b2', 'bG2', 'f', 'mu'), powers)] = sp.factor(sp.cancel(M))
    return out


def p13_matrix():
    """``{name: R(nu1)}`` of the P13 blocks; the matrix is ``R tan(pi nu1) / pi``.

    The renormalisation convention: the terms of b2 and of the renormalised operators must
    be scaleless here (asserted) and are dropped.
    """
    factor, orders = model.LOOP_TERMS['13']
    gens = BIAS + RENORMALISED + (f,)
    S = factor * kernels.Z(K_VEC) * kernels.Z(K_VEC, P_VEC, -P_VEC, renormalised=True)
    Sm = split_monomials(sp.expand(S), gens)
    dropped = dropped_symbols()
    out = {}
    for expo, c in Sm.items():
        powers = {str(g): e for g, e in zip(gens, expo)}
        avg = sp.together(los_average(c))
        num, den = sp.fraction(avg)
        for (jm,), cm in sp.Poly(sp.expand(num), mu).terms():
            N = numerator_over_Dm(sp.cancel(cm / den))
            # N(k, p, x) / D-  with x -> (k^2 + p^2 - q^2) / (2 k p), D- = q^2
            Nq = sp.cancel(N.subs(x, (k**2 + p**2 - q**2) / (2 * k * p)))
            R = sp.Integer(0)
            for (ek, ep, eq), c_ in laurent_terms(Nq, (k, p, q)).items():
                assert ek % 2 == 0 and ep % 2 == 0 and eq % 2 == 0 and ek + ep + eq == 2
                C = eq // 2
                if C >= 1:
                    continue                 # I(., 1 - C <= 0) = 0: scaleless
                assert C == 0
                B = ep // 2
                # I(nu1 - B, 1) / I(nu1, 1) = (nu1 - 1) / (nu1 - B - 1)
                R += c_ * (nu1 - 1) / (nu1 - B - 1)
            # I(nu1, 1) = tan(pi nu1) / (8 pi (nu1 - 1)): R = pi x block / tan(pi nu1)
            R = sp.factor(sp.cancel(R / (8 * (nu1 - 1))))
            if any(powers[str(s)] for s in dropped):
                if R != 0:
                    raise AssertionError(('renormalised term not scaleless', expo, jm, R))
                continue
            powers['mu'] = jm
            if R != 0:
                out[_name('13', ('b1', 'bG2', 'bGamma3', 'f', 'mu'), powers)] = R
    return out


def uv_limit():
    """``(Z3_UV, bias_group_limit)``: ``6 lim (p/k)^2 <Z3(k, p, -p)>`` of the matter-group terms,
    and the same limit of the bias-group terms (b2, bG2, bGamma3), which the runtime does not
    need (their decomposition bias makes the power-law integrals UV convergent).

    Asserts that the O(1) and O(k/p) terms of the direction average vanish for the matter
    group (in the bias group, b2 has the zero-lag O(1) term).
    """
    factor, orders = model.LOOP_TERMS['13']
    Z3 = kernels.Z(K_VEC, P_VEC, -P_VEC)
    gens = BIAS + (f,)
    Sm = split_monomials(sp.expand(Z3), gens)
    eps = sp.Symbol('eps', positive=True)
    out = {'matter': sp.Integer(0), 'bias': sp.Integer(0)}
    for expo, c in Sm.items():
        mono = sp.Mul(*[g**e for g, e in zip(gens, expo)])
        group = 'bias' if any(e and g in BIAS_GROUP for g, e in zip(gens, expo)) else 'matter'
        avg = sp.cancel(los_average(c).subs(k, eps * p))
        if avg.has(p):
            raise AssertionError('the kernel is not homogeneous of degree 0')
        series = []
        for order in range(3):
            coeff = sp.cancel(sp.diff(avg, eps, order).subs(eps, 0) / sp.factorial(order))
            series.append(sp.integrate(sp.expand(coeff), (x, -1, 1)) / 2)
        if group == 'matter' and (sp.expand(series[0]) != 0 or sp.expand(series[1]) != 0):
            raise AssertionError(('UV: O(1) or O(k/p) term of the direction average', expo))
        out[group] += factor * mono * series[2]
    return sp.expand(out['matter']), sp.expand(out['bias'])


def uv_rows(z3_uv):
    """``[(powers of model.MONOMIAL_COLUMNS, Fraction)]`` of ``Z1 Z3_UV``, sorted by the powers
    of (mu, f, b1) (``sorted(Poly(.., mu, f, b1).terms())``)."""
    z1 = kernels.Z(K_VEC)
    poly = sp.Poly(sp.expand(z1 * z3_uv), mu, f, kernels.BIAS['b1'])
    rows = []
    for (m, fp, b), c in sorted(poly.terms()):
        powers = {'mu': m, model.GROWTH_RATE: fp, 'b1': b}
        rows.append((tuple(powers.get(col, 0) for col in model.MONOMIAL_COLUMNS), to_fraction(c)))
    return rows
