"""The prototype's hand-typed kernels (experiments/symbolic_tables_20260924/kernels.py of the
ps_1loop_jax repo), kept as the reference of tests/test_kernels.py only.

F2/G2 in closed form, F3/G3 by the n = 3 recursion with closed-form sub-kernels, K_G2, K_G3,
Z1, Z2 and the unsymmetrised Z3 of jax_eft_power_spectrum.tex, section "Eulerian perturbation
theory".  Terms that carry F2/G2 of a vanishing momentum sum are dropped as in the prototype.
"""
import itertools

import sympy as sp

from pt_autogen.vectors import vsum, dot, norm2, ndot

f = sp.Symbol('f')
b1, b2, bG2, b3, bdG2, bG3, bGam3 = sp.symbols('b1 b2 bG2 b3 bdG2 bG3 bGamma3')


def F2(u, v):
    if (u + v).is_zero():
        return sp.Integer(0)
    uv = dot(u, v)
    return (sp.Rational(5, 7) + sp.Rational(1, 2) * uv * (1 / norm2(u) + 1 / norm2(v))
            + sp.Rational(2, 7) * uv**2 / (norm2(u) * norm2(v)))


def G2(u, v):
    if (u + v).is_zero():
        return sp.Integer(0)
    uv = dot(u, v)
    return (sp.Rational(3, 7) + sp.Rational(1, 2) * uv * (1 / norm2(u) + 1 / norm2(v))
            + sp.Rational(4, 7) * uv**2 / (norm2(u) * norm2(v)))


def alpha(k1, k2):
    return dot(k1 + k2, k1) / norm2(k1)


def beta(k1, k2):
    return norm2(k1 + k2) * dot(k1, k2) / (2 * norm2(k1) * norm2(k2))


def _FG_sym(order, qs):
    if order == 1:
        return sp.Integer(1), sp.Integer(1)
    if order == 2:
        return F2(*qs), G2(*qs)
    raise ValueError


def _FG3_unsym(q1, q2, q3):
    n = 3
    Fn = sp.Integer(0)
    Gn = sp.Integer(0)
    qs = (q1, q2, q3)
    for m in range(1, n):
        left, right = qs[:m], qs[m:]
        kl, kr = vsum(left), vsum(right)
        if (m >= 2 and kl.is_zero()) or (n - m >= 2 and kr.is_zero()):
            continue
        Fm, Gm = _FG_sym(m, left)
        Fr, Gr = _FG_sym(n - m, right)
        a = alpha(kl, kr)
        bta = beta(kl, kr)
        pref = Gm / ((2 * n + 3) * (n - 1))
        Fn += pref * ((2 * n + 1) * a * Fr + 2 * bta * Gr)
        Gn += pref * (3 * a * Fr + 2 * n * bta * Gr)
    return Fn, Gn


def FG3(q1, q2, q3):
    Fs, Gs = sp.Integer(0), sp.Integer(0)
    for perm in itertools.permutations((q1, q2, q3)):
        F_, G_ = _FG3_unsym(*perm)
        Fs += F_
        Gs += G_
    return Fs / 6, Gs / 6


def KG2(u, v):
    return dot(u, v)**2 / (norm2(u) * norm2(v)) - 1


def KG3(u, v, w):
    return (sp.Rational(1, 2) * (dot(u, v)**2 / (norm2(u) * norm2(v))
                                 + dot(v, w)**2 / (norm2(v) * norm2(w))
                                 + dot(w, u)**2 / (norm2(w) * norm2(u)))
            - dot(u, v) * dot(v, w) * dot(w, u) / (norm2(u) * norm2(v) * norm2(w))
            - sp.Rational(1, 2))


def Z1(u):
    return b1 + f * ndot(u)**2 / norm2(u)


def Z2(u, v):
    s = u + v
    ns = ndot(s)
    return (b1 * F2(u, v) + b2 / 2 + bG2 * KG2(u, v)
            + f * ns**2 / norm2(s) * G2(u, v)
            + f * ns / 2 * b1 * (ndot(u) / norm2(u) + ndot(v) / norm2(v))
            + (f * ns)**2 / 2 * ndot(u) / norm2(u) * ndot(v) / norm2(v))


def bias3_unsym(k1, k2, k3, include_renormalized=False):
    """The third-order bias terms of _Z3_unsym (without b1 F3)."""
    k23 = k2 + k3
    out = sp.Integer(0)
    if not k23.is_zero():
        F23, G23 = F2(k2, k3), G2(k2, k3)
        K1_23 = KG2(k1, k23)
        out += b2 * F23
        out += 2 * bG2 * K1_23 * F23
        out += 2 * bGam3 * K1_23 * (F23 - G23)
    if include_renormalized:
        out += b3 / 6 + bdG2 * KG2(k2, k3) + bG3 * KG3(k1, k2, k3)
    return out


def _Z3_unsym(k1, k2, k3, include_renormalized=False):
    k23 = k2 + k3
    k123 = k1 + k2 + k3
    n123 = ndot(k123)
    F3s, G3s = FG3(k1, k2, k3)
    out = b1 * F3s
    zero23 = k23.is_zero()
    if not zero23:
        F23, G23 = F2(k2, k3), G2(k2, k3)
        K1_23 = KG2(k1, k23)
        out += b2 * F23
        out += 2 * bG2 * K1_23 * F23
        out += 2 * bGam3 * K1_23 * (F23 - G23)
        out += f * n123 * b1 * ndot(k23) / norm2(k23) * G23
        out += (f * n123)**2 * ndot(k1) / norm2(k1) * ndot(k23) / norm2(k23) * G23
    else:
        F23 = sp.Integer(0)
    out += f * n123**2 / norm2(k123) * G3s
    out += f * n123 * ndot(k1) / norm2(k1) * (b1 * F23 + b2 / 2 + bG2 * KG2(k2, k3))
    out += (f * n123)**2 / 2 * b1 * ndot(k2) / norm2(k2) * ndot(k3) / norm2(k3)
    out += (f * n123)**3 / 6 * (ndot(k1) / norm2(k1)) * (ndot(k2) / norm2(k2)) * (ndot(k3) / norm2(k3))
    if include_renormalized:
        out += b3 / 6 + bdG2 * KG2(k2, k3) + bG3 * KG3(k1, k2, k3)
    return out


def symmetrise3(func, k1, k2, k3, **kw):
    tot = sp.Integer(0)
    for perm in itertools.permutations((k1, k2, k3)):
        tot += func(*perm, **kw)
    return tot / 6


def Z3_sym(k1, k2, k3, include_renormalized=False):
    return symmetrise3(_Z3_unsym, k1, k2, k3, include_renormalized=include_renormalized)
