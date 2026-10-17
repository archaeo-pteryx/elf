"""F_n, G_n, the bias-operator kernels and the redshift-space kernels Z_n, generated from
the declarations of :mod:`pt_autogen.model` (no hand-typed kernel).

* :func:`FG`: the EdS recursion from ``F_1 = G_1 = 1``, symmetrised over the momenta.
* :func:`product_kernel`: the kernel of a product of R fields with a momentum weight w,
  ``K_n(q_1..q_n) = sum over ordered partitions (S_1..S_R) of {1..n} into non-empty blocks of
  [prod_r |S_r|! / n!] w(k_{S_1}, .., k_{S_R}) prod_r K^{X_r}_{|S_r|}(S_r)``; it gives the
  galaxy density kernels ``K^g_n = sum_O b_O K^O_n`` (:func:`galaxy_kernel`, ``K^g_0 = 1``).
* :func:`Z`: the expansion of ``(1 + delta_g) exp(-i f k_z u_z)``,
  ``Z_n(q_1..q_n) = sum_{j=0}^{n} (f k mu)^j / j! sum over ordered partitions (S_0, S_1..S_j)
  of {1..n}, S_1..S_j non-empty, S_0 possibly empty, of [|S_0|! prod_i |S_i|! / n!]
  K^g_{|S_0|}(S_0) prod_{i=1}^{j} (n . k_{S_i} / k_{S_i}^2) G_{|S_i|}(S_i)``,
  ``k = q_1 + .. + q_n``, ``k mu = n . k``.

Vanishing-momentum rule (the prototype's): a kernel ``F_m``/``G_m`` of order ``m >= 2`` whose
momenta sum to zero vanishes as ``|sum|^2``, while the factors that multiply it (``alpha``/``beta``
of the recursion, the velocity factor ``n . k_S / k_S^2``, a ``K_G2`` weight) diverge at most as
``1/|sum|`` or stay bounded.  Every term that contains such a sub-kernel is dropped, and
``F_n = G_n = 0`` at vanishing total momentum.  This is what ``Z_3(k, p, -p)`` needs.
"""
import itertools
import math
from fractions import Fraction
from functools import lru_cache

import sympy as sp

from . import model
from .vectors import vsum, ndot, norm2

f = sp.Symbol(model.GROWTH_RATE)
BIAS = {name: sp.Symbol(name) for name in model.BIAS_NAMES + model.RENORMALISED_NAMES}


def _sym_rational(c):
    """A sympy Rational of an int or Fraction."""
    c = Fraction(c)
    return sp.Rational(c.numerator, c.denominator)


def _canonical(qs):
    """Momenta in a fixed order: the symmetrised kernels do not depend on it."""
    return tuple(sorted(qs))


# ---------------------------------------------------------------------------
# SPT kernels

def _FG_unsym(qs):
    """``(F_n, G_n)`` of the recursion for the ordered momenta ``qs``, before symmetrisation."""
    n = len(qs)
    d, (a_F, b_F), (a_G, b_G) = model.recursion_coefficients(n)
    Fn = sp.Integer(0)
    Gn = sp.Integer(0)
    for m in range(1, n):
        left, right = qs[:m], qs[m:]
        k1, k2 = vsum(left), vsum(right)
        if (m >= 2 and k1.is_zero()) or (n - m >= 2 and k2.is_zero()):
            continue                     # vanishing-momentum rule
        F_l, G_l = FG(left)
        F_r, G_r = FG(right)
        a = model.alpha(k1, k2)
        b = model.beta(k1, k2)
        pref = G_l / d
        Fn += pref * (a_F * a * F_r + b_F * b * G_r)
        Gn += pref * (a_G * a * F_r + b_G * b * G_r)
    return Fn, Gn


@lru_cache(maxsize=None)
def _FG_sym(qs):
    n = len(qs)
    if n == 1:
        return (sp.Integer(model.FIRST_ORDER_KERNEL['F']),
                sp.Integer(model.FIRST_ORDER_KERNEL['G']))
    if vsum(qs).is_zero():
        return sp.Integer(0), sp.Integer(0)
    F_s = sp.Integer(0)
    G_s = sp.Integer(0)
    for perm in itertools.permutations(qs):
        F_, G_ = _FG_unsym(perm)
        F_s += F_
        G_s += G_
    return F_s / math.factorial(n), G_s / math.factorial(n)


def _check_order(n):
    if not 1 <= n <= model.MAX_ORDER:
        raise ValueError(f'order {n} outside 1..{model.MAX_ORDER} (model.MAX_ORDER)')


def FG(qs):
    """Symmetrised ``(F_n, G_n)`` of the momenta ``qs`` (a sequence of :class:`Vec`)."""
    _check_order(len(qs))
    return _FG_sym(_canonical(qs))


def F(*qs):
    """``F_n(q_1..q_n)``."""
    return FG(qs)[0]


def G(*qs):
    """``G_n(q_1..q_n)``."""
    return FG(qs)[1]


def field_kernel(field, qs):
    """Kernel of order ``len(qs)`` of the field ``field`` (``model.FIELDS``)."""
    return {'F': F, 'G': G}[model.FIELDS[field]](*qs)


# ---------------------------------------------------------------------------
# composite operators

@lru_cache(maxsize=None)
def ordered_partitions(n, nblocks, allow_empty_first=False):
    """Ordered partitions ``(S_1..S_R)`` of ``range(n)`` into ``nblocks`` blocks (tuples of indices).

    Every block is non-empty, except the first one if ``allow_empty_first``.
    """
    out = []
    for assign in itertools.product(range(nblocks), repeat=n):
        blocks = tuple(tuple(i for i in range(n) if assign[i] == b) for b in range(nblocks))
        if all(blocks[1:]) and (allow_empty_first or blocks[0]):
            out.append(blocks)
    return tuple(out)


def _partition_weight(blocks, n):
    """``prod_r |S_r|! / n!``."""
    return sp.Rational(math.prod(math.factorial(len(b)) for b in blocks), math.factorial(n))


def product_kernel(product, qs):
    """Kernel of ``product`` (a :class:`model.Product`) at the momenta ``qs``."""
    n = len(qs)
    total = sp.Integer(0)
    for blocks in ordered_partitions(n, len(product.fields)):
        subs = [tuple(qs[i] for i in b) for b in blocks]
        ks = [vsum(s) for s in subs]
        if any(len(s) >= 2 and kk.is_zero() for s, kk in zip(subs, ks)):
            continue                     # vanishing-momentum rule
        term = _partition_weight(blocks, n) * product.weight(*ks)
        for field, s in zip(product.fields, subs):
            term = term * field_kernel(field, s)
        total += term
    return product.coefficient * total


def operator_kernel(operator, qs):
    """Kernel of ``operator`` (a :class:`model.Operator`) at the momenta ``qs``."""
    out = sp.Integer(0)
    for product in operator.products:
        out += product_kernel(product, qs)
    return _sym_rational(operator.normalisation) * out


@lru_cache(maxsize=None)
def _galaxy_kernel(qs, renormalised):
    if len(qs) == 0:
        return sp.Integer(1)
    operators = model.BIAS_OPERATORS + (model.RENORMALISED_OPERATORS if renormalised else ())
    out = sp.Integer(0)
    for name, operator in operators:
        out += BIAS[name] * operator_kernel(operator, qs)
    return out


def galaxy_kernel(qs, renormalised=False):
    """``K^g_n(qs) = sum_O b_O K^O_n``, ``K^g_0 = 1``; ``renormalised`` adds the operators of
    ``model.RENORMALISED_OPERATORS``."""
    return _galaxy_kernel(_canonical(qs), renormalised)


# ---------------------------------------------------------------------------
# redshift space

def Z(*qs, renormalised=False):
    """Redshift-space kernel ``Z_n(q_1..q_n)`` (module docstring), symmetric in the momenta."""
    n = len(qs)
    _check_order(n)
    kmu = ndot(vsum(qs))
    total = sp.Integer(0)
    for j in range(n + 1):
        part = sp.Integer(0)
        for blocks in ordered_partitions(n, j + 1, allow_empty_first=True):
            subs = [tuple(qs[i] for i in b) for b in blocks]
            ks = [vsum(s) for s in subs]
            if any(len(s) >= 2 and kk.is_zero() for s, kk in zip(subs[1:], ks[1:])):
                continue                 # vanishing-momentum rule
            term = _partition_weight(blocks, n) * galaxy_kernel(subs[0], renormalised)
            for s, kk in zip(subs[1:], ks[1:]):
                term = term * ndot(kk) / norm2(kk) * G(*s)
            part += term
        total += (f * kmu)**j / math.factorial(j) * part
    return total
