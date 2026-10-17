"""Declared inputs of the EPT table generator: everything it cannot derive, in one place.

The generator derives the kernels (:mod:`pt_autogen.kernels`), the loop integrands and the
tables (:mod:`pt_autogen.ept_hankel`, :mod:`pt_autogen.ept_matrix`) from these declarations only.

* dynamics: the EdS SPT recursion with ``F_1 = G_1 = 1``;
* the bias operators of the galaxy density and the renormalisation convention;
* the redshift-space mapping and the observable;
* the conventions of the numeric tables that the runtime (``src/elf/ept.py``) reads (the meaning of the entries
  is stated in the docstrings of :mod:`pt_autogen.ept_hankel` and :mod:`pt_autogen.ept_matrix`).
"""
from fractions import Fraction
from typing import Callable, NamedTuple, Tuple

import sympy as sp

from .vectors import dot, norm2

# ---------------------------------------------------------------------------
# dynamics (Einstein-de Sitter SPT)

MAX_ORDER = 3
"""Highest perturbative order of the one-loop power spectrum (P13 needs Z_3)."""

FIRST_ORDER_KERNEL = {'F': 1, 'G': 1}
"""``F_1 = G_1 = 1``."""


def recursion_coefficients(n):
    """``(d, (a_F, b_F), (a_G, b_G))`` of the EdS recursion at order ``n >= 2``.

    ``F_n(q_1..q_n) = sum_{m=1}^{n-1} G_m(q_1..q_m) [a_F alpha(k_1, k_2) F_{n-m}(q_{m+1}..q_n)
    + b_F beta(k_1, k_2) G_{n-m}(q_{m+1}..q_n)] / d`` and the same for ``G_n`` with
    ``(a_G, b_G)``; ``k_1 = q_1 + .. + q_m``, ``k_2`` the rest, then symmetrised.
    """
    return (2 * n + 3) * (n - 1), (2 * n + 1, 2), (3, 2 * n)


def alpha(k1, k2):
    """``alpha(k_1, k_2) = (k_1 + k_2) . k_1 / k_1^2``."""
    return dot(k1 + k2, k1) / norm2(k1)


def beta(k1, k2):
    """``beta(k_1, k_2) = |k_1 + k_2|^2 (k_1 . k_2) / (2 k_1^2 k_2^2)``."""
    return norm2(k1 + k2) * dot(k1, k2) / (2 * norm2(k1) * norm2(k2))


FIELDS = {'delta': 'F', 'theta': 'G'}
"""The fields of the operators and their kernels: the density ``delta`` (``F_n``) and the
velocity divergence ``theta = -div v / (f a H)`` (``G_n``)."""

# ---------------------------------------------------------------------------
# bias operators


def unit_weight(*ks):
    """Momentum weight 1 (plain products such as ``delta^2``)."""
    return 1


def K_G2(ka, kb):
    """``K_G2(k_a, k_b) = (k_a . k_b)^2 / (k_a^2 k_b^2) - 1`` (the Galileon ``G_2``)."""
    return dot(ka, kb)**2 / (norm2(ka) * norm2(kb)) - 1


def K_G2_last_two(ka, kb, kc):
    """``K_G2(k_b, k_c)``: the weight of ``delta G_2[delta]``."""
    return K_G2(kb, kc)


def K_G3(ka, kb, kc):
    """``K_G3 = (1/2)[mu_ab^2 + mu_bc^2 + mu_ca^2] - mu_ab mu_bc mu_ca - 1/2`` (``G_3``),
    ``mu_ab^2 = (k_a . k_b)^2 / (k_a^2 k_b^2)``."""
    return (sp.Rational(1, 2) * (dot(ka, kb)**2 / (norm2(ka) * norm2(kb))
                                 + dot(kb, kc)**2 / (norm2(kb) * norm2(kc))
                                 + dot(kc, ka)**2 / (norm2(kc) * norm2(ka)))
            - dot(ka, kb) * dot(kb, kc) * dot(kc, ka) / (norm2(ka) * norm2(kb) * norm2(kc))
            - sp.Rational(1, 2))


class Product(NamedTuple):
    """``coefficient x int w(k_1..k_R) X_1(k_1) .. X_R(k_R)``: a product of fields with a weight."""

    coefficient: int
    fields: Tuple[str, ...]
    weight: Callable


class Operator(NamedTuple):
    """``normalisation x sum of products``."""

    label: str
    normalisation: Fraction
    products: Tuple[Product, ...]


BIAS_OPERATORS = (
    ('b1', Operator('delta', Fraction(1), (Product(1, ('delta',), unit_weight),))),
    ('b2', Operator('delta^2 / 2', Fraction(1, 2), (Product(1, ('delta', 'delta'), unit_weight),))),
    ('bG2', Operator('G2[delta]', Fraction(1), (Product(1, ('delta', 'delta'), K_G2),))),
    ('bGamma3', Operator('Gamma3 = G2[delta] - G2[theta]', Fraction(1),
                         (Product(1, ('delta', 'delta'), K_G2), Product(-1, ('theta', 'theta'), K_G2)))),
)
"""``delta_g = sum_O b_O O``: (coefficient name, operator) of the galaxy density."""

RENORMALISED_OPERATORS = (
    ('b3', Operator('delta^3 / 6', Fraction(1, 6), (Product(1, ('delta',) * 3, unit_weight),))),
    ('bdG2', Operator('delta G2[delta]', Fraction(1), (Product(1, ('delta',) * 3, K_G2_last_two),))),
    ('bG3', Operator('G3[delta]', Fraction(1), (Product(1, ('delta',) * 3, K_G3),))),
)
"""The third-order operators dropped by the renormalisation convention (the prototype's
``include_renormalized``).  Their one-loop contribution to P is zero-lag, a constant times
sigma^2 Z_1 P(k) (absorbed in b1), or zero.  The generator includes them and asserts that
they leave the tables unchanged; the zero-lag part of P13 is dropped, which also removes
every b2 term of P13."""

BIAS_NAMES = tuple(name for name, _ in BIAS_OPERATORS)
RENORMALISED_NAMES = tuple(name for name, _ in RENORMALISED_OPERATORS)

# ---------------------------------------------------------------------------
# redshift space and the observable

GROWTH_RATE = 'f'
"""Symbol of the growth rate.  It enters through the redshift-space mapping: the redshift-space
density is the expansion of ``(1 + delta_g) exp(-i f k_z u_z)`` (:func:`pt_autogen.kernels.Z`) with the
line-of-sight velocity ``u_z(k) = i (n . k / k^2) theta(k)`` in units of ``f a H``, so the
velocity kernel at order ``|S|`` is ``(n . k_S / k_S^2) G_|S|``."""

LOOP_TERMS = {'22': (2, (2, 2)), '13': (6, (1, 3))}
"""The one-loop terms of the observable, the auto power spectrum of ``delta_g`` in redshift space
(parity even).  ``block: (factor, kernel orders)``: ``P22 = 2 int_p Z_2(p, k-p)^2 P(p) P(|k-p|)``
and ``P13 = 6 Z_1(k) P(k) int_p Z_3(k, p, -p) P(p)``.  The cross spectrum of two tracers follows
from the auto polynomial by the runtime's ``cross_bias_factor`` (total bias degree <= 2)."""

MATRIX_UV_TERM = True
"""The matrix method adds ``k^2 P(k) pk_int Z_1 Z3_UV`` (``pk_int = (1/2 pi^2) int dk P``) for
the matter group (monomials without b2, bG2, bGamma3), whose power-law integrals diverge in
the UV; ``Z3_UV = 6 lim_{p -> inf} (p/k)^2 <Z_3(k, p, -p)>`` over the directions of p."""

# ---------------------------------------------------------------------------
# conventions of the numeric tables

MONOMIAL_COLUMNS = ('mu', 'f', 'b1', 'b2', 'bG2', 'bGamma3')
"""Columns 0-5 of a table row are these powers, column 6 the coefficient (float64)."""

TERM_ORDER = ('b1', 'b2', 'bG2', 'bGamma3', 'f', 'mu')
"""The rows of a block are sorted by their powers of these variables in descending
lexicographic order (the order of sympy's ``as_ordered_terms``); blocks are
zero-padded with rows of zeros to one length."""

MAX_BIAS_DEGREE = 2
"""Total bias degree of every monomial (A/B split of ``cross_bias_factor``, phase 0)."""
