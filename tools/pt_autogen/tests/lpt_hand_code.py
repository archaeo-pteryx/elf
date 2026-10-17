"""The hand-written LPT component definitions of elf, kept as the fixture of the generator's tests.

Copied verbatim from elf's working tree (branch pt_autogen, 2026-10-17, before the table-driven runtime replaced them):
the static definitions of ``src/elf/lpt_defs.py`` (the bias degrees of the templates, the moment indices, the enums
Template, LinearKind, Component and Row, ``_COMPONENTS``, the template and linear matrices, the row slices and
``_KSPACE_ROWS``) and the method ``LPT._component_coefficients`` of ``src/elf/lpt.py`` as a module-level function (its
unused parameter ``self`` kept).  Only ``jnp.int32`` became ``_np.int32`` (the same dtype), so that the fixture needs no
JAX.  ``trace`` evaluates the function on sympy symbols (the method of experiments/lpt_basis_contraction_20261010).
The tests compare the generator's components, templates, linear kinds, component table and k-space rows with these.
"""
from enum import IntEnum

import numpy as _np
import sympy as sp

# Powers of (b1, b2, bG2, bGamma3) of the 12 bias templates (total degree <= 2, as cross_bias_factor needs).
# Row t is template t of _TEMPLATE_MATRIX. Template.CTR, the Zel'dovich counterterm, has no
# row. LPT._get_lpt_bias_factors turns the rows into the 12 bias weights of a tracer pair.
_LPT_BIAS_DEGREES = _np.array([
    [0, 0, 0, 0],   # 1
    [1, 0, 0, 0],   # b1
    [2, 0, 0, 0],   # b1^2
    [0, 1, 0, 0],   # b2
    [1, 1, 0, 0],   # b1 b2
    [0, 2, 0, 0],   # b2^2
    [0, 0, 1, 0],   # bG2
    [1, 0, 1, 0],   # b1 bG2
    [0, 1, 1, 0],   # b2 bG2
    [0, 0, 2, 0],   # bG2^2
    [0, 0, 0, 1],   # bGamma3
    [1, 0, 0, 1],   # b1 bGamma3
], dtype=_np.int32)


# Indices into the tuple of utils_lpt.get_lpt_moments.
# Each moment is an (lmax + 1, nq) array over the order l of j_l(kq) and q. MQa stands for
# the power (Khat.qhat)^a and NQb for (nhat.qhat)^b, with K = k + f (k.n) n. They are the
# keys of the component dicts of LPT._component_coefficients.
MQ0, MQ1, MQ2, MQ3, MQ4, NQ1, NQ2, MQ1NQ1, MQ2NQ1 = range(9)


class Template(IntEnum):
    """Templates, the rows of ``_TEMPLATE_MATRIX``.

    ONE ... B1_BGAMMA3 are the 12 bias monomials (rows of ``_LPT_BIAS_DEGREES``). ONE,
    the monomial 1, is the matter part. CTR is the Zel'dovich counterterm base.
    """
    ONE = 0
    B1 = 1
    B1_B1 = 2
    B2 = 3
    B1_B2 = 4
    B2_B2 = 5
    BG2 = 6
    B1_BG2 = 7
    B2_BG2 = 8
    BG2_BG2 = 9
    BGAMMA3 = 10
    B1_BGAMMA3 = 11
    CTR = 12


class LinearKind(IntEnum):
    """Kind of a component's linear part, the columns of ``_LINEAR_MATRIX`` (``LPT._linear_rows``)."""
    ZA = 0
    B1 = 1
    XI = 2


class Component(IntEnum):
    """The 23 components of the final integrand, the columns of ``_TEMPLATE_MATRIX`` (see ``_COMPONENTS``)."""
    ZA = 0
    AA = 1
    A22 = 2
    A13 = 3
    W112 = 4
    U10 = 5
    A_U = 6
    A10 = 7
    XI = 8
    A_XI = 9
    U_U = 10
    U11 = 11
    U20 = 12
    XI_U = 13
    XI_XI = 14
    UPS = 15
    V10 = 16
    V12 = 17
    CHI = 18
    ZETA = 19
    UB3 = 20
    THETA = 21
    CTR = 22


# The 23 components of the final integrand (order of LPT._component_coefficients): the
# templates each belongs to and the kind of its linear part.
# A component is one term in the braces of Eqs. (pkmu_lpt_tree) and (pkmu_lpt_1loop), without
# its bias coefficient (kk A = k_i k_j A_ij and k.U = k_i U_i, with redshift-space displacements):
#   ZA     1 - (1/2) kk A^> (Zel'dovich)   AA     (1/8) kkkk A^> A^>
#   A22    -(1/2) kk A^(22)                A13    -(1/2) kk (A^(13) + A^(31))
#   W112   -(i/6) kkk W                    U10    2i k.(U^lin + U^(3))
#   A_U    -i kk A^> k.U^lin               A10    -kk A^10
#   XI     xi_lin                          A_XI   -(1/2) kk A^> xi_lin
#   U_U    -kk U^lin U^lin                 U11    i k.U^11
#   U20    i k.U^20                        XI_U   2i xi_lin k.U^lin
#   XI_XI  (1/2) xi_lin^2                  UPS    -kk Upsilon
#   V10    2i k.V^10                       V12    2i k.V^12
#   CHI    chi                             ZETA   zeta
#   UB3    2i k.U_Gamma3                   THETA  2 theta
#   CTR    the ZA term, weighted by the Zel'dovich counterterm shape
# Every component also carries the factor e^{-(1/2) kk A^<}. 'templates' are rows of
# _TEMPLATE_MATRIX. 'linear' is the kind of the component's linear-theory term (a column of
# _LINEAR_MATRIX), None if it has none. A component's index is its column of W in
# LPT._final_integral, and the order must match LPT._component_coefficients.
_COMPONENTS = (
    # component          templates                                linear
    (Component.ZA,       (Template.ONE,),                         LinearKind.ZA),
    (Component.AA,       (Template.ONE,),                         None),
    (Component.A22,      (Template.ONE,),                         None),
    (Component.A13,      (Template.ONE,),                         None),
    (Component.W112,     (Template.ONE,),                         None),
    (Component.U10,      (Template.B1,),                          LinearKind.B1),
    (Component.A_U,      (Template.B1,),                          None),
    (Component.A10,      (Template.B1,),                          None),
    (Component.XI,       (Template.B1_B1,),                       LinearKind.XI),
    (Component.A_XI,     (Template.B1_B1,),                       None),
    (Component.U_U,      (Template.B1_B1, Template.B2),           None),
    (Component.U11,      (Template.B1_B1,),                       None),
    (Component.U20,      (Template.B2,),                          None),
    (Component.XI_U,     (Template.B1_B2,),                       None),
    (Component.XI_XI,    (Template.B2_B2,),                       None),
    (Component.UPS,      (Template.BG2,),                         None),
    (Component.V10,      (Template.BG2,),                         None),
    (Component.V12,      (Template.B1_BG2,),                      None),
    (Component.CHI,      (Template.B2_BG2,),                      None),
    (Component.ZETA,     (Template.BG2_BG2,),                     None),
    (Component.UB3,      (Template.BGAMMA3,),                     None),
    (Component.THETA,    (Template.B1_BGAMMA3,),                  None),
    (Component.CTR,      (Template.CTR,),                         LinearKind.ZA),
)
assert [component for component, *_ in _COMPONENTS] == list(Component)
N_COMPONENTS = len(_COMPONENTS)
# Component.XI_XI, the b2^2 component, has its xi_lin^2 / 2 transformed separately.
# LPT._integrands subtracts the undamped xi_lin^2 / 2 from the l = 0 integrand, and the rows
# Row.B2SQ_RESIDUAL and Row.B2SQ_CONST of _KSPACE_ROWS add it back in k space.

# M[t, c] = 1 if component c belongs to template t.
# Shape (13, 23). Rows are templates (Template), columns are components (Component).
# LPT._pkmu_true uses (template weights) @ M as the component weights W. W = M gives the
# 13 templates (get_pkmu_templates_from_pt_terms).
_TEMPLATE_MATRIX = _np.array(
    [[1.0 if t in templates else 0.0 for _, templates, _ in _COMPONENTS] for t in Template])

# N[c, kind] = 1 if component c has a linear part of that kind.
# Shape (23, 3). The kinds of LPT._linear_rows are LinearKind.ZA (orders l = 0, 1, 2), B1
# (l = 1) and XI (l = 0). They come back as (1 + f mu^2)^2 P, 2 (1 + f mu^2) P and P.
# LPT._integrands uses W @ N, the weight of each kind.
_LINEAR_MATRIX = _np.array(
    [[1.0 if linear == kind else 0.0 for kind in LinearKind] for _, _, linear in _COMPONENTS])


class Row(IntEnum):
    """Rows of ``PTTerms.rows`` (the table in :class:`PTTerms`).

    ``TREE_ROWS``, ``MATTER_ROWS`` and ``BIAS_ROWS`` live on the core q grid, the last
    five rows (CHI_OUTSIDE ... THETA_BALL) on the core k grid.
    """
    X_LIN = 0
    Y_LIN = 1
    X_LIN_LT = 2
    Y_LIN_LT = 3
    X_LIN_GT = 4
    Y_LIN_GT = 5
    XI_LIN = 6
    U_LIN = 7
    X22 = 8
    Y22 = 9
    X13 = 10
    Y13 = 11
    V1 = 12
    V3 = 13
    T = 14
    U3 = 15
    U11 = 16
    U20 = 17
    X10 = 18
    Y10 = 19
    V10 = 20
    V12 = 21
    X_UPS = 22
    Y_UPS = 23
    CHI = 24
    ZETA = 25
    UB3 = 26
    THETA = 27
    CHI_OUTSIDE = 28
    ZETA_OUTSIDE = 29
    B2SQ_RESIDUAL = 30
    B2SQ_CONST = 31
    THETA_BALL = 32


TREE_ROWS = slice(Row.X_LIN, Row.U_LIN + 1)
MATTER_ROWS = slice(Row.X22, Row.T + 1)
BIAS_ROWS = slice(Row.U3, Row.THETA + 1)
BIAS_OFFSET = Row.U3   # corrs_bias[i] is rows[BIAS_OFFSET + i]

# k-space rows of pt_terms added after the final transform (constants are read at k_min).
# Columns: 'component' is the component whose weight W[:, component] multiplies the row,
# 'row' the row of pt_terms.rows, 'factor' a fixed multiplier. 'interpolated' rows are
# Hermite-interpolated in ln k at the true k, the others are constants. 'k0_const' rows are
# skipped when subtract_k0_const=True. Used by LPT._final_integral.
_KSPACE_ROWS = (
    # component          row                 factor  interpolated  k0_const
    (Component.CHI,      Row.CHI_OUTSIDE,    1.0,    True,         False),   # -int 4 pi q^2 chi dq
    (Component.ZETA,     Row.ZETA_OUTSIDE,   1.0,    True,         False),   # -int 4 pi q^2 zeta dq
    (Component.XI_XI,    Row.B2SQ_RESIDUAL,  1.0,    True,         False),   # H_0[xi^2 / 2](k) - C_q
    (Component.XI_XI,    Row.B2SQ_CONST,     1.0,    False,        True),    # I0
    (Component.THETA,    Row.THETA_BALL,     2.0,    False,        False),   # D_theta (the component is 2 theta)
)


def _component_coefficients(
    self, k_i, mu_j, f, Kfac, K, Ksq, corrs_tree, corrs_matter_1loop, corrs_bias
):
    """The 23 components as ``{moment index: (nq,) coefficient}`` dicts, in ``_COMPONENTS`` order.

    This is the only definition of the integrand. Component c contributes
    sum_j c_j^c(q) M_j to the integrand, where M_j are the moments (MQ0 ...) of
    ``_angular_integral_factors``. The coefficients contain the k, K, f and mu factors
    of the terms listed above ``_COMPONENTS``.
    """
    X_lin_gt, Y_lin_gt = corrs_tree[Row.X_LIN_GT], corrs_tree[Row.Y_LIN_GT]
    xi_lin, U_lin = corrs_tree[Row.XI_LIN], corrs_tree[Row.U_LIN]
    X22, Y22, X13, Y13, V1, V3, T = corrs_matter_1loop
    U3, U11, U20, X10, Y10 = corrs_bias[Row.U3 - BIAS_OFFSET:Row.Y10 - BIAS_OFFSET + 1]
    V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta = corrs_bias[Row.V10 - BIAS_OFFSET:]

    mu2 = mu_j**2

    ZA = {MQ0: 1.0 - 0.5 * Ksq * X_lin_gt, MQ2: -0.5 * Ksq * Y_lin_gt}

    aa = Ksq**2 / 8.0
    AA = {MQ0: aa * X_lin_gt**2,
          MQ2: aa * 2.0 * X_lin_gt * Y_lin_gt,
          MQ4: aa * Y_lin_gt**2}

    p22 = -0.5 * k_i**2
    A22 = {MQ0: p22 * (Kfac**2 + 2 * f * (1 + f) * mu2 + f**2 * mu2) * X22,
           MQ2: p22 * Kfac**2 * Y22,
           MQ1NQ1: p22 * 2 * f * Kfac * mu_j * Y22,
           NQ2: p22 * f**2 * mu2 * Y22}

    A13 = {MQ0: p22 * 2 * (Kfac**2 + 2 * f * (1 + f) * mu2) * X13,
           MQ2: p22 * 2 * Kfac**2 * Y13,
           MQ1NQ1: p22 * 2 * 2 * f * Kfac * mu_j * Y13}

    p112 = 0.5 * k_i**3
    W112 = {MQ1: p112 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu2) * V1 + Kfac**2 * Kfac * V3),
            NQ1: p112 * Kfac**2 * f * mu_j * V3,
            MQ3: p112 * Kfac**2 * Kfac * T,
            MQ2NQ1: p112 * Kfac**2 * f * mu_j * T}

    U10 = {MQ1: -2.0 * K * (U_lin + U3), NQ1: -4.0 * f * k_i * mu_j * U3}

    A_U = {MQ1: Ksq * K * X_lin_gt * U_lin, MQ3: Ksq * K * Y_lin_gt * U_lin}

    A10 = {MQ0: -Ksq * X10 - f * k_i**2 * mu_j * (1 + f) * mu_j * X10,
           MQ2: -Ksq * Y10,
           MQ1NQ1: -f * k_i**2 * mu_j * Kfac * Y10}

    XI = {MQ0: xi_lin}
    A_XI = {MQ0: -0.5 * Ksq * X_lin_gt * xi_lin, MQ2: -0.5 * Ksq * Y_lin_gt * xi_lin}
    U_U = {MQ2: -Ksq * U_lin**2}
    U11c = {MQ1: -K * U11, NQ1: -f * k_i * mu_j * U11}
    U20c = {MQ1: -K * U20, NQ1: -f * k_i * mu_j * U20}
    XI_U = {MQ1: -2.0 * K * xi_lin * U_lin}
    XI_XI = {MQ0: 0.5 * xi_lin**2}

    UPS = {MQ0: -Ksq * X_Upsilon, MQ2: -Ksq * Y_Upsilon}
    V10c = {MQ1: -2.0 * K * V10, NQ1: -2.0 * f * k_i * mu_j * V10}
    V12c = {MQ1: -2.0 * K * V12}
    CHI = {MQ0: chi}
    ZETA = {MQ0: zeta}
    UB3 = {MQ1: -2.0 * K * Ub3}
    THETA = {MQ0: 2.0 * theta}

    CTR = {MQ0: 1.0 - 0.5 * Ksq * X_lin_gt, MQ2: -0.5 * Ksq * Y_lin_gt}

    components = dict(ZA=ZA, AA=AA, A22=A22, A13=A13, W112=W112, U10=U10, A_U=A_U,
                      A10=A10, XI=XI, A_XI=A_XI, U_U=U_U, U11=U11c, U20=U20c,
                      XI_U=XI_U, XI_XI=XI_XI, UPS=UPS, V10=V10c, V12=V12c, CHI=CHI,
                      ZETA=ZETA, UB3=UB3, THETA=THETA, CTR=CTR)
    return tuple(components[component.name] for component, *_ in _COMPONENTS)


# ---------------------------------------------------------------------------------------------------------------
# the hand code's stage A and R sources (src/elf/lpt.py, LPT.__init__ and LPT._set_hankel)

LN_LIST = [[0, 0], [0, -2], [1, -1], [2, 0], [2, -2], [3, -1], [4, 0]]
R_LN = [(0, 0), (2, 0), (4, 0), (1, -1), (3, -1)]
MOMENT_ORDER = ('MQ0', 'MQ1', 'MQ2', 'MQ3', 'MQ4', 'NQ1', 'NQ2', 'MQ1NQ1', 'MQ2NQ1')


def trace(row_symbols, k, mu, f, Kfac):
    """{component name: {moment name: expression}} of ``_component_coefficients`` on sympy symbols, the rows
    ``row_symbols`` (a dict by Row name) and the per-point variables, with K -> k Kfac and Ksq -> k^2 Kfac^2."""
    K, Ksq = sp.symbols('K Ksq')
    rsym = [row_symbols[Row(i).name] for i in range(BIAS_ROWS.stop)]
    comps = _component_coefficients(None, k, mu, f, Kfac, K, Ksq, rsym[TREE_ROWS], rsym[MATTER_ROWS],
                                     rsym[BIAS_ROWS])
    subs = {K: k * Kfac, Ksq: k**2 * Kfac**2}
    out = {}
    for c, comp in enumerate(comps):
        out[Component(c).name] = {MOMENT_ORDER[j]: sp.expand(sp.nsimplify(sp.expand(sp.sympify(e)), rational=True)
                                                             .subs(subs)) for j, e in comp.items()}
    return out
