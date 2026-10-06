"""LPT backend: the one-loop LPT galaxy power spectrum P(k, mu) in redshift space.

The model is Eqs. (pkmu_lpt_tree) and (pkmu_lpt_1loop) of the paper, with the
correlators (cumulants) of its appendix "Explicit formulas for LPT power spectrum".
Below, H_l^n[S](q) = int dk k^{n+2} S(k) j_l(kq) / (2 pi^2) and xi_l^n = H_l^n[P]
(the paper's xi^l_n).

Stage 1, ``get_pt_terms(pk_data, h)`` -> :class:`PTTerms`, depends only on P_lin and h.
Its ``rows`` (:class:`Row`) are 28 correlators on the q grid and five k-space rows.
The FFTlog chain:
  A. ``_get_xi_ln``: P(k) -> xi_l^n(q) for the (l, n) of ``ln_list``, for P and
     P_lt = P exp(-(k/k_IR)^2).
  B. ``_get_Qs_Rs``: products of xi_l^n -> Q_1, Q_2, Q_5(k) (q -> k), and
     transforms of xi_l^n times P -> R_1, R_2(k).
  C. ``_transform_sources``: linear combinations S(k) of Q_n and R_n -> rows
     H_l^n[S](q) (k -> q), for ``_get_corrs_matter_1loop`` and ``_get_corrs_bias``.
The other rows are local in q. ``_get_corrs_tree`` builds the tree rows from xi_l^n.
Products of xi_l^n give chi, zeta, X/Y_Upsilon and, through the paper's identities,
Y22, U20, V10 and V12. Ub3 = -(24/5) U3. ``_get_kspace_rows`` builds the k-space rows.

Stage 2, for each (k, mu), ``_pkmu_true`` -> ``_final_integral`` does the final q -> k
integral. ``_integrands`` combines the rows with the angular moments of
``utils_lpt.get_lpt_moments`` into one integrand per order l of j_l(kq), minus its
linear part. ``_final_integral`` contracts it with the transform rows of
``_FinalTransform`` interpolated at k. It then adds back the linear part (Kaiser
factor times P_lin(k)) and the k-space rows. ``base.PowerSpectrum`` adds the
stochastic term, the k-space counterterms and the AP volume factor.

Tables. ``_COMPONENTS`` lists the 23 components (terms) of the integrand
(:class:`Component`). ``_TEMPLATE_MATRIX`` (13 x 23) assigns them to the templates
(:class:`Template`), which are the 12 bias monomials of ``_LPT_BIAS_DEGREES`` and the
Zel'dovich counterterm. ``_LINEAR_MATRIX`` (23 x 3) gives the kind
(:class:`LinearKind`) of each component's linear part. ``_KSPACE_ROWS`` pairs the
k-space rows with components. The one engine ``_final_integral(..., W)`` weights the
components with W (n_out, 23). W = identity gives the components, W =
``_TEMPLATE_MATRIX`` the templates, and W = (bias and counterterm weights) @
``_TEMPLATE_MATRIX`` the spectrum.
"""

from enum import IntEnum
from typing import NamedTuple, Optional

import numpy as _np

import jax
from jax import jit
from functools import partial

import jax.numpy as jnp

from . import fftlog
from . import spline

from .base import PowerSpectrum
from .utils import cross_bias_factor
from .utils_lpt import (
    make_G00_coeffs,
    get_lpt_moments,
    compute_V_mu,
)


# Powers of (b1, b2, bG2, bGamma3) of the 12 bias templates (total degree <= 2, as cross_bias_factor needs).
# Row t is template t of _TEMPLATE_MATRIX. Template.CTR, the Zel'dovich counterterm, has no
# row. LPT._get_lpt_bias_factors turns the rows into the 12 bias weights of a tracer pair.
_LPT_BIAS_DEGREES = jnp.array([
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
], dtype=jnp.int32)


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
N_ROWS = len(Row)

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


class _RSDGeometry(NamedTuple):
    """Redshift-space geometry of one mu (:func:`_rsd_geometry`), with K = k + f (k.n) n."""
    Kfac: jnp.ndarray               # |K| / k
    c: jnp.ndarray                  # cosine of the angle between K and k
    s: jnp.ndarray                  # sine of that angle
    c2: jnp.ndarray                 # c^2
    s2: jnp.ndarray                 # s^2
    A_mu: jnp.ndarray               # (1 + f) mu / Kfac, the line of sight along K
    B_mu: jnp.ndarray               # sqrt(1 - mu^2) / Kfac, the line of sight across K


def _rsd_geometry(f, mu, sin_mu):
    """The :class:`_RSDGeometry` of ``mu`` (``sin_mu`` its AD-regular sine)."""
    Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
    c = (1 + f * mu**2) / Kfac
    s = f * mu * sin_mu / Kfac
    return _RSDGeometry(Kfac=Kfac, c=c, s=s, c2=c**2, s2=s**2,
                        A_mu=(1 + f) * mu / Kfac, B_mu=sin_mu / Kfac)


def _weighted_coeffs(components, W):
    """Moment coefficients of the outputs, ``{j: sum_c W[:, c] c_j^c}``, each ``(n_out, nq)``.

    Summed in component order on purpose (the low-k l = 0 integrand amplifies the last bit).
    """
    coeffs = {}
    for c, component in enumerate(components):
        for j, c_j in component.items():
            term = W[:, c, None] * c_j
            coeffs[j] = coeffs[j] + term if j in coeffs else term
    return coeffs


def _apply_coeffs(coeffs, moments):
    """``sum_j c_j[q] m_j[l, q]``, shape ``(L, nq)``."""
    integrand = None
    for j, c_j in coeffs.items():
        term = c_j[None, :] * moments[j]
        integrand = term if integrand is None else integrand + term
    return integrand


class PTTerms(NamedTuple):
    """LPT PT terms, which depend only on ``P_lin`` and ``h``.

    ``rows`` ``(33, nfft)`` are 28 q-space correlators and five k-space rows, named by
    :class:`Row`.  ``pk_nw_data``, ``Sigma2`` and ``dSigma2`` are the no-wiggle data of
    the 'linear_ir_resum' counterterm base (None for the other bases and for ``h=None``).

    Rows 0-27 live on the core q grid ``LPT._q``, rows 28-32 on the core k grid
    ``LPT._k``. Vector and tensor cumulants are stored as their scalar coefficients,
    e.g. U_i = qhat_i U and A_ij = X delta_ij + Y qhat_i qhat_j. The builders return
    the blocks ``TREE_ROWS``, ``MATTER_ROWS``, ``BIAS_ROWS`` and the k-space rows, so
    ``corrs_bias[i]`` is row ``BIAS_OFFSET + i``.

    ======  =============================  ==================================================
    row     Row                            definition (paper symbols, xi_l^n = H_l^n[P])
    ======  =============================  ==================================================
    0, 1    X_LIN, Y_LIN                   A^lin_ij
    2, 3    X_LIN_LT, Y_LIN_LT             the same from P_lt = P exp(-(k/k_IR)^2), i.e. A^<
    4, 5    X_LIN_GT, Y_LIN_GT             rows 0, 1 minus rows 2, 3, i.e. A^>
    6       XI_LIN                         xi_lin = xi_0^0
    7       U_LIN                          U^lin = -xi_1^{-1}
    8, 9    X22, Y22                       A^(22)_ij
    10, 11  X13, Y13                       A^(13)_ij
    12-14   V1, V3, T                      W^(112)_ijk (V2 = V1)
    15      U3                             U^(3)
    16      U11                            U^11
    17      U20                            U^20
    18, 19  X10, Y10                       A^10_ij
    20      V10                            V^10_G2
    21      V12                            V^12_G2
    22, 23  X_UPS, Y_UPS                   Upsilon_G2,ij (X_Upsilon, Y_Upsilon)
    24      CHI                            chi_G2
    25      ZETA                           zeta_G2
    26      UB3                            U_Gamma3 = -(24/5) U3
    27      THETA                          theta_Gamma3
    28      CHI_OUTSIDE                    -4 pi int dq q^2 chi over the core q grid (constant in k)
    29      ZETA_OUTSIDE                   the same for zeta
    30      B2SQ_RESIDUAL                  4 pi int dq q^2 [j_0(kq) - 1] xi_lin^2 / 2, both terms on the same q samples
    31      B2SQ_CONST                     I0 = (1/2) int dk k^2 P^2 / (2 pi^2) (constant in k)
    32      THETA_BALL                     4 pi int_0^{q_min} dq q^2 theta = (96 pi / 5) q_min^2 U3(q_min)
    ======  =============================  ==================================================

    Rows 28 and 29 equal the integrals over q outside the core q grid, since the
    integrals over all q vanish.
    """

    rows: jnp.ndarray
    pk_nw_data: Optional[jnp.ndarray]
    Sigma2: Optional[jnp.ndarray]
    dSigma2: Optional[jnp.ndarray]


class _XiTransform(NamedTuple):
    """The P -> xi_l^n transform (forward, k -> q; ``LPT._get_xi_ln``).

    Stage A data, one entry per (l, n) row of ``LPT.ln_list``. Every row has the kernel bias
    ``nu`` (the kernel ``HankelGrids.u_m[l]``) and the input bias ``nu - (n + 3)``, and lands
    on ``HankelGrids.q_padded[l]``.
    """
    forward: fftlog.ForwardTransform  # the seven rows (nu_kernel = nu)
    log_q_rows: jnp.ndarray         # (7, n_padded) ln of each row's output grid q_padded[l]


class _SourceTransform(NamedTuple):
    """The Q/R sources (backward, q -> k) and their k -> q transforms to the correlator rows.

    Stage B (``LPT._get_Qs_Rs``) and stage C (``LPT._transform_sources``) data. The
    order index l runs over 0 ... max(4, ``LPT.lmax``). ``u_m`` is ``(n_l, n_padded // 2 + 1)``,
    and ``u_m[0]`` serves the Q sources and the b2^2 row. The end weights live on
    ``k_padded``. ``u_m_src`` and ``q_src_padded`` are indexed ``[group, l]``. Group 0
    has bias ``nu`` and group 1 has ``nu_n0`` (sources with n >= 0). Each kernel uses the
    low-ringing phase of its own order.
    """
    u_m: jnp.ndarray                # backward kernels onto the common grid (l = 0 phase)
    four_pi_q3_padded: jnp.ndarray  # 4 pi q^3 on _q_padded (with d ln q it gives 4 pi q^2 dq)
    end_weights_Q: jnp.ndarray      # support [k_min, 2 k_max]
    end_weights_R: jnp.ndarray      # support [k_min, k_max]
    R_source_rows: jnp.ndarray      # (5,) ln_list rows of R: (l, n) = (0,0) (2,0) (4,0) (1,-1) (3,-1)
    R_source_ells: jnp.ndarray      # (5,) their orders l
    u_m_src: jnp.ndarray            # k -> q kernels [bias group (nu, nu_n0), l]
    q_src_padded: jnp.ndarray       # (2, n_l, n_padded) output q grids of u_m_src
    log_q_src_padded: jnp.ndarray   # ln of q_src_padded


class _FinalTransform(NamedTuple):
    """The final q -> k transform of the (k, mu) integrand (``LPT._final_integral``).

    q runs over the core q grid ``LPT._q`` (nq = nfft), k over the core k grid ``LPT._k``
    and l over 0 ... ``LPT.lmax``. Every order uses the kernel with the l = 0 phase, so
    the integrand stays on the common q grid.
    """
    pk_select_kernel: jnp.ndarray   # [l, k, j]: transform of the unit source at core node j, end weights folded in
    pk_select_slopes: jnp.ndarray   # ln k slopes of pk_select_kernel (axis 1) for spline.hermite
    end_weights_q_core: jnp.ndarray # (nq,) trapezoid weights of the core q grid, 1/2 at both ends
    four_pi_q3: jnp.ndarray         # (nq,) 4 pi q^3
    ell_indices: jnp.ndarray        # (lmax + 1,) the orders l as floats
    q_inv_pows: jnp.ndarray         # q^{-l} [l, q]


class _AngularIntegralFactors(NamedTuple):
    """Per-(k, mu) factors of the final integrand (``LPT._angular_integral_factors``)."""
    Kfac: jnp.ndarray               # |K| / k for K = k + f (k.n) n
    K: jnp.ndarray                  # |K| = k Kfac
    Ksq: jnp.ndarray                # K^2
    ir_damping: jnp.ndarray         # (nq,) exp(-K^2 (X_lin_lt + c^2 Y_lin_lt) / 2), the IR factor the moments leave out
    ell_factors: jnp.ndarray        # (lmax + 1, nq) (-2/k)^l q^{-l}
    moments: tuple                  # the nine (lmax + 1, nq) arrays MQ0 ... of utils_lpt.get_lpt_moments


class LPT(PowerSpectrum):

    def __init__(self,
                 kmin_fft=1e-5,
                 kmax_fft=1e2,
                 nfft=512,
                 pad_mode='zero-pad',
                 fftlog_settings=None,
                 ngauss=4,
                 ells=(0, 2, 4),
                 counterterm_base='linear_ir_resum',
                 irres_method='DST',
                 r_bao=110.0,
                 lambda_ir=0.2,
                 subtract_k0_const=False,
                 lmax=5,
                 k_IR=0.2,
                 bias_basis='bG2',
                 ):
        """One-loop Lagrangian-PT galaxy power spectrum in redshift space.

        ``Params.bias`` is Lagrangian (b1, b2, bG2, bGamma3), with b_s2 in the third slot
        for ``bias_basis='bs2'``.  Shared options as in :class:`base.PowerSpectrum`.
        ``pad_mode='power-law'`` continues P only for P -> xi, and the Q/R sources stay
        cut at 2 k_max / k_max.  ``irres_method``, ``r_bao`` and ``lambda_ir`` only shape
        the 'linear_ir_resum' counterterm base.
        ``lmax`` (>= 2) is the order of the j_l expansion of the angular integrals of the
        final integral, not the multipole order of ``ells``.
        ``k_IR`` sets the split ``P_lt = P e^{-(k/k_IR)^2}`` inside the LPT body, whereas
        ``lambda_ir`` is the upper limit of the Sigma^2 integral of the counterterm base.
        ``bias_basis`` is 'bG2' or 'bs2'.

        The attributes ``ln_list`` (the (l, n) of the forward transforms xi_l^n) and
        ``G00_coeffs`` (the angular coefficients of ``utils_lpt.make_G00_coeffs``) are
        internal tables.
        """

        if bias_basis not in ('bG2', 'bs2'):
            raise ValueError("bias_basis must be 'bG2' or 'bs2'")
        if isinstance(lmax, (int, _np.integer)) and not isinstance(lmax, bool) and lmax < 2:
            raise ValueError(
                "lmax must be >= 2: the linear part of the final integrand, which is "
                "replaced by linear theory, has orders l = 0, 1, 2"
            )
        super().__init__(kmin_fft=kmin_fft, kmax_fft=kmax_fft, nfft=nfft, pad_mode=pad_mode,
                         fftlog_settings=fftlog_settings, ngauss=ngauss, ells=ells,
                         counterterm_base=counterterm_base, irres_method=irres_method,
                         r_bao=r_bao, lambda_ir=lambda_ir, subtract_k0_const=subtract_k0_const)

        self.lmax = lmax
        self.k_IR = k_IR
        self.bias_basis = bias_basis

        self.G00_coeffs = make_G00_coeffs(self.lmax)

        ln_list = [[0, 0], [0, -2], [1, -1], [2, 0], [2, -2], [3, -1], [4, 0]]
        self.ln_list = jnp.array(ln_list)
        self._set_hankel(max(int(jnp.max(self.ln_list[:, 0])), self.lmax))

    def _set_hankel(self, lmax):
        """Grids, kernels and static support weights of the four transforms.

        ``self._hankel`` (:class:`fftlog.HankelGrids`) holds the padded k grid and the
        per-order q grids. The stage data go to ``_xi_transform`` (A), ``_source_transform``
        (B and C) and ``_final_transform`` (the final integral). The common q grid
        ``_q_padded`` is the l = 0 output grid, and ``_q`` is its core. ``lmax`` is the
        highest order needed, max(4, ``self.lmax``).
        """
        fs = self.fftlog_settings
        l_list = list(range(lmax + 1))

        self._hankel = fftlog.HankelGrids(self._k, fs, self.pad_mode, lmax)
        self._set_xi_ells_transform()
        dln = self._hankel.grid.dln

        self._q_padded = self._hankel.q_padded[0]
        self._q = self._hankel.grid.crop(self._q_padded)
        self._log_q_padded = jnp.log(self._q_padded)

        forward = fftlog.forward_transform(self._hankel.grid, fs.nu, self.ln_list[:, 0], self.ln_list[:, 1],
                                           nu_kernel=fs.nu)
        self._xi_transform = _XiTransform(forward=forward, log_q_rows=jnp.log(forward.q))

        u_m = jnp.array([
            fftlog.hankel_setup(l, fs.nu, self._hankel.grid, lnxy=self._hankel.lnxy[0])[2] for l in l_list
        ])
        # R sources stay on their own grid q_padded[l]: the conjugate relation lands them on k_padded
        r_ln = [(0, 0), (2, 0), (4, 0), (1, -1), (3, -1)]
        ln_np = _np.asarray(self.ln_list)
        k_ref = _np.asarray(self._hankel.k_padded)
        for l, _n in r_ln:
            k_back = _np.asarray(self._hankel.q_grids[l].conjugate(self._hankel.lnxy[l]).x)
            assert _np.allclose(k_back, k_ref, rtol=1e-12), (
                f'R-source backward output grid for ell={l} is not the common '
                'k grid; the conjugate relation of the low-ringing phase is '
                'broken')
        setups_n0 = [fftlog.hankel_setup(l, fs.nu_n0, self._hankel.grid) for l in l_list]
        q_src_padded = jnp.stack([self._hankel.q_padded, jnp.array([s[1].x for s in setups_n0])])
        self._source_transform = _SourceTransform(
            u_m=u_m,
            four_pi_q3_padded=4.0 * jnp.pi * self._q_padded**3,
            end_weights_Q=fftlog.trapezoid_weights('core_to_2kmax', self._nfft, self._hankel.npad, dln),
            end_weights_R=fftlog.trapezoid_weights('core', self._nfft, self._hankel.npad, dln),
            R_source_rows=jnp.asarray(
                [int(_np.flatnonzero((ln_np[:, 0] == l) & (ln_np[:, 1] == n))[0])
                 for l, n in r_ln], dtype=jnp.int32),
            R_source_ells=jnp.asarray([l for l, _ in r_ln], dtype=jnp.int32),
            u_m_src=jnp.stack([self._hankel.u_m, jnp.array([s[2] for s in setups_n0])]),
            q_src_padded=q_src_padded,
            log_q_src_padded=jnp.log(q_src_padded),
        )

        end_weights_q_core = fftlog.trapezoid_weights('core', self._nfft, 0, dln)
        basis = fftlog.pad(jnp.eye(self._q.shape[0], dtype=self._q.dtype), self._hankel.q_grids[0], 'zero-pad')

        def build_one(u_m_l):
            pk = fftlog.hankel(fs.nu, basis, self._q_padded, self._hankel.k_padded, u_m_l, self._hankel.npad)
            return pk.T

        ell_indices = jnp.arange(self.lmax + 1, dtype=self._q.dtype)
        pk_select_kernel = jax.vmap(build_one)(u_m[:self.lmax + 1]) * end_weights_q_core
        self._final_transform = _FinalTransform(
            pk_select_kernel=pk_select_kernel,
            pk_select_slopes=spline.slopes(self._logk_fft, pk_select_kernel, axis=1),
            end_weights_q_core=end_weights_q_core,
            four_pi_q3=4.0 * jnp.pi * self._q**3,
            ell_indices=ell_indices,
            q_inv_pows=self._q[None, :] ** (-ell_indices[:, None]),
        )

    # Chain: P -> xi_l^n (k -> q), xi products -> Q/R sources (q -> k), sources ->
    # correlator rows H_l^n[S] (k -> q).  Each transform uses the low-ringing phase of its
    # own order, and outputs meet through the conjugate grid relation or by resampling.

    def _resample_to_common_q_padded(self, log_q_rows, rows):
        """Resample rows from their own grids (``log_q_rows``) onto the common padded q grid.

        ``log_q_rows`` and ``rows`` are ``(n_rows, n_padded)``. Cubic Hermite in ln q
        (``spline.interp1d``), one row at a time.
        """
        log_q = self._log_q_padded
        return jax.vmap(lambda x, row: spline.interp1d(log_q, x, row))(log_q_rows, rows)

    def _get_xi_ln(self, arrays):
        """P -> xi_l^n of ``arrays`` ``(n_arrays, nfft)``: ``(xi_ln_common, xi_ln_own_grid)``.

        Stage A. ``arrays`` are spectra on the core k grid (P and P_lt in ``get_pt_terms``),
        transformed for the seven (l, n) of ``ln_list``.
        ``xi_ln_common[i, l, n]`` is resampled onto ``_q_padded``; ``xi_ln_own_grid``
        (rows of ``ln_list``) stays on ``q_padded[l]``.  Both carry the plain mask,
        not ``w_xi``: the pointwise rows would be wrong with a halved q_max node.
        ``xi_ln_common`` has shape ``(n_arrays, 5, 5, n_padded)``. n = -1, -2 sit at the
        Python indices -1, -2, and the entries not in ``ln_list`` are zero.
        """
        arrays = jnp.atleast_2d(jnp.asarray(arrays))
        ells = self.ln_list[:, 0]
        ns = self.ln_list[:, 1]

        fx = self._hankel.pad_input(arrays / (2 * jnp.pi**2))              # (n_arrays, n_padded)
        raw = self._xi_transform.forward.apply(fx[:, None, :], self._hankel.grid)   # (n_arrays, n_ln, n_padded)

        log_q_rows = self._xi_transform.log_q_rows
        xis = jax.vmap(lambda row: self._resample_to_common_q_padded(log_q_rows, row))(raw)
        xis = xis * self._hankel.mask_xi
        raw = raw * self._hankel.mask_xi
        xi_ln = jnp.zeros((arrays.shape[0], 5, 5, self._hankel.k_padded.shape[0]))
        return xi_ln.at[:, ells, ns].set(xis), raw

    def _get_Qs_Rs(self, xi_ln_common, xi_ln_own_grid, pk_lin_padded):
        """Q/R sources on the padded k grid: ``(Q1, Q2, Q5)``, ``(R1, R2)``.

        Q (products of different l) from ``xi_ln_common``; R from ``xi_ln_own_grid``
        with the per-l kernel.  Inputs carry the mask only; ``w_xi`` is applied here once
        per transform input (on the product for Q: it is one function cut at q_max).

        Stage B. One l = 0 transform of two xi products gives (paper Eq. Q_as_ft)
        Q_1 = 4 pi int dq q^2 j_0(kq) zeta / 2 and
        Q_1 - Q_2 = (2/5) k^2 4 pi int dq q^2 j_0(kq) [(xi_1^{-1})^2 - (xi_3^{-1})^2].
        Q_5 = (Q_1 + Q_2) / 2. R_1 and R_1 + R_2 are P(k) times combinations of
        k^2 int dq q xi_l^0(q) j_l(kq) (l = 0, 2, 4) and k^3 int dq q xi_l^{-1}(q) j_l(kq)
        (l = 1, 3). Q is cut to [k_min, 2 k_max], R to [k_min, k_max].
        """
        nu = self.fftlog_settings.nu
        integrands = jnp.stack([
            8/15 * xi_ln_common[0,0]**2 - 16/21 * xi_ln_common[2,0]**2
            + 8/35 * xi_ln_common[4,0]**2,
            xi_ln_common[1,-1]**2 - xi_ln_common[3,-1]**2,
        ], axis=0) * self._hankel.w_xi
        res = fftlog.hankel(
            nu, self._source_transform.four_pi_q3_padded * integrands, self._q_padded,
            self._hankel.k_padded, self._source_transform.u_m[0], self._hankel.npad, crop=False,
        )
        res = res * self._source_transform.end_weights_Q
        Q1 = res[0]
        Q2 = Q1 - 2/5 * self._hankel.k_padded**2 * res[1]
        Q5 = (Q1 + Q2) / 2
        Qs = jnp.stack([Q1, Q2, Q5], axis=0)

        ells = self._source_transform.R_source_ells
        q_l = self._hankel.q_padded[ells]
        xis = xi_ln_own_grid[self._source_transform.R_source_rows] * self._hankel.w_xi
        pk_list = fftlog.hankel(
            nu, q_l**2 * xis, q_l,
            self._hankel.k_padded, self._hankel.u_m[ells], self._hankel.npad, crop=False,
        ) * (pk_lin_padded * self._source_transform.end_weights_R)
        pk_00, pk_20, pk_40, pk_1m1, pk_3m1 = pk_list
        k = self._hankel.k_padded
        R1 = k**2 * (8/15*pk_00 - 16/21*pk_20 + 8/35*pk_40)
        R3 = (k**2 * (2/5*pk_00 - 6/7*pk_20 + 16/35*pk_40)
              + k**3 * (2/5*pk_1m1 - 2/5*pk_3m1))
        R2 = R3 - R1
        Rs = jnp.stack([R1, R2], axis=0)
        return Qs, Rs

    def _transform_sources(self, sources, *, ells, ns):
        """``H_l^n[S](q) = int dk k^{n+2} S(k) j_l(kq) / (2 pi^2)`` of each source on the common padded q grid.

        Stage C. ``sources`` is a list of ``(N,)`` sources on the padded k grid
        (N = n_padded) and ``ns`` holds one n per source. ``ells`` holds one order per
        source, giving ``(n_src, N)``, or one tuple of orders per source, giving
        ``(n_src, n_l, N)`` (one rfft per source, one irfft per order).

        Sources with ``n >= 0`` use the bias ``nu_n0``, the others ``nu``. Each output is
        resampled from its own grid onto ``_q_padded``.
        """
        sources = jnp.stack(sources, axis=0)
        ells = _np.asarray(ells)
        ns = _np.asarray(ns)
        group = (ns >= 0).astype(int)                 # bias group: 0 nu, 1 nu_n0
        nu = _np.where(group == 1, self.fftlog_settings.nu_n0, self.fftlog_settings.nu)
        fx = sources * self._hankel.k_padded[None, :] ** (jnp.asarray(ns)[:, None] + 3) / (2 * jnp.pi**2)
        if ells.ndim == 2:
            fx = fx[:, None, :]
            group = _np.broadcast_to(group[:, None], ells.shape)
            nu = nu[:, None]
        nu = float(nu.flat[0]) if _np.all(nu == nu.flat[0]) else jnp.asarray(nu)[..., None]
        d = fftlog.hankel(
            nu, fx, self._hankel.k_padded, self._source_transform.q_src_padded[group, ells],
            self._source_transform.u_m_src[group, ells], self._hankel.npad, crop=False,
        )
        rows = self._resample_to_common_q_padded(
            self._source_transform.log_q_src_padded[group, ells].reshape(-1, d.shape[-1]),
            d.reshape(-1, d.shape[-1]))
        return rows.reshape(d.shape)

    def _get_corrs_tree(self, xi_ln, xi_ln_lt, pk_int, pk_int_lt):
        """Tree rows (``TREE_ROWS``): ``X = (2/3)(pk_int - xi_0^{-2} - xi_2^{-2})``, ``Y = 2 xi_2^{-2}``.

        Y as ``(2q/5)(xi_1^{-1} + xi_3^{-1})`` (n = -2 is inaccurate at small q); X keeps
        ``xi_2^{-2}``, whose error cancels that of ``xi_0^{-2}``.

        ``xi_ln`` and ``xi_ln_lt`` are the stage-A outputs of P and P_lt on the core q grid.
        ``pk_int`` = int dk P / (2 pi^2), and likewise ``pk_int_lt``. The lt rows use P_lt,
        and the gt rows are the full rows minus the lt rows. xi_lin = xi_0^0 and
        U_lin = -xi_1^{-1}.
        """
        q = self._q
        X_lin = 2/3 * (pk_int - xi_ln[0,-2] - xi_ln[2,-2])
        Y_lin = 2 * q / 5 * (xi_ln[1,-1] + xi_ln[3,-1])

        X_lin_lt = 2/3 * (pk_int_lt - xi_ln_lt[0,-2] - xi_ln_lt[2,-2])
        Y_lin_lt = 2 * q / 5 * (xi_ln_lt[1,-1] + xi_ln_lt[3,-1])

        X_lin_gt = X_lin - X_lin_lt
        Y_lin_gt = Y_lin - Y_lin_lt

        xi_lin = xi_ln[0,0]
        U_lin = - xi_ln[1,-1]

        corrs = jnp.stack([X_lin, Y_lin, X_lin_lt, Y_lin_lt,
                           X_lin_gt, Y_lin_gt, xi_lin, U_lin], axis=0)
        return corrs

    def _get_corrs_matter_1loop(self, Qs, Rs, Y_lin):
        """Matter one-loop rows X22, Y22, X13, Y13, V1, V3, T (``MATTER_ROWS``), returns ``(corrs, M22)``.

        Through stage C. The sources are ``S_22 = (9/98) Q1``, ``S_13 = (5/21) R1`` and
        W_T, W_1, W_3, the brackets of the paper's T, V1, V3 with their prefactors.
        ``X22 = (2/3)(M22 - H_0^{-2}[S_22] - Y22/2)``,
        ``X13 = (2/3)(M13 - H_0^{-2}[S_13] - H_2^{-2}[S_13])`` and
        ``Y13 = 2 H_2^{-2}[S_13]``. ``Y_lin`` (``Row.Y_LIN``) enters through Y22.

        Zero-lag constant ``M = int dk S / (2 pi^2)``: the grid moment of the padded,
        end-weighted source.  Identities:
        ``Y22 = (18/49) Y_lin^2 / q^2``, ``M13 = (70/27) M22``.
        n = -3 rows from n = -2 pairs:
        ``T = (q/7)(H_2^{-2} + H_4^{-2})[W_T]``, ``H_1^{-3}[W] = (q/3)(H_0^{-2} + H_2^{-2})[W]``,
        ``V_i = H_1^{-3}[W_i] - T/5``.
        """
        crop = self._hankel.grid.crop
        Q1, Q2 = Qs[0], Qs[1]
        R1, R2 = Rs[0], Rs[1]
        q = self._q

        S_22 = 9/98 * Q1
        S_13 = 5/21 * R1
        W_T = -3/7 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2)
        W_1 = 3/35 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2)
        W_3 = -3/35 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2)

        H0_S22_padded = self._transform_sources([S_22], ells=[0], ns=[-2])[0]
        H_pairs_padded = self._transform_sources(
            [S_13, W_T, W_1, W_3],
            ells=[(0, 2), (2, 4), (0, 2), (0, 2)],
            ns=[-2, -2, -2, -2],
        )
        H_pairs = crop(H_pairs_padded)
        moment = self._hankel.grid.moment
        M22 = moment(S_22, 1) / (2 * jnp.pi**2)
        M13 = 70/27 * M22
        H0_S22 = crop(H0_S22_padded)
        (H0_S13, H2_S13), (H2_WT, H4_WT), (H0_W1, H2_W1), (H0_W3, H2_W3) = H_pairs

        Y22 = 18/49 * Y_lin**2 / q**2
        X22 = 2/3 * (M22 - H0_S22 - 0.5 * Y22)
        X13 = 2/3 * (M13 - H0_S13 - H2_S13)
        Y13 = 2 * H2_S13

        T = q / 7 * (H2_WT + H4_WT)
        H1m3_W1 = q / 3 * (H0_W1 + H2_W1)
        H1m3_W3 = q / 3 * (H0_W3 + H2_W3)
        V1 = H1m3_W1 - 0.2 * T
        V3 = H1m3_W3 - 0.2 * T

        corrs = jnp.stack([X22, Y22, X13, Y13, V1, V3, T], axis=0)
        return corrs, M22

    def _get_corrs_bias(self, Qs, Rs, xi_ln, M22):
        """Bias rows (``BIAS_ROWS``) from the Q/R sources and the core xi output ``xi_ln``.

        Identities: ``M10 = (28/9) M22``,
        ``V10 = (24/35)((xi_1^{-1})^2 - (xi_3^{-1})^2)/q``, ``U20 = -(6/7) U_lin^2 / q``,
        ``Ub3 = -(24/5) U3``, ``theta = H_0^0[-(8/7) R1]`` (``F_G2 = -(20/7) R1``).

        Stage C gives ``U3 = H_1^{-1}[-(5/21) R1]``, ``U11 = H_1^{-1}[-(6/7)(R1 + R2)]``,
        ``X10 = M10 - H_0^{-2}[S_10a] - H_2^{-2}[S_10b]``,
        ``Y10 = 3 H_2^{-2}[S_10b]`` and theta, with
        ``S_10a = (2/7)(Q5 + 2 R2)`` and ``S_10b = (1/7)(2 Q5 + 3 R1 + 4 R2)``.
        The other rows are local in ``xi_ln``. chi and zeta are the paper's chi_G2 and
        zeta_G2. ``X_Upsilon = 4 J3^2`` and ``Y_Upsilon = 12 J2^2 - 4 J3^2 - (4/3) U_lin^2``,
        where J2, J3 are the paper's J_2, J_3 up to sign.
        ``V12 = (14/3) V10 + (2/5) d/dq[(xi_1^{-1})^2 - (xi_3^{-1})^2]``.
        """
        Q5 = Qs[2]
        R1, R2 = Rs[0], Rs[1]
        q = self._q

        S_10a = 2/7 * (Q5 + 2 * R2)
        S_10b = 1/7 * (2 * Q5 + 3 * R1 + 4 * R2)
        crop = self._hankel.grid.crop
        H_bias_padded = self._transform_sources(
            [
                -5/21 * R1,
                -6/7 * (R1 + R2),
                S_10a,
                S_10b,
            ],
            ells=[1, 1, 0, 2],
            ns=[-1, -1, -2, -2],
        )
        H_bias = crop(H_bias_padded)
        # separate call: batching theta with the rows above changes their FFT rounding
        theta = crop(self._transform_sources([-8/7 * R1], ells=[0], ns=[0])[0])
        M10 = 28/9 * M22

        U3, U11 = H_bias[0], H_bias[1]
        H0_S10a, H2_S10b = H_bias[2], H_bias[3]
        X10 = M10 - H0_S10a - H2_S10b
        Y10 = 3 * H2_S10b

        U_lin = -xi_ln[1, -1]
        U20 = -(6.0/7.0) * U_lin**2 / q

        J2 = 2/15 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
        J3 = -1/5 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
        X_Upsilon = 4 * J3**2
        Y_Upsilon = 12 * J2**2 - 4 * J3**2 - (4.0/3.0) * U_lin**2

        V10 = 24/35 * (xi_ln[1,-1]**2 - xi_ln[3,-1]**2) / q
        # d/dq[(xi_3^{-1})^2 - (xi_1^{-1})^2], with d xi_l^n / dq = (l/q) xi_l^n - xi_{l+1}^{n+1}
        dq_xi3sq_minus_xi1sq = (
            2.0 * xi_ln[3,-1] * (3.0*xi_ln[3,-1]/q - xi_ln[4,0])
            - 2.0 * xi_ln[1,-1] * (xi_ln[1,-1]/q - xi_ln[2,0])
        )
        V12 = (14.0/3.0) * V10 - (2.0/5.0) * dq_xi3sq_minus_xi1sq
        chi = 4/3 * (xi_ln[2,0]**2 - xi_ln[0,0]**2)
        zeta = 2 * (8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2)

        Ub3 = -24/5 * U3

        return jnp.stack([U3, U11, U20, X10, Y10, V10, V12, X_Upsilon, Y_Upsilon,
                          chi, zeta, Ub3, theta], axis=0)

    def _get_kspace_rows(self, chi, zeta, U3, xi_lin_padded, pk_lin):
        """The k-space rows ``Row.CHI_OUTSIDE`` ... ``Row.THETA_BALL`` on the core k grid (see :class:`PTTerms`).

        The chi and zeta rows are ``-4 pi int dq q^2 corr`` over the core q grid (core
        trapezoid), which equals the integral outside it. The b2^2 rows are the residual
        ``H_0[xi^2/2](k) - C_q`` (the paper's Eq. m0_residual) and the constant
        ``I0 = (1/2) int dk k^2 P^2 / (2 pi^2)``.
        Here ``H_0[xi^2/2](k) = 4 pi int dq q^2 j_0(kq) xi_lin^2 / 2`` is a q -> k
        transform of the padded xi_0^0 ``xi_lin_padded``, and ``C_q`` is its k -> 0 limit,
        the end-weighted sum of the same samples that enter the transform (as for the EPT
        m = 0 blocks), so the residual is as accurate as the transform. I0, the same limit
        by Parseval's theorem, is the k-space weighted sum (end weight on P^2 once) and is
        added once.
        The theta row is the q < q_min ball ``D_theta = (96 pi / 5) q_min^2 U3(q_min)``.
        """
        q_core = fftlog.LogGrid(self._q)
        w_core = self._final_transform.end_weights_q_core

        def outside_core_constant(corr):
            return jnp.broadcast_to(-4 * jnp.pi * q_core.moment(w_core * corr, 3), self._k.shape)

        source = 0.5 * xi_lin_padded**2
        b2sq_direct = fftlog.hankel(
            self.fftlog_settings.nu,
            self._source_transform.four_pi_q3_padded * source * self._hankel.w_xi,
            self._q_padded, self._hankel.k_padded, self._source_transform.u_m[0], self._hankel.npad, crop=True,
        )
        # C_q: the k -> 0 limit of b2sq_direct as the end-weighted sum of the same samples
        b2sq_direct_k0 = 4 * jnp.pi * self._hankel.q_grids[0].moment(source * self._hankel.w_xi, 3)
        # the end weight multiplies P^2 once
        b2sq_const = 0.5 * self._hankel.grid.moment(
            self._hankel.w_input * fftlog.pad(pk_lin, self._hankel.grid, self._hankel.pad_mode)**2, 3) / (2 * jnp.pi**2)
        theta_ball = (96 * jnp.pi / 5) * self._q[0]**2 * U3[0]
        return jnp.stack([
            outside_core_constant(chi),
            outside_core_constant(zeta),
            b2sq_direct - b2sq_direct_k0,
            jnp.broadcast_to(b2sq_const, self._k.shape),
            jnp.broadcast_to(theta_ball, self._k.shape),
        ], axis=0)

    @partial(jit, static_argnames=['self'])
    def get_pt_terms(self, pk_data, h):
        """The :class:`PTTerms` of ``pk_data``.

        The rows (:class:`Row`) are ``TREE_ROWS`` (X/Y_lin, X/Y_lin_lt, X/Y_lin_gt, xi_lin,
        U_lin), ``MATTER_ROWS`` (X22, Y22, X13, Y13, V1, V3, T) and ``BIAS_ROWS`` (U3, U11,
        U20, X10, Y10, V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta) on the core
        q grid, then the k-space rows of :meth:`_get_kspace_rows`.

        The steps are stage A (``_get_xi_ln``) of P and P_lt, ``_get_corrs_tree``, stage B
        (``_get_Qs_Rs``), ``_get_corrs_matter_1loop`` and ``_get_corrs_bias`` through
        stage C, and ``_get_kspace_rows``. :class:`PTTerms` has the table of the rows.

        ``h`` enters only the no-wiggle data of the 'linear_ir_resum' counterterm base.
        ``h=None`` skips them (the template and component methods do so), and such
        ``pt_terms`` cannot be fed to the spectrum methods under that base.
        """
        pk_lin = self._pk_at(self._k, pk_data)
        pk_lin_lt = pk_lin * jnp.exp(-(self._k / self.k_IR)**2)

        pk_int = self._hankel.grid.moment(self._hankel.pad_input(pk_lin), 1) / (2 * jnp.pi**2)
        pk_int_lt = self._hankel.grid.moment(self._hankel.pad_input(pk_lin_lt), 1) / (2 * jnp.pi**2)

        xi_ln_padded, xi_raw_padded = self._get_xi_ln(
            jnp.stack([pk_lin, pk_lin_lt], axis=0))
        xi_ln = self._hankel.grid.crop(xi_ln_padded[0])
        xi_ln_lt = self._hankel.grid.crop(xi_ln_padded[1])

        corrs_tree = self._get_corrs_tree(xi_ln, xi_ln_lt, pk_int, pk_int_lt)

        Qs, Rs = self._get_Qs_Rs(xi_ln_padded[0], xi_raw_padded[0],
                                 fftlog.pad(pk_lin, self._hankel.grid, 'zero-pad'))
        corrs_matter_1loop, M22 = self._get_corrs_matter_1loop(Qs, Rs, corrs_tree[Row.Y_LIN])
        corrs_bias = self._get_corrs_bias(Qs, Rs, xi_ln, M22)

        # padded xi_0^0, not corrs_tree[Row.XI_LIN]: the lower padded band gives the q < q_min part
        kspace_rows = self._get_kspace_rows(
            corrs_bias[Row.CHI - BIAS_OFFSET], corrs_bias[Row.ZETA - BIAS_OFFSET],
            corrs_bias[Row.U3 - BIAS_OFFSET], xi_ln_padded[0][0, 0], pk_lin)
        rows = jnp.concatenate([corrs_tree, corrs_matter_1loop, corrs_bias, kspace_rows], axis=0)

        if self.counterterm_base == 'linear_ir_resum' and h is not None:
            pk_nw_data, Sigma2, dSigma2 = self._get_ir_data(pk_data, h)
        else:
            pk_nw_data = Sigma2 = dSigma2 = None
        return PTTerms(rows=rows, pk_nw_data=pk_nw_data, Sigma2=Sigma2, dSigma2=dSigma2)

    def _angular_integral_factors(self, k_i, V, geometry, corrs_tree):
        """Per-(k, mu) factors of the integrand, an :class:`_AngularIntegralFactors`.

        ``K = k Kfac`` is the norm of K = k + f (k.n) n. ``ir_damping`` is the factor
        exp(-K^2 (X_lin_lt + c^2 Y_lin_lt) / 2) of e^{-(1/2) kk A^<} that the moments
        leave out. ``ell_factors[l]`` = (-2/k)^l q^{-l}, and ``moments`` are the nine arrays
        (MQ0 ...) of ``utils_lpt.get_lpt_moments``. ``V`` is the angular table
        (``utils_lpt.compute_V_mu``) and ``geometry`` the :class:`_RSDGeometry` of this mu.
        """
        q = self._q
        X_lin_lt, Y_lin_lt = corrs_tree[Row.X_LIN_LT], corrs_tree[Row.Y_LIN_LT]
        geo = geometry

        K = k_i * geo.Kfac
        Ksq = K**2

        A = k_i * q * geo.c
        B = -0.5 * Ksq * Y_lin_lt
        C = k_i * q * geo.s
        # = exp(-K^2 (X + Y)/2) exp(-B s^2), combined so large K gives no 0 * inf
        ir_damping = jnp.exp(-0.5 * Ksq * (X_lin_lt + geo.c2 * Y_lin_lt))
        ft = self._final_transform
        ell_factors = ((-2.0 / k_i) ** ft.ell_indices)[:, None] * ft.q_inv_pows
        moments = get_lpt_moments(A, B, C, geo.c2, geo.s2, geo.A_mu, geo.B_mu, V, c=geo.c, s=geo.s)
        return _AngularIntegralFactors(Kfac=geo.Kfac, K=K, Ksq=Ksq, ir_damping=ir_damping,
                                       ell_factors=ell_factors, moments=moments)

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

    def _linear_rows(self, mu_j, f, V, geometry, K, Ksq, ell_factors, corrs_tree):
        """Linear part of the final integrand (JVP at pt_terms = 0), its weight and transform.

        The weight is ``lin_weight = exp(-K^2 (X_lt + Y_lt)(q_max) / 2)``. The rows are
        Zel'dovich l = 0, 1, 2, b1 (l = 1) and xi (l = 0), whose transforms are
        ``(1 + f mu^2)^2 P``, ``2 (1 + f mu^2) P`` and ``P``.
        Returns ``(lin_weight, (za0, za1, za2), b1, xi, (kaiser_za, kaiser_b1, kaiser_xi))``.

        The kinds are the columns of ``_LINEAR_MATRIX`` (:class:`LinearKind`). The rows are
        linear-theory terms (first order in the tree rows), on the core q grid.
        ``geometry`` is the :class:`_RSDGeometry` of this mu, and ``K``, ``Ksq`` and
        ``ell_factors`` are those of ``_angular_integral_factors``.
        """
        X_lt, Y_lt = corrs_tree[Row.X_LIN_LT], corrs_tree[Row.Y_LIN_LT]
        X_gt, Y_gt = corrs_tree[Row.X_LIN_GT], corrs_tree[Row.Y_LIN_GT]
        xi_lin, U_lin = corrs_tree[Row.XI_LIN], corrs_tree[Row.U_LIN]

        half_Ksq = 0.5 * Ksq
        V000, V001, V010 = V[0, 0, 0], V[0, 0, 1], V[0, 1, 0]

        lin_weight = jnp.exp(-half_Ksq * (X_lt[-1] + Y_lt[-1]))
        za0 = -half_Ksq * (V000 * (X_lt + X_gt + Y_lt) - Y_lt * (V001 + geometry.s2 * V000))
        za1 = ell_factors[1] * half_Ksq * (V010 * Y_lt + 0.5 * V000 * Y_gt)
        za2 = half_Ksq * geometry.c2 * V000 * Y_gt
        b1 = -2.0 * K * geometry.c * V000 * U_lin
        xi = V000 * xi_lin

        kaiser = 1 + f * mu_j**2
        return lin_weight, (za0, za1, za2), b1, xi, (kaiser**2, 2.0 * kaiser, jnp.ones_like(kaiser))

    def _get_lpt_bias_factors(self, bias_a, bias_b=None):
        """Bias monomials of the 12 templates (``bias_b=None``: auto).

        bs2 -> bG2 is applied per tracer before the cross symmetrisation (they do not commute).
        """
        if bias_b is None:
            bias_b = bias_a

        def to_bG2_basis(bias):
            b1, b2, btidal, bGamma3 = bias
            if self.bias_basis == 'bs2':
                b2 = b2 + (4.0 / 3.0) * btidal
            return jnp.stack([b1, b2, btidal, bGamma3])

        return cross_bias_factor(
            _LPT_BIAS_DEGREES, to_bG2_basis(bias_a), to_bG2_basis(bias_b)
        )

    def _require_pt_terms_rows(self, pt_terms):
        n_rows = pt_terms.rows.shape[0]
        if n_rows < N_ROWS:
            raise ValueError(
                f"pt_terms.rows has {n_rows} rows, but LPT needs the {N_ROWS} rows of "
                "LPT.get_pt_terms. Recompute pt_terms with get_pt_terms."
            )

    def _integrands(self, k_i, mu_j, f, V, geometry, corrs_q, W):
        """Integrands ``(L, nq)`` of the outputs with component weights ``W`` ``(n_out, 23)`` at one (k, mu).

        ``g_o = ir_damping ell_factors sum_j (sum_c W[o, c] c_j^c) M_j - lin_weight (linear rows)
        - W[o, Component.XI_XI] xi^2/2``.  Returns ``(g, lin)`` with ``lin * P_lin(k)`` the k-space add-back.

        c_j^c are the component coefficients of ``_component_coefficients``. M_j,
        ``ir_damping`` and ``ell_factors`` are those of ``_angular_integral_factors``. The
        linear rows come from ``_linear_rows``, weighted per kind by ``W @ _LINEAR_MATRIX``.
        ``g`` is a tuple of n_out arrays. ``lin`` has shape ``(n_out,)``. ``V`` and ``geometry``
        are the angular table and the :class:`_RSDGeometry` of this mu.
        """
        corrs_tree, corrs_matter_1loop, corrs_bias = corrs_q
        factors = self._angular_integral_factors(k_i, V, geometry, corrs_tree)
        components = self._component_coefficients(
            k_i, mu_j, f, factors.Kfac, factors.K, factors.Ksq,
            corrs_tree, corrs_matter_1loop, corrs_bias)
        coeffs = _weighted_coeffs(components, W)                        # {j: (n_out, nq)}

        lin_weight, za, b1, xi, kaiser = self._linear_rows(
            mu_j, f, V, geometry, factors.K, factors.Ksq, factors.ell_factors, corrs_tree)
        zero = jnp.zeros_like(xi)
        rows = (za, (zero, b1, zero), (xi, zero, zero))                 # [kind][l]
        w_kind = W @ jnp.asarray(_LINEAR_MATRIX)                        # (n_out, 3)
        lin = lin_weight * (w_kind @ jnp.stack(kaiser))
        b2sq = 0.5 * corrs_tree[Row.XI_LIN]**2

        # separate arrays per output: a leading output axis breaks XLA's fusion of the moments
        g = []
        for o in range(W.shape[0]):
            g_o = _apply_coeffs({j: c_j[o] for j, c_j in coeffs.items()}, factors.moments)
            g_o = g_o * (factors.ir_damping[None, :] * factors.ell_factors)
            for l in range(3):
                delta = -lin_weight * (w_kind[o, LinearKind.ZA] * rows[LinearKind.ZA][l]
                                       + w_kind[o, LinearKind.B1] * rows[LinearKind.B1][l]
                                       + w_kind[o, LinearKind.XI] * rows[LinearKind.XI][l])
                g_o = g_o.at[l].add(delta - W[o, Component.XI_XI] * b2sq if l == 0 else delta)
            g.append(g_o)
        return tuple(g), lin

    def _final_integral(self, k_true, mu_true, sin_true, rows, f, pk_data, W):
        """Final (k, mu) integral of the outputs with component weights ``W``, ``(n_out, nk, nmu)``.

        ``W`` is an ``(n_out, 23)`` array or a function ``(k_i, mu_j) -> W``.  ``mu_true``
        must be folded to ``|mu|`` (the moments use ``sqrt(s^2) = |s|``).  ``sin_true`` is
        passed through so its AD-regular value, not ``sqrt(1 - mu^2)``, reaches the moments.

        For each (k, mu) it calls ``_integrands`` and subtracts the l = 0 integrand's value
        at q_max from the l = 0 integrand. It contracts 4 pi q^3 g_l(q) with
        ``pk_select_kernel`` Hermite-interpolated at ln k, summing over l and q. It then
        adds ``lin * P_lin(k)`` and the ``_KSPACE_ROWS`` rows times their component weights.
        W = identity gives the components, ``_TEMPLATE_MATRIX`` the templates, and the
        function ``component_weights`` of ``_pkmu_true`` the spectrum. ``rows`` is ``PTTerms.rows``.
        """
        ft = self._final_transform
        corrs_q = (rows[TREE_ROWS], rows[MATTER_ROWS], rows[BIAS_ROWS])
        kspace_rows = {row: rows[row] if interpolated else rows[row, 0]
                       for _, row, _, interpolated, _ in _KSPACE_ROWS}
        kspace_slopes = {row: spline.slopes(self._logk_fft, rows[row])
                         for _, row, _, interpolated, _ in _KSPACE_ROWS if interpolated}

        def angular_table(mu_j, sin_j):
            return compute_V_mu(_rsd_geometry(f, mu_j, sin_j).s2, self.G00_coeffs, self.lmax)

        V_all = jax.vmap(angular_table)(mu_true, sin_true)       # (nmu, 5, L, L), once per mu
        mu_true = jnp.broadcast_to(mu_true, k_true.shape)
        pk_lin_true = self._pk_at(k_true, pk_data)

        def per_mu(k_i, mu_j, V, sin_j, p_i):
            # the geometry once per (k, mu), shared by the integral factors and the linear rows
            geometry = _rsd_geometry(f, mu_j, sin_j)
            logk = jnp.log(k_i)
            W_ij = W(k_i, mu_j) if callable(W) else W
            g, lin = self._integrands(k_i, mu_j, f, V, geometry, corrs_q, W_ij)
            kernel = spline.hermite(logk, self._logk_fft, ft.pk_select_kernel, ft.pk_select_slopes, axis=1)
            out = []
            for g_o in g:
                g_o = ft.four_pi_q3 * g_o.at[0, :].subtract(g_o[0, -1])
                out.append(jnp.sum(jnp.sum(g_o * kernel, axis=-1), axis=-1))
            out = jnp.stack(out) + lin * p_i
            for component, row, factor, interpolated, k0_const in _KSPACE_ROWS:
                if k0_const and self.subtract_k0_const:
                    continue
                if interpolated:
                    value = spline.hermite(logk, self._logk_fft, kspace_rows[row], kspace_slopes[row])
                else:
                    value = kspace_rows[row]
                out = out + W_ij[:, component] * (factor * value)
            return out

        def per_k(k_row, mu_row, p_row):
            return jax.vmap(per_mu)(k_row, mu_row, V_all, sin_true, p_row)

        out = jax.vmap(per_k)(k_true, mu_true, pk_lin_true)       # (nk, nmu, n_out)
        return jnp.moveaxis(out, -1, 0)

    def _pkmu_true(self, k_true, mu_true, sin_true, pt_terms, pk_data, params):
        """Tree level plus one loop at the true (k, mu), the Zel'dovich counterterms folded into the weights.

        The component weights at each (k, mu) are the 13 template weights times
        ``_TEMPLATE_MATRIX``. The template weights are the 12 bias monomials of
        ``_get_lpt_bias_factors`` and ``Counterterms.zeldovich_weight`` (leading plus NLO
        shape for 'zeldovich', zero for the k-space bases, which ``base`` adds).
        """
        self._require_pt_terms_rows(pt_terms)
        f = params.f
        bias_facs = self._get_lpt_bias_factors(params.bias_a, params.bias_b)
        M = jnp.asarray(_TEMPLATE_MATRIX)

        def component_weights(k_i, mu_j):
            ctr_weight = self._counterterms.zeldovich_weight(k_i, mu_j, f, params.ctr)
            row = jnp.concatenate([bias_facs, jnp.stack([ctr_weight])])
            return (row @ M)[None, :]

        return self._final_integral(k_true, mu_true, sin_true, pt_terms.rows, f, pk_data,
                                    component_weights)[0]

    @partial(jit, static_argnames=['self'])
    def get_pkmu_components(self, k, mu, pk_data, f):
        """The 23 components of the final integrand on the (k, mu) grid, ``(23, nk, nmu)``.

        Diagnostic without AP, in the order of :class:`Component` (``_COMPONENTS``).
        ``_final_integral`` with W = identity, so no bias or counterterm weights are applied.
        """
        pt_terms = self.get_pt_terms(pk_data, None)
        k_true, mu_true, sin_true = self._true_coordinates(k, mu, 1.0, 1.0)
        return self._final_integral(k_true, mu_true, sin_true, pt_terms.rows, f, pk_data,
                                    jnp.eye(N_COMPONENTS))

    @partial(jit, static_argnames=['self'])
    def get_pkmu_templates_from_pt_terms(self, k, mu, pt_terms, pk_data, f, alpha_perp=1.0, alpha_para=1.0):
        """Bias-independent templates ``(13, nk, nmu)``, indexed by :class:`Template`.

        Template 0 (``Template.ONE``, bias monomial 1) is the matter part, templates 1-11
        multiply the other bias monomials of ``_LPT_BIAS_DEGREES``, and template 12
        (``Template.CTR``) is the Zel'dovich counterterm base. This is the template API
        needed for ``counterterm_base='zeldovich'`` and for fast variation of the biases.
        Recombining the templates into P(k, mu) is the caller's job. Template 12 applies
        only to the 'zeldovich' base (the other bases are added in k space), and no volume
        factor ``1 / (alpha_perp^2 alpha_para)`` is applied.
        ``pk_data`` must be the spectrum ``pt_terms`` was computed from.

        ``_final_integral`` with W = ``_TEMPLATE_MATRIX``.
        """
        self._require_pt_terms_rows(pt_terms)
        k_true, mu_true, sin_true = self._true_coordinates(k, mu, alpha_perp, alpha_para)
        return self._final_integral(k_true, mu_true, sin_true, pt_terms.rows, f, pk_data,
                                    jnp.asarray(_TEMPLATE_MATRIX))

    def get_pkmu_templates(self, k, mu, pk_data, f, alpha_perp=1.0, alpha_para=1.0):
        """Templates from ``pk_data`` (see :meth:`get_pkmu_templates_from_pt_terms`)."""
        pt_terms = self.get_pt_terms(pk_data, None)
        return self.get_pkmu_templates_from_pt_terms(k, mu, pt_terms, pk_data, f, alpha_perp, alpha_para)
