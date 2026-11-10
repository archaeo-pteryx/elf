"""LPT backend: the one-loop LPT galaxy power spectrum P(k, mu) in redshift space.

The model is Eqs. (pkmu_lpt_tree) and (pkmu_lpt_1loop) of the paper, with the
correlators (cumulants) of its appendix "Explicit formulas for LPT power spectrum".
Below, H_l^n[S](q) = int dk k^{n+2} S(k) j_l(kq) / (2 pi^2) and xi_l^n = H_l^n[P]
(the paper's xi^l_n).

The chain.  tools/pt_autogen derives the integrand from the generating function and writes
the plain-text tables ``tables/lpt/*.dat``, which ``lpt_tables`` loads.  The
runtime reads them in three places.

Step 1, ``get_pt_terms(pk_data, h)`` -> :class:`PTTerms`, depends only on P_lin and h.
Its ``rows`` (:class:`Row`) are 28 correlators on the q grid and five k-space rows, made
by ``lpt_stages.pt_rows`` from the recipes of ``rows_recipes.dat`` (and the units and
sources of ``rows_sources.dat``, the rules of ``rows_rules.dat``):
  A. ``_get_xi_ln``: P(k) -> xi_l^n(q) for the (l, n) of ``ln_list``, for P and
     P_lt = P exp(-(k/k_IR)^2).
  B. the stage-B units of rows_sources.dat.  A unit of kind 22 is a product of xi_l^n
     transformed q -> k, a unit of kind 13 a transform of xi_l^n times P (the kinds are
     labels, as in P22 and P13).  The sources Q_1, Q_2 (k) are made of the 22 units and
     R_1, R_2 of the 13 units.
  C. ``_transform_sources``: combinations S(k) of the sources -> H_l^n[S](q) (k -> q).
The other rows are products of xi_l^n, zero-lag constants (grid moments) and references
to other rows, and the k-space rows come from the rules of rows_rules.dat.

Step 2, for each (k, mu), ``_pkmu_true`` -> ``_final_integral`` does the final q -> k
integral, its angular integrals in the Legendre representation 'leg' (``angular``, the
table ``angular.dat``).  K = k + f mu k n is the redshift-space wavevector of the point
(n the line of sight, ``lpt_defs``), X_lin_lt and Y_lin_lt are the rows ``X_LIN_LT`` and
``Y_LIN_LT``, and a moment j is a product of powers of Khat.qhat and n.qhat in the
integrand (``lpt_tables.MOMENTS``).  ``_integrands`` gives one
integrand per order L <= lmax of j_L(kq), g_L = D_X sum_j c_j C_L^j minus its linear part,
with D_X = exp(-K^2 X_lin_lt / 2), the angular factors C_L^j of the moments j (the matrix
G of ``angular.angular_matrix`` once per mu times the exponential moments
``angular.exp_moments`` of B = -K^2 Y_lin_lt / 2) and the moment coefficients c_j, the
coefficients of the component table (``basis_rows.dat``, ``component_entries.dat``) summed
with the component weights (``_components_from_table``, ``_weighted_coeffs``).
``_final_integral`` contracts it with the kernels of the final transform at k
(``_point_kernel``: the circulant sequences of the FFTlog kernels of ``_FinalTransform``,
interpolated in ln k). It then adds back the linear part (Kaiser factor times P_lin(k)) and
the k-space rows. ``base.PowerSpectrum`` adds the stochastic term, the k-space
counterterms and the AP volume factor.  The derivatives of the point loop come from the
Legendre tangent rule ``lpt_rule`` or from plain autodiff of the same loop, as the
constructor's ``tangent_rule`` fixes.

Tables (module ``lpt_tables``, from ``components.dat``). ``_COMPONENTS`` lists the 23
components (terms) of the integrand (:class:`Component`). ``_TEMPLATE_MATRIX`` (13 x 23)
assigns them to the templates (:class:`Template`), which are the 12 bias monomials of
``_LPT_BIAS_DEGREES`` and the Zel'dovich counterterm. ``_LINEAR_MATRIX`` (23 x 3) gives
the kind (:class:`LinearKind`) of each component's linear part. ``_KSPACE_ROWS``
(``rows_rules.dat``) pairs the k-space rows with components. The one engine
``_final_integral(..., W)`` weights the components with W (n_out, 23). W = identity gives
the components, W = ``_TEMPLATE_MATRIX`` the templates, and W = (bias and counterterm
weights) @ ``_TEMPLATE_MATRIX`` the spectrum.
"""

from typing import NamedTuple, Optional

import numpy as _np

import jax
from jax import jit
from functools import partial

import jax.numpy as jnp

from . import angular
from . import fftlog
from . import lpt_rule
from . import lpt_stages
from . import spline

from .base import PowerSpectrum
# The definitions of the final integrand, the tables (lpt_tables) and the hand-written geometry (lpt_defs), are
# imported by name, so that lpt.Row, lpt.Component, ... are attributes of this module.
from .lpt_tables import (_LPT_BIAS_DEGREES, Template, LinearKind, Component, _COMPONENTS, N_COMPONENTS,
                         _TEMPLATE_MATRIX, _LINEAR_MATRIX, Row, TREE_ROWS, _KSPACE_ROWS, MOMENTS, MOMENT_POLYNOMIALS,
                         COMPONENT_ENTRIES, entry_scalars, row_basis)
from .lpt_defs import _rsd_geometry
from .utils import cross_bias_factor

N_ROWS = len(Row)


def _components_from_table(s, R):
    """The coefficients of the components from the component table, ``[c][j]`` -> ``(nq,)``.

    c_j^c(q) = sum over the entries e = (c, j, t) of tables/lpt/component_entries.dat of s_e R_t(q), with ``s`` the
    per-point factors of ``lpt_tables.entry_scalars`` and ``R`` the basis rows of ``lpt_tables.row_basis``.  One dict
    {j: c_j^c} per component c, in Component order.
    """
    components = []
    for entries in COMPONENT_ENTRIES:
        coefficients = {}
        for j, e, t in entries:
            term = s[e] * R[t]
            coefficients[j] = coefficients[j] + term if j in coefficients else term
        components.append(coefficients)
    return components


def _weighted_coeffs(components, W):
    """The moment coefficients of the outputs, c^o_j = sum_c W[o, c] c_j^c, one ``(n_out, nq)`` array per moment j in
    the order of MOMENTS."""
    coeffs = [None] * len(MOMENTS)
    for c, component in enumerate(components):
        for j, c_j in component.items():
            term = W[:, c, None] * c_j
            coeffs[j] = term if coeffs[j] is None else coeffs[j] + term
    return coeffs


class PTTerms(NamedTuple):
    """LPT PT terms, which depend only on ``P_lin`` and ``h``.

    ``rows`` ``(33, nfft)`` are 28 q-space correlators and five k-space rows, named by
    :class:`Row`.  ``pk_nw_data`` (the ``(2, n)`` no-wiggle table, k in row 0 and P_nw in
    row 1), ``Sigma2`` and ``dSigma2`` are the no-wiggle data of the 'linear_ir_resum'
    counterterm base (None for the other bases and for ``h=None``).

    Rows 0-27 live on the core q grid ``LPT._q``, rows 28-32 on the core k grid
    ``LPT._k``. Vector and tensor cumulants are stored as their scalar coefficients,
    e.g. U_i = qhat_i U and A_ij = X delta_ij + Y qhat_i qhat_j. The rows are evaluated from
    the recipes of tables/lpt/rows_recipes.dat (``lpt_stages``), ``TREE_ROWS`` being the rows
    linear in P_lin.

    ======  =============================  ==================================================
    row     Row                            definition (paper symbols, xi_l^n = H_l^n[P])
    ======  =============================  ==================================================
    0, 1    X_LIN, Y_LIN                   A^lin_ij
    2, 3    X_LIN_LT, Y_LIN_LT             the same from P_lt = P exp(-(k/k_IR)^2), i.e. A^< (= rows 0, 1 for k_IR None)
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

    Stage A data, one slot per (l, n) of ``LPT.ln_list``: ``slots[(l, n)]`` is the row of
    ``forward`` and of the weights of ``resampling`` and the index of xi_l^n in the outputs of
    ``_get_xi_ln`` (``LPT._xi_slot``).  The orders are stored by slot, so any l >= 0 and integer n have
    their own place.  Every row has the kernel bias
    ``nu`` (the kernel ``HankelGrids.u_m[l]``) and the input bias ``nu - (n + 3)``, and lands
    on ``HankelGrids.q_padded[l]``.  ``resampling`` takes the rows from there onto the common
    padded q grid ``LPT._q_padded`` with the fixed weights of ``spline.resampling`` (the cubic
    interpolation of ``spline.interp1d`` in ln q).
    """
    forward: fftlog.ForwardTransform  # one row per slot
    resampling: spline.Resampling   # rows onto _q_padded, weights (n_pairs, n_padded, width)
    slots: dict                     # (l, n) -> slot


class _SourceTransform(NamedTuple):
    """The Q/R sources (backward, q -> k) and their k -> q transforms to the correlator rows.

    Stage B (``lpt_stages.stage_b``) and stage C (``LPT._transform_sources``) data. The
    order index l runs over 0 ... ``lpt_stages.MAX_ORDER``, the orders of every stage.  The
    22 units (kind '22' of ``lpt_stages.UNITS``, which make the Q sources) and the b2^2 row are
    transformed with the l = 0 kernel ``HankelGrids.u_m[0]``, the 13 units (kind '13', which make
    the R sources) with ``HankelGrids.u_m[l]`` (the conjugate relation lands them on ``k_padded``).
    The end weights live on ``k_padded``. ``u_m_src``, ``q_src_padded``, ``q_src_pow`` and the weights
    of ``resampling`` are indexed ``[group, l]``, ``k_pow`` ``[group]``. Group 0 has bias ``nu``
    and group 1 has ``nu_n0`` (sources with n >= 0).  ``k_pow`` and ``q_src_pow`` are the power
    factors of the k -> q transforms of each group (``fftlog.hankel``).
    Each kernel uses the low-ringing phase of its own order.  ``resampling`` takes the outputs
    from ``q_src_padded`` onto the common padded q grid ``LPT._q_padded`` with the fixed weights
    of ``spline.resampling`` (the cubic interpolation of ``spline.interp1d`` in ln q).
    """
    four_pi_q3_padded: jnp.ndarray  # 4 pi q^3 on _q_padded (with d ln q it gives 4 pi q^2 dq)
    end_weights_Q: jnp.ndarray      # support [k_min, 2 k_max] ([k_min, 2 k_cut] with k_cut)
    end_weights_R: jnp.ndarray      # support [k_min, k_max] ([k_min, k_cut] with k_cut)
    R_source_rows: jnp.ndarray      # stage-A slots of the 13 units of stage B, (l, n) = lpt_stages.UNITS_13_LN
    R_source_ells: jnp.ndarray      # their orders l
    u_m_src: jnp.ndarray            # k -> q kernels [bias group (nu, nu_n0), l]
    q_src_padded: jnp.ndarray       # (2, n_l, n_padded) output q grids of u_m_src
    k_pow: jnp.ndarray              # (2, n_padded) k_padded^-nu and k_padded^-nu_n0 of the two bias groups
    q_src_pow: jnp.ndarray          # (2, n_l, n_padded) q_src_padded[g]^-nu_g
    resampling: spline.Resampling   # q_src_padded onto _q_padded, weights (2, n_l, n_padded, width)


class _FinalTransform(NamedTuple):
    """The final q -> k transform of the (k, mu) integrand (``LPT._final_integral``).

    q runs over the core q grid ``LPT._q`` (nq = nfft, the nodes q_j) and the order L over 0 ... ``LPT.lmax``.  Every
    order uses the kernel u_L with the phase of the order 0, so the integrand stays on the common q grid.  The FFTlog
    transform of the unit source at q_j (zero padding by p nodes on each side, N = nq + 2p, kernel bias nu), read at
    the core k node k_n, is

        kern_L(k_n)[j] = k_n^{-nu} q_j^{-nu} w_j beta_L[(n + j + 2p) mod N],      beta_L = irfft(conj(u_L), N),

    with w the end weights of the core q grid.  It is the circulant sequence beta_L times factors of the two grids.
    Between the nodes ``LPT._point_kernel`` interpolates beta_L with the cubic Hermite in the index n (node slopes the
    periodic central differences, as ``spline.slopes`` on the uniform ln k grid) and multiplies by the exact k^{-nu}.
    ``beta`` and ``beta_slopes`` hold beta_L[(s + 2p) mod N] and its slopes for s = 0 .. 2 nq - 1, the window every
    point reads by contiguous slices.
    """
    beta: jnp.ndarray               # (lmax + 1, 2 nq) beta_L[(s + 2p) mod N]
    beta_slopes: jnp.ndarray        # (lmax + 1, 2 nq) (beta_L[s + 1] - beta_L[s - 1]) / 2 of the same window (periodic)
    q_factor: jnp.ndarray           # (nq,) q_j^{-nu} w_j
    end_weights_q_core: jnp.ndarray # (nq,) trapezoid weights w of the core q grid, 1/2 at both ends
    four_pi_q3: jnp.ndarray         # (nq,) 4 pi q^3
    log_k0: float                   # ln k of the first core k node
    dlnk: float                     # the spacing of the core ln k grid


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
                 k_cut=None,
                 lmax=8,
                 k_IR=0.2,
                 bias_basis='bG2',
                 tangent_rule=True,
                 ):
        """One-loop Lagrangian-PT galaxy power spectrum in redshift space.

        ``Params.bias`` is Lagrangian (b1, b2, bG2, bGamma3), with b_s2 in the third slot
        for ``bias_basis='bs2'``.  Shared options as in :class:`base.PowerSpectrum`.
        ``pad_mode='power-law'`` continues P only for P -> xi, and the Q/R sources stay
        cut at 2 k_max / k_max.  ``irres_method``, ``r_bao`` and ``lambda_ir`` only shape
        the 'linear_ir_resum' counterterm base.
        ``lmax`` (2 <= lmax <= 40, the orders of tables/lpt/angular.dat) is the order of the
        j_L expansion of the angular integrals of the final integral in the Legendre
        representation (L <= lmax), not the multipole order of ``ells``.  The default 8
        keeps P_0 and P_2 within 5e-5 and P_4 within 2e-3 of their peaks at k <= 0.3 (against
        the exact angular integral).  12 lowers this to 5e-6 for P_0 and P_2 and 5e-5 for
        P_4, and P_4 at k up to 0.5 needs 16 to stay below 1e-4.
        ``k_IR`` sets the split ``P_lt = P e^{-(k/k_IR)^2}`` inside the LPT body.  A^< from P_lt
        stays in the exponent and A^> = A^lin - A^< is expanded.  The paper uses 0.2.  ``lambda_ir``
        is a different scale, the upper limit of the Sigma^2 integral of the counterterm base.
        ``k_IR=None`` makes no split, as velocileptors' kIR=None.  The whole A^lin stays in the
        exponent, P_lt = P, and the GT rows and the components ``AA``, ``A_U`` and ``A_XI`` vanish
        identically.  The converged result depends on this choice at the 10% level at k = 0.2, z = 0.
        ``bias_basis`` is 'bG2' or 'bs2'.
        ``tangent_rule`` is the differentiation strategy of the final integral, fixed at
        construction: True, the Legendre tangent rule of ``lpt_rule`` (a ``jax.custom_jvp``
        around the point loop), False, plain autodiff of the same point loop, for tests and
        diagnosis (the same values, slow gradients on CPU).

        The attributes ``ln_list`` (the (l, n) of the forward transforms xi_l^n, in slot
        order) and ``_angular`` (the :class:`angular.AngularPlan` of the moments) are internal
        tables.
        """

        if bias_basis not in ('bG2', 'bs2'):
            raise ValueError("bias_basis must be 'bG2' or 'bs2'")
        if k_IR is not None and not k_IR > 0:
            raise ValueError('k_IR must be positive or None (no IR split)')
        if lmax < 2:
            raise ValueError(
                "lmax must be >= 2: the linear part of the final integrand, which is "
                "replaced by linear theory, has orders L = 0, 1, 2"
            )
        super().__init__(kmin_fft=kmin_fft, kmax_fft=kmax_fft, nfft=nfft, pad_mode=pad_mode,
                         fftlog_settings=fftlog_settings, ngauss=ngauss, ells=ells,
                         counterterm_base=counterterm_base, irres_method=irres_method,
                         r_bao=r_bao, lambda_ir=lambda_ir, subtract_k0_const=subtract_k0_const,
                         k_cut=k_cut)

        self.lmax = lmax
        self.k_IR = k_IR
        self.bias_basis = bias_basis
        self.tangent_rule = tangent_rule

        self._angular = angular.AngularPlan(self.lmax, tuple(MOMENT_POLYNOMIALS[m] for m in MOMENTS))

        # the kernels of the stages reach their highest order (lpt_stages.MAX_ORDER), the final transform has its own
        self._set_hankel()

    def _set_hankel(self):
        """Grids, kernels and static support weights of the four transforms.

        ``self._hankel`` (:class:`fftlog.HankelGrids`) holds the padded k grid and the
        per-order q grids. The stage data go to ``_xi_transform`` (A), ``_source_transform``
        (B and C) and ``_final_transform`` (the final integral). The common q grid
        ``_q_padded`` is the l = 0 output grid, and ``_q`` is its core. The kernels of
        ``_hankel`` and ``_source_transform`` reach the highest order of every stage,
        ``lpt_stages.MAX_ORDER`` (stage A, the 13 units of stage B, the H terms of stage C).
        The final transform builds its own kernels of the orders up to ``self.lmax`` on the
        l = 0 grid. Stage A is planned for the orders ``lpt_stages.STAGE_A`` (``_set_xi_transform``).
        """
        self._hankel = fftlog.HankelGrids(self._k, self.fftlog_settings, self.pad_mode, lpt_stages.MAX_ORDER,
                                          self._n_cut)
        self._set_xi_ells_transform()

        self._q_padded = self._hankel.q_padded[0]
        self._q = self._hankel.grid.crop(self._q_padded)
        self._log_q_padded = jnp.log(self._q_padded)

        self._set_xi_transform(lpt_stages.STAGE_A)
        self._set_source_transform()
        self._set_final_transform()

    def _set_xi_transform(self, ln_list):
        """Plan stage A for the orders ``ln_list`` ((l, n) pairs, each once): ``ln_list`` and ``_xi_transform``.

        Slot i holds the order ``ln_list[i]`` (``_xi_slot``), whatever its l and n.
        """
        pairs = [(int(l), int(n)) for l, n in ln_list]
        if len(set(pairs)) != len(pairs):
            raise ValueError(f'stage A lists an order twice: {pairs}')
        self.ln_list = jnp.array(pairs)
        forward = fftlog.forward_transform(self._hankel.grid, self.fftlog_settings.nu, self.ln_list[:, 0],
                                           self.ln_list[:, 1])
        resampling = spline.resampling(_np.asarray(self._log_q_padded), _np.log(_np.asarray(forward.q)))
        self._xi_transform = _XiTransform(forward=forward, resampling=resampling,
                                          slots={ln: i for i, ln in enumerate(pairs)})

    def _xi_slot(self, l, n):
        """The slot of xi_l^n in the outputs of ``_get_xi_ln``.

        ValueError for an order that stage A does not transform (not in ``ln_list``): it has
        no slot to read, rather than another order's or a zero array.
        """
        slot = self._xi_transform.slots.get((int(l), int(n)))
        if slot is None:
            raise ValueError(f'stage A has no xi_l^n of (l, n) = ({l}, {n}), its orders (ln_list) are '
                             f'{list(self._xi_transform.slots)}')
        return slot

    def _set_source_transform(self):
        """``_source_transform``, the data of stages B and C (:class:`_SourceTransform`)."""
        dln = self._hankel.grid.dln
        # the units of kind 13 stay on their own grids q_padded[l], and the conjugate relation lands them on k_padded
        r_ln = lpt_stages.UNITS_13_LN
        k_ref = _np.asarray(self._hankel.k_padded)
        for l, _n in r_ln:
            k_back = _np.asarray(self._hankel.q_grids[l].conjugate(self._hankel.lnxy[l]).x)
            assert _np.allclose(k_back, k_ref, rtol=1e-12), (
                f'R-source backward output grid for ell={l} is not the common '
                'k grid; the conjugate relation of the low-ringing phase is '
                'broken')
        # an exception to the bias rule, the sources with n >= 0 (theta) at nu_n0 on the low-ringing grids of (l, nu_n0)
        setups_n0 = [fftlog.hankel_setup(l, self.fftlog_settings.nu_n0, self._hankel.grid)
                     for l in range(lpt_stages.MAX_ORDER + 1)]
        q_src_padded = jnp.stack([self._hankel.q_padded, jnp.array([s[1].x for s in setups_n0])])
        nu_groups = (self.fftlog_settings.nu, self.fftlog_settings.nu_n0)
        k_padded = self._hankel.k_padded
        self._source_transform = _SourceTransform(
            four_pi_q3_padded=4.0 * jnp.pi * self._q_padded**3,
            end_weights_Q=fftlog.trapezoid_weights('core_to_2kmax', self._nfft, self._hankel.npad, dln, self._n_cut),
            end_weights_R=fftlog.trapezoid_weights('core', self._nfft, self._hankel.npad, dln, self._n_cut),
            R_source_rows=jnp.asarray([self._xi_slot(l, n) for l, n in r_ln], dtype=jnp.int32),
            R_source_ells=jnp.asarray([l for l, _ in r_ln], dtype=jnp.int32),
            u_m_src=jnp.stack([self._hankel.u_m, jnp.array([s[2] for s in setups_n0])]),
            q_src_padded=q_src_padded,
            k_pow=jnp.stack([k_padded ** (-nu) for nu in nu_groups]),
            q_src_pow=jnp.stack([q_src_padded[g] ** (-nu) for g, nu in enumerate(nu_groups)]),
            resampling=spline.resampling(_np.asarray(self._log_q_padded), _np.log(_np.asarray(q_src_padded))),
        )

    def _set_final_transform(self):
        """``_final_transform``, the window of the circulant sequences of the final kernels
        (:class:`_FinalTransform`)."""
        dln = self._hankel.grid.dln
        # The final transform is an exception to the bias rule, with every order on the phase of the order 0 (one q
        # grid for the integrand).  Its kernels u_L are kept as the circulant sequences beta_L on the window of
        # _FinalTransform.
        nq, p = self._nfft, self._hankel.npad
        n_padded = nq + 2 * p
        u_m = jnp.array([fftlog.hankel_setup(l, self.fftlog_settings.nu, self._hankel.grid,
                                             lnxy=self._hankel.lnxy[0])[2] for l in range(self.lmax + 1)])
        beta = jnp.fft.irfft(jnp.conj(u_m), n=n_padded)
        beta_slopes = 0.5 * (jnp.roll(beta, -1, axis=1) - jnp.roll(beta, 1, axis=1))
        window = (_np.arange(2 * nq) + 2 * p) % n_padded
        end_weights_q_core = fftlog.trapezoid_weights('core', nq, 0, dln)
        log_k = _np.log(_np.asarray(self._k))
        self._final_transform = _FinalTransform(
            beta=beta[:, window],
            beta_slopes=beta_slopes[:, window],
            q_factor=self._q**(-self.fftlog_settings.nu) * end_weights_q_core,
            end_weights_q_core=end_weights_q_core,
            four_pi_q3=4.0 * jnp.pi * self._q**3,
            log_k0=float(log_k[0]),
            dlnk=float((log_k[-1] - log_k[0]) / (nq - 1)),
        )

    # Chain: P -> xi_l^n (k -> q), xi products -> Q/R sources (q -> k), sources ->
    # correlator rows H_l^n[S] (k -> q).  Each transform uses the low-ringing phase of its
    # own order, and outputs meet through the conjugate grid relation or by resampling.

    def _get_xi_ln(self, arrays):
        """P -> xi_l^n of ``arrays`` ``(n_arrays, nfft)``: ``(xi_ln_common, xi_ln_own_grid)``.

        Stage A. ``arrays`` are spectra on the core k grid (P and P_lt in ``get_pt_terms``),
        transformed for the (l, n) of ``ln_list``.  Both outputs have the shape
        ``(n_arrays, n_pairs, n_padded)`` and hold xi_l^n in the slot ``_xi_slot(l, n)``:
        ``xi_ln_common`` resampled onto ``_q_padded`` (the fixed weights of ``spline.resampling``,
        the same cubic interpolation), ``xi_ln_own_grid`` on ``q_padded[l]``.
        Both carry the plain support mask ``HankelGrids.mask_xi``, not the end weights
        ``HankelGrids.w_xi``, because the rows that are pointwise products of these outputs
        would be wrong with a halved q_max node.
        """
        arrays = jnp.atleast_2d(jnp.asarray(arrays))
        fx = self._hankel.pad_input(arrays / (2 * jnp.pi**2))              # (n_arrays, n_padded)
        raw = self._xi_transform.forward.apply(fx[:, None, :])          # (n_arrays, n_pairs, n_padded)

        xis = spline.resample(self._xi_transform.resampling, raw)
        return xis * self._hankel.mask_xi, raw * self._hankel.mask_xi

    def _transform_sources(self, sources, *, ells, ns):
        """``H_l^n[S](q) = int dk k^{n+2} S(k) j_l(kq) / (2 pi^2)`` of each source on the common padded q grid.

        Stage C. ``sources`` is a list of ``(N,)`` sources on the padded k grid
        (N = n_padded) and ``ns`` holds one n per source. ``ells`` holds one order per
        source, giving ``(n_src, N)``, or one tuple of orders per source, giving
        ``(n_src, n_l, N)`` (one rfft per source, one irfft per order).

        Sources with ``n >= 0`` use the bias ``nu_n0``, the others ``nu``. Each output is
        resampled from its own grid onto ``_q_padded`` with the fixed weights of
        ``spline.resampling`` (the same cubic interpolation).
        """
        ells = _np.asarray(ells)
        ns = _np.asarray(ns)
        n_l = self._source_transform.u_m_src.shape[1]
        if ells.max() >= n_l:          # the gathers [group, ells] below would clip it to the last order
            raise ValueError(f'_transform_sources has the kernels of the orders l <= {n_l - 1}, not '
                             f'{sorted({int(l) for l in ells.flat if l >= n_l})}')
        st, k = self._source_transform, self._hankel.k_padded
        group = (ns >= 0).astype(int)                 # bias group: 0 nu, 1 nu_n0
        fx = jnp.stack([s * k**(int(n) + 3) for s, n in zip(sources, ns)]) / (2 * jnp.pi**2)
        x_pow = st.k_pow[group]                       # (n_src, N), k^-nu of the bias group of each source
        if ells.ndim == 2:
            fx, x_pow = fx[:, None, :], x_pow[:, None, :]
            group = _np.broadcast_to(group[:, None], ells.shape)
        d = fftlog.hankel(fx, x_pow, st.u_m_src[group, ells], st.q_src_pow[group, ells], self._hankel.npad, crop=False)
        plan = self._source_transform.resampling
        return spline.resample(plan._replace(weights=plan.weights[group, ells]), d)

    @partial(jit, static_argnames=['self'])
    def get_pt_terms(self, pk_data, h):
        """The :class:`PTTerms` of ``pk_data`` (the ``(2, n)`` array of the input table, k [h/Mpc] in row 0 and P_lin
        in row 1).

        The rows (:class:`Row`) are the tree rows (``TREE_ROWS``), the one-loop rows on the core q grid and the
        k-space rows, evaluated from the recipes of tables/lpt/rows_recipes.dat by ``lpt_stages.pt_rows`` (stage A
        ``_get_xi_ln`` of P and P_lt, the stage-B units and sources, stage C through ``_transform_sources``, the
        products, constants and rules).  :class:`PTTerms` has the table of the rows.

        ``h`` enters only the no-wiggle data of the 'linear_ir_resum' counterterm base.
        ``h=None`` skips them (the template and component methods do so), and such
        ``pt_terms`` cannot be fed to the spectrum methods under that base.
        """
        rows = lpt_stages.pt_rows(self, self._pk_core(self._prepare(pk_data)))
        if self.counterterm_base == 'linear_ir_resum' and h is not None:
            pk_nw_data, Sigma2, dSigma2 = self._get_ir_data(pk_data, h)
        else:
            pk_nw_data = Sigma2 = dSigma2 = None
        return PTTerms(rows=rows, pk_nw_data=pk_nw_data, Sigma2=Sigma2, dSigma2=dSigma2)

    def _linear_rows(self, mu_j, f, geometry, K, Ksq, corrs_tree):
        """Linear part of the final integrand (its JVP at pt_terms = 0), its weight and transforms.

        Returns ``(lin_weight, rows, kaiser)``.  The weight is
        ``lin_weight = exp(-K^2 (X_lt + Y_lt)(q_max) / 2)``.  ``rows`` are the triples
        ``(kind, L, row)`` of the linear rows in the orders L of j_L(kq).  At B = 0 the angular
        factor C_L^j is the plane-wave coefficient of the moment's polynomial, nonzero for
        finitely many L, so the Zel'dovich rows sit at L = 0,
        -(K^2/2) [X_lt + X_gt + (Y_lt + Y_gt)/3], and L = 2, (K^2/3) P_2(c) (Y_lt + Y_gt), the b1
        row at L = 1, -2 K c U_lin, and the xi row at L = 0, xi_lin.  ``kaiser`` holds the
        transforms of the kinds over P, ``((1 + f mu^2)^2, 2 (1 + f mu^2), 1)``.

        The kinds are the columns of ``_LINEAR_MATRIX`` (:class:`LinearKind`, the order of
        ``kaiser``). The rows are linear-theory terms (first order in the tree rows), on the
        core q grid.  ``geometry`` is the :class:`_RSDGeometry` of this mu, and ``K``, ``Ksq``
        are |K| and K^2 of the point.

        Hand-written.  Which components have a linear part of which kind comes from tables/lpt/components.dat
        (``_LINEAR_MATRIX``).  The rows of each kind, their expansion in L and their Kaiser factors are written here.
        """
        X_lt, Y_lt = corrs_tree[Row.X_LIN_LT], corrs_tree[Row.Y_LIN_LT]
        X_gt, Y_gt = corrs_tree[Row.X_LIN_GT], corrs_tree[Row.Y_LIN_GT]
        xi_lin, U_lin = corrs_tree[Row.XI_LIN], corrs_tree[Row.U_LIN]

        lin_weight = jnp.exp(-0.5 * Ksq * (X_lt[-1] + Y_lt[-1]))
        Y_sum = Y_lt + Y_gt
        rows = ((LinearKind.ZA, 0, -0.5 * Ksq * (X_lt + X_gt + Y_sum / 3.0)),
                (LinearKind.ZA, 2, (Ksq / 3.0) * (0.5 * (3.0 * geometry.c**2 - 1.0)) * Y_sum),
                (LinearKind.B1, 1, -2.0 * K * geometry.c * U_lin),
                (LinearKind.XI, 0, xi_lin))

        kaiser = 1 + f * mu_j**2
        return lin_weight, rows, (kaiser**2, 2.0 * kaiser, jnp.ones_like(kaiser))

    def _integrands(self, k_i, mu_j, f, G_rows, geometry, corrs_q, W):
        """Integrands ``(lmax + 1, nq)`` of the outputs with component weights ``W`` ``(n_out, 23)`` at one (k, mu).

        ``g_o[L] = D_X sum_{rows r = (j, L)} c^o_j C_r - lin_weight (linear rows of order L)
        - [L = 0] W[o, Component.XI_XI] xi^2/2`` with D_X = exp(-K^2 X_lt / 2), the angular factors
        C_r = C_L^j of the rows of ``self._angular`` (``angular.banded_coefficients`` of ``G_rows`` and
        the exponential moments J of B = -K^2 Y_lt / 2) and the moment coefficients
        c^o_j(q) = sum_c W[o, c] c_j^c(q) of the component coefficients c_j^c of the basis rows of
        ``corrs_q = (tree rows, basis rows R_t)`` (``_components_from_table``, ``_weighted_coeffs``).
        The linear rows come from ``_linear_rows``, weighted per kind by ``W @ _LINEAR_MATRIX``.
        The term xi_lin^2 / 2 that the component table gives to XI_XI is subtracted from the L = 0
        integrand, because the k-space rows B2SQ_RESIDUAL and B2SQ_CONST are its transform.
        Returns ``(g, lin)`` with ``lin * P_lin(k)`` the k-space add-back: ``g`` a tuple of n_out
        arrays (a leading output axis breaks XLA's fusion), ``lin`` of shape ``(n_out,)``.
        ``G_rows`` and ``geometry`` are the angular matrix and the :class:`_RSDGeometry` of this mu.
        """
        corrs_tree, R = corrs_q
        plan = self._angular
        K = k_i * geometry.Kfac
        Ksq = K**2
        D_X = jnp.exp(-0.5 * Ksq * corrs_tree[Row.X_LIN_LT])
        # the exponential moments materialized once per point, otherwise XLA repeats the end of the J_0 series in the
        # fusion of every order L that reads J
        J = jax.lax.optimization_barrier(tuple(angular.exp_moments(-0.5 * Ksq * corrs_tree[Row.Y_LIN_LT], plan.nm)))
        C = angular.banded_coefficients(plan, G_rows, J)
        coef = _weighted_coeffs(_components_from_table(entry_scalars(k_i, mu_j, f, geometry.Kfac), R), W)

        lin_weight, linear, kaiser = self._linear_rows(mu_j, f, geometry, K, Ksq, corrs_tree)
        w_kind = W @ jnp.asarray(_LINEAR_MATRIX)                        # (n_out, 3)
        lin = lin_weight * (w_kind @ jnp.stack(kaiser))
        b2sq = 0.5 * corrs_tree[Row.XI_LIN]**2

        g = []
        for o in range(W.shape[0]):
            orders = []
            for L, rows_L in enumerate(plan.rows_of_order):
                g_L = coef[plan.row_moment[rows_L[0]]][o] * C[rows_L[0]]
                for r in rows_L[1:]:
                    g_L = g_L + coef[plan.row_moment[r]][o] * C[r]
                g_L = D_X * g_L
                for kind, L_row, row in linear:
                    if L_row == L:
                        g_L = g_L - lin_weight * (w_kind[o, kind] * row)
                if L == 0:
                    g_L = g_L - W[o, Component.XI_XI] * b2sq
                orders.append(g_L)
            g.append(jnp.stack(orders))
        return tuple(g), lin

    def _kspace_tables(self, rows):
        """The ``_KSPACE_ROWS`` rows (interpolated rows whole, constants at k_min) and their ln k slopes."""
        kspace_rows = {row: rows[row] if interpolated else rows[row, 0]
                       for _, row, _, interpolated, _ in _KSPACE_ROWS}
        kspace_slopes = {row: spline.slopes(self._logk_fft, rows[row])
                         for _, row, _, interpolated, _ in _KSPACE_ROWS if interpolated}
        return kspace_rows, kspace_slopes

    def _kspace_terms(self, logk, kspace):
        """``((component, value), ..)`` of the ``_KSPACE_ROWS`` rows at ``logk``, value = factor times the row
        (interpolated in ln k, or the constant), without the k0_const rows when ``subtract_k0_const`` is set.
        ``kspace`` is the pair of ``_kspace_tables``."""
        kspace_rows, kspace_slopes = kspace
        terms = []
        for component, row, factor, interpolated, k0_const in _KSPACE_ROWS:
            if k0_const and self.subtract_k0_const:
                continue
            value = (spline.hermite(logk, self._logk_fft, kspace_rows[row], kspace_slopes[row]) if interpolated
                     else kspace_rows[row])
            terms.append((component, factor * value))
        return terms

    def _point_kernel(self, k, derivative=False, weighted=False):
        """The kernels of the final transform at ``k``, ``(kern (lmax + 1, nq), d kern / d ln k or None)``, both times
        4 pi q^3 if ``weighted`` (the kernel KW of the tangent rule ``lpt_rule``).

        kern_L(k)[j] = k^{-nu} q_j^{-nu} w_j beta_L(n + theta + j + 2p) (``_FinalTransform``), the sequence beta_L
        interpolated by the cubic Hermite at the index position n + theta of k on the core grid,
        n + theta = (ln k - ln k_0) / dlnk with n clipped to the core intervals 0 .. nq - 2 (as spline.hermite clips).
        At the core nodes the kernels are the FFTlog transforms of the unit sources.  The derivative is the Hermite
        weights' derivative over dlnk minus nu times the kernels.

        The forward reads the window by four slices of length nq (y0, m0, y1, m1) and the tangent rule by two of
        length nq + 1 (beta and its slopes, the nodes being their offset sub-slices).  The values are the same; the
        forms are XLA's best for each (in the forward XLA copies the offset sub-slices, 1.1x the time for four chains
        under vmap, and in the rule four slices are four gathers, 11 MiB more scratch memory).
        """
        ft = self._final_transform
        nq = ft.q_factor.shape[0]
        position = (jnp.log(k) - ft.log_k0) / ft.dlnk
        n = jnp.clip(jnp.floor(position), 0, nq - 2)
        theta, start = position - n, n.astype(jnp.int32)
        if weighted:
            seq = jax.lax.dynamic_slice_in_dim(ft.beta, start, nq + 1, axis=1)
            slopes = jax.lax.dynamic_slice_in_dim(ft.beta_slopes, start, nq + 1, axis=1)
            nodes = (seq[:, :-1], slopes[:, :-1], seq[:, 1:], slopes[:, 1:])
        else:
            cut = lambda a, s: jax.lax.dynamic_slice_in_dim(a, s, nq, axis=1)
            nodes = (cut(ft.beta, start), cut(ft.beta_slopes, start), cut(ft.beta, start + 1),
                     cut(ft.beta_slopes, start + 1))
        scale = k**(-self.fftlog_settings.nu) * (ft.four_pi_q3 * ft.q_factor if weighted else ft.q_factor)
        hw = spline.hermite_weights(theta, 1.0)
        kern = scale * (hw[0] * nodes[0] + hw[1] * nodes[1] + hw[2] * nodes[2] + hw[3] * nodes[3])
        if not derivative:
            return kern, None
        dhw = spline.hermite_weight_derivatives(theta, 1.0)
        d_kern = scale * (dhw[0] * nodes[0] + dhw[1] * nodes[1] + dhw[2] * nodes[2] + dhw[3] * nodes[3]) / ft.dlnk
        return kern, d_kern - self.fftlog_settings.nu * kern

    def _point_value(self, k_i, mu_j, G_rows, sin_j, p_i, W_ij, f, corrs_q, kspace):
        """Output ``(n_out,)`` of the final integral at one (k, mu) (the body of the point loop).

        ``G_rows`` is the angular matrix of this mu, ``p_i`` P_lin at ``k_i``, ``W_ij`` the component weights
        ``(n_out, 23)`` of the point, ``corrs_q`` the pair (tree rows, basis rows R_t) and ``kspace`` the pair of
        ``_kspace_tables``.  It calls ``_integrands`` and subtracts from the L = 0 integrand its value at q_max (a
        constant in q transforms to delta_D(k)).  It contracts 4 pi q^3 g_L(q) with the kernels of ``_point_kernel``
        at k, summing over L and q.  It then adds ``lin * P_lin(k)`` and the ``_KSPACE_ROWS`` rows times their
        component weights.
        """
        ft = self._final_transform
        # the geometry once per (k, mu), shared by the integrand and the linear rows
        geometry = _rsd_geometry(f, mu_j, sin_j)
        logk = jnp.log(k_i)
        g, lin = self._integrands(k_i, mu_j, f, G_rows, geometry, corrs_q, W_ij)
        kernel = self._point_kernel(k_i)[0]
        out = []
        for g_o in g:
            g_o = ft.four_pi_q3 * g_o.at[0, :].subtract(g_o[0, -1])
            out.append(jnp.sum(jnp.sum(g_o * kernel, axis=-1), axis=-1))
        out = jnp.stack(out) + lin * p_i
        for component, value in self._kspace_terms(logk, kspace):
            out = out + W_ij[:, component] * value
        return out

    def _point_loop(self, W_fn, k_true, mu_true, sin_true, pk_lin_true, rows, f, W_args):
        """The point loop ``(nk, nmu, n_out)``: ``_point_value`` at every (k, mu).

        ``k_true``, ``pk_lin_true`` ``(nk, nmu)``; ``mu_true``, ``sin_true`` ``(nmu,)``; ``rows``
        ``PTTerms.rows``.  The weights are ``W_fn(k_i, mu_j, *W_args)`` ``(n_out, 23)`` at each point
        (a callable W, closure-converted) or, for ``W_fn=None``, the array ``W_args[0]``.
        """
        corrs_q = (rows[TREE_ROWS], row_basis(rows))       # the basis rows R_t once per call, outside the loop
        kspace = self._kspace_tables(rows)
        # the angular matrix G (n_rows, nm) once per mu
        G_of = lambda mu_j, sin_j: angular.angular_matrix(self._angular, _rsd_geometry(f, mu_j, sin_j))
        G_all = jax.vmap(G_of)(mu_true, sin_true)
        mu_b = jnp.broadcast_to(mu_true, k_true.shape)

        def per_mu(k_i, mu_j, G_rows, sin_j, p_i):
            W_ij = W_fn(k_i, mu_j, *W_args) if W_fn is not None else W_args[0]
            return self._point_value(k_i, mu_j, G_rows, sin_j, p_i, W_ij, f, corrs_q, kspace)

        def per_k(k_row, mu_row, p_row):
            return jax.vmap(per_mu)(k_row, mu_row, G_all, sin_true, p_row)

        return jax.vmap(per_k)(k_true, mu_b, pk_lin_true)

    def _final_integral(self, k_true, mu_true, sin_true, rows, f, pk_lin_true, W):
        """Final (k, mu) integral of the outputs with component weights ``W``, ``(n_out, nk, nmu)``.

        ``W`` is an ``(n_out, 23)`` array or a function ``(k_i, mu_j) -> W``.  ``mu_true`` is
        the ``|mu|`` that ``base`` folds (the result is even in mu).  ``sin_true`` is passed
        through so its AD-regular value, not ``sqrt(1 - mu^2)``, reaches the angular matrix.
        ``pk_lin_true`` ``(nk, nmu)`` is P_lin at ``k_true``.

        ``_point_value`` is evaluated at each (k, mu).  W = identity gives the components,
        ``_TEMPLATE_MATRIX`` the templates, and the function ``component_weights`` of
        ``_pkmu_true`` the spectrum. ``rows`` is ``PTTerms.rows``.

        With ``tangent_rule`` (the constructor's) the point loop is a ``jax.custom_jvp``
        (``_point_loop``, rule ``lpt_rule.point_loop_jvp``), otherwise JAX differentiates it
        directly.  A callable W is closure-converted, so the values it closes over (biases,
        counterterms, f) are arguments of the custom_jvp, and the primal evaluates it at every
        point.
        """
        if callable(W):
            W_fn, W_args = jax.closure_convert(W, k_true[0, 0], mu_true[0])
            W_args = tuple(W_args)
        else:
            W_fn, W_args = None, (jnp.asarray(W),)
        if self.tangent_rule:
            loop = jax.custom_jvp(partial(self._point_loop, W_fn))
            loop.defjvp(partial(lpt_rule.point_loop_jvp, self, W_fn), symbolic_zeros=True)
        else:
            loop = partial(self._point_loop, W_fn)
        out = loop(k_true, mu_true, sin_true, pk_lin_true, rows, f, W_args)   # (nk, nmu, n_out)
        return jnp.moveaxis(out, -1, 0)

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

    def _pkmu_true(self, k_true, mu_true, sin_true, pt_terms, linear, params):
        """Tree level plus one loop at the true (k, mu), the Zel'dovich counterterms folded into the weights.

        The component weights at each (k, mu) are the 13 template weights times
        ``_TEMPLATE_MATRIX``. The template weights are the 12 bias monomials of
        ``_get_lpt_bias_factors`` and ``Counterterms.zeldovich_weight`` (leading plus NLO
        shape for 'zeldovich', zero for the k-space bases, which ``base`` adds).  ``linear``
        is the tuple of ``_linear_at``, whose P_lin enters the linear part.
        """
        self._require_pt_terms_rows(pt_terms)
        f = params.f
        bias_facs = self._get_lpt_bias_factors(params.bias_a, params.bias_b)
        M = jnp.asarray(_TEMPLATE_MATRIX)

        def component_weights(k_i, mu_j):
            ctr_weight = self._counterterms.zeldovich_weight(k_i, mu_j, f, params.ctr)
            row = jnp.concatenate([bias_facs, jnp.stack([ctr_weight])])
            return (row @ M)[None, :]

        return self._final_integral(k_true, mu_true, sin_true, pt_terms.rows, f, linear[0],
                                    component_weights)[0]

    @partial(jit, static_argnames=['self'])
    def get_pkmu_components(self, k, mu, pk_data, f):
        """The 23 components of the final integrand on the (k, mu) grid, ``(23, nk, nmu)``.

        Diagnostic without AP, in the order of :class:`Component` (``_COMPONENTS``).
        ``_final_integral`` with W = identity, so no bias or counterterm weights are applied.
        ``pk_data`` is the ``(2, n)`` input table.
        """
        pt_terms = self.get_pt_terms(pk_data, None)
        k_true, mu_true, sin_true = self._true_coordinates(k, mu, 1.0, 1.0)
        pk_lin_true = self._pk_true(self._prepare(pk_data), k_true)
        return self._final_integral(k_true, mu_true, sin_true, pt_terms.rows, f, pk_lin_true,
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
        ``pk_data`` (the ``(2, n)`` input table) must be the spectrum ``pt_terms`` was computed
        from.

        ``_final_integral`` with W = ``_TEMPLATE_MATRIX``.
        """
        self._require_pt_terms_rows(pt_terms)
        k_true, mu_true, sin_true = self._true_coordinates(k, mu, alpha_perp, alpha_para)
        pk_lin_true = self._pk_true(self._prepare(pk_data), k_true)
        return self._final_integral(k_true, mu_true, sin_true, pt_terms.rows, f, pk_lin_true,
                                    jnp.asarray(_TEMPLATE_MATRIX))

    def get_pkmu_templates(self, k, mu, pk_data, f, alpha_perp=1.0, alpha_para=1.0):
        """Templates from ``pk_data``, the ``(2, n)`` input table (see
        :meth:`get_pkmu_templates_from_pt_terms`)."""
        pt_terms = self.get_pt_terms(pk_data, None)
        return self.get_pkmu_templates_from_pt_terms(k, mu, pt_terms, pk_data, f, alpha_perp, alpha_para)
