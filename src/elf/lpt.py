"""LPT backend: the one-loop LPT galaxy power spectrum P(k, mu) in redshift space.

The model is Eqs. (pkmu_lpt_tree) and (pkmu_lpt_1loop) of the paper, with the
correlators (cumulants) of its appendix "Explicit formulas for LPT power spectrum".
Below, H_l^n[S](q) = int dk k^{n+2} S(k) j_l(kq) / (2 pi^2) and xi_l^n = H_l^n[P]
(the paper's xi^l_n).

The chain.  tools/pt_autogen derives the integrand from the generating function and writes
the plain-text tables ``tables/lpt/*.dat``, which ``lpt_tables`` loads and checks.  The
runtime reads them in three places.

Stage 1, ``get_pt_terms(pk_data, h)`` -> :class:`PTTerms`, depends only on P_lin and h.
Its ``rows`` (:class:`Row`) are 28 correlators on the q grid and five k-space rows, made
by ``lpt_stages.pt_rows`` from the recipes of ``rows_recipes.dat`` (and the units and
sources of ``rows_sources.dat``, the rules of ``rows_rules.dat``):
  A. ``_get_xi_ln``: P(k) -> xi_l^n(q) for the (l, n) of ``ln_list``, for P and
     P_lt = P exp(-(k/k_IR)^2).
  B. the stage-B units, products of xi_l^n (q -> k) and transforms of xi_l^n times P,
     and the sources Q_1, Q_2, R_1, R_2 (k) made of them.
  C. ``_transform_sources``: combinations S(k) of the sources -> H_l^n[S](q) (k -> q).
The other rows are products of xi_l^n, zero-lag constants (grid moments) and references
to other rows, and the k-space rows come from the rules of rows_rules.dat.

Stage 2, for each (k, mu), ``_pkmu_true`` -> ``_final_integral`` does the final q -> k
integral. ``_integrands`` combines the rows with the angular moments of
``utils_lpt.get_lpt_moments`` into one integrand per order l of j_l(kq), minus its
linear part. ``_final_integral`` contracts it with the transform rows of
``_FinalTransform`` interpolated at k. It then adds back the linear part (Kaiser
factor times P_lin(k)) and the k-space rows. ``base.PowerSpectrum`` adds the
stochastic term, the k-space counterterms and the AP volume factor.  The derivatives of
the point loop come from the tangent rule ``lpt_rule`` (``USE_TANGENT_RULE``), which
reads the component table (``basis_rows.dat``, ``component_entries.dat``) and the psi
family of the moments (``psi_terms.dat``, ``scalar_layer.dat``).

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

from . import fftlog
from . import lpt_rule
from . import lpt_stages
from . import spline

from .base import PowerSpectrum
# The definitions of the final integrand, imported by name (lpt.Row, lpt.Component, ... keep working): the tables
# (lpt_tables) and the hand-written geometry and moment indices (lpt_defs).
from .lpt_tables import (_LPT_BIAS_DEGREES, Template, LinearKind, Component, _COMPONENTS, N_COMPONENTS,
                         _TEMPLATE_MATRIX, _LINEAR_MATRIX, Row, TREE_ROWS, _KSPACE_ROWS, COMPONENT_ENTRIES,
                         entry_scalars, row_basis)
from .lpt_defs import _RSDGeometry, _rsd_geometry
from .utils import cross_bias_factor
from .utils_lpt import (
    make_G00_coeffs,
    get_lpt_moments,
    compute_V_mu,
)

USE_TANGENT_RULE = True
"""For tests and diagnosis.  With True (the default) the point loop of ``LPT._final_integral`` is a ``jax.custom_jvp``
differentiated by the tangent rule ``lpt_rule.point_loop_jvp``, with False JAX differentiates the same loop directly
(the same values, slow gradients).  Read at trace time, so a change applies to functions traced afterwards."""

N_ROWS = len(Row)


def _components_from_table(s, R):
    """The components as ``{moment index: (nq,) coefficient}`` dicts, in Component order, from the component table.

    c_j^c(q) = sum over the entries e = (c, j, t) of s_e R_t(q) (tables/lpt/component_entries.dat), with ``s`` the
    per-point factors of ``lpt_tables.entry_scalars`` and ``R`` the basis rows of ``lpt_tables.row_basis``, summed per
    component and moment in table order.
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
    e.g. U_i = qhat_i U and A_ij = X delta_ij + Y qhat_i qhat_j. The rows are evaluated from
    the recipes of tables/lpt/rows_recipes.dat (``lpt_stages``), ``TREE_ROWS`` being the rows
    linear in P_lin.

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

    Stage A data, one slot per (l, n) of ``LPT.ln_list``: ``slots[(l, n)]`` is the row of
    ``forward`` and ``log_q_rows`` and the index of xi_l^n in the outputs of ``_get_xi_ln``
    (``LPT._xi_slot``).  The orders are stored by slot, not at the index [l, n] of a fixed
    array, so any l >= 0 and integer n have their own place.  Every row has the kernel bias
    ``nu`` (the kernel ``HankelGrids.u_m[l]``) and the input bias ``nu - (n + 3)``, and lands
    on ``HankelGrids.q_padded[l]``.
    """
    forward: fftlog.ForwardTransform  # one row per slot (nu_kernel = nu)
    log_q_rows: jnp.ndarray         # (n_pairs, n_padded) ln of each row's output grid q_padded[l]
    slots: dict                     # (l, n) -> slot


class _SourceTransform(NamedTuple):
    """The Q/R sources (backward, q -> k) and their k -> q transforms to the correlator rows.

    Stage B (``lpt_stages.stage_b``) and stage C (``LPT._transform_sources``) data. The
    order index l runs over 0 ... max(``lpt_stages.MAX_ORDER``, ``LPT.lmax``), the orders of
    every stage and of the final integral. ``u_m`` is ``(n_l, n_padded // 2 + 1)``,
    and ``u_m[0]`` serves the Q sources and the b2^2 row. The end weights live on
    ``k_padded``. ``u_m_src`` and ``q_src_padded`` are indexed ``[group, l]``. Group 0
    has bias ``nu`` and group 1 has ``nu_n0`` (sources with n >= 0). Each kernel uses the
    low-ringing phase of its own order.
    """
    u_m: jnp.ndarray                # backward kernels onto the common grid (l = 0 phase)
    four_pi_q3_padded: jnp.ndarray  # 4 pi q^3 on _q_padded (with d ln q it gives 4 pi q^2 dq)
    end_weights_Q: jnp.ndarray      # support [k_min, 2 k_max]
    end_weights_R: jnp.ndarray      # support [k_min, k_max]
    R_source_rows: jnp.ndarray      # stage-A slots of the 13 units of stage B, (l, n) = lpt_stages.UNITS_13_LN
    R_source_ells: jnp.ndarray      # their orders l
    u_m_src: jnp.ndarray            # k -> q kernels [bias group (nu, nu_n0), l]
    q_src_padded: jnp.ndarray       # (2, n_l, n_padded) output q grids of u_m_src
    log_q_src_padded: jnp.ndarray   # ln of q_src_padded


class _FinalTransform(NamedTuple):
    """The final q -> k transform of the (k, mu) integrand (``LPT._final_integral``).

    q runs over the core q grid ``LPT._q`` (nq = nfft), k over the core k grid ``LPT._k``
    and l over 0 ... ``LPT.lmax``. Every order uses the kernel with the l = 0 phase, so
    the integrand stays on the common q grid.  ``pk_select_kernel`` and ``pk_select_slopes``
    are the one stored form of the kernel (8 L nfft^2 bytes each): the forward
    (``_point_value``, spline.hermite) and the tangent rule (``lpt_rule.point_kernel``) read
    them in place, no copy is kept.
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

        The attributes ``ln_list`` (the (l, n) of the forward transforms xi_l^n, in slot
        order) and ``G00_coeffs`` (the angular coefficients of ``utils_lpt.make_G00_coeffs``)
        are internal tables.
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

        # the kernels reach the highest order of every stage (lpt_stages.MAX_ORDER) and of the final integral
        self._set_hankel(max(lpt_stages.MAX_ORDER, self.lmax))

    def _set_hankel(self, lmax):
        """Grids, kernels and static support weights of the four transforms.

        ``self._hankel`` (:class:`fftlog.HankelGrids`) holds the padded k grid and the
        per-order q grids. The stage data go to ``_xi_transform`` (A), ``_source_transform``
        (B and C) and ``_final_transform`` (the final integral). The common q grid
        ``_q_padded`` is the l = 0 output grid, and ``_q`` is its core. ``lmax`` is the
        highest order needed, that of every stage (``lpt_stages.MAX_ORDER``: stage A, the 13
        units of stage B, the H terms of stage C) and of the final integral (``self.lmax``).
        Stage A is planned for the orders ``lpt_stages.STAGE_A`` (``_set_xi_transform``).
        """
        l_list = list(range(lmax + 1))

        self._hankel = fftlog.HankelGrids(self._k, self.fftlog_settings, self.pad_mode, lmax)
        self._set_xi_ells_transform()
        dln = self._hankel.grid.dln

        self._q_padded = self._hankel.q_padded[0]
        self._q = self._hankel.grid.crop(self._q_padded)
        self._log_q_padded = jnp.log(self._q_padded)

        self._set_xi_transform(lpt_stages.STAGE_A)

        u_m = jnp.array([
            fftlog.hankel_setup(l, self.fftlog_settings.nu, self._hankel.grid, lnxy=self._hankel.lnxy[0])[2] for l in l_list
        ])
        # the 13 units of stage B stay on their own grid q_padded[l]: the conjugate relation lands them on k_padded
        r_ln = lpt_stages.UNITS_13_LN
        k_ref = _np.asarray(self._hankel.k_padded)
        for l, _n in r_ln:
            k_back = _np.asarray(self._hankel.q_grids[l].conjugate(self._hankel.lnxy[l]).x)
            assert _np.allclose(k_back, k_ref, rtol=1e-12), (
                f'R-source backward output grid for ell={l} is not the common '
                'k grid; the conjugate relation of the low-ringing phase is '
                'broken')
        setups_n0 = [fftlog.hankel_setup(l, self.fftlog_settings.nu_n0, self._hankel.grid) for l in l_list]
        q_src_padded = jnp.stack([self._hankel.q_padded, jnp.array([s[1].x for s in setups_n0])])
        self._source_transform = _SourceTransform(
            u_m=u_m,
            four_pi_q3_padded=4.0 * jnp.pi * self._q_padded**3,
            end_weights_Q=fftlog.trapezoid_weights('core_to_2kmax', self._nfft, self._hankel.npad, dln),
            end_weights_R=fftlog.trapezoid_weights('core', self._nfft, self._hankel.npad, dln),
            R_source_rows=jnp.asarray([self._xi_slot(l, n) for l, n in r_ln], dtype=jnp.int32),
            R_source_ells=jnp.asarray([l for l, _ in r_ln], dtype=jnp.int32),
            u_m_src=jnp.stack([self._hankel.u_m, jnp.array([s[2] for s in setups_n0])]),
            q_src_padded=q_src_padded,
            log_q_src_padded=jnp.log(q_src_padded),
        )

        end_weights_q_core = fftlog.trapezoid_weights('core', self._nfft, 0, dln)
        basis = fftlog.pad(jnp.eye(self._q.shape[0], dtype=self._q.dtype), self._hankel.q_grids[0], 'zero-pad')

        def build_one(u_m_l):
            pk = fftlog.hankel(self.fftlog_settings.nu, basis, self._q_padded, self._hankel.k_padded, u_m_l, self._hankel.npad)
            return pk.T

        ell_indices = jnp.arange(self.lmax + 1, dtype=self._q.dtype)
        pk_select_kernel = jax.vmap(build_one)(u_m[:self.lmax + 1]) * end_weights_q_core
        pk_select_slopes = spline.slopes(self._logk_fft, pk_select_kernel, axis=1)
        self._final_transform = _FinalTransform(
            pk_select_kernel=pk_select_kernel,
            pk_select_slopes=pk_select_slopes,
            end_weights_q_core=end_weights_q_core,
            four_pi_q3=4.0 * jnp.pi * self._q**3,
            ell_indices=ell_indices,
            q_inv_pows=self._q[None, :] ** (-ell_indices[:, None]),
        )

    def _set_xi_transform(self, ln_list):
        """Plan stage A for the orders ``ln_list`` ((l, n) pairs, each once): ``ln_list`` and ``_xi_transform``.

        Slot i holds the order ``ln_list[i]`` (``_xi_slot``), whatever its l and n.
        """
        pairs = [(int(l), int(n)) for l, n in ln_list]
        if len(set(pairs)) != len(pairs):
            raise ValueError(f'stage A lists an order twice: {pairs}')
        self.ln_list = jnp.array(pairs)
        forward = fftlog.forward_transform(self._hankel.grid, self.fftlog_settings.nu, self.ln_list[:, 0],
                                           self.ln_list[:, 1], nu_kernel=self.fftlog_settings.nu)
        self._xi_transform = _XiTransform(forward=forward, log_q_rows=jnp.log(forward.q),
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
        transformed for the (l, n) of ``ln_list``.  Both outputs have the shape
        ``(n_arrays, n_pairs, n_padded)`` and hold xi_l^n in the slot ``_xi_slot(l, n)``:
        ``xi_ln_common`` resampled onto ``_q_padded``, ``xi_ln_own_grid`` on ``q_padded[l]``.
        Both carry the plain mask, not ``w_xi``: the pointwise rows would be wrong with a
        halved q_max node.
        """
        arrays = jnp.atleast_2d(jnp.asarray(arrays))
        fx = self._hankel.pad_input(arrays / (2 * jnp.pi**2))              # (n_arrays, n_padded)
        raw = self._xi_transform.forward.apply(fx[:, None, :], self._hankel.grid)   # (n_arrays, n_pairs, n_padded)

        log_q_rows = self._xi_transform.log_q_rows
        xis = jax.vmap(lambda row: self._resample_to_common_q_padded(log_q_rows, row))(raw)
        return xis * self._hankel.mask_xi, raw * self._hankel.mask_xi

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
        n_l = self._source_transform.u_m_src.shape[1]
        if ells.max() >= n_l:          # the gathers [group, ells] below would clip it to the last order
            raise ValueError(f'_transform_sources has the kernels of the orders l <= {n_l - 1}, not '
                             f'{sorted({int(l) for l in ells.flat if l >= n_l})}')
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

    @partial(jit, static_argnames=['self'])
    def get_pt_terms(self, pk_data, h):
        """The :class:`PTTerms` of ``pk_data``.

        The rows (:class:`Row`) are the tree rows (``TREE_ROWS``), the one-loop rows on the core q grid and the
        k-space rows, evaluated from the recipes of tables/lpt/rows_recipes.dat by ``lpt_stages.pt_rows`` (stage A
        ``_get_xi_ln`` of P and P_lt, the stage-B units and sources, stage C through ``_transform_sources``, the
        products, constants and rules).  :class:`PTTerms` has the table of the rows.

        ``h`` enters only the no-wiggle data of the 'linear_ir_resum' counterterm base.
        ``h=None`` skips them (the template and component methods do so), and such
        ``pt_terms`` cannot be fed to the spectrum methods under that base.
        """
        rows = lpt_stages.pt_rows(self, self._pk_at(self._k, pk_data))
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

        Hand-written.  Which components have a linear part of which kind, and the tree rows of
        each kind, come from tables/lpt/components.dat (derived by tools/pt_autogen from the cumulant
        expansion, lpt_components), the expansion of those terms in l and their Kaiser factors
        are written here.
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

        c_j^c are the component coefficients of the component table (``_components_from_table``, from
        ``corrs_q = (tree rows, basis rows R_t)``). M_j,
        ``ir_damping`` and ``ell_factors`` are those of ``_angular_integral_factors``. The
        linear rows come from ``_linear_rows``, weighted per kind by ``W @ _LINEAR_MATRIX``.
        ``g`` is a tuple of n_out arrays. ``lin`` has shape ``(n_out,)``. ``V`` and ``geometry``
        are the angular table and the :class:`_RSDGeometry` of this mu.
        """
        corrs_tree, R = corrs_q
        factors = self._angular_integral_factors(k_i, V, geometry, corrs_tree)
        components = _components_from_table(entry_scalars(k_i, mu_j, f, factors.Kfac), R)
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

    def _angular_table(self, f, mu_j, sin_j):
        """The angular table V ``(5, L, L)`` of one mu (``utils_lpt.compute_V_mu``)."""
        return compute_V_mu(_rsd_geometry(f, mu_j, sin_j).s2, self.G00_coeffs, self.lmax)

    def _kspace_tables(self, rows):
        """The ``_KSPACE_ROWS`` rows (interpolated rows whole, constants at k_min) and their ln k slopes."""
        kspace_rows = {row: rows[row] if interpolated else rows[row, 0]
                       for _, row, _, interpolated, _ in _KSPACE_ROWS}
        kspace_slopes = {row: spline.slopes(self._logk_fft, rows[row])
                         for _, row, _, interpolated, _ in _KSPACE_ROWS if interpolated}
        return kspace_rows, kspace_slopes

    def _point_value(self, k_i, mu_j, V, sin_j, p_i, W_ij, f, corrs_q, kspace):
        """Output ``(n_out,)`` of the final integral at one (k, mu) (the body of the point loop)."""
        ft = self._final_transform
        kspace_rows, kspace_slopes = kspace
        # the geometry once per (k, mu), shared by the integral factors and the linear rows
        geometry = _rsd_geometry(f, mu_j, sin_j)
        logk = jnp.log(k_i)
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

    def _point_loop(self, W_fn, k_true, mu_true, sin_true, pk_lin_true, rows, f, W_args):
        """The point loop ``(nk, nmu, n_out)``: ``_point_value`` at every (k, mu).

        ``k_true``, ``pk_lin_true`` ``(nk, nmu)``; ``mu_true``, ``sin_true`` ``(nmu,)``; ``rows``
        ``PTTerms.rows``.  The weights are ``W_fn(k_i, mu_j, *W_args)`` ``(n_out, 23)`` at each point
        (a callable W, closure-converted) or, for ``W_fn=None``, the array ``W_args[0]``.
        """
        corrs_q = (rows[TREE_ROWS], row_basis(rows))       # the basis rows R_t once per call, outside the loop
        kspace = self._kspace_tables(rows)
        V_all = jax.vmap(lambda m, s: self._angular_table(f, m, s))(mu_true, sin_true)   # once per mu
        mu_b = jnp.broadcast_to(mu_true, k_true.shape)

        def per_mu(k_i, mu_j, V, sin_j, p_i):
            W_ij = W_fn(k_i, mu_j, *W_args) if W_fn is not None else W_args[0]
            return self._point_value(k_i, mu_j, V, sin_j, p_i, W_ij, f, corrs_q, kspace)

        def per_k(k_row, mu_row, p_row):
            return jax.vmap(per_mu)(k_row, mu_row, V_all, sin_true, p_row)

        return jax.vmap(per_k)(k_true, mu_b, pk_lin_true)

    def _final_integral(self, k_true, mu_true, sin_true, rows, f, pk_data, W):
        """Final (k, mu) integral of the outputs with component weights ``W``, ``(n_out, nk, nmu)``.

        ``W`` is an ``(n_out, 23)`` array or a function ``(k_i, mu_j) -> W``.  ``mu_true`` is
        the ``|mu|`` that ``base`` folds (the result is even in mu).  ``sin_true`` is passed
        through so its AD-regular value, not ``sqrt(1 - mu^2)``, reaches the moments.

        For each (k, mu) it calls ``_integrands`` and subtracts the l = 0 integrand's value
        at q_max from the l = 0 integrand. It contracts 4 pi q^3 g_l(q) with
        ``pk_select_kernel`` Hermite-interpolated at ln k, summing over l and q. It then
        adds ``lin * P_lin(k)`` and the ``_KSPACE_ROWS`` rows times their component weights.
        W = identity gives the components, ``_TEMPLATE_MATRIX`` the templates, and the
        function ``component_weights`` of ``_pkmu_true`` the spectrum. ``rows`` is ``PTTerms.rows``.

        The point loop is a ``jax.custom_jvp`` (``_point_loop``, rule ``lpt_rule.point_loop_jvp``;
        plain autodiff with ``USE_TANGENT_RULE = False``).  A callable W is closure-converted, so the
        values it closes over (biases, counterterms, f) are arguments of the custom_jvp, and the
        primal evaluates it at every point as before.
        """
        pk_lin_true = self._pk_at(k_true, pk_data)
        if callable(W):
            W_fn, W_args = jax.closure_convert(W, k_true[0, 0], mu_true[0])
            W_args = tuple(W_args)
        else:
            W_fn, W_args = None, (jnp.asarray(W),)
        if USE_TANGENT_RULE:
            loop = jax.custom_jvp(partial(self._point_loop, W_fn))
            loop.defjvp(partial(lpt_rule.point_loop_jvp, self, W_fn), symbolic_zeros=True)
        else:
            loop = partial(self._point_loop, W_fn)
        out = loop(k_true, mu_true, sin_true, pk_lin_true, rows, f, W_args)   # (nk, nmu, n_out)
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
