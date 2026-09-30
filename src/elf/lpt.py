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
from .eft_terms import Counterterms


# Powers of (b1, b2, bG2, bGamma3) of the 12 bias templates (total degree <= 2, as cross_bias_factor needs).
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
MQ0, MQ1, MQ2, MQ3, MQ4, NQ1, NQ2, MQ1NQ1, MQ2NQ1 = range(9)
N_MOMENTS = 9


# The 23 parts of the final integrand (order of LPT._lpt_moment_coefficient_parts): the
# templates each belongs to (0-11 bias monomials, 12 the Zel'dovich counterterm tree) and
# the kind of its linear part.
_PARTS = (
    # name       templates  linear
    ('ZA',       (0,),      'za'),
    ('AA',       (0,),      None),
    ('A22',      (0,),      None),
    ('A13',      (0,),      None),
    ('W112',     (0,),      None),
    ('U10',      (1,),      'b1'),
    ('A_U',      (1,),      None),
    ('A10',      (1,),      None),
    ('XI',       (2,),      'xi'),
    ('A_XI',     (2,),      None),
    ('U_U',      (2, 3),    None),
    ('U11',      (2,),      None),
    ('U20',      (3,),      None),
    ('XI_U',     (4,),      None),
    ('XI_XI',    (5,),      None),
    ('UPS',      (6,),      None),
    ('V10',      (6,),      None),
    ('V12',      (7,),      None),
    ('CHI',      (8,),      None),
    ('ZETA',     (9,),      None),
    ('UB3',      (10,),     None),
    ('THETA',    (11,),     None),
    ('CTR',      (12,),     'za'),
)
N_PARTS = len(_PARTS)
N_TEMPLATES = 13
XI_XI = 14   # the b2^2 part, whose xi_lin^2 / 2 is transformed separately

# M[t, p] = 1 if part p belongs to template t.
_TEMPLATE_MATRIX = _np.array(
    [[1.0 if t in part[1] else 0.0 for part in _PARTS] for t in range(N_TEMPLATES)])

# N[p, kind] = 1 if part p has a linear part of that kind.
_LINEAR_MATRIX = _np.array(
    [[1.0 if part[2] == kind else 0.0 for kind in ('za', 'b1', 'xi')] for part in _PARTS])

# k-space rows of corrs added after the final transform (constants are read at k_min).
_DC_ROWS = (
    # part  row  factor  interpolated  k0_limit
    (18,    28,  1.0,    True,         False),   # CHI:   -int 4 pi q^2 chi dq
    (19,    29,  1.0,    True,         False),   # ZETA:  -int 4 pi q^2 zeta dq
    (14,    30,  1.0,    True,         False),   # XI_XI: H_0[xi^2 / 2](k) - I0
    (14,    31,  1.0,    False,        True),    # XI_XI: I0
    (21,    32,  2.0,    False,        False),   # THETA: D_theta; the part is 2 theta
)
N_CORRS_ROWS = 33


def _rsd_factors(f, mu, sin_mu):
    """``(Kfac, c, s)``: ``|K| / k`` for ``K = k + f (k.n) n`` and the cosine and sine of the angle (K, k)."""
    Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
    return Kfac, (1 + f * mu**2) / Kfac, f * mu * sin_mu / Kfac


def _weighted_coeffs(parts, W):
    """Moment coefficients of the outputs: ``{j: sum_p W[:, p] c_j^p}``, each ``(n_out, nq)``.

    Summed in part order on purpose: the low-k l = 0 integrand amplifies the last bit.
    """
    coeffs = {}
    for p, part in enumerate(parts):
        for j, c_j in part.items():
            term = W[:, p, None] * c_j
            coeffs[j] = coeffs[j] + term if j in coeffs else term
    return coeffs


def _apply_coeffs(coeffs, moments):
    """``sum_j c_j[q] m_j[l, q]``, shape ``(L, nq)``."""
    integrand = None
    for j, c_j in coeffs.items():
        term = c_j[None, :] * moments[j]
        integrand = term if integrand is None else integrand + term
    return integrand


class Corrs(NamedTuple):
    """LPT intermediate quantities; depend only on ``P_lin`` and ``h``.

    ``rows`` ``(33, nfft)``: 28 q-space correlators and five k-space rows;
    ``pk_nw_data``, ``Sigma2``, ``dSigma2``: IR-resummed counterterm base (None
    unless ``counterterm_base='linear_ir_resum'``).
    """

    rows: jnp.ndarray
    pk_nw_data: Optional[jnp.ndarray]
    Sigma2: Optional[jnp.ndarray]
    dSigma2: Optional[jnp.ndarray]


class _XiTransform(NamedTuple):
    """The P -> xi_l^n transform (forward, k -> q; ``LPT._get_xi_ln``)."""
    nu_rows: jnp.ndarray
    log_q_rows: jnp.ndarray


class _SourceTransform(NamedTuple):
    """The Q/R sources (backward, q -> k) and their k -> q transforms to the correlator rows."""
    u_m: jnp.ndarray                # backward kernels onto the common grid (l = 0 phase)
    four_pi_q3_padded: jnp.ndarray
    end_weights_Q: jnp.ndarray      # support [k_min, 2 k_max]
    end_weights_R: jnp.ndarray      # support [k_min, k_max]
    R_source_rows: jnp.ndarray
    R_source_ells: jnp.ndarray
    u_m_src: jnp.ndarray            # k -> q kernels [bias group (nu, nu_n0), l]
    q_src_padded: jnp.ndarray
    log_q_src_padded: jnp.ndarray


class _FinalTransform(NamedTuple):
    """The final q -> k transform of the (k, mu) integrand (``LPT._evaluate``)."""
    pk_select_kernel: jnp.ndarray   # [l, k, j]: transform of the unit source at core node j, end weights folded in
    pk_select_slopes: jnp.ndarray
    end_weights_q_core: jnp.ndarray
    four_pi_q3: jnp.ndarray
    ell_indices: jnp.ndarray
    q_inv_pows: jnp.ndarray         # q^{-l} [l, q]


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
                 subtract_k0_limit=False,
                 lmax=5,
                 k_IR=0.2,
                 bias_basis='bG2',
                 ):
        """One-loop Lagrangian-PT galaxy power spectrum in redshift space.

        ``Params.bias`` is Lagrangian (b1, b2, bG2, bGamma3); shared options as in
        :class:`base.PowerSpectrum`.  ``pad_mode='power-law'`` continues P only
        for P -> xi; the Q/R sources stay cut at 2 k_max / k_max.  ``irres_method``,
        ``r_bao``, ``lambda_ir`` only shape the 'linear_ir_resum' counterterm base.
        ``lmax`` (>= 2): order of the angular expansion of the final integral.
        ``k_IR``: split ``P_lt = P e^{-(k/k_IR)^2}`` inside the LPT body (not
        ``lambda_ir``; overridable per call).  ``bias_basis``: 'bG2' or 'bs2'.
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
                         r_bao=r_bao, lambda_ir=lambda_ir, subtract_k0_limit=subtract_k0_limit)

        self.lmax = lmax
        self.k_IR = k_IR
        self.bias_basis = bias_basis

        self.G00_coeffs = make_G00_coeffs(self.lmax)

        ln_list = [[0, 0], [0, -2], [1, -1], [2, 0], [2, -2], [3, -1], [4, 0]]
        self.ln_list = jnp.array(ln_list)
        self._set_hankel(max(int(jnp.max(self.ln_list[:, 0])), self.lmax))

    def _set_hankel(self, lmax):
        """Grids, kernels and static support weights of the four transforms."""
        fs = self.fftlog_settings
        l_list = list(range(lmax + 1))

        hg = self._hankel = fftlog.HankelGrids(self._k, fs, self.pad_mode, lmax)
        self._set_xi_ells_transform()
        dln = hg.grid.dln

        self._q_padded = hg.q_padded[0]
        self._q = hg.grid.crop(self._q_padded)
        self._log_q_padded = jnp.log(self._q_padded)

        self._xi_transform = _XiTransform(
            nu_rows=fs.nu - (self.ln_list[:, 1] + 3).astype(jnp.float64),
            log_q_rows=jnp.log(hg.q_padded[self.ln_list[:, 0]]),
        )

        u_m = jnp.array([
            fftlog.hankel_setup(l, fs.nu, hg.grid, lnxy=hg.lnxy[0])[2] for l in l_list
        ])
        # R sources stay on their own grid q_padded[l]: the conjugate relation lands them on k_padded
        r_ln = [(0, 0), (2, 0), (4, 0), (1, -1), (3, -1)]
        ln_np = _np.asarray(self.ln_list)
        k_ref = _np.asarray(hg.k_padded)
        for l, _n in r_ln:
            k_back = _np.asarray(hg.q_grids[l].conjugate(hg.lnxy[l]).x)
            assert _np.allclose(k_back, k_ref, rtol=1e-12), (
                f'R-source backward output grid for ell={l} is not the common '
                'k grid; the conjugate relation of the low-ringing phase is '
                'broken')
        setups_n0 = [fftlog.hankel_setup(l, fs.nu_n0, hg.grid) for l in l_list]
        q_src_padded = jnp.stack([hg.q_padded, jnp.array([s[1].x for s in setups_n0])])
        self._source_transform = _SourceTransform(
            u_m=u_m,
            four_pi_q3_padded=4.0 * jnp.pi * self._q_padded**3,
            end_weights_Q=fftlog.trapezoid_weights('core_to_2kmax', self._nfft, hg.npad, dln),
            end_weights_R=fftlog.trapezoid_weights('core', self._nfft, hg.npad, dln),
            R_source_rows=jnp.asarray(
                [int(_np.flatnonzero((ln_np[:, 0] == l) & (ln_np[:, 1] == n))[0])
                 for l, n in r_ln], dtype=jnp.int32),
            R_source_ells=jnp.asarray([l for l, _ in r_ln], dtype=jnp.int32),
            u_m_src=jnp.stack([hg.u_m, jnp.array([s[2] for s in setups_n0])]),
            q_src_padded=q_src_padded,
            log_q_src_padded=jnp.log(q_src_padded),
        )

        end_weights_q_core = fftlog.trapezoid_weights('core', self._nfft, 0, dln)
        basis = fftlog.pad(jnp.eye(self._q.shape[0], dtype=self._q.dtype), hg.q_grids[0], 'zero-pad')

        def build_one(u_m_l):
            pk = fftlog.hankel(fs.nu, basis, self._q_padded, hg.k_padded, u_m_l, hg.npad)
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
    # correlator rows D_l^n (k -> q).  Each transform uses the low-ringing phase of its
    # own order; outputs meet through the conjugate grid relation or by resampling.

    def _resample_to_common_q_padded(self, log_q_rows, rows):
        """Resample rows from their own grids (``log_q_rows``) onto the common padded q grid."""
        log_q = self._log_q_padded
        return jax.vmap(lambda x, row: spline.interp1d(log_q, x, row))(log_q_rows, rows)

    def _get_xi_ln(self, arrays):
        """P -> xi_l^n of ``arrays`` ``(n_arrays, nfft)``: ``(xi_ln_common, xi_ln_own_grid)``.

        ``xi_ln_common[i, l, n]`` is resampled onto ``_q_padded``; ``xi_ln_own_grid``
        (rows of ``ln_list``) stays on ``q_padded[l]``.  Both carry the plain mask,
        not ``w_xi``: the pointwise rows would be wrong with a halved q_max node.
        """
        hg = self._hankel
        arrays = jnp.atleast_2d(jnp.asarray(arrays))
        ells = self.ln_list[:, 0]
        ns = self.ln_list[:, 1]

        fx = hg.pad_input(arrays / (2 * jnp.pi**2))              # (n_arrays, n_padded)
        raw = fftlog.hankel(
            self._xi_transform.nu_rows[:, None], fx[:, None, :], hg.k_padded, None,
            hg.u_m[ells], hg.npad, crop=False,
            y_pow=hg.q_padded[ells] ** (-self.fftlog_settings.nu),
        )                                                          # (n_arrays, n_ln, n_padded)

        log_q_rows = self._xi_transform.log_q_rows
        xis = jax.vmap(lambda row: self._resample_to_common_q_padded(log_q_rows, row))(raw)
        xis = xis * hg.mask_xi
        raw = raw * hg.mask_xi
        xi_ln = jnp.zeros((arrays.shape[0], 5, 5, hg.k_padded.shape[0]))
        return xi_ln.at[:, ells, ns].set(xis), raw

    def _get_Qs_Rs(self, xi_ln_common, xi_ln_own_grid, pk_lin_padded):
        """Q/R sources on the padded k grid: ``(Q1, Q2, Q5)``, ``(R1, R2)``.

        Q (products of different l) from ``xi_ln_common``; R from ``xi_ln_own_grid``
        with the per-l kernel.  Inputs carry the mask only; ``w_xi`` is applied here once
        per transform input (on the product for Q: it is one function cut at q_max).
        """
        hg = self._hankel
        st = self._source_transform
        nu = self.fftlog_settings.nu
        integrands = jnp.stack([
            8/15 * xi_ln_common[0,0]**2 - 16/21 * xi_ln_common[2,0]**2
            + 8/35 * xi_ln_common[4,0]**2,
            xi_ln_common[1,-1]**2 - xi_ln_common[3,-1]**2,
        ], axis=0) * hg.w_xi
        res = fftlog.hankel(
            nu, st.four_pi_q3_padded * integrands, self._q_padded,
            hg.k_padded, st.u_m[0], hg.npad, crop=False,
        )
        res = res * st.end_weights_Q
        Q1 = res[0]
        Q2 = Q1 - 2/5 * hg.k_padded**2 * res[1]
        Q5 = (Q1 + Q2) / 2
        Qs = jnp.stack([Q1, Q2, Q5], axis=0)

        ells = st.R_source_ells
        q_l = hg.q_padded[ells]
        xis = xi_ln_own_grid[st.R_source_rows] * hg.w_xi
        pk_list = fftlog.hankel(
            nu, q_l**2 * xis, q_l,
            hg.k_padded, hg.u_m[ells], hg.npad, crop=False,
        ) * (pk_lin_padded * st.end_weights_R)
        pk_00, pk_20, pk_40, pk_1m1, pk_3m1 = pk_list
        k = hg.k_padded
        R1 = k**2 * (8/15*pk_00 - 16/21*pk_20 + 8/35*pk_40)
        R3 = (k**2 * (2/5*pk_00 - 6/7*pk_20 + 16/35*pk_40)
              + k**3 * (2/5*pk_1m1 - 2/5*pk_3m1))
        R2 = R3 - R1
        Rs = jnp.stack([R1, R2], axis=0)
        return Qs, Rs

    def _transform_sources(self, sources, ells, ns):
        """``D_l^n(q) = int dk k^{n+2} S(k) j_l(kq) / (2 pi^2)`` on the common padded q grid.

        ``ells`` ``(n_src,)`` or ``(n_src, n_l)`` (one rfft per source, one irfft
        per order), or scalars for one ``(N,)`` source; rows with ``n >= 0`` use ``nu_n0``.
        """
        hg = self._hankel
        st = self._source_transform
        fs = self.fftlog_settings
        ells = _np.asarray(ells)
        ns = _np.asarray(ns)
        group = (ns >= 0).astype(int)                 # bias group: 0 nu, 1 nu_n0
        nu = _np.where(group == 1, fs.nu_n0, fs.nu)
        if ns.ndim == 0:
            fx = sources * hg.k_padded ** (jnp.asarray(ns) + 3) / (2 * jnp.pi**2)
        else:
            fx = sources * hg.k_padded[None, :] ** (jnp.asarray(ns)[:, None] + 3) / (2 * jnp.pi**2)
        if ells.ndim == 2:
            fx = fx[:, None, :]
            group = _np.broadcast_to(group[:, None], ells.shape)
            nu = nu[:, None]
        nu = float(nu.flat[0]) if _np.all(nu == nu.flat[0]) else jnp.asarray(nu)[..., None]
        d = fftlog.hankel(
            nu, fx, hg.k_padded, st.q_src_padded[group, ells],
            st.u_m_src[group, ells], hg.npad, crop=False,
        )
        if d.ndim == 1:
            return spline.interp1d(self._log_q_padded, st.log_q_src_padded[group, ells], d)
        rows = self._resample_to_common_q_padded(
            st.log_q_src_padded[group, ells].reshape(-1, d.shape[-1]),
            d.reshape(-1, d.shape[-1]))
        return rows.reshape(d.shape)

    def _get_corrs_tree(self, xi_ln, xi_ln_lt, pk_int, pk_int_lt):
        """Tree rows (corrs 0-7): ``X = (2/3)(pk_int - xi_0^{-2} - xi_2^{-2})``, ``Y = 2 xi_2^{-2}``.

        Y as ``(2q/5)(xi_1^{-1} + xi_3^{-1})`` (n = -2 is inaccurate at small q); X keeps
        ``xi_2^{-2}``, whose error cancels that of ``xi_0^{-2}``.
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
        """Matter one-loop rows X22, Y22, X13, Y13, V1, V3, T; returns ``(corrs, M22)``.

        Zero-lag constant ``M = int dk S / (2 pi^2)``: the grid moment of the padded,
        end-weighted source.  zero-pad identities: ``Y22 = (18/49) Y_lin^2 / q^2``,
        ``M13 = (70/27) M22`` (power-law: Y22 transformed, M13 from its own source).
        n = -3 rows from n = -2 pairs:
        ``T = (q/7)(H_2^{-2} + H_4^{-2})[W_T]``, ``H_1^{-3}[W] = (q/3)(H_0^{-2} + H_2^{-2})[W]``,
        ``V_i = H_1^{-3}[W_i] - T/5``.
        """
        crop = self._hankel.grid.crop
        Q1, Q2 = Qs[0], Qs[1]
        R1, R2 = Rs[0], Rs[1]
        q = self._q

        source_22 = 9/98 * Q1
        source_13 = 5/21 * R1
        W_T = -3/7 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2)
        W_1 = 3/35 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2)
        W_3 = -3/35 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2)

        # a batch of one: the unbatched call is slower
        D_22 = self._transform_sources(source_22[None, :], [0], [-2])[0]
        D_pairs_padded = self._transform_sources(
            jnp.stack([source_13, W_T, W_1, W_3], axis=0),
            [[0, 2], [2, 4], [0, 2], [0, 2]],
            [-2, -2, -2, -2],
        )
        D_pairs = crop(D_pairs_padded)
        moment = self._hankel.grid.moment
        M22 = moment(source_22, 1) / (2 * jnp.pi**2)
        if self.pad_mode == 'power-law':
            M13 = moment(source_13, 1) / (2 * jnp.pi**2)
        else:
            M13 = 70/27 * M22
        xi_ln_22_0m2 = crop(D_22)
        (xi_ln_13_0m2, xi_ln_13_2m2), (H2_WT, H4_WT), (H0_W1, H2_W1), (H0_W3, H2_W3) = D_pairs

        if self.pad_mode == 'power-law':
            Y22 = 2 * crop(self._transform_sources(source_22, 2, -2))
        else:
            Y22 = 18/49 * Y_lin**2 / q**2
        X22 = 2/3 * (M22 - xi_ln_22_0m2 - 0.5 * Y22)
        X13 = 2/3 * (M13 - xi_ln_13_0m2 - xi_ln_13_2m2)
        Y13 = 2 * xi_ln_13_2m2

        T = q / 7 * (H2_WT + H4_WT)
        H1m3_W1 = q / 3 * (H0_W1 + H2_W1)
        H1m3_W3 = q / 3 * (H0_W3 + H2_W3)
        V1 = H1m3_W1 - 0.2 * T
        V3 = H1m3_W3 - 0.2 * T

        corrs = jnp.stack([X22, Y22, X13, Y13, V1, V3, T], axis=0)
        return corrs, M22

    def _get_corrs_bias(self, Qs, Rs, xi_ln, M22):
        """Bias rows (corrs 15-27) from the Q/R sources and the core xi output ``xi_ln``.

        zero-pad identities: ``M10 = (28/9) M22``, ``V10 = (24/35)((xi_1^{-1})^2 - (xi_3^{-1})^2)/q``
        (power-law: M10 from its own source, V10 transformed); ``U20 = -(6/7) U_lin^2 / q``;
        ``Ub3 = -(24/5) U3``, ``theta = H_0^0[-(8/7) R1]`` (``F_G2 = -(20/7) R1``).
        """
        Q5 = Qs[2]
        R1, R2 = Rs[0], Rs[1]
        q = self._q

        source_10a = 2/7 * (Q5 + 2 * R2)
        source_10b = 1/7 * (2 * Q5 + 3 * R1 + 4 * R2)
        crop = self._hankel.grid.crop
        xi_bias_padded = self._transform_sources(
            jnp.stack([
                -5/21 * R1,
                -6/7 * (R1 + R2),
                source_10a,
                source_10b,
            ], axis=0),
            [1, 1, 0, 2],
            [-1, -1, -2, -2],
        )
        xi_bias = crop(xi_bias_padded)
        # separate call: batching theta with the rows above changes their FFT rounding
        theta = crop(self._transform_sources(-8/7 * R1, 0, 0))
        if self.pad_mode == 'power-law':
            M10 = self._hankel.grid.moment(source_10a, 1) / (2 * jnp.pi**2)
        else:
            M10 = 28/9 * M22

        U3, U11 = xi_bias[0], xi_bias[1]
        xi_ln_A10_0m2, xi_ln_A10_2m2 = xi_bias[2], xi_bias[3]
        X10 = M10 - xi_ln_A10_0m2 - xi_ln_A10_2m2
        Y10 = 3 * xi_ln_A10_2m2

        U_lin = -xi_ln[1, -1]
        U20 = -(6.0/7.0) * U_lin**2 / q

        J2 = 2/15 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
        J3 = -1/5 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
        X_Upsilon = 4 * J3**2
        Y_Upsilon = 12 * J2**2 - 4 * J3**2 - (4.0/3.0) * U_lin**2

        if self.pad_mode == 'power-law':
            V10 = crop(self._transform_sources(3/7 * Qs[0], 1, -1))
        else:
            V10 = 24/35 * (xi_ln[1,-1]**2 - xi_ln[3,-1]**2) / q
        d_q_delta = (
            2.0 * xi_ln[3,-1] * (3.0*xi_ln[3,-1]/q - xi_ln[4,0])
            - 2.0 * xi_ln[1,-1] * (xi_ln[1,-1]/q - xi_ln[2,0])
        )
        V12 = (14.0/3.0) * V10 - (2.0/5.0) * d_q_delta
        chi = 4/3 * (xi_ln[2,0]**2 - xi_ln[0,0]**2)
        zeta = 2 * (8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2)

        Ub3 = -24/5 * U3

        return jnp.stack([U3, U11, U20, X10, Y10, V10, V12, X_Upsilon, Y_Upsilon,
                          chi, zeta, Ub3, theta], axis=0)

    def _get_dc_rows(self, chi, zeta, U3, xi_lin_padded, pk_lin):
        """k-space rows 28-32: chi/zeta DC ``-4 pi int q^2 corr dq`` (core trapezoid),
        b2^2 residual ``H_0[xi^2/2](k) - I0`` and ``I0 = pk_int2 / 2``, and
        ``D_theta = (96 pi / 5) q_min^2 U3(q_min)`` (the q < q_min ball of theta).
        """
        hg = self._hankel
        st = self._source_transform
        shape = self._k.shape
        q_core = fftlog.LogGrid(self._q)
        w_core = self._final_transform.end_weights_q_core

        def dc_correction(corr):
            return jnp.broadcast_to(-4 * jnp.pi * q_core.moment(w_core * corr, 3), shape)

        source = 0.5 * xi_lin_padded**2
        b2sq_direct = fftlog.hankel(
            self.fftlog_settings.nu,
            st.four_pi_q3_padded * source * hg.w_xi,
            self._q_padded, hg.k_padded, st.u_m[0], hg.npad, crop=True,
        )
        # the end weight multiplies P^2 once
        b2sq_dc = 0.5 * hg.grid.moment(
            hg.w_input * fftlog.pad(pk_lin, hg.grid, hg.pad_mode)**2, 3) / (2 * jnp.pi**2)
        theta_dc = (96 * jnp.pi / 5) * self._q[0]**2 * U3[0]
        return jnp.stack([
            dc_correction(chi),
            dc_correction(zeta),
            b2sq_direct - b2sq_dc,
            jnp.broadcast_to(b2sq_dc, shape),
            jnp.broadcast_to(theta_dc, shape),
        ], axis=0)

    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data, h, k_IR=None):
        """The :class:`Corrs` of ``pk_data`` (``k_IR=None``: the constructor's value).

        Rows: 0-7 tree (X/Y_lin, X/Y_lin_lt, X/Y_lin_gt, xi_lin, U_lin), 8-14 matter
        (X22, Y22, X13, Y13, V1, V3, T), 15-27 bias (U3, U11, U20, X10, Y10, V10, V12,
        X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta) on the core q grid; 28-32 k-space
        rows of :meth:`_get_dc_rows`.  ``h`` enters only the counterterm base.
        """
        k_IR = self.k_IR if k_IR is None else k_IR
        hg = self._hankel
        pk_lin = self._pk_at(self._k, pk_data)
        pk_lin_lt = pk_lin * jnp.exp(-(self._k / k_IR)**2)

        pk_int = hg.grid.moment(hg.pad_input(pk_lin), 1) / (2 * jnp.pi**2)
        pk_int_lt = hg.grid.moment(hg.pad_input(pk_lin_lt), 1) / (2 * jnp.pi**2)

        xi_ln_padded, xi_raw_padded = self._get_xi_ln(
            jnp.stack([pk_lin, pk_lin_lt], axis=0))
        xi_ln = hg.grid.crop(xi_ln_padded[0])
        xi_ln_lt = hg.grid.crop(xi_ln_padded[1])

        corrs_tree = self._get_corrs_tree(xi_ln, xi_ln_lt, pk_int, pk_int_lt)

        Qs, Rs = self._get_Qs_Rs(xi_ln_padded[0], xi_raw_padded[0],
                                 fftlog.pad(pk_lin, hg.grid, 'zero-pad'))
        corrs_matter_1loop, M22 = self._get_corrs_matter_1loop(Qs, Rs, corrs_tree[1])
        corrs_bias = self._get_corrs_bias(Qs, Rs, xi_ln, M22)

        # padded xi_0^0, not corrs_tree[6]: the lower padded band gives the q < q_min part
        dc_rows = self._get_dc_rows(corrs_bias[9], corrs_bias[10], corrs_bias[0],
                                    xi_ln_padded[0][0, 0], pk_lin)
        rows = jnp.concatenate([corrs_tree, corrs_matter_1loop, corrs_bias, dc_rows], axis=0)

        if self.counterterm_base == 'linear_ir_resum':
            pk_nw_data, Sigma2, dSigma2 = self._get_ir_data(pk_data, h)
        else:
            pk_nw_data = Sigma2 = dSigma2 = None
        return Corrs(rows=rows, pk_nw_data=pk_nw_data, Sigma2=Sigma2, dSigma2=dSigma2)

    def _get_lpt_kinematics(self, k_i, mu_j, f, V_mu, corrs_tree):
        q = self._q
        X_lin_lt, Y_lin_lt = corrs_tree[2], corrs_tree[3]

        V_mu, sin_mu = V_mu

        Kfac, c, s = _rsd_factors(f, mu_j, sin_mu)
        K = k_i * Kfac
        Ksq = K**2
        A_mu = (1 + f) * mu_j / Kfac
        B_mu = sin_mu / Kfac

        A = k_i * q * c
        B = -0.5 * Ksq * Y_lin_lt
        C = k_i * q * s
        # = exp(-K^2 (X + Y)/2) exp(-B s^2); combined so large K gives no 0 * inf
        base = jnp.exp(-0.5 * Ksq * (X_lin_lt + c**2 * Y_lin_lt))
        ft = self._final_transform
        weights = ((-2.0 / k_i) ** ft.ell_indices)[:, None] * ft.q_inv_pows
        moments = get_lpt_moments(
            A, B, C, c**2, s**2, A_mu, B_mu, self.G00_coeffs, self.lmax, V_mu,
            c=c, s=s,
        )
        return Kfac, K, Ksq, base, weights, moments

    def _lpt_moment_coefficient_parts(
        self, k_i, mu_j, f, Kfac, K, Ksq, corrs_tree, corrs_matter_1loop, corrs_bias
    ):
        """The 23 parts as ``{moment index: (nq,) coefficient}`` dicts, in ``_PARTS`` order."""
        X_lin_gt, Y_lin_gt = corrs_tree[4], corrs_tree[5]
        xi_lin, U_lin = corrs_tree[6], corrs_tree[7]
        X22, Y22, X13, Y13, V1, V3, T = corrs_matter_1loop
        U3, U11, U20, X10, Y10 = corrs_bias[0:5]
        V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta = corrs_bias[5:]

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
        U11p = {MQ1: -K * U11, NQ1: -f * k_i * mu_j * U11}
        U20p = {MQ1: -K * U20, NQ1: -f * k_i * mu_j * U20}
        XI_U = {MQ1: -2.0 * K * xi_lin * U_lin}
        XI_XI = {MQ0: 0.5 * xi_lin**2}

        UPS = {MQ0: -Ksq * X_Upsilon, MQ2: -Ksq * Y_Upsilon}
        V10p = {MQ1: -2.0 * K * V10, NQ1: -2.0 * f * k_i * mu_j * V10}
        V12p = {MQ1: -2.0 * K * V12}
        CHI = {MQ0: chi}
        ZETA = {MQ0: zeta}
        UB3 = {MQ1: -2.0 * K * Ub3}
        THETA = {MQ0: 2.0 * theta}

        CTR = {MQ0: 1.0 - 0.5 * Ksq * X_lin_gt, MQ2: -0.5 * Ksq * Y_lin_gt}

        parts = dict(ZA=ZA, AA=AA, A22=A22, A13=A13, W112=W112, U10=U10, A_U=A_U,
                     A10=A10, XI=XI, A_XI=A_XI, U_U=U_U, U11=U11p, U20=U20p,
                     XI_U=XI_U, XI_XI=XI_XI, UPS=UPS, V10=V10p, V12=V12p, CHI=CHI,
                     ZETA=ZETA, UB3=UB3, THETA=THETA, CTR=CTR)
        return tuple(parts[name] for name, *_ in _PARTS)

    def _lpt_linear_parts(self, k_i, mu_j, f, V_mu, corrs_tree, weights):
        """Linear part of the final integrand (JVP at corrs = 0), its weight ``D`` and transform.

        ``D = exp(-K^2 (X_lt + Y_lt)(q_max) / 2)``; rows: Zel'dovich l = 0, 1, 2, b1 (l = 1),
        xi (l = 0), whose transforms are ``(1 + f mu^2)^2 P``, ``2 (1 + f mu^2) P``, ``P``.
        Returns ``(D, (za0, za1, za2), b1, xi, (kaiser_za, kaiser_b1, kaiser_xi))``.
        """
        V, sin_mu = V_mu
        X_lt, Y_lt, X_gt, Y_gt = corrs_tree[2], corrs_tree[3], corrs_tree[4], corrs_tree[5]
        xi_lin, U_lin = corrs_tree[6], corrs_tree[7]

        Kfac, c, s = _rsd_factors(f, mu_j, sin_mu)
        K = k_i * Kfac
        half_Ksq = 0.5 * K**2
        V000, V001, V010 = V[0, 0, 0], V[0, 0, 1], V[0, 1, 0]

        D = jnp.exp(-half_Ksq * (X_lt[-1] + Y_lt[-1]))
        za0 = -half_Ksq * (V000 * (X_lt + X_gt + Y_lt) - Y_lt * (V001 + s**2 * V000))
        za1 = weights[1] * half_Ksq * (V010 * Y_lt + 0.5 * V000 * Y_gt)
        za2 = half_Ksq * c**2 * V000 * Y_gt
        b1 = -2.0 * K * c * V000 * U_lin
        xi = V000 * xi_lin

        kaiser = 1 + f * mu_j**2
        return D, (za0, za1, za2), b1, xi, (kaiser**2, 2.0 * kaiser, jnp.ones_like(kaiser))

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

    def _require_zeldovich_base(self, method):
        if self._counterterms.needs_kspace_base:
            raise ValueError(
                f"{method} needs counterterm_base='zeldovich' (the fused Zel'dovich counterterm "
                f"template); use get_pkmu/get_pk_ells for counterterm_base={self.counterterm_base!r}"
            )

    def _require_corrs_rows(self, corrs):
        if corrs.rows.shape[0] < N_CORRS_ROWS:
            raise ValueError(
                f"corrs has {corrs.rows.shape[0]} rows, but this LPT instance requires "
                f"at least {N_CORRS_ROWS}. Recompute corrs with the same LPT options."
            )

    def _integrands(self, k_i, mu_j, f, V_mu, corrs_q, W):
        """Integrands ``(L, nq)`` of the outputs with part weights ``W`` ``(n_out, 23)`` at one (k, mu).

        ``g_o = base w_l sum_j (sum_p W[o, p] c_j^p) M_j - D (linear rows) - W[o, XI_XI] xi^2/2``;
        returns ``(g, lin)`` with ``lin * P_lin(k)`` the k-space add-back.
        """
        corrs_tree, corrs_matter_1loop, corrs_bias = corrs_q
        Kfac, K, Ksq, base, w, moments = self._get_lpt_kinematics(k_i, mu_j, f, V_mu, corrs_tree)
        parts = self._lpt_moment_coefficient_parts(
            k_i, mu_j, f, Kfac, K, Ksq, corrs_tree, corrs_matter_1loop, corrs_bias)
        coeffs = _weighted_coeffs(parts, W)                             # {j: (n_out, nq)}

        D, za, b1, xi, kaiser = self._lpt_linear_parts(k_i, mu_j, f, V_mu, corrs_tree, w)
        zero = jnp.zeros_like(xi)
        rows = (za, (zero, b1, zero), (xi, zero, zero))                 # [kind][l]
        w_kind = W @ jnp.asarray(_LINEAR_MATRIX)                        # (n_out, 3)
        lin = D * (w_kind @ jnp.stack(kaiser))
        b2sq = 0.5 * corrs_tree[6]**2

        # separate arrays per output: a leading output axis breaks XLA's fusion of the moments
        g = []
        for o in range(W.shape[0]):
            g_o = _apply_coeffs({j: c_j[o] for j, c_j in coeffs.items()}, moments)
            g_o = g_o * (base[None, :] * w)
            for l in range(3):
                delta = -D * (w_kind[o, 0] * rows[0][l] + w_kind[o, 1] * rows[1][l]
                              + w_kind[o, 2] * rows[2][l])
                g_o = g_o.at[l].add(delta - W[o, XI_XI] * b2sq if l == 0 else delta)
            g.append(g_o)
        return tuple(g), lin

    def _evaluate(self, k_true, mu_true, sin_true, rows, f, pk_data, W):
        """Final (k, mu) integral of the outputs with part weights ``W``, ``(n_out, nk, nmu)``.

        ``W``: ``(n_out, 23)`` array or ``(k_i, mu_j) -> W``.  ``mu_true`` must be folded to
        ``|mu|`` (the moments use ``sqrt(s^2) = |s|``); ``sin_true`` is passed through so its
        AD-regular value, not ``sqrt(1 - mu^2)``, reaches the moments.
        """
        ft = self._final_transform
        corrs_q = (rows[0:8], rows[8:15], rows[15:28])
        dc_rows = {row: rows[row] if interpolated else rows[row, 0]
                   for _, row, _, interpolated, _ in _DC_ROWS}
        dc_slopes = {row: spline.slopes(self._logk_fft, rows[row])
                     for _, row, _, interpolated, _ in _DC_ROWS if interpolated}

        def V_mu_of(mu_j, sin_j):
            _, _, s = _rsd_factors(f, mu_j, sin_j)
            return (compute_V_mu(s**2, self.G00_coeffs, self.lmax), sin_j)

        V_all_mu = jax.vmap(V_mu_of)(mu_true, sin_true)
        mu_true = jnp.broadcast_to(mu_true, k_true.shape)
        pk_lin_true = self._pk_at(k_true, pk_data)

        def per_mu(k_i, mu_j, V_mu, p_i):
            logk = jnp.log(k_i)
            W_ij = W(k_i, mu_j) if callable(W) else W
            g, lin = self._integrands(k_i, mu_j, f, V_mu, corrs_q, W_ij)
            kernel = spline.hermite(logk, self._logk_fft, ft.pk_select_kernel, ft.pk_select_slopes, axis=1)
            out = []
            for g_o in g:
                g_o = ft.four_pi_q3 * g_o.at[0, :].subtract(g_o[0, -1])
                out.append(jnp.sum(jnp.sum(g_o * kernel, axis=-1), axis=-1))
            out = jnp.stack(out) + lin * p_i
            for part, row, factor, interpolated, k0_limit in _DC_ROWS:
                if k0_limit and self.subtract_k0_limit:
                    continue
                if interpolated:
                    value = spline.hermite(logk, self._logk_fft, dc_rows[row], dc_slopes[row])
                else:
                    value = dc_rows[row]
                out = out + W_ij[:, part] * (factor * value)
            return out

        def per_k(k_row, mu_row, p_row):
            return jax.vmap(per_mu)(k_row, mu_row, V_all_mu, p_row)

        out = jax.vmap(per_k)(k_true, mu_true, pk_lin_true)       # (nk, nmu, n_out)
        return jnp.moveaxis(out, -1, 0)

    def _pkmu_true(self, k_true, mu_true, sin_true, corrs, pk_data, params):
        """Tree level plus one loop at the true (k, mu); Zel'dovich counterterms folded into the weights."""
        self._require_corrs_rows(corrs)
        f = params.f
        bias_facs = self._get_lpt_bias_factors(params.bias, params.bias_b)
        ctr_leading, ctr_nlo = Counterterms.split_coefficients(params.ctr)
        M = jnp.asarray(_TEMPLATE_MATRIX)

        def weights(k_i, mu_j):
            if self._counterterms.needs_kspace_base:
                ctr_k2_shape = jnp.zeros_like(k_i)
                nlo_shape = jnp.zeros_like(k_i)
            else:
                ctr_k2_shape = self._counterterms.leading_shape(k_i, mu_j, f, ctr_leading)
                nlo_shape = Counterterms.nlo_shape(k_i, mu_j, f, ctr_nlo)
            row = jnp.concatenate([bias_facs, jnp.stack([
                ctr_k2_shape + nlo_shape,
            ])])
            return (row @ M)[None, :]

        return self._evaluate(k_true, mu_true, sin_true, corrs.rows, f, pk_data, weights)[0]

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=None):
        """P(k, mu) at the observed (k, mu), AP included, ``(nk, nmu)``; ``k_IR`` as in :meth:`get_corrs`."""
        corrs = self.get_corrs(pk_data, params.h, k_IR)
        return self.get_pkmu_from_corrs(k, mu, corrs, pk_data, params, alpha_perp, alpha_para)

    def get_pk_ells(self, k, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=None):
        """Multipoles ``P_ell(k)``, ``(nells, nk)``; ``k_IR`` as in :meth:`get_corrs`."""
        corrs = self.get_corrs(pk_data, params.h, k_IR)
        return self.get_pk_ells_from_corrs(k, corrs, pk_data, params, alpha_perp, alpha_para)

    @partial(jit, static_argnames=['self', 'kmax'])
    def get_xi_ells(self, r, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=None, kmax=None):
        """``xi_ell(r)``, ``(nells, nr)``: as :meth:`PowerSpectrum.get_xi_ells`; ``k_IR`` as in :meth:`get_corrs`."""
        corrs = self.get_corrs(pk_data, params.h, k_IR)
        return self._get_xi_ells_from_corrs(r, corrs, pk_data, params, alpha_perp, alpha_para, kmax)

    @partial(jit, static_argnames=['self'])
    def get_pkmu_components(self, k, mu, pk_data, f, k_IR=None):
        """The 23 parts of ``_PARTS`` on the (k, mu) grid, ``(23, nk, nmu)`` (diagnostic, no AP)."""
        self._require_zeldovich_base('get_pkmu_components')
        corrs = self.get_corrs(pk_data, None, k_IR)
        k_true, mu_true, sin_true = self._true_coordinates(k, mu, 1.0, 1.0)
        return self._evaluate(k_true, mu_true, sin_true, corrs.rows, f, pk_data, jnp.eye(N_PARTS))

    @partial(jit, static_argnames=['self'])
    def get_pkmu_terms_from_corrs(self, k, mu, corrs, pk_data, f, alpha_perp=1.0, alpha_para=1.0):
        """Bias-independent templates ``(13, nk, nmu)``: 12 bias monomials and the matter tree.

        Contract with :meth:`combine_pkmu_terms` (which applies the volume factor);
        ``pk_data`` must be the spectrum ``corrs`` was computed from.
        """
        self._require_zeldovich_base('get_pkmu_terms_from_corrs')
        self._require_corrs_rows(corrs)
        k_true, mu_true, sin_true = self._true_coordinates(k, mu, alpha_perp, alpha_para)
        return self._evaluate(k_true, mu_true, sin_true, corrs.rows, f, pk_data,
                              jnp.asarray(_TEMPLATE_MATRIX))

    def get_pkmu_terms(self, k, mu, pk_data, f, alpha_perp=1.0, alpha_para=1.0, k_IR=None):
        """Templates from ``pk_data`` (see :meth:`get_pkmu_terms_from_corrs`)."""
        self._require_zeldovich_base('get_pkmu_terms')
        corrs = self.get_corrs(pk_data, None, k_IR)
        return self.get_pkmu_terms_from_corrs(k, mu, corrs, pk_data, f, alpha_perp, alpha_para)

    @partial(jit, static_argnames=['self'])
    def combine_pkmu_terms(self, k, mu, pkmu_terms, params, alpha_perp=1.0, alpha_para=1.0):
        """P(k, mu) from the templates of :meth:`get_pkmu_terms_from_corrs` (template 12 is the counterterm base)."""
        self._require_zeldovich_base('combine_pkmu_terms')
        bias_facs = self._get_lpt_bias_factors(params.bias, params.bias_b)
        k_true, mu_true, _ = self._true_coordinates(k, mu, alpha_perp, alpha_para)
        mu_k = jnp.broadcast_to(mu_true, k_true.shape)                  # (nk, nmu)
        pkmu = jnp.tensordot(bias_facs, pkmu_terms[:12], axes=(0, 0))   # (nk, nmu)
        return self._add_stoch_ctr_volume(pkmu, k_true, mu_k, params, pkmu_terms[12],
                                          alpha_perp, alpha_para)
