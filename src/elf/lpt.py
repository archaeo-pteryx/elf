import os
import numpy as _np

import jax
from jax import jit
from functools import partial

import jax.numpy as jnp

from . import fftlog
from . import spline

from .utils import get_pk, cross_bias_factor
from .utils_lpt import (
    make_G00_coeffs,
    get_lpt_moments,
    compute_V_mu,
)
from .multipole import (
    prepare_mu_gauleg,
    get_legendre_multipoles,
    get_k_mu_true_for_ap,
    get_k_mu_true_sin_for_ap,
)
from . import ir_resum
from .eft_terms import Counterterms, stochasticity


# Bias monomials multiplying the 12 LPT term templates, in the order produced by
# ``LPT._get_lpt_bias_factors``.  Columns are the powers of
# (b1, b2, bG2, bGamma3).  Every monomial has total degree <= 2, which is the
# regime ``cross_bias_factor`` covers, so the auto -> cross symmetrisation rule
# is literally the same code path the EPT backend uses.
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


# Indices into the tuple returned by utils_lpt.get_lpt_moments:
# (mq0, mq1, mq2, mq3, mq4, nq1, nq2, mq1_nq1, mq2_nq1)
MQ0, MQ1, MQ2, MQ3, MQ4, NQ1, NQ2, MQ1NQ1, MQ2NQ1 = range(9)
N_MOMENTS = 9


def _sum_coeffs(dicts):
    """Sum a list of {moment index: (nq,) coefficient} dictionaries."""
    out = {}
    for d in dicts:
        for j, v in d.items():
            out[j] = out[j] + v if j in out else v
    return out


def _weighted_sum_coeffs(dicts, weights):
    """sum_t weight_t * dict_t, again as a {moment index: coefficient} dict."""
    out = {}
    for d, weight in zip(dicts, weights):
        for j, v in d.items():
            term = weight * v
            out[j] = out[j] + term if j in out else term
    return out


def _apply_coeffs(coeffs, moments):
    """Contract a coefficient dictionary with the moments: sum_j c_j[q] m_j[l, q].

    ``moments`` is the nine-element tuple from ``utils_lpt.get_lpt_moments``,
    each entry of shape (L, nq); the result is a single (L, nq) integrand.
    """
    integrand = None
    for j, c_j in coeffs.items():
        term = c_j[None, :] * moments[j]
        integrand = term if integrand is None else integrand + term
    return integrand


class LPT:

    def __init__(self,
                 kmin_fft=1e-5,
                 kmax_fft=1e2,
                 nfft=512,
                 pad_mode='zero-pad',
                 fftlog_settings=None,
                 lmax=5,
                 ngauss=4,
                 counterterm_base='linear_ir_resum',
                 irres_method='DST',
                 r_bao=110.0,
                 lambda_ir=0.2,
                 bias_basis='bG2',
                 subtract_k0_limit=False,
                 ):
        """One-loop Lagrangian-PT galaxy power spectrum in redshift space.

        Parameters
        ----------
        kmin_fft, kmax_fft : float, defaults 1e-5 and 1e2
            Ends of the internal logarithmic k grid, in h/Mpc.
        nfft : int, default 512
            Number of nodes on that grid.  Must be even: the FFTlog pipeline
            uses rfft/irfft with an implicit even length.
        pad_mode : {'zero-pad', 'power-law'}, default 'zero-pad'
            Model of the input P(k) outside the tabulated range.

            'zero-pad' is the band-limited model: P is exactly zero outside
            [k_min, k_max], and the k_min and k_max nodes enter the forward
            FFTlog sum with the trapezoid end weight 1/2 (see
            ``fftlog.trapezoid_weights``).  Consistently with that, the physical
            supports of the Q/R sources -- [k_min, 2 k_max] for the
            mode-coupling rows Q and [k_min, k_max] for the rows R that carry
            an explicit P_lin(k) factor -- also carry the half weight at their
            end nodes.

            'power-law' is a hybrid model: only the P -> xi transform is
            computed from the P continued across the padded band by its
            endpoint log-slopes, while the sources Q and R are still cut at the
            same places (2 k_max and k_max).  That cut is part of the model
            definition: the tail beyond it is ~1e-3 of the zero-lag constants.
        fftlog_settings : fftlog.FFTlogSettings or None, default None
            Numerical constants of the FFTlog pipeline (bias ``nu``, width of
            the padded band, node ``q_star`` at which the zero-lag constants
            are read); ``None`` means the validated defaults
            ``FFTlogSettings()``.
        lmax : int, default 5
            Order of the angular-moment expansion of the final (k, mu)
            integral.  The truncation error at the default is ~5e-4 (P0) and
            ~2e-3 (P2, P4) for k <= 0.2 h/Mpc, growing to ~6e-3 and ~1.4e-2 at
            k = 0.3 h/Mpc; use lmax >= 8 if k = 0.3 h/Mpc must be accurate.
        ngauss : int, default 4
            Number of Gauss-Legendre points on mu in [0, 1] used for the
            multipole projection.
        counterterm_base : {'linear_ir_resum', 'linear', 'zeldovich'}, default 'linear_ir_resum'
            Spectrum multiplying the k^2 and k^4 counterterms.  'zeldovich' is
            fused into the same final q integral as the rest of the model
            rather than being computed separately.
        irres_method : {'DST', 'SG', 'WH'}, default 'DST'
            Wiggle/no-wiggle split of the IR-resummed linear counterterm base.
            In LPT it is used only when ``counterterm_base='linear_ir_resum'``.
        r_bao : float, default 110.0
            BAO scale of that IR resummation, in Mpc/h.  In LPT it is used only
            when ``counterterm_base='linear_ir_resum'``.
        lambda_ir : float, default 0.2
            Upper limit of the k integral of the BAO damping Sigma^2 (and
            delta Sigma^2) of that IR resummation, in h/Mpc.  In LPT it is used
            only when ``counterterm_base='linear_ir_resum'``; it affects the
            linear counterterm base only, never the LPT body.  Not to be
            confused with the ``k_IR`` argument of the methods (the scale of the
            split ``P = P e^{-(k/k_IR)^2} + ...`` inside the LPT body).
        bias_basis : {'bG2', 'bs2'}, default 'bG2'
            Convention of the second-order tidal bias in the ``bias`` vector;
            'bs2' is converted to the 'bG2' basis internally.
        subtract_k0_limit : bool, default False
            Subtract the k -> 0 constant of the b2^2 term, i.e. its
            shot-noise-like piece.
        """

        if pad_mode not in fftlog.INPUT_SUPPORT:
            raise ValueError("pad_mode must be 'zero-pad' or 'power-law'")
        if nfft % 2:
            raise ValueError(
                "nfft must be even: the FFTlog pipeline uses rfft/irfft with an "
                "implicit even length"
            )
        if bias_basis not in ('bG2', 'bs2'):
            raise ValueError("bias_basis must be 'bG2' or 'bs2'")

        self.lmax = lmax
        self._counterterms = Counterterms(counterterm_base)
        if irres_method not in ('DST', 'SG', 'WH'):
            raise ValueError("irres_method must be 'DST', 'SG', or 'WH'")
        self.irres_method = irres_method
        self.r_bao = r_bao
        self.lambda_ir = lambda_ir
        self.bias_basis = bias_basis
        self._subtract_k0_limit = subtract_k0_limit

        # preparation for Gauss-Legendre quadrature
        self._mu_quad, self._legendre_weights = prepare_mu_gauleg(ngauss)

        # preparation for FFT
        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self.fftlog_settings = fftlog.FFTlogSettings() if fftlog_settings is None else fftlog_settings
        self._pad_mode = pad_mode
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._initialize_lpt()

    @property
    def counterterm_base(self):
        """Counterterm base fixed at construction time."""
        return self._counterterms.base

    def _initialize_lpt(self):
        # precompute the coefficients of G00
        self.G00_coeffs = make_G00_coeffs(self.lmax)

        # (l, n) for which xi_ln's are cached; UV-safe basis uses n in {-2,-1,0} only.
        ln_list = [[0, 0], [0, -2], [1, -1], [2, 0], [2, -2], [3, -1], [4, 0]]
        self.ln_list = jnp.array(ln_list)
        lmax = max(int(jnp.max(self.ln_list[:, 0])), self.lmax)
        self._set_hankel(lmax)

    def _set_hankel(self, lmax):
        """Grids, kernels, static support weights and the node ``q_star``.

        The chain of transforms and the support rules are described above
        :meth:`_get_xi_ln`.  Support weights (``fftlog.trapezoid_weights``):

        * ``_end_weights_input``: the input model (``pad_mode``) -- 1/2 on
          k_min and k_max for 'zero-pad', on the padded corners for
          'power-law'.  The FFTlog's m = 0 coefficient is the rectangle rule
          over the padded grid, so the node at which the input's support ends
          must carry weight 1/2, or an O(dln) Euler-Maclaurin endpoint term
          survives in every correlator.
        * ``_mask_xi`` / ``_end_weights_xi``: the P -> xi outputs, restricted
          to ``'lower_and_core'`` (the lower part of the padded band is the
          physical q -> 0 plateau, the upper part is the rounding floor of an
          already-decayed transform).  The plain mask is applied to the
          outputs; the end weights only to the copies that are transformed
          again (the Q/R source inputs and the b2^2 residual source): that
          support ends at the interior node q_max, and for the n = -1 rows
          (xi_1^{-1}, xi_3^{-1}) the transformed function q^2 xi does not
          decay, so the end node would carry the full O(dln) term (verified
          2026-09-23: the backward pk_1m1 / pk_3m1 errors against an exact
          reference for the transform alone drop from 2.5e-5 / 1.2e-4 to
          2.7e-7 / 2.0e-6).
        * ``_end_weights_Q`` / ``_end_weights_R``: the physical supports of the
          sources -- [k_min, 2 k_max] for the mode-coupling rows Q and
          [k_min, k_max] for the rows R, whose explicit P_lin(k) factor is what
          cuts them.
        * ``_end_weights_q_core``: the source of the final q -> k transform,
          which lives on the core q grid and is zero-padded (support
          [q_min, q_max]).  With the half weights the transform's m = 0
          coefficient is the same trapezoid sum as the chi/zeta DC constant
          (``_get_qspace_dc_integral``), so the k -> 0 limit of those templates
          cancels exactly.

        ``q_star`` is the node at which the zero-lag moment
        ``M = lim_{q->0} D_0(q)`` is read off the *padded* forward output: the
        node of ``_q_padded`` closest to ``q_min * q_star_factor`` (one decade
        inside the core edge by default).  This is far enough from the padded
        corner (where the ``q^{-nu}`` unbiasing amplifies the FFT floor and
        ``D_0`` is still in its transition) and close enough to ``q = 0`` that
        the ``q*^2`` and ``q*^4`` Taylor terms in ``_get_zero_lag_constant``
        recover the zero-lag value to ~1e-9.  A scan of the reading against an
        independent reference (2026-09-19) showed the corner-based rule
        (``q* k_supp = 6e-3``, ``k_supp = 2 kmax_fft``) sat on an accidental
        zero crossing.
        """
        fs = self.fftlog_settings
        l_list = list(range(lmax + 1))

        self._npad = max(1, int(fs.npad_factor * self._nfft))
        self._grid = fftlog.LogGrid.from_core(self._k, self._npad)
        self._k_padded = self._grid.x
        dln = self._grid.dln
        self._dln_k = float(dln)

        # Low-ringing output phase per ell
        lnxy = jnp.array([
            fftlog.low_ringing_phase(l, fs.nu, self._grid)
            for l in l_list
        ])
        self._lnxy_hankel = lnxy

        # Ell-specific forward output grids (used for xi_ln forward transforms)
        self._q_xi_padded = jnp.array([
            fftlog.output_grid(self._grid, lnxy[l]) for l in l_list
        ])

        # Common q-grid (ell=0) for backward transforms and all q-space products
        self._q_padded = self._q_xi_padded[0]           # shape (nfft_padded,)
        self._q = self._grid.crop(self._q_padded)       # shape (nfft,)
        self._ell_indices = jnp.arange(self.lmax + 1, dtype=self._q.dtype)
        self._q_inv_pows = self._q[None, :] ** (-self._ell_indices[:, None])
        # Precomputed constants used inside JIT-compiled methods
        self._4pi_q3 = 4.0 * jnp.pi * self._q**3
        self._4pi_q3_padded = 4.0 * jnp.pi * self._q_padded**3
        self._logk_fft = jnp.log(self._k)
        self._log_q_padded = jnp.log(self._q_padded)
        self._log_q_xi_padded = jnp.log(self._q_xi_padded)

        # Forward kernels: ell-specific output phase, ell-specific G_l kernel
        self._u_m_xi = jnp.array([
            fftlog.hankel_kernel(l, fs.nu, self._grid, lnxy[l]) for l in l_list
        ])
        # Backward kernels: ell=0 output phase, ell-specific G_l kernel
        self._u_m = jnp.array([
            fftlog.hankel_kernel(l, fs.nu, self._grid, lnxy[0]) for l in l_list
        ])
        # Mellin bias of the forward transform of row (l, n) of ``ln_list``:
        # the k^{n+3} factor of xi_l^n is absorbed into the bias, the output
        # is unbiased with q^{-nu}.
        self._nu_xi_rows = fs.nu - (self.ln_list[:, 1] + 3).astype(jnp.float64)

        # Static support weights (see the docstring).
        self._end_weights_input = fftlog.trapezoid_weights(
            fftlog.INPUT_SUPPORT[self._pad_mode], self._nfft, self._npad, dln)
        self._end_weights_xi = fftlog.trapezoid_weights('lower_and_core', self._nfft, self._npad, dln)
        self._mask_xi = jnp.where(self._end_weights_xi > 0, 1.0, 0.0)
        self._end_weights_Q = fftlog.trapezoid_weights('core_to_2kmax', self._nfft, self._npad, dln)
        self._end_weights_R = fftlog.trapezoid_weights('core', self._nfft, self._npad, dln)
        self._end_weights_q_core = fftlog.trapezoid_weights('core', self._nfft, 0, dln)

        # Node at which the zero-lag constants are read (see the docstring).
        q_target = float(self._q[0]) * fs.q_star_factor
        iq = int(_np.argmin(_np.abs(
            _np.log(_np.asarray(self._q_padded)) - _np.log(q_target))))
        self._iq_star = iq
        self._q_star = float(self._q_padded[iq])

        # R sources: which (l, n) rows of ``ln_list`` they are, and the
        # order l of the backward kernel each one uses.  They are transformed
        # on their *own* P -> xi output grid ``_q_xi_padded[l]`` with the
        # low-ringing phase of their own order (``_u_m_xi[l]``); the conjugate
        # grid relation then lands every row on the common ``_k_padded`` with
        # no interpolation anywhere in the chain (the same construction the EPT
        # P13 blocks use).  The assertion below is that relation:
        # ``output_grid(output_grid(k, lnxy_l), lnxy_l) == k``.
        r_ln = [(0, 0), (2, 0), (4, 0), (1, -1), (3, -1)]
        ln_np = _np.asarray(self.ln_list)
        self._R_source_rows = jnp.asarray(
            [int(_np.flatnonzero((ln_np[:, 0] == l) & (ln_np[:, 1] == n))[0])
             for l, n in r_ln], dtype=jnp.int32)
        self._R_source_ells = jnp.asarray([l for l, _ in r_ln], dtype=jnp.int32)
        k_ref = _np.asarray(self._k_padded)
        for l, _n in r_ln:
            k_back = _np.asarray(fftlog.output_grid(
                self._q_xi_padded[l], self._lnxy_hankel[l]))
            assert _np.allclose(k_back, k_ref, rtol=1e-12), (
                f'R-source backward output grid for ell={l} is not the common '
                'k grid; the conjugate relation of the low-ringing phase is '
                'broken')

        self._set_pk_select_kernel()

    def _set_pk_select_kernel(self):
        """Kernel of the final q -> k transform, one column per core q node.

        ``_pk_select_kernel[l, k, j]`` is the transform of the unit source at
        q node j, so ``sum_j g_j K[l, :, j]`` is the transform of ``g``.  The
        source lives on the core q grid and is zero-padded; the columns are
        scaled by the trapezoid end weights ``_end_weights_q_core`` here, once,
        so every caller of :meth:`get_pk_batched_interp` transforms the
        end-weighted source at no extra cost.
        """
        basis = fftlog.pad(jnp.eye(self._q.shape[0], dtype=self._q.dtype), self._npad, 'zero-pad')

        def build_one(u_m):
            pk = fftlog.hankel(
                self.fftlog_settings.nu, basis, self._q_padded, self._k_padded,
                u_m, self._npad,
            )
            return pk.T

        self._pk_select_kernel = (
            jax.vmap(build_one)(self._u_m[:self.lmax + 1])
            * self._end_weights_q_core
        )

    def _get_qspace_dc_integral(self, corr):
        """Compute 4*pi*int q^2 corr(q) dq on the internal log-q grid."""
        y = self._4pi_q3 * corr
        dlnq = jnp.log(self._q[1] / self._q[0])
        return dlnq * (jnp.sum(y) - 0.5 * (y[0] + y[-1]))

    def _get_lpt_dc_correction(self, corr):
        """Return the additive zero-DC correction for a scalar l=0 source.

        The desired correction is the k-independent ``-P(k=0)`` offset.  Earlier versions computed it as ``H[j0-1] - H[j0]``.  
        That is accurate at low k, but the non-decaying ``j0-1`` tail develops the same high-k endpoint contamination seen in the EPT P22 residual blocks.
        The DC offset itself is a simple q-space scalar, so compute it directly and broadcast it.
        """
        dc = self._get_qspace_dc_integral(corr)
        return jnp.broadcast_to(-dc, self._k.shape)

    def _get_grid_pk_int(self, pk_lin):
        """``(1/2 pi^2) int dk P(k)`` as the zero-lag moment of the FFTlog input.

        The constant is the ``m = 0`` coefficient of the discrete transform of
        the *same* padded, end-weighted array that feeds the forward FFTlog
        (``_pad_input``), i.e. ``sum_j dln k_j P_j / (2 pi^2)``.  Reading it off
        that array -- rather than integrating an independently extrapolated
        copy of ``P`` over an unrelated k range -- keeps the constant and the
        q-dependent part of the tree correlators consistent: ``X_lin,lt(q->0)``
        is then the physical ``q^2 M_2 / 15`` instead of a negative offset left
        over from the band mismatch.
        """
        return fftlog.grid_moment(
            self._pad_input(pk_lin), self._k_padded, self._grid.dln, 1
        ) / (2 * jnp.pi**2)

    def _get_grid_pk_int2(self, pk_lin):
        """``(1/2 pi^2) int dk k^2 P^2(k)`` on the same band as the FFTlog input.

        Same band-limited definition as :meth:`_get_grid_pk_int`, i.e.
        ``sum_j dln k_j^3 P_j^2 / (2 pi^2)`` over the padded input model.  The
        trapezoid end weights belong to the input ``P`` and are applied once,
        not once per factor, so the weighted array is formed as
        ``w * (pad P)^2`` rather than ``(pad P * w)^2``.
        """
        padded = fftlog.pad(pk_lin, self._npad, self._pad_mode, x=self._k)
        return fftlog.grid_moment(
            self._end_weights_input * padded**2, self._k_padded, self._grid.dln, 3
        ) / (2 * jnp.pi**2)

    def _get_lpt_b2sq_residual_pk(self, xi_lin, pk_lin):
        """Return H_0[0.5 xi^2](k) - I0 and I0 for the b2^2 DC split.

        ``xi_lin`` is ``xi_0^0`` on the padded common q grid (the
        ``'lower_and_core'`` forward output: the lower part of the padded band
        kept, the upper part zeroed).  The source ``4 pi q^3 xi^2 / 2`` carries
        the trapezoid end weights of that support (``_end_weights_xi``), so the
        transform's m = 0 coefficient is a trapezoid sum, and the lower part of
        the padded band supplies the q < q_min part of the integral -- the same
        treatment as the EPT P22 m = 0 blocks.
        """
        source = 0.5 * xi_lin**2
        direct = fftlog.hankel(
            self.fftlog_settings.nu,
            self._4pi_q3_padded * source * self._end_weights_xi,
            self._q_padded, self._k_padded, self._u_m[0], self._npad, crop=True,
        )
        dc = 0.5 * self._get_grid_pk_int2(pk_lin)
        return direct - dc, dc

    def _require_corrs_rows(self, corrs):
        min_rows = 32
        if corrs.shape[0] < min_rows:
            raise ValueError(
                f"corrs has {corrs.shape[0]} rows, but this LPT instance requires "
                f"at least {min_rows}. Recompute corrs with the same LPT options."
            )

    def _interp_k_array(self, array, i0, t):
        idx = jnp.clip(i0 + jnp.arange(-1, 3), 0, self._k.shape[0] - 1)
        rows = jnp.take(array, idx, axis=-1)
        linear = (1.0 - t) * rows[..., 1] + t * rows[..., 2]
        cubic = spline.cubic_interp_patch(jnp.moveaxis(rows, -1, 0), t)
        use_cubic = (i0 >= 1) & (i0 <= self._k.shape[0] - 3)
        return jnp.where(use_cubic, cubic, linear)

    def get_pk_batched_interp(self, arrays, i0, t):
        arrays = jnp.asarray(arrays)
        L = arrays.shape[-2]
        idx = jnp.clip(i0 + jnp.arange(-1, 3), 0, self._k.shape[0] - 1)
        kernel_rows = jnp.take(self._pk_select_kernel[:L], idx, axis=1)
        linear = (1.0 - t) * kernel_rows[:, 1, :] + t * kernel_rows[:, 2, :]
        cubic = spline.cubic_interp_patch(jnp.moveaxis(kernel_rows, 1, 0), t)
        use_cubic = (i0 >= 1) & (i0 <= self._k.shape[0] - 3)
        weights = jnp.where(use_cubic, cubic, linear)
        return jnp.sum(arrays * weights, axis=-1)

    def _get_V_mu_for_mu(self, mu_j, f, sin_mu=None):
        Kfac = jnp.sqrt(1 + f * (2 + f) * mu_j**2)
        if sin_mu is None:
            sin_mu = jnp.sqrt(jnp.maximum(1 - mu_j**2, 0.0))
        s = f * mu_j * sin_mu / Kfac
        s2 = s**2
        # sin_mu travels with the cached contraction so that the caller's
        # AD-regular value (not a re-derived sqrt(1-mu^2)) reaches the moments.
        return (compute_V_mu(s2, self.G00_coeffs, self.lmax), sin_mu)

    def _get_lpt_weights(self, k_i):
        return ((-2.0 / k_i) ** self._ell_indices)[:, None] * self._q_inv_pows

    def _pad_input(self, array):
        """Padded, end-weighted copy of a core-grid input array.

        This is the array the forward FFTlog actually transforms, and it is
        also the array whose zero-lag moment defines the ``pk_int``-type
        constants (see :meth:`_get_grid_pk_int`), which is why it is factored
        out here rather than inlined in the rfft.

        ``_end_weights_input`` carries the trapezoid end weights of the input
        model: 1/2 on the k_min and k_max nodes for the band-limited
        ('zero-pad') model, 1/2 on the padded corners for the continued
        ('power-law') one.  The array may carry leading batch axes.
        """
        return fftlog.pad(array, self._npad, self._pad_mode, x=self._k) * self._end_weights_input

    # ------------------------------------------------------------------
    # The chain of FFTlog transforms
    # ------------------------------------------------------------------
    # The three chained transforms are
    #   the P -> xi transform (the linear correlation functions):
    #       P(k)             -> xi_l^n(q)       (forward,  k -> q)
    #   the Q/R sources (backward transforms of xi products and of q^2 xi):
    #       xi combinations  -> Q(k), R(k)      (backward, q -> k)
    #   the k -> q transforms of the sources (the correlator rows):
    #       Q/R combinations -> D_l^n(q)        (forward,  k -> q)
    # Only the input P(k) is padded (``pad_mode``); each transform hands its
    # full padded output to the next one, restricted to the support of the
    # quantity it represents by the static weights of ``_set_hankel`` (see
    # ``fftlog.trapezoid_weights``).
    #
    # Output-grid rule: every transform uses the low-ringing phase of its own
    # order l, so its output lives on that order's grid; outputs are brought
    # together either through the conjugate grid relation (free, exact) or by
    # resampling (a sub-node cubic interpolation).  Concretely
    #   * the P -> xi transform produces xi_l^n on the per-l grid
    #     ``_q_xi_padded[l]``;
    #   * the R sources are transformed straight off that grid with the
    #     same per-l phase, which by ``output_grid(output_grid(k, lnxy_l),
    #     lnxy_l) == k`` lands them back on the common ``_k_padded`` with no
    #     interpolation at all -- the same chain the EPT P13 blocks use;
    #   * the Q sources are *products* of xi's of different l, so they do
    #     need one common q grid, and use the resampled xi copy;
    #   * the source -> correlator transforms again go per l and resample the
    #     D_l^n onto the common q grid, and the final (k, mu) evaluation
    #     interpolates the selected rows off the l = 0 grid.

    def _resample_to_common_q_padded(self, ells, rows):
        """Move ell-specific padded forward outputs onto the common padded q grid.

        The ell-specific grids differ from the common one only by the constant
        low-ringing phase ``exp(lnxy[l] - lnxy[0]) < e^{dln}``, so this is an
        interpolation by less than one node; the outermost node is extrapolated
        by the same cubic.
        """
        log_q = self._log_q_padded

        def interp_one(ell, row):
            return spline.interp1d(log_q, self._log_q_xi_padded[ell], row)

        return jax.vmap(interp_one)(ells, rows)

    def _get_xi_ln(self, arrays):
        """The P -> xi transform on the padded grid, returned on both grid layouts.

        ``arrays`` is ``(n_arrays, nfft)`` on the core k grid.  Returns
        ``(xi_ln_common, xi_ln_own_grid)`` where

        * ``xi_ln_common`` is ``(n_arrays, 5, 5, n_padded)`` indexed as
          ``xi_ln[i, l, n]`` on the *common* padded q grid ``_q_padded``
          (the l = 0 low-ringing grid), obtained by resampling; it is what the
          pointwise rows and the Q products need, because those combine
          different l at the same q;
        * ``xi_ln_own_grid`` is ``(n_arrays, n_ln, n_padded)``, one row per
          entry of ``ln_list``, each still on *its own* forward output grid
          ``_q_xi_padded[l]`` -- no interpolation has touched it.  The R
          sources are transformed straight off these grids.

        Every input row is padded once with ``pad_mode`` and carries the
        trapezoid end weights of that model (:meth:`_pad_input`).  Row (l, n)
        of ``ln_list`` is the transform of ``P k^{n+3} / (2 pi^2)`` with the
        ``k^{n+3}`` factor absorbed into the Mellin bias
        (``_nu_xi_rows = nu - (n + 3)``) and the output unbiased with
        ``q^{-nu}``.  Both outputs keep the lower part of their padded band
        (the physical ``q -> 0`` plateau, rule (i)) while the upper part is
        zeroed (rule (ii)); that mask is index-based, so the same one applies
        to the per-l rows.

        The returned arrays carry the plain ``'lower_and_core'`` mask, *not*
        the trapezoid end weights of that support: ``xi_ln_common`` is cropped
        to the core for the pointwise (tree, chi/zeta) rows, where a halved
        q_max node would be wrong.  The end weights ``_end_weights_xi`` (also
        index-based) are applied by the caller to the copies that are handed to
        :meth:`_get_Qs_Rs`, i.e. only to the arrays that are transformed
        again.
        """
        arrays = jnp.atleast_2d(jnp.asarray(arrays))
        ells = self.ln_list[:, 0]
        ns = self.ln_list[:, 1]

        fx = self._pad_input(arrays / (2 * jnp.pi**2))          # (n_arrays, n_padded)
        raw = fftlog.hankel(
            self._nu_xi_rows[:, None], fx[:, None, :], self._k_padded, None,
            self._u_m_xi[ells], self._npad, crop=False,
            y_pow=self._q_xi_padded[ells] ** (-self.fftlog_settings.nu),
        )                                                          # (n_arrays, n_ln, n_padded)

        xis = jax.vmap(lambda row: self._resample_to_common_q_padded(ells, row))(raw)
        xis = xis * self._mask_xi
        raw = raw * self._mask_xi
        xi_ln = jnp.zeros((arrays.shape[0], 5, 5, self._k_padded.shape[0]))
        return xi_ln.at[:, ells, ns].set(xis), raw

    def _get_Qs_Rs(self, xi_ln_common, xi_ln_own_grid, pk_lin_padded):
        """The Q/R sources on the padded k grid: Q1,Q2,Q5 and R1,R2,F_G2.

        The Q rows keep their physical support ``[k_min, 2 k_max]`` and drop
        the lower part of the padded band (rule (iii)); the R rows carry an
        explicit factor ``P_lin(k)``, which is zero outside the core, so they
        need no extra mask.

        The two groups take their xi input in different layouts.  The Q
        rows are pointwise products of ``xi_l^n`` of different l and therefore
        need them tabulated at the same q: they use ``xi_ln_common``, the
        resampled copy on the common grid ``_q_padded``.  The R rows transform
        one ``q^2 xi_l^n`` each, so they can stay on the per-l forward grid:
        they use ``xi_ln_own_grid`` (rows of ``ln_list``, each on
        ``_q_xi_padded[l]``) with the backward kernel of that same order and
        phase, ``_u_m_xi[l]``, whose conjugate output grid is exactly the common
        ``_k_padded``.  That is the interpolation-free chain the EPT P13 blocks
        use; for l = 0 it is identical to a transform on the common grid with
        the common phase, so ``pk_00`` is unchanged bit for bit.

        Both supports end at an interior node of the padded grid, so both carry
        the trapezoid end weights of :func:`fftlog.trapezoid_weights`:
        ``_end_weights_Q`` (1/2 at k_min and at 2 k_max, and zero outside) and
        ``_end_weights_R`` (1/2 at k_min and k_max), applied to the R rows
        through their common ``P_lin(k)`` factor.

        ``xi_ln_common`` and ``xi_ln_own_grid`` are expected to carry the end
        weights ``_end_weights_xi`` of their own support (the lower part of the
        padded band plus the core) already; the caller applies them, because
        the unweighted P -> xi output is also used pointwise on the core.  They
        matter for the n = -1 rows, whose ``q^2 xi`` does not decay at q_max
        (verified 2026-09-23: pk_1m1 / pk_3m1 errors against an exact reference
        for the transform alone, 2.5e-5 / 1.2e-4 -> 2.7e-7 / 2.0e-6).
        """
        nu = self.fftlog_settings.nu
        integrands = jnp.stack([
            8/15 * xi_ln_common[0,0]**2 - 16/21 * xi_ln_common[2,0]**2
            + 8/35 * xi_ln_common[4,0]**2,
            xi_ln_common[1,-1]**2 - xi_ln_common[3,-1]**2,
        ], axis=0)
        res = fftlog.hankel(
            nu, self._4pi_q3_padded * integrands, self._q_padded,
            self._k_padded, self._u_m[0], self._npad, crop=False,
        )
        res = res * self._end_weights_Q
        Q1 = res[0]
        Q2 = Q1 - 2/5 * self._k_padded**2 * res[1]
        Q5 = (Q1 + Q2) / 2
        Qs = jnp.stack([Q1, Q2, Q5], axis=0)

        ells = self._R_source_ells
        q_l = self._q_xi_padded[ells]
        xis = xi_ln_own_grid[self._R_source_rows]
        pk_list = fftlog.hankel(
            nu, q_l**2 * xis, q_l,
            self._k_padded, self._u_m_xi[ells], self._npad, crop=False,
        ) * (pk_lin_padded * self._end_weights_R)
        pk_00, pk_20, pk_40, pk_1m1, pk_3m1 = pk_list
        k = self._k_padded
        R1 = k**2 * (8/15*pk_00 - 16/21*pk_20 + 8/35*pk_40)
        R3 = (k**2 * (2/5*pk_00 - 6/7*pk_20 + 16/35*pk_40)
              + k**3 * (2/5*pk_1m1 - 2/5*pk_3m1))
        R2 = R3 - R1
        F_G2_raw = -32/21 * k**2 * (pk_00 - 10/7*pk_20 + 3/7*pk_40)
        Rs = jnp.stack([R1, R2, F_G2_raw], axis=0)
        return Qs, Rs

    def _get_xi_ln_batched(self, ells, ns, sources):
        """k -> q transforms of padded sources: D_l^n(q) on the common padded q grid.

        ``sources`` already live on the padded k grid and vanish outside their
        physical support, so no padding is applied here.
        """
        ells = jnp.asarray(ells, dtype=jnp.int32)
        ns = jnp.asarray(ns)
        fx = sources * self._k_padded[None, :] ** (ns[:, None] + 3) / (2 * jnp.pi**2)
        d = fftlog.hankel(
            self.fftlog_settings.nu, fx, self._k_padded, self._q_xi_padded[ells],
            self._u_m_xi[ells], self._npad, crop=False,
        )
        return self._resample_to_common_q_padded(ells, d)

    def _get_zero_lag_constant(self, source, D0):
        """Zero-lag moment of a Q/R source, read from the same discrete operator.

        With ``D_0(q) = int dlnk k S(k) j_0(kq) / (2 pi^2)`` and
        ``j_0(x) = 1 - x^2/6 + x^4/120 - x^6/5040 + ...``,

            M = lim_{q->0} D_0(q) = D_0(q*) + q*^2 M_2 / 6 - q*^4 M_4 / 120 + O(q*^6)
            M_2 = sum_k dln k^3 S(k) / (2 pi^2)
            M_4 = sum_k dln k^5 S(k) / (2 pi^2).

        Reading ``D_0`` at ``q*`` (on its ``q -> 0`` plateau, see
        ``_set_hankel``) instead of quadrating ``S`` separately keeps the
        constant and the ``q``-dependent rows consistent to machine precision,
        since both come out of the same transform.

        Returns ``(M_hat, D0_at_q_star, M_2, M_4, remainder_bound)`` with
        ``remainder_bound = q*^6 / (5040 * 2 pi^2) * sum_k dln k^7 |S(k)|``, a
        modulus bound on the first neglected (sixth-order) Taylor term.  The
        sums run over the whole padded k grid: the source is already zero
        outside its physical support, so a log-trapezoid there is the
        trapezoid over that support.
        """
        k = self._k_padded
        dln = self._dln_k
        two_pi_sq = 2 * jnp.pi**2
        y2 = k**3 * source / two_pi_sq
        y4 = k**5 * source / two_pi_sq
        M_2 = dln * (jnp.sum(y2, axis=-1) - 0.5 * (y2[..., 0] + y2[..., -1]))
        M_4 = dln * (jnp.sum(y4, axis=-1) - 0.5 * (y4[..., 0] + y4[..., -1]))
        D0_star = D0[..., self._iq_star]
        q_star = self._q_star
        M_hat = D0_star + q_star**2 * M_2 / 6.0 - q_star**4 * M_4 / 120.0
        remainder = (q_star**6 / 5040.0) * dln * jnp.sum(
            k**7 * jnp.abs(source) / two_pi_sq, axis=-1)
        return M_hat, D0_star, M_2, M_4, remainder

    def _get_corrs_matter_1loop(self, Qs, Rs):
        """X22/Y22/X13/Y13/V1/V3/T rows from the padded Q/R sources.

        Returns ``(corrs, moments)``: the q -> 0 constants of X22 and X13 are
        read off the padded transform (:meth:`_get_zero_lag_constant`), and
        ``moments`` holds their breakdown for :meth:`get_corrs_diagnostics`.
        """
        Q1, Q2 = Qs[0], Qs[1]
        R1, R2 = Rs[0], Rs[1]

        source_22 = 9/98 * Q1
        source_13 = 5/21 * R1
        xi_A = self._get_xi_ln_batched(
            jnp.array([0, 2, 0, 2]),
            jnp.array([-2, -2, -2, -2]),
            jnp.stack([source_22, source_22, source_13, source_13], axis=0),
        )
        moment_22 = self._get_zero_lag_constant(source_22, xi_A[0])
        moment_13 = self._get_zero_lag_constant(source_13, xi_A[2])
        xi_ln_22_0m2, xi_ln_22_2m2, xi_ln_13_0m2, xi_ln_13_2m2 = self._grid.crop(xi_A)

        X22 = 2/3 * (moment_22[0] - xi_ln_22_0m2 - xi_ln_22_2m2)
        Y22 = 2 * xi_ln_22_2m2
        X13 = 2/3 * (moment_13[0] - xi_ln_13_0m2 - xi_ln_13_2m2)
        Y13 = 2 * xi_ln_13_2m2

        xi_W = self._grid.crop(self._get_xi_ln_batched(
            jnp.array([3, 1, 1]),
            jnp.array([-3, -3, -3]),
            jnp.stack([
                -3/7 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2),
                3/35 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2),
                -3/35 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2),
            ], axis=0),
        ))
        T = xi_W[0]
        V1 = xi_W[1] - 0.2 * T
        V3 = xi_W[2] - 0.2 * T

        corrs = jnp.stack([X22, Y22, X13, Y13, V1, V3, T], axis=0)
        moments = {'S22': (source_22,) + moment_22, 'S13': (source_13,) + moment_13}
        return corrs, moments

    def _get_corrs_bias(self, Qs, Rs, xi_ln):
        """Bias rows from the padded Q/R sources and the core P -> xi output.

        ``xi_ln`` is the *core* P -> xi output: every row built from it (U20,
        X/Y_Upsilon, chi, zeta, V12) is a purely algebraic q-space combination.
        Returns ``(corrs, moments)`` like :meth:`_get_corrs_matter_1loop`.
        """
        Q1, Q2, Q5 = Qs[0], Qs[1], Qs[2]
        R1, R2, F_G2_raw = Rs[0], Rs[1], Rs[2]

        source_10a = 2/7 * (Q5 + 2 * R2)
        source_10b = 1/7 * (2 * Q5 + 3 * R1 + 4 * R2)
        xi_bias_padded = self._get_xi_ln_batched(
            jnp.array([1, 1, 0, 2, 1, 1, 0]),
            jnp.array([-1, -1, -2, -2, -1, -1, 0]),
            jnp.stack([
                -5/21 * R1,
                -6/7 * (R1 + R2),
                source_10a,
                source_10b,
                3/7 * Q1,
                -2/5 * F_G2_raw,
                2/5 * F_G2_raw,
            ], axis=0),
        )
        # The X10 constant is the q -> 0 limit of the same ell=0 transform whose
        # q-dependent part is subtracted below, so that X10(q -> 0) = 0 holds by
        # construction.
        moment_10 = self._get_zero_lag_constant(source_10a, xi_bias_padded[2])
        xi_bias = self._grid.crop(xi_bias_padded)

        U3, U11 = xi_bias[0], xi_bias[1]
        xi_ln_A10_0m2, xi_ln_A10_2m2 = xi_bias[2], xi_bias[3]
        X10 = moment_10[0] - xi_ln_A10_0m2 - xi_ln_A10_2m2
        Y10 = 3 * xi_ln_A10_2m2

        # U20 from Ulin via Schmittfull-Vlah identity
        U_lin = -xi_ln[1, -1]
        U20 = self._get_U20_from_Ulin(U_lin)

        # X_Upsilon, Y_Upsilon from J2/J3 algebraic identities
        J2 = 2/15 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
        J3 = -1/5 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
        X_Upsilon = 4 * J3**2
        Y_Upsilon = 12 * J2**2 - 4 * J3**2 - (4.0/3.0) * U_lin**2

        V10 = xi_bias[4]
        V12 = self._get_V12_G2_from_V10(V10, xi_ln)
        chi = 4/3 * (xi_ln[2,0]**2 - xi_ln[0,0]**2)
        zeta = 2 * (8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2)

        Ub3 = xi_bias[5]
        theta = xi_bias[6]

        corrs = jnp.stack([U3, U11, U20, X10, Y10, V10, V12, X_Upsilon, Y_Upsilon,
                           chi, zeta, Ub3, theta], axis=0)
        return corrs, {'S10': (source_10a,) + moment_10}

    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data, k_IR=0.2):
        """The 32 q-space rows (``corrs``) the final (k, mu) integral needs.

        ``k_IR`` (h/Mpc) is the scale of the split of the linear spectrum into
        ``P_lt = P e^{-(k/k_IR)^2}`` and the rest, which is kept exponentiated
        and expanded respectively inside the LPT body.  It is a different
        quantity from ``lambda_ir``, the upper limit of the Sigma^2 integral of
        the IR-resummed linear counterterm base.

        Rows: 0-7 tree (X/Y_lin, X/Y_lin_lt, X/Y_lin_gt, xi_lin, U_lin), 8-14
        matter one-loop, 15-27 bias, 28/29 chi/zeta DC corrections, 30 the b2^2
        residual ``H_0[xi^2/2](k) - I0``, 31 the constant ``I0``; all on the
        core q grid (the DC rows on the core k grid).
        """
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_lin_lt = pk_lin * jnp.exp(-(self._k / k_IR)**2)

        # Band-limited zero-lag moments of the very arrays the forward FFTlog
        # transforms; see ``_get_grid_pk_int``.
        pk_int = self._get_grid_pk_int(pk_lin)
        pk_int_lt = self._get_grid_pk_int(pk_lin_lt)

        # The P -> xi transform.  Both rows use the same input model
        # (``pad_mode``) and the same trapezoid end weights.
        xi_ln_padded, xi_raw_padded = self._get_xi_ln(
            jnp.stack([pk_lin, pk_lin_lt], axis=0))
        xi_ln = self._grid.crop(xi_ln_padded[0])
        xi_ln_lt = self._grid.crop(xi_ln_padded[1])

        corrs_tree = self.get_corrs_tree(xi_ln, xi_ln_lt, pk_int, pk_int_lt)

        # The Q/R sources and their k -> q transforms.  The arrays handed to
        # the source construction additionally carry the trapezoid end weights
        # ``_end_weights_xi`` of their own support (the lower part of the
        # padded band plus the core): that support ends at the interior node
        # q_max, and for the n = -1 rows q^2 xi does not decay there, so
        # without the half weight the end node leaves an O(Delta) term in every
        # source moment.  The weights belong to the arrays that are
        # *transformed*, hence they are applied here and not to the cropped
        # core rows ``xi_ln`` / ``xi_ln_lt`` above, which are only ever used
        # pointwise.  ``_end_weights_xi`` is index-based, so the same array
        # weights the common-grid copy (used by the Q products) and the per-l
        # raw rows (used by the R transforms).
        Qs, Rs = self._get_Qs_Rs(xi_ln_padded[0] * self._end_weights_xi,
                                 xi_raw_padded[0] * self._end_weights_xi,
                                 fftlog.pad(pk_lin, self._npad, 'zero-pad'))
        corrs_matter_1loop, _ = self._get_corrs_matter_1loop(Qs, Rs)
        corrs_bias, _ = self._get_corrs_bias(Qs, Rs, xi_ln)

        chi = corrs_bias[9]
        zeta = corrs_bias[10]

        chi_dc_correction = self._get_lpt_dc_correction(chi)
        zeta_dc_correction = self._get_lpt_dc_correction(zeta)

        # b2^2 residual from the padded xi_0^0 (lower part of the padded band
        # kept), like the EPT P22 m = 0 blocks; ``corrs_tree[6]`` is its core
        # crop.
        b2sq_residual_pk, b2sq_dc = self._get_lpt_b2sq_residual_pk(
            xi_ln_padded[0][0, 0], pk_lin)
        b2sq_dc_arr = jnp.broadcast_to(b2sq_dc, self._k.shape)

        return jnp.concatenate([
            corrs_tree, corrs_matter_1loop, corrs_bias,
            chi_dc_correction[None, :],
            zeta_dc_correction[None, :],
            b2sq_residual_pk[None, :],
            b2sq_dc_arr[None, :],
        ], axis=0)

    def get_corrs_diagnostics(self, pk_data):
        """Breakdown of the zero-lag constants of the loop correlators.

        For each of the three Q/R sources whose ``q -> 0`` constant enters
        X22 / X13 / X10 (``'S22'``, ``'S13'``, ``'S10'``), the returned dict
        holds ``M_hat`` (the adopted value), ``D0_at_q_star``, ``M_2``,
        ``M_4`` and ``remainder_bound`` (bound on the neglected sixth-order
        Taylor term); see :meth:`_get_zero_lag_constant`.  ``q_star`` and
        ``i_q_star`` give the reading node.  The constants do not depend on
        ``k_IR``.  Not jitted: this is a diagnostic.
        """
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        xi_ln_padded, xi_raw_padded = self._get_xi_ln(pk_lin[None, :])
        Qs, Rs = self._get_Qs_Rs(xi_ln_padded[0] * self._end_weights_xi,
                                 xi_raw_padded[0] * self._end_weights_xi,
                                 fftlog.pad(pk_lin, self._npad, 'zero-pad'))
        _, moments = self._get_corrs_matter_1loop(Qs, Rs)
        moments.update(self._get_corrs_bias(Qs, Rs, self._grid.crop(xi_ln_padded[0]))[1])

        out = {}
        for name, (source, M_hat, D0_star, M_2, M_4, remainder) in moments.items():
            out[name] = dict(
                M_hat=M_hat,
                D0_at_q_star=D0_star,
                M_2=M_2,
                M_4=M_4,
                remainder_bound=remainder,
            )
        out['q_star'] = self._q_star
        out['i_q_star'] = self._iq_star
        return out

    def _get_lpt_dc_terms(self, corrs):
        chi_dc_correction = corrs[28]
        zeta_dc_correction = corrs[29]
        b2sq_residual_pk = corrs[30]
        b2sq_dc = corrs[31, 0]
        return chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc

    def _apply_lpt_dc_to_term_vector(
        self, pkmu_vals, i0, t, chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc,
        chi_index, zeta_index, b2sq_index,
    ):
        pkmu_vals = pkmu_vals.at[chi_index].add(self._interp_k_array(chi_dc_correction, i0, t))
        pkmu_vals = pkmu_vals.at[zeta_index].add(self._interp_k_array(zeta_dc_correction, i0, t))
        pkmu_vals = pkmu_vals.at[b2sq_index].add(self._interp_k_array(b2sq_residual_pk, i0, t))
        if not self._subtract_k0_limit:
            pkmu_vals = pkmu_vals.at[b2sq_index].add(b2sq_dc)
        return pkmu_vals

    def _get_lpt_dc_scalar(self, bias_facs, i0, t, chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc):
        val = jnp.array(0.0, dtype=self._k.dtype)
        val = val + bias_facs[8] * self._interp_k_array(chi_dc_correction, i0, t)
        val = val + bias_facs[9] * self._interp_k_array(zeta_dc_correction, i0, t)
        val = val + bias_facs[5] * self._interp_k_array(b2sq_residual_pk, i0, t)
        if not self._subtract_k0_limit:
            val = val + bias_facs[5] * b2sq_dc
        return val

    def _get_lpt_kinematics(self, k_i, mu_j, f, V_mu, corrs_tree):
        q = self._q
        X_lin_lt, Y_lin_lt = corrs_tree[2], corrs_tree[3]

        V_mu, sin_mu = V_mu

        Kfac = jnp.sqrt(1 + f * (2 + f) * mu_j**2)
        K = k_i * Kfac
        Ksq = K**2
        c = (1 + f * mu_j**2) / Kfac
        s = f * mu_j * sin_mu / Kfac
        A_mu = (1 + f) * mu_j / Kfac
        B_mu = sin_mu / Kfac

        A = k_i * q * c
        B = -0.5 * Ksq * Y_lin_lt
        C = k_i * q * s
        base = jnp.exp(-0.5 * Ksq * (X_lin_lt + Y_lin_lt))
        weights = self._get_lpt_weights(k_i)
        moments = get_lpt_moments(
            A, B, C, c**2, s**2, A_mu, B_mu, self.G00_coeffs, self.lmax, V_mu,
            c=c, s=s,
        )
        return Kfac, K, Ksq, base, weights, moments

    def _lpt_moment_coefficient_parts(
        self, k_i, mu_j, f, Kfac, K, Ksq, corrs_tree, corrs_matter_1loop, corrs_bias
    ):
        """Coefficient of each moment in each of the 23 named sub-integrands.

        This is the single definition of the LPT integrand algebra.  Each
        sub-integrand is *linear* in the nine moment arrays returned by
        ``utils_lpt.get_lpt_moments``, so rather than materialising one
        ``(L, nq)`` array per sub-integrand the algebra is carried as its
        q-dependent coefficients: entry ``j`` of the dictionary returned for a
        part is the ``(nq,)`` factor multiplying moment ``j`` in that part, and
        absent keys are exact zeros.  A caller that only wants a weighted sum of
        the parts can therefore combine the coefficients first -- at most nine
        ``(nq,)`` vectors -- and touch the ``(L, nq)`` moments exactly once,
        which is what the bias-folded hot path
        ``_get_lpt_bias_combined_integrand`` does.

        The comment above each entry restates the sub-integrand in its
        ``(L, nq)`` form so that the coefficients can be checked term by term.
        """
        X_lin_gt, Y_lin_gt = corrs_tree[4], corrs_tree[5]
        xi_lin, U_lin = corrs_tree[6], corrs_tree[7]
        X22, Y22, X13, Y13, V1, V3, T = corrs_matter_1loop
        U3, U11, U20, X10, Y10 = corrs_bias[0:5]
        V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta = corrs_bias[5:]

        mu2 = mu_j**2

        # integrand_ZA = mq0 - 0.5*Ksq*(mq0*X_lin_gt + mq2*Y_lin_gt)
        ZA = {MQ0: 1.0 - 0.5 * Ksq * X_lin_gt, MQ2: -0.5 * Ksq * Y_lin_gt}

        # integrand_AA = Ksq^2/8 * (mq0*X_gt^2 + 2*mq2*X_gt*Y_gt + mq4*Y_gt^2)
        aa = Ksq**2 / 8.0
        AA = {MQ0: aa * X_lin_gt**2,
              MQ2: aa * 2.0 * X_lin_gt * Y_lin_gt,
              MQ4: aa * Y_lin_gt**2}

        # integrand_A22 = -0.5*k^2*((Kfac^2 + 2 f (1+f) mu^2 + f^2 mu^2)*mq0*X22
        #                 + (Kfac^2*mq2 + 2 f Kfac mu*mq1_nq1 + f^2 mu^2*nq2)*Y22)
        p22 = -0.5 * k_i**2
        A22 = {MQ0: p22 * (Kfac**2 + 2 * f * (1 + f) * mu2 + f**2 * mu2) * X22,
               MQ2: p22 * Kfac**2 * Y22,
               MQ1NQ1: p22 * 2 * f * Kfac * mu_j * Y22,
               NQ2: p22 * f**2 * mu2 * Y22}

        # integrand_A13 = -0.5*k^2*(2*(Kfac^2 + 2 f (1+f) mu^2)*mq0*X13
        #                 + 2*(Kfac^2*mq2 + 2 f Kfac mu*mq1_nq1)*Y13)
        A13 = {MQ0: p22 * 2 * (Kfac**2 + 2 * f * (1 + f) * mu2) * X13,
               MQ2: p22 * 2 * Kfac**2 * Y13,
               MQ1NQ1: p22 * 2 * 2 * f * Kfac * mu_j * Y13}

        # integrand_W112 = 0.5*k^3*(2*Kfac*(Kfac^2 + f (1+f) mu^2)*mq1*V1
        #                  + Kfac^2*(Kfac*mq1 + f mu*nq1)*V3
        #                  + Kfac^2*(Kfac*mq3 + f mu*mq2_nq1)*T)
        p112 = 0.5 * k_i**3
        W112 = {MQ1: p112 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu2) * V1 + Kfac**2 * Kfac * V3),
                NQ1: p112 * Kfac**2 * f * mu_j * V3,
                MQ3: p112 * Kfac**2 * Kfac * T,
                MQ2NQ1: p112 * Kfac**2 * f * mu_j * T}

        # integrand_U10 = -2*(K*mq1*(U_lin + U3) + 2 f k mu*nq1*U3)
        U10 = {MQ1: -2.0 * K * (U_lin + U3), NQ1: -4.0 * f * k_i * mu_j * U3}

        # integrand_A_U = Ksq*(mq1*X_gt + mq3*Y_gt)*(K*U_lin)
        A_U = {MQ1: Ksq * K * X_lin_gt * U_lin, MQ3: Ksq * K * Y_lin_gt * U_lin}

        # integrand_A10 = -Ksq*(X10*mq0 + Y10*mq2)
        #                 - f*k^2*mu*((1+f)*mu*mq0*X10 + Kfac*mq1_nq1*Y10)
        A10 = {MQ0: -Ksq * X10 - f * k_i**2 * mu_j * (1 + f) * mu_j * X10,
               MQ2: -Ksq * Y10,
               MQ1NQ1: -f * k_i**2 * mu_j * Kfac * Y10}

        # integrand_xi = mq0*xi_lin
        XI = {MQ0: xi_lin}
        # integrand_A_xi = -0.5*Ksq*(mq0*X_gt + mq2*Y_gt)*xi_lin
        A_XI = {MQ0: -0.5 * Ksq * X_lin_gt * xi_lin, MQ2: -0.5 * Ksq * Y_lin_gt * xi_lin}
        # integrand_U_U = -Ksq*mq2*U_lin^2
        U_U = {MQ2: -Ksq * U_lin**2}
        # integrand_U11 = -(K*mq1 + f k mu*nq1)*U11
        U11p = {MQ1: -K * U11, NQ1: -f * k_i * mu_j * U11}
        # integrand_U20 = -(K*mq1 + f k mu*nq1)*U20
        U20p = {MQ1: -K * U20, NQ1: -f * k_i * mu_j * U20}
        # integrand_xi_U = -2*K*mq1*xi_lin*U_lin
        XI_U = {MQ1: -2.0 * K * xi_lin * U_lin}
        # integrand_xi_xi = 0.5*mq0*xi_lin^2
        XI_XI = {MQ0: 0.5 * xi_lin**2}

        # integrand_Upsilon = -Ksq*(mq0*X_Upsilon + mq2*Y_Upsilon)
        UPS = {MQ0: -Ksq * X_Upsilon, MQ2: -Ksq * Y_Upsilon}
        # integrand_V10 = -2*(K*mq1 + f k mu*nq1)*V10
        V10p = {MQ1: -2.0 * K * V10, NQ1: -2.0 * f * k_i * mu_j * V10}
        # integrand_V12 = -2*K*mq1*V12
        V12p = {MQ1: -2.0 * K * V12}
        # integrand_chi = mq0*chi
        CHI = {MQ0: chi}
        # integrand_zeta = mq0*zeta
        ZETA = {MQ0: zeta}
        # integrand_Ub3 = -2*K*mq1*Ub3
        UB3 = {MQ1: -2.0 * K * Ub3}
        # integrand_theta = 2*mq0*theta
        THETA = {MQ0: 2.0 * theta}

        # The diagnostic counterterm component is the Zel'dovich base, i.e.
        # integrand_ctr is literally integrand_ZA.  Linear-family counterterms
        # need raw pk_data and are therefore assembled only in the high-level
        # pk_data paths, not from a bare corrs array.
        CTR = {MQ0: 1.0 - 0.5 * Ksq * X_lin_gt, MQ2: -0.5 * Ksq * Y_lin_gt}

        return (ZA, AA, A22, A13, W112, U10, A_U, A10, XI, A_XI, U_U, U11p,
                U20p, XI_U, XI_XI, UPS, V10p, V12p, CHI, ZETA, UB3, THETA,
                CTR)

    def _lpt_template_coefficients(self, parts):
        """Group the 23 coefficient dictionaries into the 13 templates.

        Same grouping as the ``(L, nq)`` form used to apply, but performed on
        the coefficients, so a caller that only wants a weighted sum of the
        templates never builds a single ``(L, nq)`` array per template.
        """
        (ZA, AA, A22, A13, W112, U10, A_U, A10, XI, A_XI, U_U, U11, U20, XI_U,
         XI_XI, UPS, V10, V12, CHI, ZETA, UB3, THETA, CTR) = parts
        return (
            _sum_coeffs([ZA, AA, A22, A13, W112]),   #  0  1
            _sum_coeffs([U10, A_U, A10]),            #  1  b1
            _sum_coeffs([XI, A_XI, U_U, U11]),       #  2  b1^2
            _sum_coeffs([U_U, U20]),                 #  3  b2
            XI_U,                                    #  4  b1 b2
            XI_XI,                                   #  5  b2^2
            _sum_coeffs([UPS, V10]),                 #  6  bG2
            V12,                                     #  7  b1 bG2
            CHI,                                     #  8  b2 bG2
            ZETA,                                    #  9  bG2^2
            UB3,                                     # 10  bGamma3
            THETA,                                   # 11  b1 bGamma3
            CTR,                                     # 12  counterterm base (Zel'dovich)
        )

    def _get_lpt_sub_integrands(
        self, k_i, mu_j, f, V_mu, corrs_tree, corrs_matter_1loop, corrs_bias
    ):
        """Return the 23 named sub-integrands as a plain tuple of (L, nq) arrays.

        Thin wrapper around ``_lpt_moment_coefficient_parts``: it contracts each
        part's coefficients with the moments.  Only the diagnostic component
        path needs all 24 arrays separately; the template and bias-folded paths
        combine the coefficients first and never build them.
        """
        Kfac, K, Ksq, base, weights, moments = self._get_lpt_kinematics(
            k_i, mu_j, f, V_mu, corrs_tree
        )
        parts = self._lpt_moment_coefficient_parts(
            k_i, mu_j, f, Kfac, K, Ksq, corrs_tree, corrs_matter_1loop, corrs_bias
        )
        integrands = tuple(_apply_coeffs(coeffs, moments) for coeffs in parts)
        return integrands, base, weights

    @staticmethod
    def _get_lpt_template_weights(bias_facs, ctr_k2_shape, nlo_shape):
        """Weights for the 13 templates, shared by the fused and template paths.

        The 12 bias monomials, then the counterterm base: both the leading
        ``k^2`` and the NLO ``k^4`` shapes ride on the same template 12.
        """
        return [bias_facs[i] for i in range(12)] + [ctr_k2_shape + nlo_shape]

    @partial(jit, static_argnames=['self'])
    def get_pkmu_components(self, k, mu, pk_data, f, k_IR=0.2):
        """The 23 named LPT components on the (k, mu) grid (diagnostic).

        ``k_IR`` is the scale of the lt/gt split of the linear spectrum inside the
        LPT body (see :meth:`get_corrs`), not the Sigma^2 limit ``lambda_ir``.
        """
        if self._counterterms.needs_kspace_base:
            raise ValueError(
                "the counterterm component (index 22) is the Zel'dovich base, which is "
                f"not the counterterm of counterterm_base={self.counterterm_base!r}: that "
                "one lives in k space and needs the raw pk_data, not corrs.  Build a "
                "separate LPT(counterterm_base='zeldovich') for this diagnostic (the "
                "other 23 components do not depend on the choice), or use "
                "get_pkmu/get_pk_ells for the full spectrum."
            )
        k  = jnp.atleast_1d(k)
        mu = jnp.abs(jnp.atleast_1d(mu))  # P even in mu; moments need mu >= 0

        corrs = self.get_corrs(pk_data, k_IR)
        corrs_tree, corrs_matter_1loop, corrs_bias = corrs[0:8], corrs[8:15], corrs[15:28]
        chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc = self._get_lpt_dc_terms(corrs)

        logk_fft = self._logk_fft
        sin_mu = jnp.sqrt(jnp.maximum(1 - mu**2, 0.0))
        V_all_mu = jax.vmap(
            lambda mu_j, sin_j: self._get_V_mu_for_mu(mu_j, f, sin_j)
        )(mu, sin_mu)

        def per_k(k_i):
            logk = jnp.log(k_i)
            # preparation for linear interpolation
            i0   = jnp.searchsorted(logk_fft, logk, side='right') - 1
            i0   = jnp.clip(i0, 0, logk_fft.size - 2)
            t    = (logk - logk_fft[i0]) / (logk_fft[i0+1] - logk_fft[i0])

            def per_mu(mu_j, V_mu):
                parts, base, weights = self._get_lpt_sub_integrands(
                    k_i, mu_j, f, V_mu, corrs_tree, corrs_matter_1loop, corrs_bias
                )
                integrands = jnp.stack(parts, axis=0)
                g = integrands * (base[None, None, :] * weights[None, :, :]) # (ncomp, L, nq)
                g = g.at[14, 0, :].add(-0.5 * corrs_tree[6]**2)
                g = g.at[:,0,:].subtract(g[:,0,-1][:,None])
                g = self._4pi_q3[None, None, :] * g

                vals = self.get_pk_batched_interp(g, i0, t)
                pkmu_vals = jnp.sum(vals, axis=1)  # (ncomp,)
                return self._apply_lpt_dc_to_term_vector(
                    pkmu_vals, i0, t, chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc,
                    chi_index=18, zeta_index=19, b2sq_index=14,
                )

            return jax.vmap(per_mu)(mu, V_all_mu)  # (nmu, ncomp)

        pkmu_terms = jax.vmap(per_k)(k)                    # (nk, nmu, ncomp)
        pkmu_terms = jnp.transpose(pkmu_terms, (2, 0, 1))  # (ncomp, nk, nmu)

        return pkmu_terms

    def _get_lpt_weighted_term_integrands(
        self, k_i, mu_j, f, V_mu, corrs_tree, corrs_matter_1loop, corrs_bias
    ):
        """The 15 LPT templates multiplied by ``base`` and the ell weights.

        The 24 sub-integrands are never built: the templates are assembled from
        the moment coefficients (:meth:`_lpt_template_coefficients`) and each
        one is contracted with the moments once.
        """
        Kfac, K, Ksq, base, w, moments = self._get_lpt_kinematics(
            k_i, mu_j, f, V_mu, corrs_tree
        )
        parts = self._lpt_moment_coefficient_parts(
            k_i, mu_j, f, Kfac, K, Ksq, corrs_tree, corrs_matter_1loop, corrs_bias
        )
        templates = self._lpt_template_coefficients(parts)
        integrands = jnp.stack(
            [_apply_coeffs(coeffs, moments) for coeffs in templates], axis=0
        )
        weighted = integrands * (base[None, None, :] * w[None, :, :])
        xi_lin = corrs_tree[6]
        return weighted.at[5, 0, :].add(-0.5 * xi_lin**2)

    def _get_lpt_bias_combined_integrand(
        self, k_i, mu_j, f, V_mu, bias_facs, ctr_k2_shape, nlo_shape,
        corrs_tree, corrs_matter_1loop, corrs_bias,
    ):
        """Production hot path: fold the bias factors and both counterterms into
        a single ``(L, nq)`` integrand inside the per-(k, mu) function.

        Exactly the same algebra as the template path, but carried out on the
        moment coefficients: the 13 templates are folded with their bias /
        counterterm weights into at most nine ``(nq,)`` coefficient vectors, and
        only then contracted with the ``(L, nq)`` moments.  Neither the
        ``(13, L, nq)`` stack of ``_get_lpt_weighted_term_integrands`` nor the
        24 individual sub-integrands are ever materialised, so the per-(k, mu)
        integrand costs at most nine multiply-adds on ``(L, nq)`` arrays instead
        of one per sub-integrand.
        """
        Kfac, K, Ksq, base, w, moments = self._get_lpt_kinematics(
            k_i, mu_j, f, V_mu, corrs_tree
        )
        parts = self._lpt_moment_coefficient_parts(
            k_i, mu_j, f, Kfac, K, Ksq, corrs_tree, corrs_matter_1loop, corrs_bias
        )
        templates = self._lpt_template_coefficients(parts)
        coeffs = _weighted_sum_coeffs(
            templates,
            self._get_lpt_template_weights(bias_facs, ctr_k2_shape, nlo_shape),
        )
        integrand = _apply_coeffs(coeffs, moments) * (base[None, :] * w)
        # DC split for b2^2: the final Hankel/dot-product path sees only the
        # scale-dependent part; H_0[0.5 xi^2] - I0 and the optional I0 add-back
        # are handled by _get_lpt_dc_scalar.
        xi_lin = corrs_tree[6]
        return integrand.at[0, :].add(-bias_facs[5] * 0.5 * xi_lin**2)

    @partial(jit, static_argnames=['self'])
    def get_pkmu_terms_from_corrs(self, k, mu, corrs, f, alpha_perp=1.0, alpha_para=1.0):
        """Bias-independent LPT templates on the (k, mu) grid.

        Returns ``(ncomp, nk, nmu)``: the 12 bias monomial templates followed by
        the three tree templates.  Contract them with
        :meth:`combine_pkmu_terms`; they depend only on the cosmology and ``f``,
        so one evaluation serves every bias/counterterm/stochastic sample and
        every tracer pair.
        """
        if self._counterterms.needs_kspace_base:
            raise ValueError(
                "bare corrs do not contain the raw linear spectrum required by "
                f"counterterm_base={self.counterterm_base!r}.  Use "
                "LPT(counterterm_base='zeldovich') for the template path, or "
                "get_pkmu/get_pk_ells, which take pk_data."
            )
        self._require_corrs_rows(corrs)
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        corrs_tree, corrs_matter_1loop, corrs_bias = corrs[0:8], corrs[8:15], corrs[15:28]
        chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc = self._get_lpt_dc_terms(corrs)

        logk_fft = self._logk_fft
        k_true, mu_true, sin_true = get_k_mu_true_sin_for_ap(
            k, mu, alpha_perp, alpha_para
        )
        # P(k, mu) is even in mu (plane-parallel); the internal moment algebra uses sqrt(s^2)=|s| and is only consistent for mu >= 0, so fold here.
        mu_true = jnp.abs(mu_true)
        V_all_mu = jax.vmap(
            lambda mu_j, sin_j: self._get_V_mu_for_mu(mu_j, f, sin_j)
        )(mu_true, sin_true)
        mu_true = jnp.broadcast_to(mu_true, k_true.shape)

        def per_point(k_i, mu_j, V_mu):
            logk = jnp.log(k_i)
            i0 = jnp.searchsorted(logk_fft, logk, side='right') - 1
            i0 = jnp.clip(i0, 0, logk_fft.size - 2)
            t = (logk - logk_fft[i0]) / (logk_fft[i0 + 1] - logk_fft[i0])

            g = self._get_lpt_weighted_term_integrands(
                k_i, mu_j, f, V_mu, corrs_tree, corrs_matter_1loop, corrs_bias
            )
            g = g.at[:, 0, :].subtract(g[:, 0, -1][:, None])
            g = self._4pi_q3[None, None, :] * g

            vals = self.get_pk_batched_interp(g, i0, t)
            pkmu_vals = jnp.sum(vals, axis=1)
            return self._apply_lpt_dc_to_term_vector(
                pkmu_vals, i0, t, chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc,
                chi_index=8, zeta_index=9, b2sq_index=5,
            )

        # Interpolate inside the per-point function: the (nmu, ncomp, L, nq)
        # integrand stack is never materialised.
        def per_k(k_true_row, mu_true_row):
            return jax.vmap(per_point)(k_true_row, mu_true_row, V_all_mu)

        pkmu_terms = jax.vmap(per_k)(k_true, mu_true)     # (nk, nmu, ncomp)
        return jnp.transpose(pkmu_terms, (2, 0, 1))       # (ncomp, nk, nmu)

    def get_pkmu_terms(self, k, mu, pk_data, f, k_IR=0.2, alpha_perp=1.0, alpha_para=1.0):
        """Bias-independent templates from ``pk_data`` (see :meth:`get_pkmu_terms_from_corrs`).

        ``k_IR`` is the scale of the lt/gt split of the linear spectrum inside the
        LPT body (see :meth:`get_corrs`), not the Sigma^2 limit ``lambda_ir``.
        """
        corrs = self.get_corrs(pk_data, k_IR)
        return self.get_pkmu_terms_from_corrs(k, mu, corrs, f, alpha_perp, alpha_para)

    @partial(jit, static_argnames=['self'])
    def combine_pkmu_terms(self, k, mu, pkmu_terms, params, alpha_perp=1.0, alpha_para=1.0):
        if self._counterterms.needs_kspace_base:
            raise ValueError(
                "the counterterm template (index 12) is the Zel'dovich base, which is "
                f"not the counterterm of counterterm_base={self.counterterm_base!r}.  Use "
                "LPT(counterterm_base='zeldovich') for the template path, or "
                "get_pkmu/get_pk_ells, which take pk_data."
            )
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        f = params.f
        bias_facs = self._get_lpt_bias_factors(params.bias, params.bias_b)

        # mapping of (k, mu)
        k_true, mu_true = get_k_mu_true_for_ap(k, mu, alpha_perp, alpha_para)
        mu_true = jnp.broadcast_to(mu_true, k_true.shape)  # (nk, nmu)

        pkmu = jnp.tensordot(bias_facs, pkmu_terms[:12], axes=(0, 0))   # (nk, nmu)

        # Counterterms: both the leading k^2 and the NLO k^4 shapes ride on the
        # Zel'dovich base (template 12).
        pkmu_ctr = self._counterterms.leading(
            k_true, mu_true, f, params.ctr, pkmu_terms[12]
        )
        pkmu_ctr = pkmu_ctr + self._counterterms.nlo(
            k_true, mu_true, f, params.ctr, pkmu_terms[12]
        )
        pkmu = pkmu + pkmu_ctr

        # stochasticity
        pkmu_stoch = stochasticity(k_true, mu_true, params.stoch)
        pkmu = pkmu + pkmu_stoch

        pkmu = pkmu / (alpha_perp**2 * alpha_para)
        return pkmu

    @partial(jit, static_argnames=['self'])
    def _get_kspace_counterterm_base(self, k, mu, pk_data, params):
        if not self._counterterms.needs_kspace_base:
            raise ValueError("the Zel'dovich counterterm base is evaluated inside the LPT integral")

        pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        if self.counterterm_base == 'linear':
            return pk

        # wiggly-non-wiggly decomposition
        pk_nw_data = ir_resum.get_pk_nw(pk_data, params.h, method=self.irres_method)
        pk_nw = get_pk(k, pk_nw_data, kmin=self._kmin, kmax=self._kmax)
        pk_w = pk - pk_nw

        # BAO damping factor in redshift space
        f = params.f
        Sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.r_bao, self.lambda_ir)
        dSigma2 = ir_resum.get_dSigma2(pk_nw_data, self.r_bao, self.lambda_ir)
        Sigma2_s = (1 + mu**2 * f * (2 + f)) * Sigma2 + f**2 * mu**2 * (mu**2 - 1) * dSigma2

        return pk_nw + jnp.exp(-k**2 * Sigma2_s) * pk_w

    @partial(jit, static_argnames=['self'])
    def get_pkmu_from_corrs(self, k, mu, corrs, params, alpha_perp=1.0, alpha_para=1.0,
                            pk_data=None):
        if self._counterterms.needs_kspace_base and pk_data is None:
            raise ValueError(
                "pk_data is required when evaluating a linear-family counterterm from corrs"
            )
        self._require_corrs_rows(corrs)
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        f = params.f
        bias_facs = self._get_lpt_bias_factors(params.bias, params.bias_b)
        ctr_leading, ctr_nlo = Counterterms.split_coefficients(params.ctr)
        corrs_tree, corrs_matter_1loop, corrs_bias = corrs[0:8], corrs[8:15], corrs[15:28]
        chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc = self._get_lpt_dc_terms(corrs)

        logk_fft = self._logk_fft
        k_true, mu_true, sin_true = get_k_mu_true_sin_for_ap(
            k, mu, alpha_perp, alpha_para
        )
        # P(k, mu) is even in mu; internal moments are mu>=0-consistent only.
        mu_true = jnp.abs(mu_true)
        V_all_mu = jax.vmap(
            lambda mu_j, sin_j: self._get_V_mu_for_mu(mu_j, f, sin_j)
        )(mu_true, sin_true)
        mu_true = jnp.broadcast_to(mu_true, k_true.shape)

        def per_k(k_true_row, mu_true_row):
            def per_mu(k_i, mu_j, V_mu):
                logk = jnp.log(k_i)
                i0 = jnp.searchsorted(logk_fft, logk, side='right') - 1
                i0 = jnp.clip(i0, 0, logk_fft.size - 2)
                t = (logk - logk_fft[i0]) / (logk_fft[i0 + 1] - logk_fft[i0])

                # For the Zel'dovich base both counterterms are folded into the
                # same (L, nq) integrand, so they cost no extra transform.  The
                # linear-family bases are added in k space after the loop.
                if self._counterterms.needs_kspace_base:
                    ctr_k2_shape = jnp.zeros_like(k_i)
                    nlo_shape = jnp.zeros_like(k_i)
                else:
                    ctr_k2_shape = self._counterterms.leading_shape(k_i, mu_j, f, ctr_leading)
                    nlo_shape = Counterterms.nlo_shape(k_i, mu_j, f, ctr_nlo)
                # Fast path: the bias factors and both counterterms are folded
                # into the moment coefficients, so this builds one (L, nq)
                # integrand and never a per-template stack.
                integrand = self._get_lpt_bias_combined_integrand(
                    k_i, mu_j, f, V_mu, bias_facs, ctr_k2_shape, nlo_shape,
                    corrs_tree, corrs_matter_1loop, corrs_bias,
                )
                g = integrand.at[0, :].subtract(integrand[0, -1])
                g = self._4pi_q3[None, :] * g

                stoch_kmu = stochasticity(k_i, mu_j, params.stoch)
                return g, i0, t, stoch_kmu

            gs, i0s, ts, stochs = jax.vmap(per_mu)(k_true_row, mu_true_row, V_all_mu)

            def interp_mu(g_mu, i0, t):
                val = jnp.sum(self.get_pk_batched_interp(g_mu, i0, t))
                return val + self._get_lpt_dc_scalar(
                    bias_facs, i0, t, chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc
                )

            return jax.vmap(interp_mu)(gs, i0s, ts) + stochs

        pkmu = jax.vmap(per_k)(k_true, mu_true)
        if self._counterterms.needs_kspace_base:
            base_pk = self._get_kspace_counterterm_base(k_true, mu_true, pk_data, params)
            pkmu = pkmu + self._counterterms.leading(k_true, mu_true, f, params.ctr, base_pk)
            pkmu = pkmu + self._counterterms.nlo(k_true, mu_true, f, params.ctr, base_pk)

        return pkmu / (alpha_perp**2 * alpha_para)

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=0.2):
        """P(k, mu) at the requested (k, mu), AP distortion included.

        ``k_IR`` is the scale of the lt/gt split of the linear spectrum inside the
        LPT body (see :meth:`get_corrs`), not the Sigma^2 limit ``lambda_ir``.
        """
        corrs = self.get_corrs(pk_data, k_IR)
        return self.get_pkmu_from_corrs(
            k, mu, corrs, params, alpha_perp, alpha_para, pk_data=pk_data
        )

    @partial(jit, static_argnames=['self'])
    def get_pk_ells_from_corrs(self, k, corrs, params, alpha_perp=1.0, alpha_para=1.0,
                               pk_data=None):
        pkmu = self.get_pkmu_from_corrs(
            k, self._mu_quad, corrs, params, alpha_perp, alpha_para, pk_data=pk_data,
        )
        pk_ells = get_legendre_multipoles(pkmu, self._legendre_weights)  # (3, nk)
        return pk_ells

    def get_pk_ells(self, k, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=0.2):
        """Legendre multipoles P_0, P_2, P_4, shape (3, nk).

        ``k_IR`` is the scale of the lt/gt split of the linear spectrum inside the
        LPT body (see :meth:`get_corrs`), not the Sigma^2 limit ``lambda_ir``.
        """
        corrs = self.get_corrs(pk_data, k_IR)
        return self.get_pk_ells_from_corrs(
            k, corrs, params, alpha_perp, alpha_para, pk_data=pk_data
        )

    def get_corrs_tree(self, xi_ln, xi_ln_lt, pk_int, pk_int_lt):
        # X_lin = 2/3 * (xi_ln[0,-2][0] - xi_ln[0,-2] - xi_ln[2,-2])
        X_lin = 2/3 * (pk_int - xi_ln[0,-2] - xi_ln[2,-2])
        Y_lin = 2 * xi_ln[2,-2]

        # X_lin_lt = 2/3 * (xi_ln_lt[0,-2][0] - xi_ln_lt[0,-2] - xi_ln_lt[2,-2])
        X_lin_lt = 2/3 * (pk_int_lt - xi_ln_lt[0,-2] - xi_ln_lt[2,-2])
        Y_lin_lt = 2 * xi_ln_lt[2,-2]

        X_lin_gt = X_lin - X_lin_lt
        Y_lin_gt = Y_lin - Y_lin_lt

        xi_lin = xi_ln[0,0]
        U_lin = - xi_ln[1,-1]

        corrs = jnp.stack([X_lin, Y_lin, X_lin_lt, Y_lin_lt,
                           X_lin_gt, Y_lin_gt, xi_lin, U_lin], axis=0)
        return corrs

    def _to_bG2_basis(self, bias):
        """Map one tracer's bias vector onto the (b1, b2, bG2, bGamma3) basis.

        Applied per tracer *before* the cross symmetrisation: the bs2 -> bG2
        shift ``b2 -> b2 + 4/3 bs2`` is linear in a single tracer's parameters
        and does not commute with the degree-2 symmetrisation.
        """
        b1, b2, btidal, bGamma3 = bias
        if self.bias_basis == 'bs2':
            b2 = b2 + (4.0 / 3.0) * btidal
        return jnp.stack([b1, b2, btidal, bGamma3])

    def _get_lpt_bias_factors(self, bias_a, bias_b=None):
        """Bias monomials for the 12 LPT term templates.

        ``bias_b=None`` reproduces the auto spectrum.  The cross case is handled
        by :func:`~.utils.cross_bias_factor`, the helper the EPT backend also
        uses, so a single rule maps auto -> cross in both backends.
        """
        if bias_b is None:
            bias_b = bias_a
        return cross_bias_factor(
            _LPT_BIAS_DEGREES, self._to_bG2_basis(bias_a), self._to_bG2_basis(bias_b)
        )

    def _get_U20_from_Ulin(self, U_lin):
        return -(6.0/7.0) * U_lin**2 / self._q

    def _get_V12_G2_from_V10(self, V10, xi_ln):
        d_q_delta = (
            2.0 * xi_ln[3,-1] * (3.0*xi_ln[3,-1]/self._q - xi_ln[4,0])
            - 2.0 * xi_ln[1,-1] * (xi_ln[1,-1]/self._q - xi_ln[2,0])
        )
        return (14.0/3.0) * V10 - (2.0/5.0) * d_q_delta

    @partial(jit, static_argnames=['self'])
    def get_xi_ells(self, r, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=0.2):
        """Approximate configuration-space multipoles xi_0, xi_2, xi_4 at ``r``.

        ``k_IR`` is the scale of the lt/gt split of the linear spectrum inside the
        LPT body (see :meth:`get_corrs`), not the Sigma^2 limit ``lambda_ir``.
        """
        r = jnp.atleast_1d(r)

        # This helper is an approximate configuration-space projection.
        # The final LPT multipoles are not numerically stable on the full FFT grid up to kmax_fft, 
        # so use a conservative spectrum grid and interpolate it back to self._k before the Hankel transform.
        k = jnp.geomspace(max(self._kmin, 1e-4), min(self._kmax, 1.0), min(self._nfft, 128))
        pk_ells = self.get_pk_ells(k, pk_data, params, alpha_perp, alpha_para, k_IR)

        pk0 = get_pk(self._k, jnp.stack([k, pk_ells[0]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk2 = get_pk(self._k, jnp.stack([k, pk_ells[1]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk4 = get_pk(self._k, jnp.stack([k, pk_ells[2]], axis=0), kmin=self._kmin, kmax=self._kmax)

        # The multipoles on the core k grid are the zero-padded source of the
        # k -> q transform; the result is resampled onto the common q grid.
        xi_ln = self._grid.crop(self._get_xi_ln_batched(
            jnp.array([0, 2, 4]), jnp.array([0, 0, 0]),
            fftlog.pad(jnp.stack([pk0, pk2, pk4], axis=0), self._npad, 'zero-pad'),
        ))
        xi0 = spline.interp1d(jnp.log(r), jnp.log(self._q), xi_ln[0])
        xi2 = spline.interp1d(jnp.log(r), jnp.log(self._q), -xi_ln[1])
        xi4 = spline.interp1d(jnp.log(r), jnp.log(self._q), xi_ln[2])

        xi_ells = jnp.stack([xi0, xi2, xi4], axis=0)
        return xi_ells
