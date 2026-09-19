import os
import numpy as _np

import jax
from jax import jit
from functools import partial

import jax.numpy as jnp

from . import fftlog
from . import spline

from .utils import get_pk, get_pk_int, get_pk_int2, cross_bias_factor
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


class LPT:

    def __init__(self,
                 kmin_fft=1e-5,
                 kmax_fft=1e2,
                 nfft=512,
                 hankel_nu=1.1,
                 hankel_npad_factor=0.5,
                 hankel_forward_pad_mode='zero-pad',
                 hankel_backward_pad_mode='zero-pad',
                 lpt_source_forward_pad_mode='zero-pad',
                 hankel_forward_mode='pld',
                 lmax=5,
                 ngauss=4,
                 counterterm_base='linear_ir_resum',
                 counterterm_irres_method='DST',
                 counterterm_r_bao=110.0,
                 counterterm_k_IR=0.2,
                 bias_basis='bG2',
                 subtract_k0_limit=False,
                 lpt_tidal_zero_dc_residual=True,
                 ):

        if hankel_forward_mode not in ('fftlog', 'pld'):
            raise ValueError("hankel_forward_mode must be 'fftlog' or 'pld'")
        if nfft % 2:
            raise ValueError(
                "nfft must be even: the FFTLog pipeline uses rfft/irfft with an "
                "implicit even length"
            )
        if bias_basis not in ('bG2', 'bs2'):
            raise ValueError("bias_basis must be 'bG2' or 'bs2'")
        valid_pad_modes = ("power-law", "zero-pad", "smooth-zero-pad")
        self._hankel_forward_pad_mode = hankel_forward_pad_mode
        self._hankel_backward_pad_mode = hankel_backward_pad_mode
        self._lpt_source_forward_pad_mode = lpt_source_forward_pad_mode
        for name, mode in (
            ("hankel_forward_pad_mode", self._hankel_forward_pad_mode),
            ("hankel_backward_pad_mode", self._hankel_backward_pad_mode),
            ("lpt_source_forward_pad_mode", self._lpt_source_forward_pad_mode),
        ):
            if mode not in valid_pad_modes:
                raise ValueError(f"{name} must be one of {valid_pad_modes}")
        if self._hankel_backward_pad_mode == 'power-law':
            # The q-space sources change sign, so a geometric continuation over
            # the guard band overflows and every multipole comes out NaN.
            raise ValueError(
                "hankel_backward_pad_mode='power-law' is not supported for the LPT "
                "backward transforms; use 'zero-pad' (default) or 'smooth-zero-pad'"
            )

        self.lmax = lmax
        self._counterterms = Counterterms(counterterm_base)
        if counterterm_irres_method not in ('DST', 'SG', 'WH'):
            raise ValueError("counterterm_irres_method must be 'DST', 'SG', or 'WH'")
        self.counterterm_irres_method = counterterm_irres_method
        self.counterterm_r_bao = counterterm_r_bao
        self.counterterm_k_IR = counterterm_k_IR
        self.bias_basis = bias_basis
        self._subtract_k0_limit = subtract_k0_limit
        self._lpt_tidal_zero_dc_residual = lpt_tidal_zero_dc_residual

        self._hankel_forward_mode = hankel_forward_mode

        # preparation for Gauss-Legendre quadrature
        self._mu_quad, self._legendre_weights = prepare_mu_gauleg(ngauss)

        # preparation for FFT
        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self._nu_hankel = hankel_nu
        self._hankel_npad_factor = hankel_npad_factor
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
        l_list = list(range(lmax + 1))

        self._npad = max(1, int(self._hankel_npad_factor * self._nfft))
        self._grid = fftlog.LogGrid.from_core(self._k, self._npad)
        self._k_padded = self._grid.x

        # Spatial window disabled: analytic residualization handles the large-q tail.
        self._w_m = fftlog.window(0, self._k_padded)

        # Low-ringing output phase per ell
        lnxy = jnp.array([
            fftlog.low_ringing_phase(l, self._nu_hankel, self._grid)
            for l in l_list
        ])

        # Ell-specific forward output grids (used for xi_ln forward transforms)
        self._q_xi_padded = jnp.array([
            fftlog.output_grid(self._grid, lnxy[l]) for l in l_list
        ])
        self._q_xi = self._grid.crop(self._q_xi_padded)

        # Common q-grid (ell=0) for backward transforms and all q-space products
        self._q_padded = self._q_xi_padded[0]   # shape (nfft_padded,)
        self._q = self._q_xi[0]                  # shape (nfft,)
        self._ell_indices = jnp.arange(self.lmax + 1, dtype=self._q.dtype)
        self._q_inv_pows = self._q[None, :] ** (-self._ell_indices[:, None])
        # Precomputed constants used inside JIT-compiled methods
        self._4pi_q3 = 4.0 * jnp.pi * self._q**3
        self._logk_fft = jnp.log(self._k)

        # Forward kernels: ell-specific output phase, ell-specific G_l kernel
        self._u_m_xi = jnp.array([
            fftlog.hankel_kernel(l, self._nu_hankel, self._grid, lnxy[l]) for l in l_list
        ])
        # Backward kernels: ell=0 output phase, ell-specific G_l kernel
        self._u_m = jnp.array([
            fftlog.hankel_kernel(l, self._nu_hankel, self._grid, lnxy[0]) for l in l_list
        ])

        self._set_hankel_forward_pld_groups()
        self._set_pk_select_kernel()

    def _set_hankel_forward_pld_groups(self):
        ln_np = _np.asarray(self.ln_list)
        group_ns = _np.asarray(sorted(set(ln_np[:, 1].tolist())), dtype=_np.int32)
        self._xi_pld_group_ns = jnp.asarray(group_ns)
        group_lookup = {int(n): i for i, n in enumerate(group_ns)}
        self._xi_pld_group_ids = jnp.asarray(
            [group_lookup[int(n)] for n in ln_np[:, 1]],
            dtype=jnp.int32,
        )

    def _set_pk_select_kernel(self):
        if self._hankel_backward_pad_mode == 'power-law':
            self._pk_select_kernel = None
            return

        basis = jnp.eye(self._q.shape[0], dtype=self._q.dtype)

        def build_one(u_m):
            pk = fftlog.hankel(
                self._nu_hankel, basis, self._q_padded, self._k_padded,
                u_m, self._npad, self._w_m, self._hankel_backward_pad_mode,
            )
            return pk.T

        self._pk_select_kernel = jax.vmap(build_one)(self._u_m[:self.lmax + 1])

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

    def _get_grid_pk_int2(self, pk_lin):
        """Compute (1/2*pi^2) int k^2 P^2(k) dk on the internal k-grid."""
        return get_pk_int2(
            jnp.stack([self._k, pk_lin], axis=0), kmin=self._kmin, kmax=self._kmax)

    def _get_lpt_b2sq_residual_pk(self, xi_lin, pk_lin):
        """Return H_0[0.5 xi^2](k) - I0 and I0 for the b2^2 DC split."""
        source = 0.5 * xi_lin**2
        direct = self.get_pk_batched(
            self._4pi_q3[None, :] * source[None, :],
            self._u_m[:1],
        )[0]
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
        if self._pk_select_kernel is None:
            pk_ffts = self.get_pk_batched(arrays, self._u_m[:L])
            return self._interp_k_array(pk_ffts, i0, t)

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

    def get_xi_ln(self, l, n, array, pad_mode=None):
        if pad_mode is None:
            pad_mode = self._hankel_forward_pad_mode
        fx = array * self._k**(n + 3) / (2 * jnp.pi**2)
        xi_ln = fftlog.hankel(
            self._nu_hankel,
            fx,
            self._k_padded,
            self._q_xi_padded[l],
            self._u_m_xi[l],
            self._npad,
            self._w_m,
            pad_mode,
        )
        xi_ln = spline.interp1d(jnp.log(self._q), jnp.log(self._q_xi[l]), xi_ln)
        return xi_ln

    def get_xi_ln_batched(self, ells, ns, arrays, pad_mode=None):
        if pad_mode is None:
            pad_mode = self._hankel_forward_pad_mode
        ells = jnp.asarray(ells, dtype=jnp.int32)
        ns = jnp.asarray(ns)
        arrays = jnp.asarray(arrays)

        fx = arrays * self._k[None, :] ** (ns[:, None] + 3) / (2 * jnp.pi**2)
        xi_ln = fftlog.hankel(
            self._nu_hankel,
            fx,
            self._k_padded,
            self._q_xi_padded[ells],
            self._u_m_xi[ells],
            self._npad,
            self._w_m,
            pad_mode,
        )
        def interp_to_common_q(ell, xi):
            return spline.interp1d(jnp.log(self._q), jnp.log(self._q_xi[ell]), xi)

        return jax.vmap(interp_to_common_q)(ells, xi_ln)

    def get_pk_ln(self, l, n, array):
        fx = array * self._q**(n + 3)
        pk_ln = fftlog.hankel(
            self._nu_hankel, fx, self._q_padded, self._k_padded, self._u_m[l],
            self._npad, self._w_m, self._hankel_backward_pad_mode,
        )
        return pk_ln

    def get_pk_batched(self, arrays, u_m):
        pks = fftlog.hankel(
            self._nu_hankel, arrays, self._q_padded, self._k_padded, u_m,
            self._npad, self._w_m, self._hankel_backward_pad_mode,
        )
        return pks

    def get_xi_ln_array(self, array):
        if self._hankel_forward_mode == 'pld':
            return self.get_xi_ln_array_pld(array)
        ls = self.ln_list[:, 0]
        ns = self.ln_list[:, 1]
        arrays = jnp.broadcast_to(array, (self.ln_list.shape[0], array.shape[0]))
        xis = self.get_xi_ln_batched(ls, ns, arrays)
        xi_ln = jnp.zeros((5, 5, self._nfft))
        xi_ln = xi_ln.at[ls, ns].set(xis)
        return xi_ln

    def get_xi_ln_array_pld(self, array):
        c_m_by_n = self._get_forward_pld_rfft_batch(array, self._hankel_forward_pad_mode)
        c_m = c_m_by_n[self._xi_pld_group_ids]  # (n_ln, nfft//2+1)
        ells = self.ln_list[:, 0]
        xis = self._apply_forward_pld_group(c_m, ells)  # (n_ln, nq)
        xi_ln = jnp.zeros((5, 5, self._nfft))
        ls, ns = self.ln_list[:, 0], self.ln_list[:, 1]
        return xi_ln.at[ls, ns].set(xis)

    def get_xi_ln_arrays(self, arrays):
        arrays = jnp.asarray(arrays)
        if arrays.ndim == 1:
            return self.get_xi_ln_array(arrays)[None, ...]
        if self._hankel_forward_mode != 'pld':
            return jax.vmap(self.get_xi_ln_array)(arrays)

        c_m_by_n = jax.vmap(
            lambda array: self._get_forward_pld_rfft_batch(
                array, self._hankel_forward_pad_mode
            )
        )(arrays)
        c_m = c_m_by_n[:, self._xi_pld_group_ids]
        ells = self.ln_list[:, 0]
        xis = jax.vmap(lambda c_m_i: self._apply_forward_pld_group(c_m_i, ells))(c_m)
        xi_ln = jnp.zeros((arrays.shape[0], 5, 5, self._nfft))
        ls, ns = self.ln_list[:, 0], self.ln_list[:, 1]
        return xi_ln.at[:, ls, ns].set(xis)

    def _get_forward_pld_rfft_batch(self, array, pad_mode):
        fx_pad = fftlog.pad(array / (2 * jnp.pi**2), self._npad, mode=pad_mode)
        nu_n = self._nu_hankel - (self._xi_pld_group_ns + 3).astype(jnp.float64)
        return fftlog.mellin_coefficients(
            fx_pad[None, :], self._k_padded, nu_n[:, None], window=self._w_m)

    def _apply_forward_pld_group(self, c_m, ells):
        c_m = jnp.atleast_2d(c_m)
        xi = fftlog.hankel_transform(
            c_m, self._u_m_xi[ells], self._q_xi_padded[ells], self._nu_hankel, self._npad)
        log_q = jnp.log(self._q)
        def interp_to_common_q(ell, row):
            return spline.interp1d(log_q, jnp.log(self._q_xi[ell]), row)
        return jax.vmap(interp_to_common_q)(ells, xi)

    def _get_lpt_dc_terms(self, corrs):
        if self._lpt_tidal_zero_dc_residual:
            chi_dc_correction = corrs[28]
            zeta_dc_correction = corrs[29]
        else:
            chi_dc_correction = jnp.zeros_like(self._k)
            zeta_dc_correction = jnp.zeros_like(self._k)
        b2sq_residual_pk = corrs[30]
        b2sq_dc = corrs[31, 0]
        return chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc

    def _apply_lpt_dc_to_term_vector(
        self, pkmu_vals, i0, t, chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc,
        chi_index, zeta_index, b2sq_index,
    ):
        if self._lpt_tidal_zero_dc_residual:
            pkmu_vals = pkmu_vals.at[chi_index].add(self._interp_k_array(chi_dc_correction, i0, t))
            pkmu_vals = pkmu_vals.at[zeta_index].add(self._interp_k_array(zeta_dc_correction, i0, t))
        pkmu_vals = pkmu_vals.at[b2sq_index].add(self._interp_k_array(b2sq_residual_pk, i0, t))
        if not self._subtract_k0_limit:
            pkmu_vals = pkmu_vals.at[b2sq_index].add(b2sq_dc)
        return pkmu_vals

    def _get_lpt_dc_scalar(self, bias_facs, i0, t, chi_dc_correction, zeta_dc_correction, b2sq_residual_pk, b2sq_dc):
        val = jnp.array(0.0, dtype=self._k.dtype)
        if self._lpt_tidal_zero_dc_residual:
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

    def _get_lpt_sub_integrands(
        self, k_i, mu_j, f, V_mu, corrs_tree, corrs_matter_1loop, corrs_bias
    ):
        """Return the 24 named sub-integrands as a plain tuple.

        This is the single definition of the LPT integrand algebra.  Returning a
        tuple rather than a stacked array is what lets the bias-folded hot path
        share it: ``_get_lpt_bias_combined_integrand`` contracts the tuple with
        Python-level arithmetic and never materialises an ``(ncomp, L, nq)``
        array, while the template paths stack it.
        """
        Kfac, K, Ksq, base, weights, moments = self._get_lpt_kinematics(
            k_i, mu_j, f, V_mu, corrs_tree
        )
        mq0, mq1, mq2, mq3, mq4, nq1, nq2, mq1_nq1, mq2_nq1 = moments

        X_lin_gt, Y_lin_gt = corrs_tree[4], corrs_tree[5]
        xi_lin, U_lin = corrs_tree[6], corrs_tree[7]
        X22, Y22, X13, Y13, V1, V3, T = corrs_matter_1loop
        U3, U11, U20, X10, Y10 = corrs_bias[0:5]
        V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta = corrs_bias[5:]

        integrand_ZA = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)
        integrand_AA = Ksq**2 / 8.0 * (
            mq0 * X_lin_gt**2 + 2 * mq2 * X_lin_gt * Y_lin_gt + mq4 * Y_lin_gt**2
        )
        integrand_A22 = -0.5 * k_i**2 * (
            (Kfac**2 + 2 * f * (1 + f) * mu_j**2 + f**2 * mu_j**2) * mq0 * X22
            + (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1 + f**2 * mu_j**2 * nq2) * Y22
        )
        integrand_A13 = -0.5 * k_i**2 * (
            2 * (Kfac**2 + 2 * f * (1 + f) * mu_j**2) * mq0 * X13
            + 2 * (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1) * Y13
        )
        integrand_W112 = 0.5 * k_i**3 * (
            2 * Kfac * (Kfac**2 + f * (1 + f) * mu_j**2) * mq1 * V1
            + Kfac**2 * (Kfac * mq1 + f * mu_j * nq1) * V3
            + Kfac**2 * (Kfac * mq3 + f * mu_j * mq2_nq1) * T
        )

        integrand_U10 = -2 * (K * mq1 * (U_lin + U3) + 2 * f * k_i * mu_j * nq1 * U3)
        integrand_A_U = Ksq * (mq1 * X_lin_gt + mq3 * Y_lin_gt) * (K * U_lin)
        integrand_A10 = -Ksq * (X10 * mq0 + Y10 * mq2)
        integrand_A10 += -f * k_i**2 * mu_j * ((1 + f) * mu_j * mq0 * X10 + Kfac * mq1_nq1 * Y10)

        integrand_xi = mq0 * xi_lin
        integrand_A_xi = -0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * xi_lin
        integrand_U11 = -(K * mq1 + f * k_i * mu_j * nq1) * U11
        integrand_U_U = -Ksq * mq2 * U_lin**2
        integrand_U20 = -(K * mq1 + f * k_i * mu_j * nq1) * U20
        integrand_xi_U = -2 * K * mq1 * xi_lin * U_lin
        integrand_xi_xi = 0.5 * mq0 * xi_lin**2

        integrand_Upsilon = -Ksq * (mq0 * X_Upsilon + mq2 * Y_Upsilon)
        integrand_V10 = -2 * (K * mq1 + f * k_i * mu_j * nq1) * V10
        integrand_V12 = -2 * K * mq1 * V12
        integrand_chi = mq0 * chi
        integrand_zeta = mq0 * zeta

        integrand_Ub3 = -2 * K * mq1 * Ub3
        integrand_theta = 2 * mq0 * theta

        # The diagnostic component is the Zel'dovich base.  Linear-family
        # counterterms need raw pk_data and are therefore assembled only in the
        # high-level pk_data paths, not from a bare corrs array.
        integrand_ctr = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)

        # b1^1 piece of the Zel'dovich (tree) spectrum.  ``integrand_U10`` keeps
        # the full U_lin + U3 combination that the b1 bias template needs; this
        # is the U_lin part on its own, appended so indices 0..22 are unchanged.
        integrand_b1_tree = -2 * K * mq1 * U_lin

        parts = (
            integrand_ZA, integrand_AA, integrand_A22, integrand_A13, integrand_W112,
            integrand_U10, integrand_A_U, integrand_A10, integrand_xi, integrand_A_xi,
            integrand_U_U, integrand_U11, integrand_U20, integrand_xi_U, integrand_xi_xi,
            integrand_Upsilon, integrand_V10, integrand_V12, integrand_chi, integrand_zeta,
            integrand_Ub3, integrand_theta, integrand_ctr, integrand_b1_tree,
        )
        return parts, base, weights

    def _group_lpt_integrands(self, parts):
        """Group the 24 sub-integrands into the 15 bias/counterterm templates.

        Pure Python tuple arithmetic, so a caller that only wants a weighted sum
        never builds the stacked array.
        """
        (ZA, AA, A22, A13, W112, U10, A_U, A10, xi, A_xi, U_U, U11, U20, xi_U,
         xi_xi, Upsilon, V10, V12, chi, zeta, Ub3, theta, ctr, b1_tree) = parts
        return (
            ZA + AA + A22 + A13 + W112,   #  0  1
            U10 + A_U + A10,              #  1  b1
            xi + A_xi + U_U + U11,        #  2  b1^2
            U_U + U20,                    #  3  b2
            xi_U,                         #  4  b1 b2
            xi_xi,                        #  5  b2^2
            Upsilon + V10,                #  6  bG2
            V12,                          #  7  b1 bG2
            chi,                          #  8  b2 bG2
            zeta,                         #  9  bG2^2
            Ub3,                          # 10  bGamma3
            theta,                        # 11  b1 bGamma3
            # Tree (Zel'dovich) templates, carrying the counterterms:
            # P_tree = t[12] + b1 t[13] + b1^2 t[14].
            ctr,                          # 12  tree b1^0
            b1_tree,                      # 13  tree b1^1
            xi,                           # 14  tree b1^2
        )

    @staticmethod
    def _get_lpt_template_weights(bias_facs, ctr_k2_shape, nlo_shape):
        """Weights for the 15 templates, shared by the fused and template paths.

        The 12 bias monomials, then the three tree templates: the k^2
        counterterm rides on template 12 alone, the k^4 FoG operator on the
        Lagrangian-biased tree ``t12 + b1 t13 + b1^2 t14``.
        """
        return [bias_facs[i] for i in range(12)] + [
            ctr_k2_shape + nlo_shape * bias_facs[0],
            nlo_shape * bias_facs[1],
            nlo_shape * bias_facs[2],
        ]

    @partial(jit, static_argnames=['self'])
    def get_pkmu_components(self, k, mu, pk_data, f, k_IR=0.2):
        if self._counterterms.needs_kspace_base:
            raise ValueError(
                "get_pkmu_components exposes the fused Zel'dovich counterterm component only; "
                "use get_pkmu/get_pk_ells for a linear-family counterterm base"
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
        """The 15 LPT templates multiplied by ``base`` and the ell weights."""
        parts, base, w = self._get_lpt_sub_integrands(
            k_i, mu_j, f, V_mu, corrs_tree, corrs_matter_1loop, corrs_bias
        )
        integrands = jnp.stack(self._group_lpt_integrands(parts), axis=0)
        weighted = integrands * (base[None, None, :] * w[None, :, :])
        xi_lin = corrs_tree[6]
        return weighted.at[5, 0, :].add(-0.5 * xi_lin**2)
    
    def _get_lpt_bias_combined_integrand(
        self, k_i, mu_j, f, V_mu, bias_facs, ctr_k2_shape, nlo_shape,
        corrs_tree, corrs_matter_1loop, corrs_bias,
    ):
        """Production hot path: fold the bias factors and both counterterms into
        a single ``(L, nq)`` integrand inside the per-(k, mu) function.

        Uses exactly the same integrands as the template path, but contracts the
        tuple with Python-level arithmetic, so the ``(15, L, nq)`` stack that
        ``_get_lpt_weighted_term_integrands`` builds is never materialised and
        XLA fuses the whole thing into one elementwise kernel.
        """
        parts, base, w = self._get_lpt_sub_integrands(
            k_i, mu_j, f, V_mu, corrs_tree, corrs_matter_1loop, corrs_bias
        )
        templates = self._group_lpt_integrands(parts)
        term_weights = self._get_lpt_template_weights(bias_facs, ctr_k2_shape, nlo_shape)

        integrand = term_weights[0] * templates[0]
        for weight, template in zip(term_weights[1:], templates[1:]):
            integrand = integrand + weight * template
        integrand = integrand * (base[None, :] * w)
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
                f"counterterm_base={self.counterterm_base!r}"
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
        corrs = self.get_corrs(pk_data, k_IR)
        return self.get_pkmu_terms_from_corrs(k, mu, corrs, f, alpha_perp, alpha_para)

    @partial(jit, static_argnames=['self'])
    def combine_pkmu_terms(self, k, mu, pkmu_terms, params, alpha_perp=1.0, alpha_para=1.0):
        if self._counterterms.needs_kspace_base:
            raise ValueError("combine_pkmu_terms currently accepts the fused Zel'dovich template only")
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        f = params.f
        bias_facs = self._get_lpt_bias_factors(params.bias, params.bias_b)

        # mapping of (k, mu)
        k_true, mu_true = get_k_mu_true_for_ap(k, mu, alpha_perp, alpha_para)
        mu_true = jnp.broadcast_to(mu_true, k_true.shape)  # (nk, nmu)

        pkmu = jnp.tensordot(bias_facs, pkmu_terms[:12], axes=(0, 0))   # (nk, nmu)

        # Counterterms.  The k^2 operator rides on the Zel'dovich matter
        # spectrum (template 12); the k^4 FoG operator rides on the
        # Lagrangian-biased Zel'dovich tree, templates 12..14 contracted with
        # (1, (b1_a+b1_b)/2, b1_a b1_b) = bias_facs[0:3].
        pkmu_ctr = self._counterterms.leading(
            k_true, mu_true, f, params.ctr, pkmu_terms[12]
        )
        tree_pk = (
            bias_facs[0] * pkmu_terms[12]
            + bias_facs[1] * pkmu_terms[13]
            + bias_facs[2] * pkmu_terms[14]
        )
        pkmu_ctr = pkmu_ctr + self._counterterms.nlo(
            k_true, mu_true, f, params.ctr, tree_pk
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
        pk_nw_data = ir_resum.get_pk_nw(pk_data, params.h, method=self.counterterm_irres_method)
        pk_nw = get_pk(k, pk_nw_data, kmin=self._kmin, kmax=self._kmax)
        pk_w = pk - pk_nw

        # BAO damping factor in redshift space
        f = params.f
        Sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.counterterm_r_bao, self.counterterm_k_IR)
        dSigma2 = ir_resum.get_dSigma2(pk_nw_data, self.counterterm_r_bao, self.counterterm_k_IR)
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
        ctr_leading, c_nlo = Counterterms.split_coefficients(params.ctr)
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
                    nlo_shape = Counterterms.nlo_shape(k_i, mu_j, f, c_nlo)
                # Section 32 fast path: bias folded inside, never builds (13, L, nq).
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
            tree_pk = self._get_lpt_nlo_tree_factor(f, mu_true, params.bias, params.bias_b) * base_pk
            pkmu = pkmu + self._counterterms.nlo(k_true, mu_true, f, params.ctr, tree_pk)

        return pkmu / (alpha_perp**2 * alpha_para)

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=0.2):
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
        """
        Legendre multipoles.
        """
        corrs = self.get_corrs(pk_data, k_IR)
        return self.get_pk_ells_from_corrs(
            k, corrs, params, alpha_perp, alpha_para, pk_data=pk_data
        )

    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data, k_IR=0.2):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_lin_lt = pk_lin * jnp.exp(-(self._k / k_IR)**2)

        # integrals of pk
        pk_int = get_pk_int(pk_data)
        pk_data_lt = jnp.stack([self._k, pk_lin_lt], axis=0)
        pk_int_lt = get_pk_int(pk_data_lt)

        # generalized correlation functions
        xi_ln, xi_ln_lt = self.get_xi_ln_arrays(jnp.stack([pk_lin, pk_lin_lt], axis=0))

        # tree-level terms
        corrs_tree = self.get_corrs_tree(xi_ln, xi_ln_lt, pk_int, pk_int_lt)

        # one-loop terms
        Qs = self.get_Qs(xi_ln)
        Rs = self.get_Rs(xi_ln, pk_lin)
        corrs_matter_1loop = self.get_corrs_matter_1loop(Qs, Rs)
        corrs_bias = self.get_corrs_bias(Qs, Rs, xi_ln)

        chi = corrs_bias[9]
        zeta = corrs_bias[10]

        if self._lpt_tidal_zero_dc_residual:
            chi_dc_correction = self._get_lpt_dc_correction(chi)
            zeta_dc_correction = self._get_lpt_dc_correction(zeta)
        else:
            chi_dc_correction = jnp.zeros_like(self._k)
            zeta_dc_correction = jnp.zeros_like(self._k)

        b2sq_residual_pk, b2sq_dc = self._get_lpt_b2sq_residual_pk(corrs_tree[6], pk_lin)
        b2sq_dc_arr = jnp.broadcast_to(b2sq_dc, self._k.shape)

        corrs = jnp.concatenate([
            corrs_tree, corrs_matter_1loop, corrs_bias,
            chi_dc_correction[None, :],
            zeta_dc_correction[None, :],
            b2sq_residual_pk[None, :],
            b2sq_dc_arr[None, :],
        ], axis=0)
        return corrs

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

    def get_Qs(self, xi_ln):
        # Compute Q1, Q2, Q5 using the original basis (backward Hankel of xi^2).
        # The UV-safe k^4 factoring amplifies FFTLog numerical noise at large k,
        # contaminating int Q1 k^2 dk / (2*pi^2) used in get_corrs_matter_1loop.
        integrands = jnp.stack([
            8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2,
            xi_ln[1,-1]**2 - xi_ln[3,-1]**2,
        ], axis=0)
        res = self.get_pk_batched(self._4pi_q3 * integrands, self._u_m[0])
        Q1 = res[0]
        Q2 = Q1 - 2/5 * self._k**2 * res[1]
        Q5 = (Q1 + Q2) / 2
        return jnp.stack([Q1, Q2, Q5], axis=0)

    def get_Rs(self, xi_ln, pk_lin):
        ells = jnp.array([0, 2, 4, 1, 3], dtype=jnp.int32)
        xis = jnp.stack([xi_ln[0,0], xi_ln[2,0], xi_ln[4,0], xi_ln[1,-1], xi_ln[3,-1]], axis=0)
        pk_list = self.get_pk_batched(self._q**2 * xis, self._u_m[ells]) * pk_lin
        pk_00, pk_20, pk_40, pk_1m1, pk_3m1 = pk_list
        k = self._k
        R1 = k**2 * (8/15*pk_00 - 16/21*pk_20 + 8/35*pk_40)
        R3 = (k**2 * (2/5*pk_00 - 6/7*pk_20 + 16/35*pk_40)
              + k**3 * (2/5*pk_1m1 - 2/5*pk_3m1))
        R2 = R3 - R1
        F_G2_raw = -32/21 * k**2 * (pk_00 - 10/7*pk_20 + 3/7*pk_40)
        return jnp.stack([R1, R2, F_G2_raw], axis=0)

    def get_corrs_matter_1loop(self, Qs, Rs):
        Q1 = Qs[0]
        Q2 = Qs[1]
        R1 = Rs[0]
        R2 = Rs[1]

        # X, Y for 1-loop A_{ij}
        xi_A = self.get_xi_ln_batched(
            jnp.array([0, 2, 0, 2]),
            jnp.array([-2, -2, -2, -2]),
            jnp.stack([9/98 * Q1, 9/98 * Q1, 5/21 * R1, 5/21 * R1], axis=0),
            pad_mode=self._lpt_source_forward_pad_mode,
        )
        xi_ln_22_0m2, xi_ln_22_2m2, xi_ln_13_0m2, xi_ln_13_2m2 = xi_A

        pk_data = jnp.stack([self._k, 9/98 * Q1], axis=0)
        xi_ln_22_q0 = get_pk_int(pk_data)
        # X22 = 2/3 * (xi_ln_22_0m2[0] - xi_ln_22_0m2 - xi_ln_22_2m2)
        X22 = 2/3 * (xi_ln_22_q0 - xi_ln_22_0m2 - xi_ln_22_2m2)
        Y22 = 2 * xi_ln_22_2m2

        pk_data = jnp.stack([self._k, 5/21 * R1], axis=0)
        xi_ln_13_q0 = get_pk_int(pk_data)
        # X13 = 2/3 * (xi_ln_13_0m2[0] - xi_ln_13_0m2 - xi_ln_13_2m2)
        X13 = 2/3 * (xi_ln_13_q0 - xi_ln_13_0m2 - xi_ln_13_2m2)
        Y13 = 2 * xi_ln_13_2m2

        # V1, V3, T for W_{ijk}
        xi_W = self.get_xi_ln_batched(
            jnp.array([3, 1, 1]),
            jnp.array([-3, -3, -3]),
            jnp.stack([
                -3/7 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2),
                3/35 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2),
                -3/35 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2),
            ], axis=0),
            pad_mode=self._lpt_source_forward_pad_mode,
        )
        T = xi_W[0]
        V1 = xi_W[1] - 0.2 * T
        V3 = xi_W[2] - 0.2 * T

        corrs = jnp.stack([X22, Y22, X13, Y13, V1, V3, T], axis=0)
        return corrs

    def get_corrs_bias(self, Qs, Rs, xi_ln):
        Q1 = Qs[0]
        Q2 = Qs[1]
        Q5 = Qs[2]
        R1 = Rs[0]
        R2 = Rs[1]
        F_G2_raw = Rs[2]

        # A10
        pk_data = jnp.stack([self._k, 2/7 * R1], axis=0)
        xi_ln_A10_q0 = get_pk_int(pk_data)

        xi_bias = self.get_xi_ln_batched(
            jnp.array([1, 1, 0, 2, 1, 1, 0]),
            jnp.array([-1, -1, -2, -2, -1, -1, 0]),
            jnp.stack([
                -5/21 * R1,
                -6/7 * (R1 + R2),
                2/7 * (Q5 + 2 * R2),
                1/7 * (2 * Q5 + 3 * R1 + 4 * R2),
                3/7 * Q1,
                -2/5 * F_G2_raw,
                2/5 * F_G2_raw,
            ], axis=0),
            pad_mode=self._lpt_source_forward_pad_mode,
        )
        U3, U11 = xi_bias[0], xi_bias[1]
        xi_ln_A10_0m2, xi_ln_A10_2m2 = xi_bias[2], xi_bias[3]
        X10 = xi_ln_A10_q0 - xi_ln_A10_0m2 - xi_ln_A10_2m2
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

        corrs = jnp.stack([U3, U11, U20, X10, Y10, V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta], axis=0)
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

    def _get_lpt_nlo_tree_factor(self, f, mu, bias_a, bias_b):
        """Factor turning the counterterm base into the tree galaxy spectrum.

        ``Counterterms.nlo`` multiplies whatever ``tree_pk`` it is given, so the
        caller supplies the full tree spectrum -- the EPT backend passes
        ``(b1_a + f mu^2)(b1_b + f mu^2) P_lin``.  Two things differ here:

        * ``b1`` is Lagrangian in this backend, so the Kaiser factor is
          ``Z1 = 1 + b1 + f mu^2`` (verified against the k -> 0 limit);
        * only the linear-family bases go through here.  The ``'zeldovich'``
          base does not rescale a matter spectrum at all: it uses the genuine
          Lagrangian-biased Zel'dovich tree assembled from templates 12..14
          (see ``_get_lpt_bias_combined_integrand`` and ``combine_pkmu_terms``).
        """
        kaiser = 1.0 + f * mu**2
        return (kaiser + bias_a[0]) * (kaiser + bias_b[0])

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
        r = jnp.atleast_1d(r)

        # This helper is an approximate configuration-space projection.
        # The final LPT multipoles are not numerically stable on the full FFT grid up to kmax_fft, 
        # so use a conservative spectrum grid and interpolate it back to self._k before the Hankel transform.
        k = jnp.geomspace(max(self._kmin, 1e-4), min(self._kmax, 1.0), min(self._nfft, 128))
        pk_ells = self.get_pk_ells(k, pk_data, params, alpha_perp, alpha_para, k_IR)

        pk0 = get_pk(self._k, jnp.stack([k, pk_ells[0]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk2 = get_pk(self._k, jnp.stack([k, pk_ells[1]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk4 = get_pk(self._k, jnp.stack([k, pk_ells[2]], axis=0), kmin=self._kmin, kmax=self._kmax)

        # get_xi_ln resamples to the common q-grid (self._q); use that as source.
        xi0 = spline.interp1d(jnp.log(r), jnp.log(self._q), self.get_xi_ln(0, 0, pk0))
        xi2 = spline.interp1d(jnp.log(r), jnp.log(self._q), -self.get_xi_ln(2, 0, pk2))
        xi4 = spline.interp1d(jnp.log(r), jnp.log(self._q), self.get_xi_ln(4, 0, pk4))

        xi_ells = jnp.stack([xi0, xi2, xi4], axis=0)
        return xi_ells
