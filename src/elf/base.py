"""The public API shared by the EPT and LPT backends.

:class:`PowerSpectrum` holds the common options and grids and implements every
public method from two backend hooks: ``get_corrs(pk_data, h)`` (quantities that
depend only on P_lin and h) and ``_pkmu_true`` (tree level plus one loop at the
true (k, mu)).  The AP map, mu folding, stochastic term, linear-family
counterterms, volume factor and the multipole/configuration-space projections
are done here.
"""

from abc import ABC, abstractmethod
from functools import partial

from jax import jit
import jax.numpy as jnp
import numpy as np

from . import fftlog
from . import ir_resum
from . import spline
from .eft_terms import Counterterms, stochasticity
from .multipole import prepare_mu_gauleg, get_legendre_multipoles, get_k_mu_true_sin_for_ap
from .utils import get_pk


class PowerSpectrum(ABC):
    """Shared spectrum operations; instantiate EPT or LPT."""

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
                 ):
        """Options shared by the EPT and LPT backends.

        kmin_fft, kmax_fft, nfft: internal log k grid in h/Mpc (nfft even).
        pad_mode: 'zero-pad' (P = 0 outside the table, end weights 1/2) or
            'power-law' (P continued by its end slopes).
        fftlog_settings: :class:`fftlog.FFTlogSettings` (None = defaults).
        ngauss, ells: Gauss-Legendre points on mu in [0, 1] (``max(ells) < 2 ngauss``)
            and the even multipole orders returned.
        counterterm_base: 'linear_ir_resum', 'linear' or 'zeldovich' (LPT only).
        irres_method ('DST', 'SG', 'WH'), r_bao [Mpc/h], lambda_ir [h/Mpc]:
            wiggle/no-wiggle split and BAO damping (Sigma^2 integrated to lambda_ir).
        subtract_k0_limit: drop the k -> 0 constant of the b2^2 term.
        """
        if nfft % 2:
            raise ValueError(
                "nfft must be even: the FFTlog pipeline uses rfft/irfft with an "
                "implicit even length"
            )
        if pad_mode not in fftlog.INPUT_SUPPORT:
            raise ValueError("pad_mode must be 'zero-pad' or 'power-law'")
        if irres_method not in ('DST', 'SG', 'WH'):
            raise ValueError("irres_method must be 'DST', 'SG', or 'WH'")

        self._counterterms = Counterterms(counterterm_base)
        self.irres_method = irres_method
        self.r_bao = r_bao
        self.lambda_ir = lambda_ir
        self.subtract_k0_limit = subtract_k0_limit

        ells = tuple(ells)
        self._mu_quad, self._legendre_weights = prepare_mu_gauleg(ngauss, ells)
        self.ells = tuple(int(ell) for ell in ells)

        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self.fftlog_settings = fftlog.FFTlogSettings() if fftlog_settings is None else fftlog_settings
        self.pad_mode = pad_mode
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._logk_fft = jnp.log(self._k)

    @property
    def counterterm_base(self):
        """Counterterm base fixed at construction time."""
        return self._counterterms.base

    @abstractmethod
    def get_corrs(self, pk_data, h):
        """Intermediate quantities that depend only on ``P_lin`` and ``h`` (backend specific)."""

    @abstractmethod
    def _pkmu_true(self, k_true, mu_true, sin_true, corrs, pk_data, params):
        """Tree level plus one loop at the true coordinates, ``(nk, nmu)``.

        ``k_true`` is ``(nk, nmu)``; ``mu_true`` (folded to ``|mu|``) and ``sin_true`` are ``(nmu,)``.
        """

    def _get_ir_data(self, pk_data, h):
        """``(pk_nw_data, Sigma2, dSigma2)`` of ``pk_data``."""
        pk_nw_data = ir_resum.get_pk_nw(pk_data, h, method=self.irres_method)
        Sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.r_bao, self.lambda_ir)
        dSigma2 = ir_resum.get_dSigma2(pk_nw_data, self.r_bao, self.lambda_ir)
        return pk_nw_data, Sigma2, dSigma2

    def _pk_at(self, k, pk_data):
        """``pk_data`` at ``k``, continued by its end slopes to ``kmin_fft``/``kmax_fft``."""
        return get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)

    @staticmethod
    def _true_coordinates(k, mu, alpha_perp, alpha_para):
        """AP map: ``(k_true (nk, nmu), |mu_true|, sin_true)`` (the spectra are even in mu)."""
        k_true, mu_true, sin_true = get_k_mu_true_sin_for_ap(k, mu, alpha_perp, alpha_para)
        return k_true, jnp.abs(mu_true), sin_true

    def _add_stoch_ctr_volume(self, pkmu, k_true, mu_k, params, ctr_base, alpha_perp, alpha_para):
        """``[pkmu + P_stoch + P_ctr[ctr_base]] / (alpha_perp^2 alpha_para)`` (``ctr_base=None``: no P_ctr)."""
        pkmu = pkmu + stochasticity(k_true, mu_k, params.stoch)
        if ctr_base is not None:
            pkmu = (pkmu + self._counterterms.leading(k_true, mu_k, params.f, params.ctr, ctr_base)
                    + self._counterterms.nlo(k_true, mu_k, params.f, params.ctr, ctr_base))
        return pkmu / (alpha_perp**2 * alpha_para)

    def _linear_at(self, k_true, mu_true, f, corrs, pk_data):
        """``(pk, pk_nw, pk_w, damp)`` at ``k_true``; the last three are None without IR data."""
        pk = self._pk_at(k_true, pk_data)
        if corrs.pk_nw_data is None:
            return pk, None, None, None
        pk_nw = self._pk_at(k_true, corrs.pk_nw_data)
        mu = jnp.broadcast_to(mu_true, k_true.shape)
        damp = ir_resum.damping_exponent(k_true, mu, f, corrs.Sigma2, corrs.dSigma2)
        return pk, pk_nw, pk - pk_nw, damp

    def _set_xi_ells_transform(self):
        """Kernels of the P_ell -> xi_ell transforms (as the EPT forward row (ell, 0))."""
        hg = self._hankel
        fs = self.fftlog_settings
        nu_eff = fs.nu_pld_n0 + 3
        u_ms, q_factors, qs = [], [], []
        for ell in self.ells:
            lnxy, q_grid, _ = fftlog.hankel_setup(ell, fs.nu, hg.grid)
            u_ms.append(fftlog.hankel_setup(ell, nu_eff, hg.grid, lnxy=lnxy)[2])
            q_factors.append(q_grid.x ** (-nu_eff))
            qs.append(q_grid.x)
        self._xi_ells_u_m = tuple(u_ms)
        self._xi_ells_q_factor = tuple(q_factors)
        self._xi_ells_q = tuple(qs)

    @partial(jit, static_argnames=['self'])
    def get_pkmu_from_corrs(self, k, mu, corrs, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """P(k, mu) at the observed (k, mu) from ``corrs``, shape ``(nk, nmu)``.

        ``[P_body + P_stoch + P_ctr](k_true, |mu_true|) / (alpha_perp^2 alpha_para)``; the
        linear-family counterterms multiply ``P_lin`` or ``P_nw + e^{-k^2 Sigma^2_s} P_w``.
        ``pk_data`` must be the spectrum ``corrs`` was computed from.
        """
        k_true, mu_true, sin_true = self._true_coordinates(k, mu, alpha_perp, alpha_para)
        pkmu = self._pkmu_true(k_true, mu_true, sin_true, corrs, pk_data, params)
        # broadcast mu to (nk, nmu): fixes the order of the reverse-mode sums
        mu_k = jnp.broadcast_to(mu_true, k_true.shape)
        base_pk = None
        if self._counterterms.needs_kspace_base:
            pk, pk_nw, pk_w, damp = self._linear_at(k_true, mu_true, params.f, corrs, pk_data)
            if self.counterterm_base == 'linear':
                base_pk = pk
            else:
                base_pk = pk_nw + jnp.exp(-damp) * pk_w
        return self._add_stoch_ctr_volume(pkmu, k_true, mu_k, params, base_pk, alpha_perp, alpha_para)

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """P(k, mu) at the observed (k, mu), AP distortion included; shape ``(nk, nmu)``."""
        corrs = self.get_corrs(pk_data, params.h)
        return self.get_pkmu_from_corrs(k, mu, corrs, pk_data, params, alpha_perp, alpha_para)

    @partial(jit, static_argnames=['self'])
    def get_pk_ells_from_corrs(self, k, corrs, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """Legendre multipoles ``P_ell(k)`` for ``ell`` in ``self.ells`` from ``corrs``; shape ``(nells, nk)``."""
        pkmu = self.get_pkmu_from_corrs(k, self._mu_quad, corrs, pk_data, params, alpha_perp, alpha_para)
        return get_legendre_multipoles(pkmu, self._legendre_weights)

    @partial(jit, static_argnames=['self'])
    def get_pk_ells(self, k, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """Legendre multipoles ``P_ell(k)`` for ``ell`` in ``self.ells``; shape ``(nells, nk)``."""
        corrs = self.get_corrs(pk_data, params.h)
        return self.get_pk_ells_from_corrs(k, corrs, pk_data, params, alpha_perp, alpha_para)

    def _xi_ells_nkeep(self, kmax):
        """Number of core k nodes with ``k <= kmax`` (``None``: ``kmax_fft / 3``)."""
        if kmax is None:
            kmax = self._kmax / 3
        k = np.asarray(self._k)
        n_keep = int(np.count_nonzero(k <= float(kmax) * (1.0 + 1e-12)))
        if n_keep < 2:
            raise ValueError("kmax must keep at least two nodes of the internal k grid")
        return n_keep

    def _get_xi_ells_from_corrs(self, r, corrs, pk_data, params, alpha_perp, alpha_para, kmax=None):
        """``xi_ell(r) = i^l int dk k^2 P_l j_l(kr) / (2 pi^2)`` of ``P_ell`` cut at ``kmax``, from ``corrs``."""
        r = jnp.atleast_1d(r)
        hg = self._hankel
        fs = self.fftlog_settings

        # P_ell on the core nodes k <= kmax, zero elsewhere; trapezoid end weights 1/2
        n_keep = self._xi_ells_nkeep(kmax)
        k = self._k[:n_keep]
        pk_ells = self.get_pk_ells_from_corrs(k, corrs, pk_data, params, alpha_perp, alpha_para)
        w = fftlog.trapezoid_weights('core', n_keep, 0, hg.grid.dln)
        n_zero = self._nfft - n_keep + hg.npad
        pk_ells = jnp.pad(pk_ells * w, [(0, 0), (hg.npad, n_zero)])

        xi_ells = []
        for i, ell in enumerate(self.ells):
            xi = fftlog.hankel(fs.nu_pld_n0, pk_ells[i] / (2 * jnp.pi**2), hg.k_padded,
                               None, self._xi_ells_u_m[i], hg.npad, crop=False,
                               y_pow=self._xi_ells_q_factor[i])
            xi = xi * hg.mask_xi
            if (ell // 2) % 2:
                xi = -xi
            xi_ells.append(spline.interp1d(jnp.log(r), jnp.log(self._xi_ells_q[i]), xi))
        return jnp.stack(xi_ells, axis=0)

    @partial(jit, static_argnames=['self', 'kmax'])
    def get_xi_ells(self, r, pk_data, params, alpha_perp=1.0, alpha_para=1.0, kmax=None):
        """Configuration-space multipoles ``xi_ell(r)`` for ``ell`` in ``self.ells``; shape ``(nells, nr)``.

        ``xi_ell`` is the transform of ``P_ell`` evaluated on the internal k nodes up to
        ``kmax`` and set to zero above; small r depends on ``kmax``.  The default
        ``kmax_fft / 3`` keeps out the high-k end, where the one-loop spectra are inaccurate.
        """
        corrs = self.get_corrs(pk_data, params.h)
        return self._get_xi_ells_from_corrs(r, corrs, pk_data, params, alpha_perp, alpha_para, kmax)
