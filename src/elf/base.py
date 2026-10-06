"""The public API shared by the EPT and LPT backends.

:class:`PowerSpectrum` holds the common options and grids and implements every
public method from two backend hooks: ``get_pt_terms(pk_data, h)`` (the PT terms,
which depend only on P_lin and h) and ``_pkmu_true`` (tree level plus one loop at
the true (k, mu)).  The AP map, mu folding, stochastic term, linear-family
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
from .params import _with_tracer_biases
from .utils import get_pk


class PowerSpectrum(ABC):
    """Shared spectrum operations (instantiate EPT or LPT)."""

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
                 ):
        """Options shared by the EPT and LPT backends.

        kmin_fft, kmax_fft, nfft: internal log k grid in h/Mpc (nfft even).
        pad_mode: 'zero-pad' (P = 0 outside the table, end weights 1/2) or
            'power-law' (P continued by its end slopes).
        fftlog_settings: :class:`fftlog.FFTlogSettings` (None = defaults).
        ngauss, ells: Gauss-Legendre points on mu in [0, 1] (``max(ells) < 2 ngauss``)
            and the even multipole orders returned.
        counterterm_base: 'linear_ir_resum', 'linear' or 'zeldovich' (LPT only).
        irres_method: method of the wiggle/no-wiggle split, 'DST' (discrete sine
            transform), 'SG' (Savitzky-Golay filter) or 'WH' (Whittaker-Henderson
            smoothing).  The split serves the EPT IR resummation and the
            'linear_ir_resum' counterterm base of both backends.
        r_bao [Mpc/h], lambda_ir [h/Mpc]: BAO scale and upper limit of the Sigma^2
            integral of the BAO damping.  ``lambda_ir`` is not LPT's ``k_IR``, the
            split ``P_lt = P e^{-(k/k_IR)^2}`` inside the LPT body.
        subtract_k0_const: drop the k -> 0 constant of the b2^2 term.
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
        self.subtract_k0_const = subtract_k0_const

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
    def get_pt_terms(self, pk_data, h):
        """The backend's ``PTTerms`` of ``pk_data``, the quantities that depend only on ``P_lin`` and ``h``.

        ``h`` enters only the no-wiggle data.  ``h=None`` skips them in LPT (its template
        and component methods do so).  EPT accepts ``h=None`` only when it needs no
        no-wiggle data (``do_irres=False`` and ``counterterm_base='linear'``).
        """

    @abstractmethod
    def _pkmu_true(self, k_true, mu_true, sin_true, pt_terms, pk_data, params):
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

    def _add_stoch_ctr_volume(self, pkmu, k_true, mu_k, params, ctr_base_pk, alpha_perp, alpha_para):
        """``[pkmu + P_stoch + P_ctr[ctr_base_pk]] / (alpha_perp^2 alpha_para)`` (no P_ctr if ``ctr_base_pk`` is None)."""
        pkmu = pkmu + stochasticity(k_true, mu_k, params.stoch)
        pkmu = self._counterterms.add_kspace(pkmu, k_true, mu_k, params.f, params.ctr, ctr_base_pk)
        return pkmu / (alpha_perp**2 * alpha_para)

    def _linear_at(self, k_true, mu_true, f, pt_terms, pk_data):
        """``(pk, pk_nw, pk_w, damp_exponent)`` at ``k_true`` (the last three None without no-wiggle data).

        ``damp_exponent`` is ``k^2 Sigma^2_s`` of :func:`ir_resum.damping_exponent`.
        """
        pk = self._pk_at(k_true, pk_data)
        if pt_terms.pk_nw_data is None:
            return pk, None, None, None
        pk_nw = self._pk_at(k_true, pt_terms.pk_nw_data)
        mu = jnp.broadcast_to(mu_true, k_true.shape)
        damp_exponent = ir_resum.damping_exponent(k_true, mu, f, pt_terms.Sigma2, pt_terms.dSigma2)
        return pk, pk_nw, pk - pk_nw, damp_exponent

    def _set_xi_ells_transform(self):
        """Kernels of the P_ell -> xi_ell transforms, the forward rows (ell, 0) at the input bias ``nu_xi_n0``."""
        hg = self._hankel
        fs = self.fftlog_settings
        forward = fftlog.forward_transform(hg.grid, fs.nu, self.ells, [0] * len(self.ells),
                                           nu_in=fs.nu_xi_n0)
        self._xi_ells_rows = tuple(forward.row(i) for i in range(len(self.ells)))

    @partial(jit, static_argnames=['self'])
    def get_pkmu_from_pt_terms(self, k, mu, pt_terms, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """P(k, mu) at the observed (k, mu) from ``pt_terms``, shape ``(nk, nmu)``.

        ``[P_body + P_stoch + P_ctr](k_true, |mu_true|) / (alpha_perp^2 alpha_para)``.  The
        linear-family counterterms multiply ``P_lin`` or ``P_nw + e^{-k^2 Sigma^2_s} P_w``.
        ``pk_data`` must be the spectrum ``pt_terms`` was computed from.  For an auto
        spectrum (``params.bias`` set) ``bias_a`` and ``bias_b`` are both set to ``bias``
        here, the only entry that reads the biases.
        """
        params = _with_tracer_biases(params)
        k_true, mu_true, sin_true = self._true_coordinates(k, mu, alpha_perp, alpha_para)
        pkmu = self._pkmu_true(k_true, mu_true, sin_true, pt_terms, pk_data, params)
        # broadcast mu to (nk, nmu): fixes the order of the reverse-mode sums
        mu_k = jnp.broadcast_to(mu_true, k_true.shape)
        # None for 'zeldovich', whose counterterms LPT folds into its integrand
        ctr_base_pk = self._counterterms.kspace_base_pk(
            lambda: self._linear_at(k_true, mu_true, params.f, pt_terms, pk_data))
        return self._add_stoch_ctr_volume(pkmu, k_true, mu_k, params, ctr_base_pk, alpha_perp, alpha_para)

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """P(k, mu) at the observed (k, mu), AP distortion included, shape ``(nk, nmu)``."""
        pt_terms = self.get_pt_terms(pk_data, params.h)
        return self.get_pkmu_from_pt_terms(k, mu, pt_terms, pk_data, params, alpha_perp, alpha_para)

    @partial(jit, static_argnames=['self'])
    def get_pk_ells_from_pt_terms(self, k, pt_terms, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """Legendre multipoles ``P_ell(k)`` for ``ell`` in ``self.ells`` from ``pt_terms``, shape ``(nells, nk)``."""
        pkmu = self.get_pkmu_from_pt_terms(k, self._mu_quad, pt_terms, pk_data, params, alpha_perp, alpha_para)
        return get_legendre_multipoles(pkmu, self._legendre_weights)

    @partial(jit, static_argnames=['self'])
    def get_pk_ells(self, k, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """Legendre multipoles ``P_ell(k)`` for ``ell`` in ``self.ells``, shape ``(nells, nk)``."""
        pt_terms = self.get_pt_terms(pk_data, params.h)
        return self.get_pk_ells_from_pt_terms(k, pt_terms, pk_data, params, alpha_perp, alpha_para)

    def _xi_ells_nkeep(self, kmax):
        """Number of core k nodes with ``k <= kmax`` (``None``: ``kmax_fft / 3``)."""
        if kmax is None:
            kmax = self._kmax / 3
        k = np.asarray(self._k)
        n_keep = int(np.count_nonzero(k <= float(kmax) * (1.0 + 1e-12)))
        if n_keep < 2:
            raise ValueError("kmax must keep at least two nodes of the internal k grid")
        return n_keep

    def _get_xi_ells_from_pt_terms(self, r, pt_terms, pk_data, params, alpha_perp, alpha_para, kmax=None):
        """``xi_ell(r) = i^l int dk k^2 P_l j_l(kr) / (2 pi^2)`` of ``P_ell`` cut at ``kmax``, from ``pt_terms``."""
        r = jnp.atleast_1d(r)
        hg = self._hankel

        # P_ell on the core nodes k <= kmax, zero elsewhere; trapezoid end weights 1/2
        n_keep = self._xi_ells_nkeep(kmax)
        k = self._k[:n_keep]
        pk_ells = self.get_pk_ells_from_pt_terms(k, pt_terms, pk_data, params, alpha_perp, alpha_para)
        w = fftlog.trapezoid_weights('core', n_keep, 0, hg.grid.dln)
        n_zero = self._nfft - n_keep + hg.npad
        pk_ells = jnp.pad(pk_ells * w, [(0, 0), (hg.npad, n_zero)])

        xi_ells = []
        for i, ell in enumerate(self.ells):
            row = self._xi_ells_rows[i]
            xi = row.apply(pk_ells[i] / (2 * jnp.pi**2), hg.grid)
            xi = xi * hg.mask_xi
            if (ell // 2) % 2:
                xi = -xi
            xi_ells.append(spline.interp1d(jnp.log(r), jnp.log(row.q), xi))
        return jnp.stack(xi_ells, axis=0)

    @partial(jit, static_argnames=['self', 'kmax'])
    def get_xi_ells(self, r, pk_data, params, alpha_perp=1.0, alpha_para=1.0, kmax=None):
        """Configuration-space multipoles ``xi_ell(r)`` for ``ell`` in ``self.ells``, shape ``(nells, nr)``.

        ``xi_ell`` is the transform of ``P_ell`` evaluated on the internal k nodes up to
        ``kmax`` and set to zero above.  ``kmax`` is this cut on ``P_ell``, applied before
        the transform.  It is unrelated to the FFTlog band ``kmax_fft``, and small r
        depends on it.  The default ``kmax_fft / 3`` keeps out the high-k end, where the
        one-loop spectra are inaccurate.
        """
        pt_terms = self.get_pt_terms(pk_data, params.h)
        return self._get_xi_ells_from_pt_terms(r, pt_terms, pk_data, params, alpha_perp, alpha_para, kmax)
