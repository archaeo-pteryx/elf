import os
import glob, re
import numpy as _np
import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp

from . import fftlog

from . import pt_coeff
from . import pt_matrix
from . import utils
from .utils import get_pk, eval_power_coeffs
from .multipole import prepare_mu_gauleg, get_legendre_multipoles, get_k_mu_true_for_ap
from .eft_terms import Counterterms, stochasticity

from . import ir_resum
from . import spline


class EPT:

    def __init__(self,
                 do_irres=True,
                 r_bao=110.,
                 lambda_ir=0.2,
                 irres_method='DST',
                 counterterm_base=None,
                 subtract_k0_limit=False,
                 method='hankel',
                 kmin_fft=1e-5,
                 kmax_fft=1e2,
                 nfft=512,
                 pad_mode='zero-pad',
                 fftlog_settings=None,
                 ngauss=4,
                 ):
        """One-loop Eulerian-PT galaxy power spectrum in redshift space.

        Parameters
        ----------
        do_irres : bool, default True
            Switch IR resummation (wiggle/no-wiggle split plus BAO damping) on
            or off for the tree-level and one-loop pieces.
        r_bao : float, default 110.
            BAO scale entering the damping factor, in Mpc/h.
        lambda_ir : float, default 0.2
            Upper limit of the k integral of the BAO damping Sigma^2 (and
            delta Sigma^2), in h/Mpc: only modes below it are resummed.
        irres_method : {'DST', 'SG', 'WH'}, default 'DST'
            Wiggle/no-wiggle split used to build P_nw -- discrete sine
            transform, Savitzky-Golay filter, or the Wallisch fitting form.
        counterterm_base : {None, 'linear', 'linear_ir_resum'}, default None
            Spectrum multiplying the counterterms.  ``None`` means
            ``'linear_ir_resum'`` if ``do_irres`` else ``'linear'``; the
            ``'zeldovich'`` base exists only in the LPT backend.
        subtract_k0_limit : bool, default False
            Subtract the k -> 0 constant of P22, i.e. the b2^2 shot-noise-like
            term, so that the loop contribution vanishes at k = 0.
        method : {'hankel', 'hybrid', 'matrix'}, default 'hankel'
            'hankel' evaluates both P22 and P13 by FFTlog; 'hybrid' keeps FFTlog
            for P22 and uses the power-law-decomposition matrices for P13;
            'matrix' uses the matrices for both.  The matrices are accurate only
            for k in [3e-3, 3] h/Mpc.
        kmin_fft, kmax_fft : float, defaults 1e-5 and 1e2
            Ends of the internal logarithmic k grid, in h/Mpc.
        nfft : int, default 512
            Number of nodes on that grid.  Must be even: the FFTlog pipeline
            uses rfft/irfft with an implicit even length.
        pad_mode : {'zero-pad', 'power-law'}, default 'zero-pad'
            Model of the input P(k) outside the tabulated range.  'zero-pad' is
            the band-limited model: P is taken to be exactly zero outside
            [k_min, k_max], and the k_min and k_max nodes enter the FFTlog sum
            with the trapezoid end weight 1/2 (see ``fftlog.trapezoid_weights``).
            'power-law' is the continued model: P is carried across the padded
            band by its endpoint log-slopes, and the end weights sit on the
            padded corners instead (where they have no measurable effect, the
            continued P being negligible there).  The two models differ by a
            counterterm-shaped piece -- k^2 P_lin(k) times a polynomial in
            mu^2 -- whose coefficients are four orders of magnitude below
            typical counterterm values, so the choice is a model statement, not
            an accuracy one.
        fftlog_settings : fftlog.FFTlogSettings or None, default None
            Numerical constants of the FFTlog pipeline (biases, width of the
            padded band, m = 0 P22 blend scale); ``None`` means the validated
            defaults ``FFTlogSettings()``.
        ngauss : int, default 4
            Number of Gauss-Legendre points on mu in [0, 1] used for the
            multipole projection.
        """

        if method not in ('matrix', 'hankel', 'hybrid'):
            raise ValueError(f"method must be 'matrix', 'hankel', or 'hybrid', got {method!r}")
        if nfft % 2:
            raise ValueError("nfft must be even: the FFTlog pipeline uses rfft/irfft with an implicit even length")
        if pad_mode not in fftlog.INPUT_SUPPORT:
            raise ValueError("pad_mode must be 'zero-pad' or 'power-law'")

        # --- IR resummation configuration ---
        self.do_irres = do_irres
        self.r_bao = r_bao
        self.lambda_ir = lambda_ir
        self.irres_method = irres_method
        if counterterm_base is None:
            counterterm_base = 'linear_ir_resum' if do_irres else 'linear'
        self._counterterms = Counterterms(counterterm_base)
        if self._counterterms.base == 'zeldovich':
            raise ValueError("EPT does not support counterterm_base='zeldovich'")

        # --- computation mode ---
        self.subtract_k0_limit = subtract_k0_limit
        self.method = method

        # --- Gauss-Legendre quadrature for mu integration ---
        self._mu_quad, self._legendre_weights = prepare_mu_gauleg(ngauss)

        # --- FFTlog grid and transform parameters ---
        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self.fftlog_settings = fftlog.FFTlogSettings() if fftlog_settings is None else fftlog_settings
        self._pad_mode = pad_mode

        # --- internal grids ---
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._mu = jnp.linspace(0., 1., 51)

        # ``x^{nu_m}`` tables of the power-law decomposition for the hybrid P13
        # blocks, built once (the matrix method builds them per call).
        if self.method == 'hybrid':
            fs = self.fftlog_settings
            self._pld_xpow_matter = fftlog.power_law_basis(self._k, fs.nu_matrix_matter)
            self._pld_xpow_p13_bias = fftlog.power_law_basis(self._k, fs.nu_matrix_p13_bias)

        self._initialize_loop_coeff()
        if self.method != 'hankel':
            self._initialize_loop_matrix()

    @property
    def counterterm_base(self):
        """Counterterm base fixed at construction time."""
        return self._counterterms.base

    def _initialize_loop_coeff(self):
        # store the names of 1-loop terms calculated with the FFTlog-based method
        fnames = sorted(glob.glob(os.path.dirname(__file__)+'/pt_coeff/22*.txt'))
        self.pkmu_coeff_names_22 = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = sorted(glob.glob(os.path.dirname(__file__)+'/pt_coeff/13*.txt'))
        self.pkmu_coeff_names_13 = [re.split('/', fname)[-1][:-4] for fname in fnames]

        lnm_list = []
        coeff_info = []
        for name in self.pkmu_coeff_names_22:
            str_list = re.split('_', name)
            l, n, m = int(str_list[1]), int(str_list[2]), int(str_list[3])
            lnm_list.append([l, n, m])
            coeff_file = glob.glob(os.path.dirname(__file__)+'/pt_coeff/%s.txt' % (name))[0]
            coeff_info.append(pt_coeff.get_coeff_info(coeff_file))

        max_len = max(len(lst) for lst in coeff_info)
        padded_coeff_info = []
        for lst in coeff_info:
            padded = lst + [[0, 0, 0, 0, 0, 0, 0.0]] * (max_len - len(lst))
            padded_coeff_info.append(padded)
        
        # P22 blocks are split statically by m: m=0 blocks carry a genuine k->0
        # constant and are residualized, m>0 blocks are plain k^m B(k).
        # ``_lnm_22_order`` maps the concatenated (m=0, m>0) result back to
        # the ``coeff_info_22`` row order.
        lnm = _np.asarray(lnm_list)
        idx_m0 = _np.flatnonzero(lnm[:, 2] == 0)
        idx_m = _np.flatnonzero(lnm[:, 2] > 0)
        self._lnm_22_m0 = jnp.asarray(lnm[idx_m0], dtype=jnp.int32)
        self._lnm_22_m = jnp.asarray(lnm[idx_m], dtype=jnp.int32)
        self._lnm_22_order = jnp.asarray(_np.argsort(_np.concatenate([idx_m0, idx_m])), dtype=jnp.int32)
        self.coeff_info_22 = jnp.array(padded_coeff_info)

        lnm_list = []
        coeff_info = []
        for name in self.pkmu_coeff_names_13:
            str_list = re.split('_', name)
            l, n, m = int(str_list[1]), int(str_list[2]), int(str_list[3])
            lnm_list.append([l, n, m])
            coeff_file = glob.glob(os.path.dirname(__file__)+'/pt_coeff/%s.txt' % (name))[0]
            coeff_info.append(pt_coeff.get_coeff_info(coeff_file))

        max_len = max(len(lst) for lst in coeff_info)
        padded_coeff_info = []
        for lst in coeff_info:
            padded = lst + [[0, 0, 0, 0, 0, 0, 0.0]] * (max_len - len(lst))
            padded_coeff_info.append(padded)
        
        self._lnm_13 = jnp.array(lnm_list)
        self.coeff_info_13 = jnp.array(padded_coeff_info)

        # set the Hankel transforms
        ln_pairs = [[0,0], [0,-2], [1,-1], [2,0], [2,-2], [3,-1], [4,0]]
        self._ln_list = jnp.array(ln_pairs)
        self._ln_list_static = tuple((int(l), int(n)) for l, n in ln_pairs)
        self._ln_index = {pair: i for i, pair in enumerate(self._ln_list_static)}
        lmax = jnp.max(self._ln_list[:, 0])
        self._set_hankel(lmax)

    def _set_hankel(self, lmax):
        """Grids, kernels and static support weights of the FFTlog chain.

        The chain is ``P(k) -> xi_l^n(q) -> P22/P13(k)``.  The input ``P(k)`` is
        padded once (``pad_mode``); the forward output ``xi_l^n`` stays on its
        padded per-l q grid and is handed to the backward transforms directly.

        * Forward transforms: one rfft of the padded input per bias group
          (``nu_pld`` for the n != 0 rows, ``nu_pld_n0`` for the n = 0 rows) is
          shared by all (l, n) rows; row (l, n) uses the kernel ``g_l`` at the
          bias ``nu_eff = nu_pld + n + 3`` (the ``k^{n+3}`` factor is absorbed
          into the bias) and the low-ringing phase of the ordinary bias ``nu``,
          so it lands on the per-l grid ``_q_padded[l]``.
        * Backward transforms: kernel ``g_l`` at ``nu`` with the same per-l
          phase, so a source on ``_q_padded[l]`` lands exactly on
          ``_k_padded`` (the conjugate grid relation); the ``m = 0`` P22 blocks
          additionally use the residual kernel ``j_0 - 1`` at ``nu_residual``
          on its own grid ``_q_p22_residual_padded``.
        * Support weights: the input carries the trapezoid end weights of its
          model (``_end_weights_input``); the forward output is restricted to
          its support ``'lower_and_core'`` (``_mask_xi``: the lower part of the
          padded band is the physical q -> 0 plateau, the upper part is the
          rounding floor of an already-decayed transform); the backward
          sources built from it (P13: ``q^2 xi_l^n``, P22: ``q^3 (xi_l^n)^2``)
          carry the end weights of that support (``_end_weights_xi``), so the
          backward transforms' m = 0 coefficient is a trapezoid sum.
        """
        fs = self.fftlog_settings
        l_list = jnp.arange(lmax + 1)

        self._npad = max(1, int(fs.npad_factor * self._nfft))
        self._grid = fftlog.LogGrid.from_core(self._k, self._npad)
        self._k_padded = self._grid.x

        # Ordinary kernels: the k^n factor is absorbed into fx, so nu does not
        # change with n.
        lnxy = jnp.array([
            fftlog.low_ringing_phase(l, fs.nu, self._grid) for l in l_list
        ])
        self._q_padded = jnp.array([
            fftlog.output_grid(self._grid, lnxy[l]) for l in l_list
        ])
        self._u_m = jnp.array([
            fftlog.hankel_kernel(l, fs.nu, self._grid, lnxy[l]) for l in l_list
        ])

        # Forward kernels of the shared-rfft (PLD) form:
        #   c_m = rfft(P(k)/(2 pi^2) * k_padded^{-nu_pld}) once per bias group
        #   u_m[l, n] = e^{-i eta lnxy[l]} g_l(nu_pld + n + 3 + i eta)
        #   xi_l^n(q) = irfft(conj(c_m u_m[l, n])) * q_padded[l]^{-(nu_pld + n + 3)}
        def build_forward(nu):
            u_ms, q_factors = [], []
            for l, n in self._ln_list_static:
                nu_eff = nu + n + 3
                # Low-ring phase from the ordinary nu, kernel bias from nu_eff
                u_ms.append(fftlog.hankel_kernel(l, nu_eff, self._grid, lnxy[l]))
                q_factors.append(self._q_padded[l] ** (-nu_eff))
            return jnp.stack(u_ms, axis=0), jnp.stack(q_factors, axis=0)

        self._xi_pld_u_m, self._xi_pld_q_factor = build_forward(fs.nu_pld)
        self._xi_pld_u_m_n0, self._xi_pld_q_factor_n0 = build_forward(fs.nu_pld_n0)
        self._xi_pld_n0_mask = jnp.array(
            [n == 0 for _, n in self._ln_list_static], dtype=bool
        )

        # Residual kernel j_0 - 1 of the m = 0 P22 blocks.  It has a different
        # convergence strip from the ordinary j_l, hence its own bias.
        lnxy_res = fftlog.low_ringing_phase(0, fs.nu_residual, self._grid)
        self._q_p22_residual_padded = fftlog.output_grid(self._grid, lnxy_res)
        self._u_m_p22_residual = fftlog.hankel_kernel(0, fs.nu_residual, self._grid, lnxy_res)

        # Static support weights (see the docstring).
        self._end_weights_input = fftlog.trapezoid_weights(
            fftlog.INPUT_SUPPORT[self._pad_mode], self._nfft, self._npad, self._grid.dln)
        self._end_weights_xi = fftlog.trapezoid_weights(
            'lower_and_core', self._nfft, self._npad, self._grid.dln)
        self._mask_xi = jnp.where(self._end_weights_xi > 0, 1.0, 0.0)

    def _initialize_loop_matrix(self):
        # store the names of 1-loop terms calculated with the FFTlog-based method
        fnames = sorted(glob.glob(os.path.dirname(__file__)+'/pt_matrix/22*.txt'))
        self.pkmu_term_names_22 = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = sorted(glob.glob(os.path.dirname(__file__)+'/pt_matrix/13*.txt'))
        self.pkmu_term_names_13 = [re.split('/', fname)[-1][:-4] for fname in fnames]

        # Group tags: matter vs bias-operator blocks, used to assign index
        # subsets; the decomposition bias of each group is set by
        # ``fftlog_settings`` (see ``_set_matrix``).
        matter = utils._NU_GROUP_MATTER_TAG
        is_matter_22 = [utils.get_nu_group_tag_from_name(name) == matter for name in self.pkmu_term_names_22]
        is_matter_13 = [utils.get_nu_group_tag_from_name(name) == matter for name in self.pkmu_term_names_13]

        # precompute the PT matrices with the right nu for each block
        matrix = self._set_matrix(self.pkmu_term_names_22 + self.pkmu_term_names_13)

        self.matrices_22 = jnp.array([matrix[name] for name in self.pkmu_term_names_22])
        self.matrices_13 = jnp.array([matrix[name] for name in self.pkmu_term_names_13])

        self._idx_22_nu1 = jnp.array([i for i, m in enumerate(is_matter_22) if m], dtype=jnp.int32)
        self._idx_22_nu2 = jnp.array([i for i, m in enumerate(is_matter_22) if not m], dtype=jnp.int32)
        self._idx_13_nu1 = jnp.array([i for i, m in enumerate(is_matter_13) if m], dtype=jnp.int32)
        self._idx_13_nu2 = jnp.array([i for i, m in enumerate(is_matter_13) if not m], dtype=jnp.int32)

        self.matrices_22_nu1 = self.matrices_22[self._idx_22_nu1]
        self.matrices_22_nu2 = self.matrices_22[self._idx_22_nu2]
        self.matrices_13_nu1 = self.matrices_13[self._idx_13_nu1]
        self.matrices_13_nu2 = self.matrices_13[self._idx_13_nu2]

        def get_degree_vector(name):
            d = utils.get_degree_dict(name)
            if name in self.pkmu_term_names_22:
                return [d['mu'], d['f'], d['b1'], d['b2'], d['bG2'], 0]
            else:
                return [d['mu'], d['f'], d['b1'], 0, d['bG2'], d['bGamma3']]

        self.degrees_22 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_22])
        self.degrees_13 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_13])

    def _set_matrix(self, names):
        """PT matrices of the given blocks at the decomposition bias of their group.

        Matter blocks use ``nu_matrix_matter``; bias-operator blocks use
        ``nu_matrix_p22_bias`` for P22 -- above -3/2, outside the convergence
        strip -3 < nu < -3/2 of those blocks, so that the decomposition
        evaluates I(k) - I(0) by analytic continuation -- and
        ``nu_matrix_p13_bias`` for P13.
        """
        fs = self.fftlog_settings
        mat = {}
        for name in names:
            mat_file = glob.glob(os.path.dirname(__file__)+'/pt_matrix/%s.txt' % (name))[0]
            if '22' in name:
                mat[name] = pt_matrix.PTMatrix22(mat_file)
            elif '13' in name:
                mat[name] = pt_matrix.PTMatrix13(mat_file)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))

        # precompute the PT matrices for appropriate FFT settings.
        eta_m = fftlog.LogGrid(self._k, 0).eta_full
        matrix = {}

        for name in names:
            if utils.get_nu_group_tag_from_name(name) == utils._NU_GROUP_MATTER_TAG:
                nu = fs.nu_matrix_matter
            elif name in self.pkmu_term_names_22:
                nu = fs.nu_matrix_p22_bias
            else:
                nu = fs.nu_matrix_p13_bias

            if '22' in name:
                nu_m = -0.5 * (nu + eta_m * 1j)
                nu_m1, nu_m2 = jnp.meshgrid(nu_m, nu_m)
                matrix[name] = mat[name](nu_m1, nu_m2).T

            elif '13' in name:
                nu_m1 = -0.5 * (nu + eta_m * 1j)
                matrix[name] = mat[name](nu_m1)
        
        return matrix
    
    def get_pkmu_grid(self, pk_data, params):
        """Compute P(k, mu) on the internal (k, mu) grid including IR resummation and counterterms.
        """

        f = params.f
        bias_a = params.bias
        bias_b = params.bias_b
        ctr = params.ctr
        stoch = params.stoch
        
        if self.do_irres:
            # tree + 1-loop
            pk_nw_data = ir_resum.get_pk_nw(pk_data, params.h, method=self.irres_method)
            pk_nw, pk_w, damp_fac = self._get_irres_components(pk_data, pk_nw_data, f)
            pkmu = self.get_pkmu_irres_LO_NLO(pk_nw, pk_w, damp_fac, f, bias_a, bias_b)

            # Counterterms use the constructor-selected base spectrum.
            if self.counterterm_base == 'linear':
                pk_ctr_base = (pk_nw + pk_w)[:, None]
            else:
                pk_ctr_base = pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None]
            pkmu_ctr_k2 = self.get_pkmu_ctr_k2(pk_ctr_base, f, ctr)
            pkmu_ctr_k4 = self.get_pkmu_ctr_k4(pk_ctr_base, f, ctr)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        else:
            pk = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
            # tree + 1-loop
            pkmu_tree = self.get_pkmu_lin(self._k, self._mu, pk_data, f, bias_a, bias_b)
            pkmu_1loop = self.get_pkmu_1loop(pk, f, bias_a, bias_b)
            pkmu = pkmu_tree + pkmu_1loop

            # Counterterms may request IR-resummed linear P even when the perturbative body itself is not IR resummed.
            if self.counterterm_base == 'linear_ir_resum':
                pk_nw_data = ir_resum.get_pk_nw(pk_data, params.h, method=self.irres_method)
                pk_nw, pk_w, damp_fac = self._get_irres_components(pk_data, pk_nw_data, f)
                pk_ctr_base = pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None]
            else:
                pk_ctr_base = pk[:, None]
            pkmu_ctr_k2 = self.get_pkmu_ctr_k2(pk_ctr_base, f, ctr)
            pkmu_ctr_k4 = self.get_pkmu_ctr_k4(pk_ctr_base, f, ctr)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        
        pkmu_stoch = self.get_pkmu_stoch(self._k, self._mu, stoch)
        pkmu = pkmu + pkmu_stoch
        
        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """Compute P(k, mu) at arbitrary (k, mu) via 2D spline interpolation of the internal grid."""
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        pkmu_grid = self.get_pkmu_grid(pk_data, params)

        # mapping of (k, mu)
        k_true, mu_true = get_k_mu_true_for_ap(k, mu, alpha_perp, alpha_para)
        # P(k, mu) of density tracers is even in mu.  The internal grid covers mu in [0, 1],
        # so reflect negative mu_true here instead of letting interp2d clamp it to mu=0.
        mu_true = jnp.abs(mu_true)

        # 2D interpolation
        xq = jnp.log(k_true)
        yq = mu_true[None, :]
        pkmu = spline.interp2d(xq, yq, jnp.log(self._k), self._mu, pkmu_grid)
        pkmu = pkmu / (alpha_perp**2 * alpha_para)

        return pkmu

    @partial(jit, static_argnames=['self'])
    def get_pk_ells(self, k, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        """Compute multipole power spectra P_0, P_2, P_4 via Gauss-Legendre quadrature."""
        pkmu = self.get_pkmu(k, self._mu_quad, pk_data, params, alpha_perp, alpha_para)
        pk_ells = get_legendre_multipoles(pkmu, self._legendre_weights)  # (3, nk)
        return pk_ells

    @partial(jit, static_argnames=['self'])
    def get_pkmu_lin(self, k, mu, pk_data, f, bias_a, bias_b):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        b1_a, b1_b = bias_a[0], bias_b[0]
        Z1_a = b1_a + f * mu**2
        Z1_b = b1_b + f * mu**2
        pk_lin = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pkmu = jnp.outer(pk_lin, Z1_a * Z1_b)

        return pkmu

    def get_pkmu_terms_22(self, p_q_1, p_q_2):
        """Compute per-block P22 terms on the internal k-grid.

        p_q_1: PLD data at ``nu_matrix_matter`` (nu1 group: matter blocks)
        p_q_2: PLD data at ``nu_matrix_p22_bias`` (nu2 group: all bias-operator blocks)
        Both groups use nu > -3/2 so the decomposition evaluates I(k)-I(0) via
        analytic continuation, so no explicit DC subtraction is needed.
        Per-k cost: O(N^2) per term
        """
        k = self._k

        diag22_1 = jnp.einsum('tnm,nj,mj->tj', self.matrices_22_nu1, p_q_1, p_q_1)
        diag22_2 = jnp.einsum('tnm,nj,mj->tj', self.matrices_22_nu2, p_q_2, p_q_2)
        diag22 = jnp.zeros((self.matrices_22.shape[0], self._nfft), dtype=diag22_1.dtype)
        diag22 = diag22.at[self._idx_22_nu1].set(diag22_1)
        diag22 = diag22.at[self._idx_22_nu2].set(diag22_2)

        return (k**3)[None, :] * jnp.real(diag22)
    
    def get_pkmu_terms_13(self, p_q_1, p_q_2, pk):
        """Compute per-block P13 terms on the internal k-grid.
        Per-k cost: O(N) for each term.
        """
        k = self._k

        Mq_13_1 = jnp.einsum('tn,nj->tj', self.matrices_13_nu1, p_q_1)
        Mq_13_2 = jnp.einsum('tn,nj->tj', self.matrices_13_nu2, p_q_2)
        Mq_13 = jnp.zeros((self.matrices_13.shape[0], self._nfft), dtype=Mq_13_1.dtype)
        Mq_13 = Mq_13.at[self._idx_13_nu1].set(Mq_13_1)
        Mq_13 = Mq_13.at[self._idx_13_nu2].set(Mq_13_2)
        pk_13  = (k**3 * pk)[None, :] * jnp.real(Mq_13)

        return pk_13
    
    @staticmethod
    def get_Z3_UV(mu, f, b1):
        return - 61./315. * b1 \
            + ((- 3./5. + 2./105. * b1) * f + (- 16./35. - 1./3. * b1) * f**2) * mu**2 \
            + ((- 46./105.) * f**2 + (- 1./3.) * f**3) * mu**4
    
    def get_pkmu_13_UV(self, pk, f, bias_a, bias_b):
        k = self._k
        mu = self._mu

        b1_a, b1_b = bias_a[0], bias_b[0]
        Z1_a = b1_a + f * mu**2
        Z1_b = b1_b + f * mu**2
        Z3_UV_a = self.get_Z3_UV(mu, f, b1_a)
        Z3_UV_b = self.get_Z3_UV(mu, f, b1_b)
        
        Z1Z3_UV = (Z1_a * Z3_UV_b + Z1_b * Z3_UV_a) / 2

        pk_int = self._get_grid_pk_int(pk)
        pkmu = jnp.outer(k**2 * pk * pk_int, Z1Z3_UV)

        return pkmu
    
    def get_pkmu_22_pld(self, pkmu_terms, pk, f, bias_a, bias_b):

        mu_pow, coeffs = eval_power_coeffs(self.degrees_22, f, bias_a, bias_b)
        mu_powers = self._mu[None, :] ** mu_pow[:, None]

        pkmu = jnp.sum(coeffs[:, None, None] * pkmu_terms[:, :, None] * mu_powers[:, None, :], axis=0)

        # The b2^2-only term was computed with \nu > -3/2 (nu3 group), so the FFTlog naturally gives I_{d2d2}(k) - I_{d2d2}(0).  
        # When subtract_k0_limit=False, add back I_{d2d2}(0) = 2 * k0_pk via the Parseval identity, which restores the full I_{d2d2}(k).
        # The contribution to P22 is b2^2/4 * I_{d2d2}(0) = b2^2/4 * 2·k0_pk = b2^2/2 * k0_pk.
        if not self.subtract_k0_limit:
            pkmu = pkmu + self._get_pkmu_22_k0_limit(pk, bias_a, bias_b)

        return pkmu  # shape: (nk, nmu)
    
    def get_pkmu_13_pld(self, pkmu_terms, pk, f, bias_a, bias_b):

        mu_pow, coeffs = eval_power_coeffs(self.degrees_13, f, bias_a, bias_b)
        mu_powers = self._mu[None, :] ** mu_pow[:, None]

        pkmu = jnp.sum(coeffs[:, None, None] * pkmu_terms[:, :, None] * mu_powers[:, None, :], axis=0)
        pkmu = pkmu + self.get_pkmu_13_UV(pk, f, bias_a, bias_b)

        return pkmu  # shape: (nk, nmu)
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop_pld(self, pk, f, bias_a, bias_b):
        fs = self.fftlog_settings
        # matter group: P22 nu1 group and P13 nu1 group
        p_q_1 = fftlog.power_law_decomposition(pk, self._k, fs.nu_matrix_matter)
        # P22 bias-operator blocks (including b2^2): nu > -3/2, so the
        # decomposition evaluates I(k)-I(0) for all blocks via analytic continuation.
        p_q_2 = fftlog.power_law_decomposition(pk, self._k, fs.nu_matrix_p22_bias)
        # P13 bias-operator blocks only (their convergence strip requires this nu)
        p_q_3 = fftlog.power_law_decomposition(pk, self._k, fs.nu_matrix_p13_bias)

        pkmu_terms_22 = self.get_pkmu_terms_22(p_q_1, p_q_2)
        pkmu_terms_13 = self.get_pkmu_terms_13(p_q_1, p_q_3, pk)

        pkmu_22 = self.get_pkmu_22_pld(pkmu_terms_22, pk, f, bias_a, bias_b)
        pkmu_13 = self.get_pkmu_13_pld(pkmu_terms_13, pk, f, bias_a, bias_b)

        return pkmu_22 + pkmu_13 # shape: (nk, nmu)

    def _pad_input(self, array):
        """Padded, end-weighted copy of a core-grid input array.

        This is the array the forward FFTlog actually transforms, and it is
        also the array whose zero-lag moments define the ``pk_int``-type
        constants (see :meth:`_get_grid_pk_int`), which is why it is factored
        out here rather than inlined in the rfft.

        ``_end_weights_input`` carries the trapezoid end weights of the input
        model: 1/2 on the k_min and k_max nodes for the band-limited
        ('zero-pad') model, 1/2 on the padded corners for the continued
        ('power-law') one.
        """
        return fftlog.pad(array, self._npad, self._pad_mode, x=self._k) * self._end_weights_input

    def _get_grid_pk_int(self, pk):
        """``(1/2 pi^2) int dk P(k)`` as the zero-lag moment of the FFTlog input.

        The constant is the ``m = 0`` coefficient of the discrete transform of
        the *same* padded, end-weighted array that feeds the forward FFTlog
        (:meth:`_pad_input`), i.e. ``sum_j dln k_j P_j / (2 pi^2)``, instead of
        an integral of a separately extrapolated copy of ``P`` over a k range
        unrelated to the FFTlog band.  That keeps the constant and the
        k-dependent loop terms consistent.  The same value is used on the
        ``method='matrix'`` path, which shares the band-limited definition.
        """
        return fftlog.grid_moment(
            self._pad_input(pk), self._k_padded, self._grid.dln, 1
        ) / (2 * jnp.pi**2)

    def _get_grid_pk_int2(self, pk):
        """``(1/2 pi^2) int dk k^2 P^2(k)`` on the same band as the FFTlog input.

        Same band-limited definition as :meth:`_get_grid_pk_int`, i.e.
        ``sum_j dln k_j^3 P_j^2 / (2 pi^2)``.  The trapezoid end weights belong
        to the input ``P`` and are applied once, not once per factor, so the
        weighted array is formed as ``w * (pad P)^2`` rather than
        ``(pad P * w)^2``.
        """
        padded = fftlog.pad(pk, self._npad, self._pad_mode, x=self._k)
        return fftlog.grid_moment(
            self._end_weights_input * padded**2, self._k_padded, self._grid.dln, 3
        ) / (2 * jnp.pi**2)

    def get_xi_ln(self, l, n, array):
        """Forward FFTlog of ``P(k) k^{n+3}`` against ``j_l`` (shared-rfft form).

        ``array`` is given on the core k grid; the result is ``xi_l^n`` on the
        padded per-l grid ``_q_padded[l]``, restricted to its support (the
        lower part of the padded band is the q -> 0 plateau and is kept, the
        upper part is zeroed; see ``_set_hankel``).
        """
        fs = self.fftlog_settings
        idx = self._ln_index[(int(l), int(n))]
        if int(n) == 0:
            nu, u_m, q_factor = fs.nu_pld_n0, self._xi_pld_u_m_n0[idx], self._xi_pld_q_factor_n0[idx]
        else:
            nu, u_m, q_factor = fs.nu_pld, self._xi_pld_u_m[idx], self._xi_pld_q_factor[idx]
        xi = fftlog.hankel(nu, self._pad_input(array / (2 * jnp.pi**2)), self._k_padded,
                           None, u_m, self._npad, crop=False, y_pow=q_factor)
        return xi * self._mask_xi

    def get_xi_ln_array(self, array):
        """All cached ``(l, n)`` forward transforms of ``array`` as ``xi_ln[l, n]``.

        Same transform as :meth:`get_xi_ln`, batched: one rfft of the padded
        input per bias group (``nu_pld``, ``nu_pld_n0``) serves every row; each
        row has its own output grid and bias, whose ``q_padded[l]**(-nu_eff)``
        factors are precomputed in ``_set_hankel``.
        """
        fs = self.fftlog_settings
        fx = self._pad_input(array / (2 * jnp.pi**2))
        xis = fftlog.hankel(fs.nu_pld, fx, self._k_padded, None, self._xi_pld_u_m,
                            self._npad, crop=False, y_pow=self._xi_pld_q_factor)
        xis_n0 = fftlog.hankel(fs.nu_pld_n0, fx, self._k_padded, None, self._xi_pld_u_m_n0,
                               self._npad, crop=False, y_pow=self._xi_pld_q_factor_n0)
        xis = jnp.where(self._xi_pld_n0_mask[:, None], xis_n0 * self._mask_xi, xis * self._mask_xi)
        xi_ln = jnp.zeros((5, 3, xis.shape[-1]))
        ls = self._ln_list[:, 0]
        ns = self._ln_list[:, 1]
        return xi_ln.at[ls, ns].set(xis)

    def get_pk_ln(self, l, n, array):
        """Backward FFTlog of a q-space source back onto the core k grid.

        ``array`` lives on the padded per-l q grid ``_q_padded[l]`` (it is a
        forward output, or a product of forward outputs); the source carries
        the trapezoid end weights ``_end_weights_xi`` of its
        ``'lower_and_core'`` support.
        """
        fx = array * self._q_padded[l]**(n + 3) * self._end_weights_xi
        return fftlog.hankel(self.fftlog_settings.nu, fx, self._q_padded[l], self._k_padded,
                             self._u_m[l], self._npad, crop=True)

    def get_pk_ln_p22_residual(self, array):
        """:meth:`get_pk_ln` with the residual kernel ``j_0 - 1`` (m = 0 P22 blocks)."""
        fx = array * self._q_p22_residual_padded**3 * self._end_weights_xi
        return fftlog.hankel(self.fftlog_settings.nu_residual, fx, self._q_p22_residual_padded,
                             self._k_padded, self._u_m_p22_residual, self._npad, crop=True)

    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop_hankel(self, pk, f, bias_a, bias_b):
        """Compute 1-loop P22 + P13 using the FFTlog-Hankel method (or hybrid for P13)."""
        xi_ln = self.get_xi_ln_array(pk)
        pkmu_22 = self.get_pkmu_22_hankel(xi_ln, pk, f, bias_a, bias_b)

        if self.method == 'hankel':
            pkmu_13 = self.get_pkmu_13_hankel(xi_ln, pk, f, bias_a, bias_b)
        else:
            # hybrid: PLD for P13
            fs = self.fftlog_settings
            p_q_1 = fftlog.power_law_decomposition(
                pk, self._k, fs.nu_matrix_matter, xpow=self._pld_xpow_matter)
            p_q_2 = fftlog.power_law_decomposition(
                pk, self._k, fs.nu_matrix_p13_bias, xpow=self._pld_xpow_p13_bias)
            pkmu_13 = self.get_pkmu_13_pld(
                self.get_pkmu_terms_13(p_q_1, p_q_2, pk), pk, f, bias_a, bias_b
            )

        return pkmu_22 + pkmu_13

    def _get_pkmu_22_k0_limit(self, pk, bias_a, bias_b):
        b2_a, b2_b = bias_a[1], bias_b[1]
        return (b2_a * b2_b) / 2. * self._get_grid_pk_int2(pk)

    def get_pkmu_22_hankel(self, xi_ln, pk, f, bias_a, bias_b):
        """P22 from per-block Hankel transforms (same structure as get_pkmu_13_hankel).

        m=0 blocks contain a genuine k->0 constant, so the quantity summed is the
        residual B_i(k)-B_i(0).  It is evaluated two ways and blended on the
        dimensionless ratio r = |B(k)-B(0)|/|B(0)|: the j0-1 kernel (accurate at
        low k) and the ordinary j0 transform minus its DC (accurate at high k).
        m>0 blocks are k^m B_i(k) and use the ordinary transform only.  The two
        groups are split at construction so no transform is computed and
        discarded.

        Why the blend, and why a Gaussian weight: the j0-1 residual kernel is
        accurate where |B(k)-B(0)| is small (low k), its absolute error is tiny
        there but grows like k^2 at high k (the kernel's non-decaying -1 tail
        makes the transform sensitive to the full-range integral B(0); this is
        the rounding floor of the residual kernel at bias ``nu_residual``).  The
        ordinary j0 transform has a near-constant small absolute error, so
        D(k)-D(k_min) is accurate where |B(k)-B(0)| is comparable to |B(0)|
        (high k; flat at the 1e-6 level for k >~ 0.1) but loses precision where
        it is tiny.  The weight on the residual estimate must vanish exactly at
        large r: the former rational form 1/(1+(r/r*)^2) only saturated at
        1/(1+(1/r*)^2) = 9e-4 and so still multiplied the k^2 error of the
        residual kernel into the blend (5e-2 of B(0) at l=4, k=30).  The
        Gaussian ``w(r) = exp(-(r/blend_rstar)^2)`` agrees with the rational form
        to second order at small r and is exactly negligible for r >~ 0.3.
        Verified 2026-09-21 (notes/ept_ja.pdf section 2026-09-20 ~ 21 in the
        ps_1loop_jax repo).
        """
        rstar = self.fftlog_settings.blend_rstar

        def get_pk_lnm_22(term):
            # ordinary j0 kernel
            l, n, m = term
            xi = xi_ln[l, n]
            source = spline.interp1d(jnp.log(self._q_padded[0]),
                                     jnp.log(self._q_padded[l]), xi * xi)
            pk_ln =  4 * jnp.pi * self.get_pk_ln(0, 0, source)
            return (self._k ** m) * pk_ln

        def get_pk_lnm_22_m0(term):
            # Low-k accurate form: B_i(k)-B_i(0) from the j0-1 residual kernel.
            l, n, _ = term
            xi = xi_ln[l, n]
            source = spline.interp1d(jnp.log(self._q_p22_residual_padded),
                                     jnp.log(self._q_padded[l]), xi * xi)
            pk_ln_j0m1 = 4 * jnp.pi * self.get_pk_ln_p22_residual(source)

            # High-k accurate form: full B_i(k) from the ordinary j0 transform,
            # then subtract its own k->0 value B_i(0) (= first grid point).
            pk_ln = get_pk_lnm_22(term)
            pk_ln_j0 = pk_ln - pk_ln[:1]

            # Per-block dimensionless crossover r = |B(k)-B(0)|/|B(0)| (B(0) is the k->0 value of the ordinary j0 transform); blend the j0-1 (at low k) and j0 (at high k) results.
            denom = jnp.maximum(jnp.abs(pk_ln[:1]), jnp.finfo(pk_ln.real.dtype).tiny)
            r_ratio = jnp.abs(pk_ln_j0) / denom
            w = jnp.exp(-(r_ratio / rstar) ** 2)

            return w * pk_ln_j0m1 + (1.0 - w) * pk_ln_j0

        pk_lnm_m0 = jax.vmap(get_pk_lnm_22_m0)(self._lnm_22_m0)   # (n_m0, nk)
        pk_lnm_m = jax.vmap(get_pk_lnm_22)(self._lnm_22_m)        # (n_m,  nk)
        pk_lnm = jnp.concatenate([pk_lnm_m0, pk_lnm_m], axis=0)[self._lnm_22_order]  # (nterms, nk)

        def compute_coeffs(entry):
            mu_pow, coeffs = eval_power_coeffs(entry[:, :6], f, bias_a, bias_b)
            coeffs = entry[:, 6] * coeffs
            mu_terms = self._mu[None, :] ** mu_pow[:, None]  # (max_len, nmu)
            return jnp.sum(coeffs[:, None] * mu_terms, axis=0)  # (nmu,)

        coeff_matrix = jax.vmap(compute_coeffs)(self.coeff_info_22)  # (nterms, nmu)
        pkmu = jnp.sum(pk_lnm[:, :, None] * coeff_matrix[:, None, :], axis=0)  # (nk, nmu)

        # The m=0 blocks above are the residualized B_i(k)-B_i(0).
        # For the unsubtracted spectrum, restore the only physical k->0 constant, the b2^2 (delta^2 . delta^2) DC = b2^2/2 * k0_pk.
        if not self.subtract_k0_limit:
            pkmu = pkmu + self._get_pkmu_22_k0_limit(pk, bias_a, bias_b)
        return pkmu  # (nk, nmu)

    def get_pkmu_13_hankel(self, xi_ln, pk, f, bias_a, bias_b):
        
        def get_pk_lnm_13(term):
            l, n, m = term
            pk_ln = self.get_pk_ln(l, -1, xi_ln[l, n])  # (nk,)
            return (self._k ** m) * pk * pk_ln

        pk_lnm = jax.vmap(get_pk_lnm_13)(self._lnm_13)  # (nterms, nk)

        def compute_coeffs(entry):
            mu_pow, coeffs = eval_power_coeffs(entry[:, :6], f, bias_a, bias_b)
            coeffs = entry[:, 6] * coeffs
            mu_terms = self._mu[None, :] ** mu_pow[:, None]  # (max_len, nmu)
            return jnp.sum(coeffs[:, None] * mu_terms, axis=0)  # (nmu,)

        coeff_matrix = jax.vmap(compute_coeffs)(self.coeff_info_13)  # (nterms, nmu)
        pkmu = jnp.sum(pk_lnm[:, :, None] * coeff_matrix[:, None, :], axis=0)  # (nk, nmu)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop(self, pk, f, bias_a, bias_b):
        """Compute 1-loop P(k, mu) = P22 + P13 on the internal grid, dispatching to matrix or Hankel path."""
        if self.method == 'matrix':
            pkmu = self.get_pkmu_1loop_pld(pk, f, bias_a, bias_b)
        else:
            pkmu = self.get_pkmu_1loop_hankel(pk, f, bias_a, bias_b)
        return pkmu
    
    def get_pkmu_irres_LO_NLO(self, pk_nw, pk_w, damp_fac, f, bias_a, bias_b):
        # LO term
        b1_a, b1_b = bias_a[0], bias_b[0]
        Z1_a = b1_a + f * self._mu**2
        Z1_b = b1_b + f * self._mu**2
        pkmu_irres_tree = (Z1_a * Z1_b)[None, :] * (pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None] * (1 + damp_fac))
        
        # NLO term
        pkmu_1loop = self.get_pkmu_1loop(pk_nw + pk_w, f, bias_a, bias_b)
        pkmu_1loop_nw = self.get_pkmu_1loop(pk_nw, f, bias_a, bias_b)
        pkmu_1loop_w = pkmu_1loop - pkmu_1loop_nw
        pkmu_irres_1loop = pkmu_1loop_nw + jnp.exp(-damp_fac) * pkmu_1loop_w

        pkmu = pkmu_irres_tree + pkmu_irres_1loop

        return pkmu
    
    def _get_irres_components(self, pk_data, pk_nw_data, f):
        k = self._k
        mu = self._mu

        # wiggly-non-wiggly decomposition
        pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_nw = get_pk(k, pk_nw_data, kmin=self._kmin, kmax=self._kmax)
        pk_w = pk - pk_nw

        # BAO damping factor in redshift space
        Sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.r_bao, self.lambda_ir)
        dSigma2 = ir_resum.get_dSigma2(pk_nw_data, self.r_bao, self.lambda_ir)
        Sigma2_s = (1 + mu**2 * f * (2 + f)) * Sigma2 + f**2 * mu**2 * (mu**2 - 1) * dSigma2
        damp_fac = jnp.outer(k**2, Sigma2_s)

        return pk_nw, pk_w, damp_fac
    
    def get_pkmu_ctr_k2(self, pk, f, ctr):
        return self._counterterms.leading(
            self._k[:, None], self._mu[None, :], f, ctr, pk
        )

    def get_pkmu_ctr_k4(self, pk, f, ctr):
        return self._counterterms.nlo(
            self._k[:, None], self._mu[None, :], f, ctr, pk
        )

    def get_pkmu_stoch(self, k, mu, stoch):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)
        return stochasticity(k[:, None], mu[None, :], stoch)

    @partial(jit, static_argnames=['self'])
    def get_xi_ells(self, r, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        r = jnp.atleast_1d(r)

        # This helper is an approximate configuration-space projection.
        k = jnp.geomspace(max(self._kmin, 1e-4), min(self._kmax, 1.0), min(self._nfft, 128))
        pk_ells = self.get_pk_ells(k, pk_data, params, alpha_perp, alpha_para)

        pk0 = get_pk(self._k, jnp.stack([k, pk_ells[0]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk2 = get_pk(self._k, jnp.stack([k, pk_ells[1]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk4 = get_pk(self._k, jnp.stack([k, pk_ells[2]], axis=0), kmin=self._kmin, kmax=self._kmax)

        # ``get_xi_ln`` returns xi_l on the padded per-l grid ``_q_padded[l]``.
        xi0 = spline.interp1d(jnp.log(r), jnp.log(self._q_padded[0]), self.get_xi_ln(0, 0, pk0))
        xi2 = spline.interp1d(jnp.log(r), jnp.log(self._q_padded[2]), -self.get_xi_ln(2, 0, pk2))
        xi4 = spline.interp1d(jnp.log(r), jnp.log(self._q_padded[4]), self.get_xi_ln(4, 0, pk4))

        xi_ells = jnp.stack([xi0, xi2, xi4], axis=0)
        return xi_ells
