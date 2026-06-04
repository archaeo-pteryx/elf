import os
import glob, re
import warnings

import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp

from .power_law_decomp import get_decomp_data
from . import hankel

from . import pt_coeff
from . import pt_matrix
from . import utils_loop
from .utils_loop import get_pk, get_pk_int, get_pk_int2
from .multipole import prepare_mu_gauleg, get_legendre_multipoles, get_k_mu_true_for_ap

from . import ir_resum
from . import spline


class PowerSpectrum1LoopEPT:

    _NU_P22_RESIDUAL = -1.7

    def __init__(self,
                 do_irres=True,
                 r_bao=110.,
                 k_IR=0.2,
                 irres_method='DST',
                 subtract_k0_limit=True,
                 method='hankel',
                 kmin_fft=1e-5,
                 kmax_fft=1e2,
                 nfft=512,
                 hankel_nu=1.1,
                 hankel_k_damp=None,
                 hankel_forward_mode='fftlog',
                 hankel_forward_pld_nu=-0.3,
                 hankel_forward_pld_nu_n0=-1.5,
                 hankel_p22_basis='uv-safe',
                 hankel_c_window_width=0.0,
                 hankel_window_factor=None,
                 hankel_npad_factor=0.5,
                 hankel_forward_pad_mode='power-law',
                 hankel_backward_pad_mode='smooth-zero-pad',
                 hankel_p22_backward_mode='standard',
                 hankel_nu_p22_backward=-2.0,
                 ngauss=4,
                 ):

        if method not in ('matrix', 'hankel', 'hybrid'):
            raise ValueError(f"method must be 'matrix', 'hankel', or 'hybrid', got {method!r}")

        # --- IR resummation configuration ---
        self.do_irres = do_irres
        self.r_bao = r_bao
        self.k_IR = k_IR
        self.irres_method = irres_method

        # --- computation mode ---
        self.subtract_k0_limit = subtract_k0_limit
        self.method = method

        # --- Gauss-Legendre quadrature for mu integration ---
        self._mu_quad, self._legendre_weights = prepare_mu_gauleg(ngauss)

        # --- FFTLog grid and transform parameters ---
        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self._nu_hankel = hankel_nu
        # k_damp: cutoff for Gaussian forward damping exp(-(k/k_damp)^2); None = no damping.
        self._hankel_k_damp = hankel_k_damp
        self._hankel_damp_kind = 'exp'
        self._hankel_damp_power = 2.0
        if hankel_forward_mode not in ('fftlog', 'pld'):
            raise ValueError("hankel_forward_mode must be 'fftlog' or 'pld'")
        self._hankel_forward_mode = hankel_forward_mode
        self._nu_hankel_forward_pld = hankel_forward_pld_nu
        self._nu_hankel_forward_pld_n0 = hankel_forward_pld_nu_n0
        if hankel_p22_basis not in ('original', 'uv-safe'):
            raise ValueError("hankel_p22_basis must be 'original' or 'uv-safe'")
        self._hankel_p22_basis = hankel_p22_basis
        self._hankel_c_window_width = hankel_c_window_width
        self._hankel_window_factor = hankel_window_factor  # None -> no spatial window
        self._hankel_npad_factor = hankel_npad_factor
        if hankel_forward_pad_mode not in ('power-law', 'zero-pad', 'smooth-zero-pad'):
            raise ValueError("hankel_forward_pad_mode must be 'power-law', 'zero-pad', or 'smooth-zero-pad'")
        if hankel_backward_pad_mode not in ('power-law', 'zero-pad', 'smooth-zero-pad'):
            raise ValueError("hankel_backward_pad_mode must be 'power-law', 'zero-pad', or 'smooth-zero-pad'")
        # Warn: original basis + fftlog mode + no k_damp is catastrophic regardless of padding.
        # k^{n+3}P(k) for n=2,3 is large at kmax_fft (e.g. k^5 P(100) \sim 4.5e6 for Planck18);
        # any FFT of this without cutoff aliases badly into low-k outputs.
        if (hankel_p22_basis == 'original'
                and hankel_k_damp is None
                and hankel_forward_mode == 'fftlog'):
            warnings.warn(
                "hankel_p22_basis='original' with no k_damp (hankel_k_damp=None) causes "
                "catastrophic aliasing in the n=2,3 forward transform blocks "
                "(k^{n+3}P(k) is large at kmax_fft). "
                "Set hankel_k_damp to a finite value (e.g. kmax_fft/90) "
                "or use hankel_p22_basis='uv-safe'.",
                UserWarning,
                stacklevel=2,
            )
        self._hankel_forward_pad_mode = hankel_forward_pad_mode
        self._hankel_backward_pad_mode = hankel_backward_pad_mode
        if hankel_p22_backward_mode not in ('standard', 'pld'):
            raise ValueError("hankel_p22_backward_mode must be 'standard' or 'pld'")
        self._hankel_p22_backward_mode = hankel_p22_backward_mode
        self._nu_p22_backward = hankel_nu_p22_backward

        # --- internal grids ---
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._mu = jnp.linspace(0., 1., 51)

        # nu for P22 matrix groups: choose \nu > -3/2 so FFTLog naturally evaluates I(k)-I(0)
        # via analytic continuation.  Convergence strip is -3 < \nu < -3/2
        # for all bias-operator P22 blocks (b2, bG2 terms); \nu=-1.0 is outside the strip for all.
        self._nu_ept_b2sq   = -1.0   # b2^2-only block (nu3 group)
        self._nu_ept_22_nu2 = -1.0   # mixed bias blocks (nu2 group): same principle
        self._pld_xpow_nu1 = self._get_pld_xpow(-0.3) if self.method == 'hybrid' else None
        self._pld_xpow_nu2 = self._get_pld_xpow(-1.6) if self.method == 'hybrid' else None

        self._initialize_loop_coeff()
        if self.method != 'hankel':
            self._initialize_loop_matrix()

    def _get_pld_xpow(self, nu):
        dln = jnp.log(self._k[1] / self._k[0])
        eta_m = 2 * jnp.pi / (self._nfft * dln) * (jnp.arange(self._nfft) - self._nfft // 2)
        nu_m = nu + eta_m * 1j
        return self._k[None, :]**nu_m[:, None]

    def _get_decomp_pq_from_xpow(self, nu, xpow, fx):
        dln = jnp.log(self._k[1] / self._k[0])
        eta_m = 2 * jnp.pi / (self._nfft * dln) * (jnp.arange(self._nfft) - self._nfft // 2)
        c_m = jnp.fft.rfft(fx * self._k**(-nu), norm='forward')
        c_m = self._k[0]**(-eta_m * 1j) * jnp.concatenate([c_m[::-1][:-1].conj(), c_m[:-1]], axis=0)
        return c_m[:, None] * xpow
    
    def _initialize_loop_coeff(self):
        # store the names of 1-loop terms calculated with the FFTLog-based method
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_coeff/22*.txt')
        self.pkmu_coeff_names_22 = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_coeff/13*.txt')
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
        
        self._lnm_list_22 = jnp.array(lnm_list)
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
        
        self._lnm_list_13 = jnp.array(lnm_list)
        self.coeff_info_13 = jnp.array(padded_coeff_info)

        # set the Hankel transforms
        self._ln_list = jnp.array([[0,0], [0,-2], [1,-1], [2,0], [2,-2], [3,-1], [4,0]])
        lmax = jnp.max(self._ln_list[:, 0])
        self._set_hankel(lmax)

    def _set_hankel(self, lmax):
        l_list = jnp.arange(lmax + 1)

        # Mellin argument for the low-ringing condition and u_m. The k^n factor
        # is absorbed into fx, so nu does not change with n.
        _mellin_arg = self._nu_hankel

        self._npad = max(1, int(self._hankel_npad_factor * self._nfft))
        self._k_padded = hankel.get_log_extrap(self._k, self._npad, self._npad)

        n_window = 0 if self._hankel_window_factor is None else int(self._hankel_window_factor * self._nfft)
        self._w_m = hankel.get_window(n_window, self._k_padded)

        nfft = len(self._k_padded)
        dln = jnp.log(self._k_padded[1] / self._k_padded[0])
        eta_m = 2 * jnp.pi * jnp.fft.rfftfreq(nfft, d=1.0) / dln

        g_l = jnp.array([hankel.get_g_l(l, _mellin_arg + 1j * eta_m) for l in l_list])

        lnxy = jnp.array([dln * jnp.angle(hankel.get_g_l(l, _mellin_arg + 1j * jnp.pi / dln)) / jnp.pi for l in l_list])
        self._lnxy = lnxy  # stored for PLD u_m construction
        self._q_padded = jnp.array([jnp.exp(lnxy[l] - dln) / self._k_padded[::-1] for l in l_list])
        self._q = jnp.array([self._q_padded[l][self._npad:-self._npad] for l in l_list])

        self._u_m = jnp.array([jnp.exp(lnxy[l])**(-1j*eta_m) * g_l[l] for l in l_list])

        self._set_hankel_forward_pld_kernel()

        # Forward damping: Gaussian exp(-(k/k_high)^2). k_high=inf -> no damping.
        self._k_high = self._hankel_k_damp if self._hankel_k_damp is not None else jnp.inf
        # No q-side damping for backward transforms.
        self._q_high = jnp.full(len(l_list), jnp.inf)
        
        c_window_width = self._hankel_c_window_width
        self._w_m_freq = hankel.c_window(jnp.arange(nfft//2+1), int(c_window_width * (nfft//2+1)))
        self._set_hankel_p22_residual()

    def _set_hankel_forward_pld_kernel(self):
        # PLD forward Hankel via irfft:
        #   c_m = rfft(P(k)*k^{-\nu_pld}/(2\pi^2)) once per \nu group (on padded grid)
        #   u_m[l,n] = exp(lnxy_std[l])^{-i\eta} * g_l(ν_pld+n+3+i\eta)   (uses standard q-grid)
        #   xi_l^n(q) = irfft(conj(c_m * u_m[l,n])) * q_padded[l]^{-(\nu_pld+n+3)}
        # This shares one rfft across all (l,n) with the same \nu, giving the same q-grid
        # as standard FFTLog (self._q[l]).
        nfft_pad = len(self._k_padded)
        dln = jnp.log(self._k_padded[1] / self._k_padded[0])
        eta_m = 2 * jnp.pi * jnp.fft.rfftfreq(nfft_pad, d=1.0) / dln

        use_split = (
            self._hankel_p22_basis == 'uv-safe'
            and self._hankel_forward_mode == 'pld'
        )

        def _build_pld_arrays(nu):
            u_ms, q_factors = [], []
            for l, n in self._ln_list:
                l_int, n_int = int(l), int(n)
                nu_eff = nu + n_int + 3
                g_l = jnp.asarray(hankel.get_g_l(l_int, nu_eff + 1j * eta_m))
                lnxy_l = self._lnxy[l_int]  # low-ring phase from standard \nu
                u_m = jnp.exp(lnxy_l * (-1j * eta_m)) * g_l
                q_factor = self._q_padded[l_int] ** (-nu_eff)  # (nfft_pad,)
                u_ms.append(u_m)
                q_factors.append(q_factor)
            return (jnp.stack(u_ms, axis=0),       # (n_ln, nrfft)
                    jnp.stack(q_factors, axis=0))   # (n_ln, nfft_pad)

        if use_split:
            self._xi_pld_u_m, self._xi_pld_q_factor = _build_pld_arrays(self._nu_hankel_forward_pld)
            self._xi_pld_u_m_n0, self._xi_pld_q_factor_n0 = _build_pld_arrays(self._nu_hankel_forward_pld_n0)
            self._xi_pld_n0_mask = jnp.array(
                [int(n) == 0 for _, n in self._ln_list], dtype=bool
            )
        else:
            self._xi_pld_u_m, self._xi_pld_q_factor = _build_pld_arrays(self._nu_hankel_forward_pld)
            self._xi_pld_u_m_n0 = None
            self._xi_pld_q_factor_n0 = None
            self._xi_pld_n0_mask = None

    def _set_hankel_p22_residual(self):
        (
            self._q_p22_residual_padded,
            self._q_p22_residual,
            self._u_m_p22_residual,
            self._q_high_p22_residual,
        ) = self._make_hankel_p22_residual(self._NU_P22_RESIDUAL)
        self._set_hankel_p22_backward_pld_kernel()

    def _set_hankel_p22_backward_pld_kernel(self):
        if self._hankel_p22_backward_mode == 'pld':
            (
                self._q_padded_back_pld,
                self._q_back_pld,
                self._u_m_back_pld,
                self._q_high_back_pld,
            ) = self._make_hankel_p22_backward_pld(self._nu_p22_backward)
        else:
            self._q_padded_back_pld = None
            self._q_back_pld = None
            self._u_m_back_pld = None
            self._q_high_back_pld = None

    def _make_hankel_p22_backward_pld(self, nu_back):
        """Build PLD backward Hankel kernels (l=0).

        The effective Mellin arg is nu_eff = \nu_back+3; u_m uses g_0(nu_eff+i*eta).
        Convergence strip: -3 < \nu_back < -1.
        """
        nfft = len(self._k_padded)
        dln = jnp.log(self._k_padded[1] / self._k_padded[0])
        eta_m = 2 * jnp.pi * jnp.fft.rfftfreq(nfft, d=1.0) / dln

        nu_eff = nu_back + 3
        g_0 = jnp.asarray(hankel.get_g_l(0, nu_eff + 1j * eta_m))
        lnxy = dln * jnp.angle(hankel.get_g_l(0, nu_eff + 1j * jnp.pi / dln)) / jnp.pi
        q_padded = jnp.exp(lnxy - dln) / self._k_padded[::-1]
        q = q_padded[self._npad:-self._npad]
        u_m = jnp.exp(lnxy)**(-1j * eta_m) * g_0
        return q_padded, q, u_m, jnp.inf

    def _make_hankel_p22_residual(self, nu):
        nfft = len(self._k_padded)
        dln = jnp.log(self._k_padded[1] / self._k_padded[0])
        eta_m = 2 * jnp.pi * jnp.fft.rfftfreq(nfft, d=1.0) / dln

        g_0 = jnp.asarray(hankel.get_g_l(0, nu + 1j * eta_m))
        lnxy = dln * jnp.angle(hankel.get_g_l(0, nu + 1j * jnp.pi / dln)) / jnp.pi
        q_padded = jnp.exp(lnxy - dln) / self._k_padded[::-1]
        q = q_padded[self._npad:-self._npad]
        u_m = jnp.exp(lnxy)**(-1j * eta_m) * g_0
        return q_padded, q, u_m, jnp.inf
    
    def _initialize_loop_matrix(self):
        # store the names of 1-loop terms calculated with the FFTLog-based method
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/22*.txt')
        self.pkmu_term_names_22 = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/13*.txt')
        self.pkmu_term_names_13 = [re.split('/', fname)[-1][:-4] for fname in fnames]

        # precompute the PT matrices
        matrix = self._set_matrix(self.pkmu_term_names_22 + self.pkmu_term_names_13)

        ## create arrays of matrices and degrees for 22 and 13, according to the degrees of \mu
        nus_22 = [utils_loop.get_nu_from_name(name) for name in self.pkmu_term_names_22]
        nus_13 = [utils_loop.get_nu_from_name(name) for name in self.pkmu_term_names_13]
        self.nus_22 = jnp.array(nus_22)
        self.nus_13 = jnp.array(nus_13)

        self.matrices_22 = jnp.array([matrix[name] for name in self.pkmu_term_names_22])
        self.matrices_13 = jnp.array([matrix[name] for name in self.pkmu_term_names_13])
        self._idx_22_nu1 = jnp.array([i for i, nu in enumerate(nus_22) if abs(nu + 0.3) < 1e-12], dtype=jnp.int32)
        self._idx_13_nu1 = jnp.array([i for i, nu in enumerate(nus_13) if abs(nu + 0.3) < 1e-12], dtype=jnp.int32)
        self._idx_13_nu2 = jnp.array([i for i, nu in enumerate(nus_13) if abs(nu + 1.6) < 1e-12], dtype=jnp.int32)
        self.matrices_13_nu1 = self.matrices_13[self._idx_13_nu1]
        self.matrices_13_nu2 = self.matrices_13[self._idx_13_nu2]

        # nu3 group: I_{d2d2} (b2^2-only) with \nu > -3/2 so FFTLog gives I(k)-I(0).
        # Convergence range for I_{d2d2} is -3 < \nu < -3/2.
        # Outside this strip, k^{2\nu+3} -> 0 as k->0, so the sum naturally evaluates
        # I(k)-I(0) via analytic continuation.
        def _is_b2sq_only(name):
            if not name.startswith('22='):
                return False
            d = utils_loop.get_degree_dict(name)
            return (d.get('b2', 0) == 2 and d.get('b1', 0) == 0
                    and d.get('bG2', 0) == 0 and d.get('f', 0) == 0
                    and d.get('mu', 0) == 0)

        b2sq_idx_set = {i for i, name in enumerate(self.pkmu_term_names_22) if _is_b2sq_only(name)}
        b2sq_names = [self.pkmu_term_names_22[i] for i in sorted(b2sq_idx_set)]

        self._idx_22_nu2 = jnp.array(
            [i for i, nu in enumerate(nus_22)
             if abs(nu + 1.6) < 1e-12 and i not in b2sq_idx_set],
            dtype=jnp.int32)
        self._idx_22_nu3 = jnp.array(sorted(b2sq_idx_set), dtype=jnp.int32)

        self.matrices_22_nu1 = self.matrices_22[self._idx_22_nu1]

        # Recompute nu2 matrices with \nu=-1.0 (outside the convergence strip -3<\nu<-3/2).
        # With nu > -3/2, the FFTLog evaluates I(k)-I(0) via analytic continuation for each
        # block, so no explicit DC subtraction is needed for subtract_k0_limit=True.
        nu2_names = [self.pkmu_term_names_22[int(i)] for i in self._idx_22_nu2]
        matrix_nu2 = self._set_matrix(
            nu2_names,
            nu_override={n: self._nu_ept_22_nu2 for n in nu2_names},
        )
        self.matrices_22_nu2 = jnp.array([matrix_nu2[n] for n in nu2_names])

        matrix_nu3 = self._set_matrix(
            b2sq_names,
            nu_override={n: self._nu_ept_b2sq for n in b2sq_names},
        )
        self.matrices_22_nu3 = jnp.array([matrix_nu3[n] for n in b2sq_names])

        def get_degree_vector(name):
            d = utils_loop.get_degree_dict(name)
            if name in self.pkmu_term_names_22:
                return [d['mu'], d['f'], d['b1'], d['b2'], d['bG2'], 0]
            else:
                return [d['mu'], d['f'], d['b1'], 0, d['bG2'], d['bGamma3']]
            
        self.degrees_22 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_22])
        self.degrees_13 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_13])
        self.degrees = jnp.concatenate([self.degrees_22, self.degrees_13], axis=0)

    def _set_matrix(self, names=[], nu_override={}):
        mat = {}
        for name in names:
            mat_file = glob.glob(os.path.dirname(__file__)+'/pt_matrix/*/*/%s.txt' % (name))[0]
            if '22' in name or 'I' in name:
                mat[name] = pt_matrix.PTMatrix22(mat_file)
            elif '13' in name or 'F' in name:
                mat[name] = pt_matrix.PTMatrix13(mat_file)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))

        # precompute the PT matrices for appropriate FFT settings.
        eta_m = 2 * jnp.pi / (self._nfft * jnp.log(self._k[1] / self._k[0])) * (jnp.arange(self._nfft) - self._nfft // 2)
        matrix = {}

        for name in names:
            nu = nu_override.get(name, utils_loop.get_nu_from_name(name))

            if '22' in name or 'I' in name:
                nu_m1 = -0.5 * (nu + eta_m * 1j)
                nu_m2 = -0.5 * (nu + eta_m * 1j)
                nu_m1, nu_m2 = jnp.meshgrid(nu_m1, nu_m2)
                matrix[name] = mat[name](nu_m1, nu_m2).T

            elif '13' in name or 'F' in name:
                nu_m1 = -0.5 * (nu + eta_m * 1j)
                matrix[name] = mat[name](nu_m1)
        
        return matrix

    @partial(jit, static_argnames=['self'])
    def get_pkmu_grid(self, pk_data, params):
        """Compute P(k, mu) on the internal (k, mu) grid including IR resummation and counterterms."""
        f = params.f
        bias = params.bias
        ctr = params.ctr
        
        if self.do_irres:
            # tree + 1-loop
            pk_nw_data = ir_resum.get_pk_nw(pk_data, params.h, method=self.irres_method)
            pk_nw, pk_w, damp_fac = self._get_irres_components(pk_data, pk_nw_data, f)
            pkmu = self.get_pkmu_irres_LO_NLO(pk_nw, pk_w, damp_fac, f, bias)

            # counterterm
            pk = pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None]
            pkmu_ctr_k2 = self.get_pkmu_ctr_k2(pk, f, ctr)
            pkmu_ctr_k4 = self.get_pkmu_ctr_k4(pk, f, bias, ctr)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        else:
            pk = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
            # tree + 1-loop
            pkmu_tree = self.get_pkmu_lin(self._k, self._mu, pk_data, f, bias)
            pkmu_1loop = self.get_pkmu_1loop(pk, f, bias)
            pkmu = pkmu_tree + pkmu_1loop

            # counterterm
            pkmu_ctr_k2 = self.get_pkmu_ctr_k2(pk[:, None], f, ctr)
            pkmu_ctr_k4 = self.get_pkmu_ctr_k4(pk[:, None], f, bias, ctr)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        
        pkmu_stoch = self.get_pkmu_stoch(self._k, self._mu, params)
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
    def get_pkmu_lin(self, k, mu, pk_data, f, bias):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        b1 = bias[0]
        Z1 = b1 + f * mu**2
        pk_lin = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pkmu = jnp.outer(pk_lin, Z1**2)

        return pkmu

    def get_pkmu_terms_22(self, p_q_1, p_q_2, p_q_3):
        """Compute per-block P22 terms on the internal k-grid.

        All three nu groups use nu > -3/2, so the FFTLog naturally evaluates I(k)-I(0)
        via analytic continuation — no explicit DC subtraction is needed.
        p_q_1: PLD data at nu=-0.3 (nu1 group)
        p_q_2: PLD data at nu=-1.0 (nu2 group, bias-operator blocks)
        p_q_3: PLD data at nu=-1.0 (nu3 group, b2²-only block)

        Per-k cost: O(N^2) for each term.  XLA compiles the
        einsum to batched BLAS GEMM -- faster than an IDFT approach for N=512 on CPU.
        """
        k = self._k

        diag22_1 = jnp.einsum('tnm,nj,mj->tj', self.matrices_22_nu1, p_q_1, p_q_1)
        diag22_2 = jnp.einsum('tnm,nj,mj->tj', self.matrices_22_nu2, p_q_2, p_q_2)
        diag22_3 = jnp.einsum('tnm,nj,mj->tj', self.matrices_22_nu3, p_q_3, p_q_3)
        diag22 = jnp.zeros((self.matrices_22.shape[0], self._nfft), dtype=diag22_1.dtype)
        diag22 = diag22.at[self._idx_22_nu1].set(diag22_1)
        diag22 = diag22.at[self._idx_22_nu2].set(diag22_2)
        diag22 = diag22.at[self._idx_22_nu3].set(diag22_3)

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
    
    def get_pkmu_13_UV(self, pk, f, bias):
        k = self._k
        mu = self._mu

        b1 = bias[0]

        Z1 = b1 + f * mu**2
        Z3_UV = - 61./315. * b1 \
            + ((- 3./5. + 2./105. * b1) * f + (- 16./35. - 1./3. * b1) * f**2) * mu**2 \
            + ((- 46./105.) * f**2 + (- 1./3.) * f**3) * mu**4
        Z1Z3_UV = Z1 * Z3_UV

        pk_data = jnp.stack([k, pk], axis=0)
        pk_int = get_pk_int(pk_data)
        pkmu = jnp.outer(k**2 * pk * pk_int, Z1Z3_UV)

        return pkmu
    
    def get_pkmu_22_pld(self, pkmu_terms, pk, f, bias):
        b1, b2, bG2, bGamma3 = bias

        powers = jnp.array([1.0, f, b1, b2, bG2, bGamma3])  # shape: (6,)
        coeffs = jnp.prod(powers[None, :] ** self.degrees_22, axis=1)
        mu_powers = self._mu[None, :] ** self.degrees_22[:, 0:1]

        pkmu = jnp.sum(coeffs[:, None, None] * pkmu_terms[:, :, None] * mu_powers[:, None, :], axis=0)

        # The b2^2-only term was computed with \nu > -3/2 (nu3 group), so the FFTLog
        # naturally gives I_{d2d2}(k) - I_{d2d2}(0).  When subtract_k0_limit=False,
        # add back I_{d2d2}(0) = 2 * k0_pk via the Parseval identity, which restores
        # the full I_{d2d2}(k).  The contribution to P22 is b2^2/4 * I_{d2d2}(0)
        # = b2^2/4 * 2·k0_pk = b2^2/2 * k0_pk.
        if not self.subtract_k0_limit:
            pkmu = pkmu + self._get_pkmu_22_k0_limit(pk, bias)

        return pkmu  # shape: (nk, nmu)
    
    def get_pkmu_13_pld(self, pkmu_terms, pk, f, bias):
        b1, b2, bG2, bGamma3 = bias

        powers = jnp.array([1.0, f, b1, b2, bG2, bGamma3])  # shape: (6,)
        coeffs = jnp.prod(powers[None, :] ** self.degrees_13, axis=1)
        mu_powers = self._mu[None, :] ** self.degrees_13[:, 0:1]

        pkmu = jnp.sum(coeffs[:, None, None] * pkmu_terms[:, :, None] * mu_powers[:, None, :], axis=0)
        pkmu = pkmu + self.get_pkmu_13_UV(pk, f, bias)

        return pkmu  # shape: (nk, nmu)
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop_pld(self, pk, f, bias):
        # nu=-0.3: P22 nu1 group and P13 nu1 group
        p_q_1, _, _ = get_decomp_data(-0.3, self._k, pk)
        # nu=-1.6: P13 nu2 group only (P13 convergence strip requires this nu)
        p_q_2, _, _ = get_decomp_data(-1.6, self._k, pk)
        # nu=-1.0: P22 nu2 (bias-operator blocks) and nu3 (b2^2-only) groups.
        # nu > -3/2 -> FFTLog evaluates I(k)-I(0) for all blocks via analytic continuation.
        p_q_3, _, _ = get_decomp_data(self._nu_ept_b2sq, self._k, pk)

        pkmu_terms_22 = self.get_pkmu_terms_22(p_q_1, p_q_3, p_q_3)
        pkmu_terms_13 = self.get_pkmu_terms_13(p_q_1, p_q_2, pk)

        pkmu_22 = self.get_pkmu_22_pld(pkmu_terms_22, pk, f, bias)
        pkmu_13 = self.get_pkmu_13_pld(pkmu_terms_13, pk, f, bias)

        return pkmu_22 + pkmu_13 # shape: (nk, nmu)
    
    def _get_xi_ln_with_k_high(self, l, n, array, k_high):
        fx = array * self._k**(n + 3) / (2 * jnp.pi**2)
        xi_ln = hankel.get_hankel(
            self._nu_hankel,
            fx,
            self._k_padded,
            self._q_padded[l],
            self._u_m[l],
            self._npad,
            k_high,
            self._w_m,
            self._w_m_freq,
            self._hankel_forward_pad_mode,
            self._hankel_damp_kind,
            self._hankel_damp_power,
        )
        return xi_ln

    def get_xi_ln(self, l, n, array):
        if self._hankel_forward_mode == 'pld':
            return self._get_xi_ln_array_pld(array)[l, n]
        return self._get_xi_ln_with_k_high(l, n, array, self._k_high)

    def get_pk_ln(self, l, n, array):
        fx = array * self._q[l]**(n + 3)
        pk_ln = hankel.get_hankel(
            self._nu_hankel,
            fx,
            self._q_padded[l],
            self._k_padded,
            self._u_m[l],
            self._npad,
            self._q_high[l],
            self._w_m,
            self._w_m_freq,
            self._hankel_backward_pad_mode,
            self._hankel_damp_kind,
            self._hankel_damp_power,
        )
        return pk_ln

    def get_pk_ln_p22_residual(self, array):
        fx = array * self._q_p22_residual**3
        pk_ln = hankel.get_hankel(
            self._NU_P22_RESIDUAL,
            fx,
            self._q_p22_residual_padded,
            self._k_padded,
            self._u_m_p22_residual,
            self._npad,
            self._q_high_p22_residual,
            self._w_m,
            self._w_m_freq,
            self._hankel_backward_pad_mode,
            self._hankel_damp_kind,
            self._hankel_damp_power,
        )
        return pk_ln

    def get_pk_ln_pld_backward(self, array):
        """PLD backward Hankel (l=0) for P22 blocks.

        array must be sampled on self._q_back_pld.
        q^3 integration measure is absorbed into u_m_back_pld via g_0(nu_back+3).
        """
        pk_ln = hankel.get_hankel_pld_backward(
            self._nu_p22_backward,
            array,
            self._q_padded_back_pld,
            self._k_padded,
            self._u_m_back_pld,
            self._npad,
            self._q_high_back_pld,
            self._w_m,
            self._w_m_freq,
            self._hankel_backward_pad_mode,
            self._hankel_damp_kind,
            self._hankel_damp_power,
        )
        return pk_ln

    def _get_xi_ln_array_with_k_high(self, array, k_high):
        def compute_ln(ln):
            l, n = ln
            return self._get_xi_ln_with_k_high(l, n, array, k_high)

        xis = jax.vmap(compute_ln)(self._ln_list)  # (n_ln, nq)
        xi_ln = jnp.zeros((5, 3, len(self._q[0])))
        ls = self._ln_list[:, 0]
        ns = self._ln_list[:, 1]
        xi_ln = xi_ln.at[ls, ns].set(xis)
        return xi_ln

    def _get_forward_pld_rfft(self, nu, array):
        """Padded rfft for PLD: rfft(array/(2 pi^2) * k_padded^{-nu} * window) with damping."""
        fx_pad = hankel.pad(array / (2 * jnp.pi**2), self._npad, mode=self._hankel_forward_pad_mode)
        fx_pad = fx_pad * hankel.get_high_x_damp(self._k_padded, self._k_high,
                                                 self._hankel_damp_kind, self._hankel_damp_power)
        c_m = jnp.fft.rfft(fx_pad * self._k_padded**(-nu) * self._w_m)
        return c_m * self._w_m_freq

    def _apply_pld_batch(self, c_m, u_m_batch, q_factor_batch):
        """Batched irfft for all (l,n) pairs -> shape (n_ln, nfft)."""
        xi_pad = jnp.fft.irfft(jnp.conj(c_m[None, :] * u_m_batch))  # (n_ln, nfft_pad)
        return (xi_pad * q_factor_batch)[:, self._npad:-self._npad]   # (n_ln, nfft)

    def _get_xi_ln_array_pld(self, array):
        c_m = self._get_forward_pld_rfft(self._nu_hankel_forward_pld, array)
        xis = self._apply_pld_batch(c_m, self._xi_pld_u_m, self._xi_pld_q_factor)
        if self._xi_pld_u_m_n0 is not None:
            c_m_n0 = self._get_forward_pld_rfft(self._nu_hankel_forward_pld_n0, array)
            xis_n0 = self._apply_pld_batch(c_m_n0, self._xi_pld_u_m_n0, self._xi_pld_q_factor_n0)
            mask = self._xi_pld_n0_mask[:, None]
            xis = jnp.where(mask, xis_n0, xis)
        xi_ln = jnp.zeros((5, 3, self._nfft))
        ls = self._ln_list[:, 0]
        ns = self._ln_list[:, 1]
        return xi_ln.at[ls, ns].set(xis)

    def get_xi_ln_array(self, array):
        if self._hankel_forward_mode == 'pld':
            return self._get_xi_ln_array_pld(array)
        return self._get_xi_ln_array_with_k_high(array, self._k_high)
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop_hankel(self, pk, f, bias):
        """Compute 1-loop P22 + P13 using the FFTLog-Hankel method (or hybrid for P13)."""
        xi_ln = self.get_xi_ln_array(pk)
        pkmu_22 = self.get_pkmu_22_hankel(xi_ln, pk, f, bias)

        if self.method == 'hankel':
            pkmu_13 = self.get_pkmu_13_hankel(xi_ln, pk, f, bias)
        else:
            # hybrid: PLD for P13
            p_q_1 = self._get_decomp_pq_from_xpow(-0.3, self._pld_xpow_nu1, pk)
            p_q_2 = self._get_decomp_pq_from_xpow(-1.6, self._pld_xpow_nu2, pk)
            pkmu_13 = self.get_pkmu_13_pld(
                self.get_pkmu_terms_13(p_q_1, p_q_2, pk), pk, f, bias
            )

        return pkmu_22 + pkmu_13

    def _get_pkmu_22_k0_limit(self, pk, bias):
        b2 = bias[1]
        pk_data = jnp.stack([self._k, pk], axis=0)
        return b2**2 / 2. * get_pk_int2(pk_data)

    def get_pkmu_22_hankel(self, xi_ln, pk, f, bias):
        b1, b2, bG2, bGamma3 = bias

        def get_pk_lnm_22(term):
            l, n, m = term
            xi = xi_ln[l, n]
            array = spline.interp1d(jnp.log(self._q[0]), jnp.log(self._q[l]), xi * xi)
            pk_ln = self.get_pk_ln(0, 0, 4 * jnp.pi * array)
            return (self._k ** m) * pk_ln # (nk,)

        pk_lnm = jax.vmap(get_pk_lnm_22)(self._lnm_list_22)  # (nterms, nk)

        def compute_coeffs(entry):
            mu_pow  = entry[:, 0].astype(jnp.int32)
            f_pow   = entry[:, 1].astype(jnp.int32)
            b1_pow  = entry[:, 2].astype(jnp.int32)
            b2_pow  = entry[:, 3].astype(jnp.int32)
            bG2_pow = entry[:, 4].astype(jnp.int32)
            coeff   = entry[:, 6]

            coeffs = coeff * (f ** f_pow) * (b1 ** b1_pow) * (b2 ** b2_pow) * (bG2 ** bG2_pow)  # (max_len,)

            mu_terms = self._mu[None, :] ** mu_pow[:, None]  # (max_len, nmu)
            return jnp.sum(coeffs[:, None] * mu_terms, axis=0)  # (nmu,)

        coeff_matrix = jax.vmap(compute_coeffs)(self.coeff_info_22)  # (nterms, nmu)

        terms = pk_lnm[:, :, None] * coeff_matrix[:, None, :]
        pkmu = jnp.sum(terms, axis=0)  # (nk, nmu)
        
        if self.subtract_k0_limit:
            pkmu = pkmu - self._get_pkmu_22_k0_limit(pk, bias)
        return pkmu  # (nk, nmu)
    
    def get_pkmu_13_hankel(self, xi_ln, pk, f, bias):
        b1, b2, bG2, bGamma3 = bias

        def get_pk_lnm_13(term):
            l, n, m = term
            pk_ln = self.get_pk_ln(l, -1, xi_ln[l, n])  # (nk,)
            return (self._k ** m) * pk * pk_ln

        pk_lnm = jax.vmap(get_pk_lnm_13)(self._lnm_list_13)  # (nterms, nk)

        def compute_coeffs(entry):
            mu_pow = entry[:, 0].astype(jnp.int32)
            f_pow = entry[:, 1].astype(jnp.int32)
            b1_pow = entry[:, 2].astype(jnp.int32)
            b2_pow = entry[:, 3].astype(jnp.int32)
            bG2_pow = entry[:, 4].astype(jnp.int32)
            bGamma3_pow = entry[:, 5].astype(jnp.int32)
            coeff = entry[:, 6]

            coeffs = coeff * (f ** f_pow) * (b1 ** b1_pow) * (b2 ** b2_pow) * (bG2 ** bG2_pow) * (bGamma3 ** bGamma3_pow)  # (max_len,)

            mu_terms = self._mu[None, :] ** mu_pow[:, None]  # (max_len, nmu)
            return jnp.sum(coeffs[:, None] * mu_terms, axis=0)  # (nmu,)

        coeff_matrix = jax.vmap(compute_coeffs)(self.coeff_info_13)  # (nterms, nmu)

        terms = pk_lnm[:, :, None] * coeff_matrix[:, None, :]
        pkmu = jnp.sum(terms, axis=0)  # (nk, nmu)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop(self, pk, f, bias):
        """Compute 1-loop P(k, mu) = P22 + P13 on the internal grid, dispatching to matrix or Hankel path."""
        if self.method == 'matrix':
            pkmu = self.get_pkmu_1loop_pld(pk, f, bias)
        else:
            pkmu = self.get_pkmu_1loop_hankel(pk, f, bias)
        return pkmu
    
    def get_pkmu_irres_LO_NLO(self, pk_nw, pk_w, damp_fac, f, bias):
        # LO term
        b1 = bias[0]
        Z1 = b1 + f * self._mu**2
        pkmu_irres_tree = (Z1**2)[None, :] * (pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None] * (1 + damp_fac))
        
        # NLO term
        pkmu_1loop = self.get_pkmu_1loop(pk_nw + pk_w, f, bias)
        pkmu_1loop_nw = self.get_pkmu_1loop(pk_nw, f, bias)
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
        Sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.r_bao, self.k_IR)
        dSigma2 = ir_resum.get_dSigma2(pk_nw_data, self.r_bao, self.k_IR)
        Sigma2_s = (1 + mu**2 * f * (2 + f)) * Sigma2 + f**2 * mu**2 * (mu**2 - 1) * dSigma2
        damp_fac = jnp.outer(k**2, Sigma2_s)

        return pk_nw, pk_w, damp_fac
    
    def get_pkmu_ctr_k2(self, pk, f, ctr):
        k = self._k
        mu = self._mu

        c0, c2, c4, _ = ctr
        ctr_k2_mu = c0 + c2 * f * mu**2 + c4 * f**2 * mu**4
        ctr_k2_fac = - 2 * jnp.outer(k**2, ctr_k2_mu)

        pkmu_ctr_k2 = ctr_k2_fac * pk
        return pkmu_ctr_k2
    
    def get_pkmu_ctr_k4(self, pk, f, bias, ctr):
        k = self._k
        mu = self._mu

        b1 = bias[0]
        cfog = ctr[3]

        ctr_k4_mu = cfog * f**4 * mu**4 * (b1 + f * mu**2)**2
        ctr_k4_fac = - jnp.outer(k**4, ctr_k4_mu)

        pkmu_ctr_k4 = ctr_k4_fac * pk
        return pkmu_ctr_k4

    def get_pkmu_stoch(self, k, mu, params):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        P_shot, a0, a2 = params.stoch
        k_nl = params.k_nl
        ndens = params.ndens

        pkmu = P_shot \
            + a0 * jnp.outer((k / k_nl)**2, mu**0) \
            + a2 * jnp.outer((k / k_nl)**2, mu**2)
        pkmu = (1. / ndens) * pkmu

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_xi_ells(self, r, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
        r = jnp.atleast_1d(r)

        k = jnp.geomspace(1e-3, 1, 128) # ad-hoc down-sampling of k
        pk_ells = self.get_pk_ells(k, pk_data, params, alpha_perp, alpha_para)

        pk0 = get_pk(self._k, jnp.stack([k, pk_ells[0]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk2 = get_pk(self._k, jnp.stack([k, pk_ells[1]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk4 = get_pk(self._k, jnp.stack([k, pk_ells[2]], axis=0), kmin=self._kmin, kmax=self._kmax)

        xi0 = spline.interp1d(jnp.log(r), jnp.log(self._q[0]), self.get_xi_ln(0, 0, pk0))
        xi2 = spline.interp1d(jnp.log(r), jnp.log(self._q[2]), -self.get_xi_ln(2, 0, pk2))
        xi4 = spline.interp1d(jnp.log(r), jnp.log(self._q[4]), self.get_xi_ln(4, 0, pk4))

        xi_ells = jnp.stack([xi0, xi2, xi4], axis=0)
        return xi_ells
