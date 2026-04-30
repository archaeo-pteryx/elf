import os
import glob, re

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

    def __init__(self,
                 do_irres=True,
                 r_bao=110.,
                 k_IR=0.2,
                 irres_method='DST',
                 subtract_k0_limit=True,
                 method='hybrid',
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=512,
                 hankel_nu=1.1,
                 hankel_k_damp=50.0,
                 hankel_q_damp=0.5,
                 hankel_c_window_width=0.70,
                 hankel_npad_factor=0.5,
                 hankel_pad_mode='zero-pad',
                 hankel_p22_richardson=False,
                 hankel_p22_richardson_factor=2,
                 hankel_p22_richardson_levels=1,
                 hankel_p22_richardson_order=4.0,
                 ngauss=4,
                 ):

        if method not in ('matrix', 'hankel', 'hybrid'):
            raise ValueError(f"method must be 'matrix', 'hankel', or 'hybrid', got {method!r}")

        self.do_irres = do_irres # flag to perform the IR resummation
        self.r_bao = r_bao
        self.k_IR = k_IR
        self.irres_method = irres_method
        self.subtract_k0_limit = subtract_k0_limit # flag to subtract k -> 0 limit from 2-2 terms
        self.method = method

        # preparation for Gauss-Legendre quadrature
        self._mu_quad, self._legendre_weights = prepare_mu_gauleg(ngauss)

        # preparation for FFT
        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self._nu_hankel = hankel_nu
        self._hankel_k_damp = hankel_k_damp
        self._hankel_q_damp = hankel_q_damp
        self._hankel_c_window_width = hankel_c_window_width
        self._hankel_npad_factor = hankel_npad_factor
        self._hankel_pad_mode = hankel_pad_mode
        self._hankel_p22_richardson = hankel_p22_richardson
        self._hankel_p22_richardson_factor = hankel_p22_richardson_factor
        self._hankel_p22_richardson_levels = hankel_p22_richardson_levels
        self._hankel_p22_richardson_order = hankel_p22_richardson_order
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._mu = jnp.linspace(0., 1., 51)

        self._initialize_loop_coeff()
        self._initialize_loop_matrix()

        self._hankel_p22_richardson_models = []
        self._hankel_p22_richardson_model = None
        if method in ('hankel', 'hybrid') and hankel_p22_richardson:
            for level in range(1, hankel_p22_richardson_levels + 1):
                factor = hankel_p22_richardson_factor ** level
                self._hankel_p22_richardson_models.append(
                    PowerSpectrum1LoopEPT(
                        do_irres=do_irres,
                        r_bao=r_bao,
                        k_IR=k_IR,
                        irres_method=irres_method,
                        subtract_k0_limit=subtract_k0_limit,
                        method=method,
                        kmin_fft=kmin_fft,
                        kmax_fft=kmax_fft,
                        nfft=factor * nfft,
                        hankel_nu=hankel_nu,
                        hankel_k_damp=hankel_k_damp,
                        hankel_q_damp=hankel_q_damp,
                        hankel_c_window_width=hankel_c_window_width,
                        hankel_npad_factor=hankel_npad_factor,
                        hankel_pad_mode=hankel_pad_mode,
                        hankel_p22_richardson=False,
                        hankel_p22_richardson_factor=hankel_p22_richardson_factor,
                        hankel_p22_richardson_levels=hankel_p22_richardson_levels,
                        hankel_p22_richardson_order=hankel_p22_richardson_order,
                        ngauss=ngauss,
                    )
                )
            self._hankel_p22_richardson_model = self._hankel_p22_richardson_models[0]
    
    def _initialize_loop_coeff(self):
        # store the names of 1-loop terms calculated with the FFTLog-based method
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_coeff/22*.txt')
        self.pkmu_coeff_names_22 = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_coeff/13*.txt')
        self.pkmu_coeff_names_13 = [re.split('/', fname)[-1][:-4] for fname in fnames]

        ln1n2_list = []
        coeff_info = []
        for name in self.pkmu_coeff_names_22:
            str_list = re.split('_', name)
            l, n1, n2 = int(str_list[1]), int(str_list[2]), int(str_list[3])
            ln1n2_list.append([l, n1, n2])
            coeff_file = glob.glob(os.path.dirname(__file__)+'/pt_coeff/%s.txt' % (name))[0]
            coeff_info.append(pt_coeff.get_coeff_info(coeff_file))

        max_len = max(len(lst) for lst in coeff_info)
        padded_coeff_info = []
        for lst in coeff_info:
            padded = lst + [[0, 0, 0, 0, 0, 0, 0.0]] * (max_len - len(lst))
            padded_coeff_info.append(padded)
        
        self.ln1n2_list = jnp.array(ln1n2_list)
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
        
        self.lnm_list = jnp.array(lnm_list)
        self.coeff_info_13 = jnp.array(padded_coeff_info)

        # set the Hankel transforms
        self.ln_list = jnp.array([[0,0], [0,-2], [0,2], [1,-1], [1,1], [1,-3], [1,3], [2,0], [2,-2], [2,2], [3,-1], [3,1], [4,0]])
        lmax = jnp.max(self.ln_list[:, 0])
        self._set_hankel(lmax)

    def _set_hankel(self, lmax):
        l_list = jnp.arange(lmax + 1)

        # Mellin argument for the low-ringing condition and u_m. The k^n factor
        # is absorbed into fx, so nu does not change with n.
        _mellin_arg = self._nu_hankel

        self._npad = max(1, int(self._hankel_npad_factor * self._nfft))
        self._k_padded = hankel.get_log_extrap(self._k, self._npad, self._npad)

        n_window = self._nfft // 4
        self._w_m = hankel.get_window(n_window, self._k_padded)

        nfft = len(self._k_padded)
        dln = jnp.log(self._k_padded[1] / self._k_padded[0])
        eta_m = 2 * jnp.pi * jnp.fft.rfftfreq(nfft, d=1.0) / dln

        g_l = jnp.array([hankel.get_g_l(l, _mellin_arg + 1j * eta_m) for l in l_list])

        lnxy = jnp.array([dln * jnp.angle(hankel.get_g_l(l, _mellin_arg + 1j * jnp.pi / dln)) / jnp.pi for l in l_list])
        self._q_padded = jnp.array([jnp.exp(lnxy[l] - dln) / self._k_padded[::-1] for l in l_list])
        self._q = jnp.array([self._q_padded[l][self._npad:-self._npad] for l in l_list])

        self._u_m = jnp.array([jnp.exp(lnxy[l])**(-1j*eta_m) * g_l[l] for l in l_list])

        self._k_high = self._kmax / self._hankel_k_damp
        self._q_high = jnp.array([self._q[l][-1] * self._hankel_q_damp for l in l_list])
        
        c_window_width = self._hankel_c_window_width
        self._w_m_freq = hankel.c_window(jnp.arange(nfft//2+1), int(c_window_width * (nfft//2+1)))

    def _initialize_loop_matrix(self):
        # store the names of 1-loop terms calculated with the FFTLog-based method
        self.pk_term_names = ['22_dd', '13_dd', 'I_d2', 'I_G2', 'I_d2_d2', 'I_G2_G2', 'I_d2_G2', 'F_G2']

        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/22*.txt')
        self.pkmu_term_names_22 = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/13*.txt')
        self.pkmu_term_names_13 = [re.split('/', fname)[-1][:-4] for fname in fnames]

        # precompute the PT matrices
        matrix = self._set_matrix(self.pk_term_names + self.pkmu_term_names_22 + self.pkmu_term_names_13)

        self.matrices_22_real = jnp.array([matrix[name] for name in ['22_dd', 'I_d2', 'I_G2', 'I_d2_d2', 'I_G2_G2', 'I_d2_G2']])
        self.matrices_13_real = jnp.array([matrix[name] for name in ['13_dd', 'F_G2']])

        ## create arrays of matrices and degrees for 22 and 13, according to the degrees of mu
        self.nus_22 = jnp.array([utils_loop.get_nu_from_name(name) for name in self.pkmu_term_names_22])
        self.nus_13 = jnp.array([utils_loop.get_nu_from_name(name) for name in self.pkmu_term_names_13])

        self.matrices_22 = jnp.array([matrix[name] for name in self.pkmu_term_names_22])
        self.matrices_13 = jnp.array([matrix[name] for name in self.pkmu_term_names_13])

        def get_degree_vector(name):
            d = utils_loop.get_degree_dict(name)
            if name in self.pkmu_term_names_22:
                return [d['mu'], d['f'], d['b1'], d['b2'], d['bG2'], 0]
            else:
                return [d['mu'], d['f'], d['b1'], 0, d['bG2'], d['bGamma3']]
            
        self.degrees_22 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_22])
        self.degrees_13 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_13])
        self.degrees = jnp.concatenate([self.degrees_22, self.degrees_13], axis=0)

    def _set_matrix(self, names=[]):
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
            nu = utils_loop.get_nu_from_name(name)

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
    def get_pk_terms(self, pk_data):
        k3 = self._k**3
        k2 = self._k**2

        # tree
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_terms = [pk_lin]

        # 1-loop matter
        p_q1, p_k1, p_k0_1 = get_decomp_data(-0.3, self._k, pk_lin)

        def compute_pk_22(matrix):
            pk = k3 * jnp.real(jnp.diag(p_q1.T @ matrix @ p_q1))
            if self.subtract_k0_limit:
                pk = pk - self._kmin**3 * jnp.real(p_k0_1 @ matrix @ p_k0_1)
            return pk

        def compute_pk_13(matrix):
            return k3 * p_k1 * jnp.real(matrix @ p_q1)

        pk_22 = compute_pk_22(self.matrices_22_real[0])
        pk_13 = compute_pk_13(self.matrices_13_real[0])
        pk_lin_int = get_pk_int(pk_data)
        pk_13 -= (61. / 315.) * k2 * p_k1 * pk_lin_int

        pk_terms += [pk_22, pk_13]

        # 1-loop bias
        p_q2, p_k2, p_k0_2 = get_decomp_data(-1.6, self._k, pk_lin)

        def compute_pk_bias(matrix):
            pk = k3 * jnp.real(jnp.diag(p_q2.T @ matrix @ p_q2))
            if self.subtract_k0_limit:
                pk -= self._kmin**3 * jnp.real(p_k0_2 @ matrix @ p_k0_2)
            return pk

        pk_bias_terms = jax.vmap(compute_pk_bias)(self.matrices_22_real[1:])

        pk_F_bias = k3 * p_k2 * jnp.real(self.matrices_13_real[1] @ p_q2)

        pk_terms += [*pk_bias_terms, pk_F_bias]
        return jnp.stack(pk_terms, axis=0)  # shape = (n_terms, nk)

    @partial(jit, static_argnames=["self"])
    def get_pk_real(self, k, pk_data, params):
        k = jnp.atleast_1d(k)

        b1, b2, bG2, bGamma3 = params.bias
        c0, c2, c4, cfog = params.ctr
        P_shot, a0, a2 = params.stoch
        k_nl = params.k_nl
        ndens = params.ndens

        coeffs = jnp.array([
            b1**2,                              # tree
            b1**2,                              # 22_dd
            b1**2,                              # 13_dd
            b1 * b2,                            # I_d2
            2 * b1 * bG2,                       # I_G2
            b2**2 / 4,                          # I_d2_d2
            bG2**2,                             # I_G2_G2
            b2 * bG2,                           # I_d2_G2
            2 * b1 * bG2 + (4/5) * b1 * bGamma3 # F_G2
        ])

        pk_terms = self.get_pk_terms(pk_data)

        # tree + 1-loop
        pk_sum = jnp.tensordot(coeffs, pk_terms, axes=1)  # shape: (nk,)

        # counterterm
        pk_ctr = -2.0 * self._k**2 * c0 * pk_terms[0]

        # interpolation
        pk_interp = jnp.interp(jnp.log(k), jnp.log(self._k), pk_sum + pk_ctr)

        # stochasticity
        pk_stoch = (P_shot + a0 * (k / k_nl)**2) / ndens

        return pk_interp + pk_stoch

    @partial(jit, static_argnames=['self'])
    def get_pkmu_grid(self, pk_data, params):
        
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
        
        # NOTE: can be removed
        # stochasticity
        pkmu_stoch = self.get_pkmu_stoch(self._k, self._mu, params)
        pkmu = pkmu + pkmu_stoch
        
        return pkmu

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, alpha_perp=1.0, alpha_para=1.0):
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

    def get_pkmu_terms_22(self, p_q_1, p_q_2, p_k0_1, p_k0_2):
        k = self._k

        mask = jnp.isclose(self.nus_22, -0.3)   # (T22,)  True→-0.3, False→-1.6
        p_q  = jnp.where(mask[:, None, None], p_q_1[None, :, :], p_q_2[None, :, :])
        p_k0 = jnp.where(mask[:, None],       p_k0_1[None, :],   p_k0_2[None, :])

        diag22 = jnp.einsum('tnm,tnj,tmj->tj', self.matrices_22, p_q, p_q)
        pk_22  = (k**3)[None, :] * jnp.real(diag22)
        
        if self.subtract_k0_limit:
            quad_k0 = jnp.einsum('tnm,tn,tm->t', self.matrices_22, p_k0, p_k0)
            pk_22   = pk_22 - (self._kmin**3) * jnp.real(quad_k0)[:, None]
        
        return pk_22
    
    def get_pkmu_terms_13(self, p_q_1, p_q_2, pk):
        k = self._k

        mask = jnp.isclose(self.nus_13, -0.3)   # (T13,)  True→-0.3, False→-1.6
        p_q  = jnp.where(mask[:, None, None], p_q_1[None, :, :], p_q_2[None, :, :])
        
        Mq_13  = jnp.einsum('tn,tnj->tj', self.matrices_13, p_q)
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

        # if not self.subtract_k0_limit:
        #     pk_data = jnp.stack([self._k, pk], axis=0)
        #     pkmu_k0 = b2**2 / 2. * get_pk_int2(pk_data)
        #     pkmu = pkmu + pkmu_k0

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

        p_q_1, _, p_k0_1 = get_decomp_data(-0.3, self._k, pk)
        p_q_2, _, p_k0_2 = get_decomp_data(-1.6, self._k, pk)

        pkmu_terms_22 = self.get_pkmu_terms_22(p_q_1, p_q_2, p_k0_1, p_k0_2)
        pkmu_terms_13 = self.get_pkmu_terms_13(p_q_1, p_q_2, pk)

        pkmu_22 = self.get_pkmu_22_pld(pkmu_terms_22, pk, f, bias)
        pkmu_13 = self.get_pkmu_13_pld(pkmu_terms_13, pk, f, bias)

        return pkmu_22 + pkmu_13 # shape: (nk, nmu)
    
    def get_xi_ln(self, l, n, array):
        fx = array * self._k**(n + 3) / (2 * jnp.pi**2)
        xi_ln = hankel.get_hankel(self._nu_hankel, fx, self._k_padded, self._q_padded[l], self._u_m[l], self._npad, self._k_high, self._w_m, self._w_m_freq, self._hankel_pad_mode)
        return xi_ln

    def get_pk_ln(self, l, n, array):
        fx = array * self._q[l]**(n + 3)
        pk_ln = hankel.get_hankel(self._nu_hankel, fx, self._q_padded[l], self._k_padded, self._u_m[l], self._npad, self._q_high[l], self._w_m, self._w_m_freq, self._hankel_pad_mode)
        return pk_ln
    
    def get_xi_ln_array(self, array):
        def compute_ln(ln):
            l, n = ln
            return self.get_xi_ln(l, n, array)
        xis = jax.vmap(compute_ln)(self.ln_list)
        xi_ln = jnp.zeros((5, 5, len(self._q[0])))
        ls = self.ln_list[:, 0]
        ns = self.ln_list[:, 1]
        xi_ln = xi_ln.at[ls, ns].set(xis)
        return xi_ln
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop_hankel(self, pk, f, bias):

        xi_ln = self.get_xi_ln_array(pk)

        pkmu_22 = self.get_pkmu_22_hankel(xi_ln, pk, f, bias)

        if self.method == 'hankel':
            pkmu_13 = self.get_pkmu_13_hankel(xi_ln, pk, f, bias)
        else:
            # hybrid: PLD for P13 — avoids catastrophic cancellation in UV-sensitive terms
            p_q_1, _, _ = get_decomp_data(-0.3, self._k, pk)
            p_q_2, _, _ = get_decomp_data(-1.6, self._k, pk)
            pkmu_13 = self.get_pkmu_13_pld(
                self.get_pkmu_terms_13(p_q_1, p_q_2, pk), pk, f, bias
            )

        return pkmu_22 + pkmu_13
    
    def _get_pkmu_22_hankel_raw(self, xi_ln, pk, f, bias, subtract_k0=True):
        b1, b2, bG2, bGamma3 = bias

        def compute_pk_ln1n2(term):
            l, n1, n2 = term
            xi1 = xi_ln[l, n1]
            xi2 = xi_ln[l, n2]

            # q[l] shares log-spacing with q[0] but is shifted; use spline to resample onto q[0].
            array = spline.interp1d(jnp.log(self._q[0]), jnp.log(self._q[l]), xi1 * xi2)
            pk_ln1n2 = self.get_pk_ln(0, 0, (-1)**l * 4 * jnp.pi * array)
            return pk_ln1n2 # (nk,)

        pk_ln1n2 = jax.vmap(compute_pk_ln1n2)(self.ln1n2_list)  # (nterms, nk)

        def compute_coeffs(entry):
            mu_pow  = entry[:, 0]
            f_pow   = entry[:, 1]
            b1_pow  = entry[:, 2]
            b2_pow  = entry[:, 3]
            bG2_pow = entry[:, 4]
            coeff   = entry[:, 6]

            coeffs = coeff * (f ** f_pow) * (b1 ** b1_pow) * (b2 ** b2_pow) * (bG2 ** bG2_pow)  # (max_len,)

            mu_terms = self._mu[None, :] ** mu_pow[:, None]  # (max_len, nmu)
            return jnp.sum(coeffs[:, None] * mu_terms, axis=0)  # (nmu,)

        coeff_matrix = jax.vmap(compute_coeffs)(self.coeff_info_22)  # (nterms, nmu)

        terms = pk_ln1n2[:, :, None] * coeff_matrix[:, None, :]
        order = jnp.argsort(jnp.abs(terms), axis=0)
        pkmu = jnp.sum(jnp.take_along_axis(terms, order, axis=0), axis=0)  # (nk, nmu)

        if subtract_k0 and self.subtract_k0_limit:
            pk_data = jnp.stack([self._k, pk], axis=0)
            pkmu_k0 = b2**2 / 2. * get_pk_int2(pk_data)
            pkmu = pkmu - pkmu_k0

        return pkmu  # (nk, nmu)

    def _interp_pkmu_to_self_grid(self, model, pkmu):
        log_k = jnp.log(model._k)
        log_k_self = jnp.log(self._k)
        return jax.vmap(
            lambda y: spline.interp1d(log_k_self, log_k, y),
            in_axes=1,
            out_axes=1,
        )(pkmu)

    def _get_pkmu_22_k0_limit(self, pk, bias):
        b2 = bias[1]
        pk_data = jnp.stack([self._k, pk], axis=0)
        return b2**2 / 2. * get_pk_int2(pk_data)

    def _get_pkmu_22_hankel_on_model(self, model, pk, f, bias):
        pk_model = spline.interp1d(jnp.log(model._k), jnp.log(self._k), pk)
        xi_ln_model = model.get_xi_ln_array(pk_model)
        pkmu_model = model._get_pkmu_22_hankel_raw(
            xi_ln_model, pk_model, f, bias, subtract_k0=False
        )
        return self._interp_pkmu_to_self_grid(model, pkmu_model)

    def get_pkmu_22_hankel(self, xi_ln, pk, f, bias):
        pkmu = self._get_pkmu_22_hankel_raw(xi_ln, pk, f, bias, subtract_k0=False)

        if self._hankel_p22_richardson_models:
            ratio = self._hankel_p22_richardson_factor ** self._hankel_p22_richardson_order
            if len(self._hankel_p22_richardson_models) == 1:
                pkmu_fine = self._get_pkmu_22_hankel_on_model(
                    self._hankel_p22_richardson_models[0], pk, f, bias
                )
                pkmu = pkmu_fine + (pkmu_fine - pkmu) / (ratio - 1.0)
            else:
                pkmu_fine = self._get_pkmu_22_hankel_on_model(
                    self._hankel_p22_richardson_models[-2], pk, f, bias
                )
                pkmu_finer = self._get_pkmu_22_hankel_on_model(
                    self._hankel_p22_richardson_models[-1], pk, f, bias
                )
                pkmu = pkmu_finer + (pkmu_finer - pkmu_fine) / (ratio - 1.0)

        if self.subtract_k0_limit:
            pkmu = pkmu - self._get_pkmu_22_k0_limit(pk, bias)

        return pkmu  # (nk, nmu)
    
    def get_pkmu_13_hankel(self, xi_ln, pk, f, bias):
        b1, b2, bG2, bGamma3 = bias

        def get_pk_ln_term(term):
            l, n, m = term
            pk_ln = self.get_pk_ln(l, -1, xi_ln[l, n])  # (nk,)
            return (self._k ** m) * pk * pk_ln

        pk_lnm = jax.vmap(get_pk_ln_term)(self.lnm_list)  # (nterms, nk)

        def compute_coeffs(entry):
            mu_pow = entry[:, 0]
            f_pow = entry[:, 1]
            b1_pow = entry[:, 2]
            b2_pow = entry[:, 3]
            bG2_pow = entry[:, 4]
            bGamma3_pow = entry[:, 5]
            coeff = entry[:, 6]

            coeffs = coeff * (f ** f_pow) * (b1 ** b1_pow) * (b2 ** b2_pow) * (bG2 ** bG2_pow) * (bGamma3 ** bGamma3_pow)  # (max_len,)

            mu_terms = self._mu[None, :] ** mu_pow[:, None]  # (max_len, nmu)
            return jnp.sum(coeffs[:, None] * mu_terms, axis=0)  # (nmu,)

        coeff_matrix = jax.vmap(compute_coeffs)(self.coeff_info_13)  # (nterms, nmu)

        terms = pk_lnm[:, :, None] * coeff_matrix[:, None, :]
        order = jnp.argsort(jnp.abs(terms), axis=0)
        pkmu = jnp.sum(jnp.take_along_axis(terms, order, axis=0), axis=0)  # (nk, nmu)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop(self, pk, f, bias):
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
        Sigma2_tot = (1 + mu**2 * f * (2 + f)) * Sigma2 + f**2 * mu**2 * (mu**2 - 1) * dSigma2
        damp_fac = jnp.outer(k**2, Sigma2_tot)

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

    # NOTE: can be removed
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
