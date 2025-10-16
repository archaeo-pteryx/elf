import os
import glob, re

import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp
import numpy as np
import quadax
import interpax

from .power_law_decomp import get_decomp_data
from . import hankel

from . import pt_coeff
from . import pt_matrix
from . import utils_loop
from .utils_loop import get_pk, get_pk_int, get_pk_int2, interp2d_separable_linear
from .utils_math import legendre

from . import ir_resum


class PowerSpectrum1LoopEPT:

    def __init__(self, 
                 do_irres=True,
                 rbao=110.,
                 ks=0.2,
                 cross=False,
                 subtract_k0_limit=True,
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=256,
                 use_hankel=True,
                 ):

        self.do_irres = do_irres # flag to perform the IR resummation
        self.rbao = rbao
        self.ks = ks
        self.cross = cross # flag to enable the calculation of cross power spectra
        self.subtract_k0_limit = subtract_k0_limit # flag to subtract k -> 0 limit from 2-2 terms

        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._mu = jnp.linspace(0., 1., 51)

        self._initialize_loop_coeff()
        self._initialize_loop_matrix()

        self.use_hankel = use_hankel
    
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
        l_list = np.arange(lmax + 1)

        self._q = 1 / self._k[::-1]
        self._nu_hankel = 1.1

        self._npad = self._nfft // 2
        self._k_padded = hankel.get_log_extrap(self._k, self._npad, self._npad)
        self._y_k = 1 / self._k_padded[::-1]
        self._q_padded = hankel.get_log_extrap(self._q, self._npad, self._npad)
        self._y_q = 1 / self._q_padded[::-1]

        nfft_k = len(self._k_padded)
        dlnx = jnp.log(self._k_padded[1] / self._k_padded[0])
        eta_m_k = 2 * jnp.pi / (nfft_k * dlnx) * jnp.arange(nfft_k//2+1)
        
        nfft_q = len(self._q_padded)
        dlnx = jnp.log(self._q_padded[1] / self._q_padded[0])
        eta_m_q = 2 * jnp.pi / (nfft_q * dlnx) * jnp.arange(nfft_q//2+1)
        
        g_l_k = jnp.array([hankel.get_g_l(l, self._nu_hankel + 1j * eta_m_k) for l in l_list])
        self._u_m_k = jnp.array([(self._k_padded[0] * self._y_k[0])**(-1j * eta_m_k) * g_l_k[l] for l in l_list])

        g_l_q = jnp.array([hankel.get_g_l(l, self._nu_hankel + 1j * eta_m_q) for l in l_list])
        self._u_m_q = jnp.array([(self._q_padded[0] * self._y_q[0])**(-1j * eta_m_q) * g_l_q[l] for l in l_list])

        self._k_high = self._kmax / 100.
        self._q_high = 1e10
        
        c_window_width = 0.25
        self._w_m_k = hankel.c_window(jnp.arange(nfft_k//2+1), int(c_window_width * (nfft_k//2+1)))
        self._w_m_q = hankel.c_window(jnp.arange(nfft_q//2+1), int(c_window_width * (nfft_q//2+1)))

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
        k = jnp.atleast_1d(k).astype(float)

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
            pk_nw_data = ir_resum.get_pk_nw_data(pk_data, params.h, khmin=7e-5, khmax=7.0, 
                                                 kmin_interp=self._kmin, kmax_interp=self._kmax)
            pk_nw, pk_w, damp_fac = self._get_irres_components(pk_data, pk_nw_data, f)
            pkmu = self.get_pkmu_irres_LO_NLO(pk_nw, pk_w, damp_fac, pk_data, pk_nw_data, f, bias)

            # counterterm
            pk = pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None]
            pkmu_ctr_k2 = self.get_pkmu_ctr_k2(pk, f, ctr)
            pkmu_ctr_k4 = self.get_pkmu_ctr_k4(pk, f, bias, ctr)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        else:
            # tree + 1-loop
            pkmu_tree = self.get_pkmu_lin(self._k, self._mu, pk_data, f, bias)
            pkmu_1loop = self.get_pkmu_1loop(pk_data, f, bias)
            pkmu = pkmu_tree + pkmu_1loop

            # counterterm
            pk = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)[:, None]
            pkmu_ctr_k2 = self.get_pkmu_ctr_k2(pk, f, ctr)
            pkmu_ctr_k4 = self.get_pkmu_ctr_k4(pk, f, bias, ctr)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        
        # NOTE: can be removed
        # stochasticity
        pkmu_stoch = self.get_pkmu_stoch(self._k, self._mu, params)
        pkmu = pkmu + pkmu_stoch
        
        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        pkmu_data = self.get_pkmu_grid(pk_data, params)
        # 2D interpolation
        pkmu = interp2d_separable_linear(jnp.log(k), mu, jnp.log(self._k), self._mu, pkmu_data)
        return pkmu

    @partial(jit, static_argnames=['self', 'num'])
    def get_pk_ell(self, k, l, pk_data, params, num=256):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.linspace(0., 1., num)

        pkmu = self.get_pkmu(k, mu, pk_data, params)

        leg = jnp.tile((2*l+1) * legendre(l, mu), (len(k), 1))
        pk_ell = quadax.simpson(pkmu * leg, x=mu, axis=1)

        return pk_ell

    @partial(jit, static_argnames=['self'])
    def get_pkmu_ref(self, k, mu, alpha_perp, alpha_para, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        # mapping of (k, mu)
        fac = jnp.sqrt(1 + mu**2 * ((alpha_perp / alpha_para)**2 - 1))
        mu_true = mu * (alpha_perp / alpha_para) / fac
        k_true = jnp.outer(k, fac) / alpha_perp

        # 2D interpolation
        pkmu_grid = self.get_pkmu_grid(pk_data, params)
        mu_tile = jnp.tile(mu_true, (len(k), 1))
        pkmu = interpax.interp2d(
            jnp.ravel(jnp.log(k_true)), jnp.ravel(mu_tile), 
            jnp.log(self._k), self._mu, pkmu_grid, 
            method='linear', extrap=True
        )
        pkmu = pkmu.reshape(len(k), len(mu)) / (alpha_perp**2 * alpha_para)

        return pkmu

    @partial(jit, static_argnames=['self', 'num'])
    def get_pk_ell_ref(self, k, l, alpha_perp, alpha_para, pk_data, params, num=256):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.linspace(0., 1., num)

        pkmu = self.get_pkmu_ref(k, mu, alpha_perp, alpha_para, pk_data, params)

        leg = jnp.tile((2*l+1) * legendre(l, mu), (len(k), 1))
        pk_ell = quadax.simpson(pkmu * leg, x=mu, axis=1)

        return pk_ell

    @partial(jit, static_argnames=['self'])
    def get_pkmu_lin(self, k, mu, pk_data, f, bias):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

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
    
    def get_pkmu_13_UV(self, pk_data, f, bias):
        k = self._k
        mu = self._mu

        b1 = bias[0]

        Z1 = b1 + f * mu**2
        Z3_UV = - 61./315. * b1 \
            + ((- 3./5. + 2./105. * b1) * f + (- 16./35. - 1./3. * b1) * f**2) * mu**2 \
            + ((- 46./105.) * f**2 + (- 1./3.) * f**3) * mu**4
        Z1Z3_UV = Z1 * Z3_UV

        pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_int = get_pk_int(pk_data)
        pkmu = jnp.outer(k**2 * pk * pk_int, Z1Z3_UV)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_22_pld(self, pkmu_terms, pk_data, f, bias):
        b1, b2, bG2, bGamma3 = bias

        powers = jnp.array([1.0, f, b1, b2, bG2, bGamma3])  # shape: (6,)
        coeffs = jnp.prod(powers[None, :] ** self.degrees_22, axis=1)
        mu_powers = self._mu[None, :] ** self.degrees_22[:, 0:1]

        pkmu = jnp.sum(coeffs[:, None, None] * pkmu_terms[:, :, None] * mu_powers[:, None, :], axis=0)

        # if not self.subtract_k0_limit:
        #     pkmu_k0 = b2**2 / 2. * get_pk_int2(pk_data)
        #     pkmu = pkmu + pkmu_k0

        return pkmu  # shape: (nk, nmu)
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_13_pld(self, pkmu_terms, pk_data, f, bias):
        b1, b2, bG2, bGamma3 = bias

        powers = jnp.array([1.0, f, b1, b2, bG2, bGamma3])  # shape: (6,)
        coeffs = jnp.prod(powers[None, :] ** self.degrees_13, axis=1)
        mu_powers = self._mu[None, :] ** self.degrees_13[:, 0:1]

        pkmu = jnp.sum(coeffs[:, None, None] * pkmu_terms[:, :, None] * mu_powers[:, None, :], axis=0)
        pkmu = pkmu + self.get_pkmu_13_UV(pk_data, f, bias)

        return pkmu  # shape: (nk, nmu)
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop_pld(self, pk_data, f, bias):

        pk = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        p_q_1, _, p_k0_1 = get_decomp_data(-0.3, self._k, pk)
        p_q_2, _, p_k0_2 = get_decomp_data(-1.6, self._k, pk)

        pkmu_terms_22 = self.get_pkmu_terms_22(p_q_1, p_q_2, p_k0_1, p_k0_2)
        pkmu_terms_13 = self.get_pkmu_terms_13(p_q_1, p_q_2, pk)

        pkmu_22 = self.get_pkmu_22_pld(pkmu_terms_22, pk_data, f, bias)
        pkmu_13 = self.get_pkmu_13_pld(pkmu_terms_13, pk_data, f, bias)

        return pkmu_22 + pkmu_13 # shape: (nk, nmu)
    
    def get_xi_ln(self, l, n, array):
        fx = array * self._k**(n + 3) / (2 * jnp.pi**2)
        xi_ln = hankel.get_hankel(self._nu_hankel, fx, self._k_padded, self._y_k, self._u_m_k[l], self._npad, self._k_high, self._w_m_k)
        return xi_ln
    
    def get_pk_ln(self, l, n, array):
        fx = array * self._q**(n + 3)
        pk_ln = hankel.get_hankel(self._nu_hankel, fx, self._q_padded, self._y_q, self._u_m_q[l], self._npad, self._q_high, self._w_m_q)
        return pk_ln
    
    def get_xi_ln_array(self, array):
        def compute_ln(ln):
            l, n = ln
            return self.get_xi_ln(l, n, array)
        xis = jax.vmap(compute_ln)(self.ln_list)
        xi_ln = jnp.zeros((5, 5, len(self._q)))
        ls = self.ln_list[:, 0]
        ns = self.ln_list[:, 1]
        xi_ln = xi_ln.at[ls, ns].set(xis)
        return xi_ln
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop_hankel(self, pk_data, f, bias):

        pk = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        xi_ln = self.get_xi_ln_array(pk)
        
        pkmu_22 = self.get_pkmu_22_hankel(xi_ln, pk_data, f, bias)
        pkmu_13 = self.get_pkmu_13_hankel(xi_ln, pk_data, f, bias)

        return pkmu_22 + pkmu_13
    
    def get_pkmu_22_hankel(self, xi_ln, pk_data, f, bias):
        b1, b2, bG2, bGamma3 = bias

        def compute_pk_ln1n2(term):
            l, n1, n2 = term
            xi1 = xi_ln[l, n1]
            xi2 = xi_ln[l, n2]
            pk_ln1n2 = self.get_pk_ln(0, 0, (-1)**l * 4 * jnp.pi * xi1 * xi2)
            return pk_ln1n2 # (nk,)

        pk_ln1n2 = jax.vmap(compute_pk_ln1n2)(self.ln1n2_list)  # (nterms, nk)

        def compute_coeffs(entry):
            mu_pow     = entry[:, 0]
            f_pow      = entry[:, 1]
            b1_pow     = entry[:, 2]
            b2_pow     = entry[:, 3]
            bG2_pow    = entry[:, 4]
            coeff      = entry[:, 6]

            coeffs = coeff * (f ** f_pow) * (b1 ** b1_pow) * (b2 ** b2_pow) * (bG2 ** bG2_pow)  # (max_len,)

            mu_terms = self._mu[None, :] ** mu_pow[:, None]  # (max_len, nmu)
            return jnp.sum(coeffs[:, None] * mu_terms, axis=0)  # (nmu,)

        coeff_matrix = jax.vmap(compute_coeffs)(self.coeff_info_22)  # (nterms, nmu)

        pkmu = jnp.sum(pk_ln1n2[:, :, None] * coeff_matrix[:, None, :], axis=0)  # (nk, nmu)

        if self.subtract_k0_limit:
            pkmu_k0 = b2**2 / 2. * get_pk_int2(pk_data)
            pkmu = pkmu - pkmu_k0

        return pkmu  # (nk, nmu)
    
    def get_pkmu_13_hankel(self, xi_ln, pk_data, f, bias):
        b1, b2, bG2, bGamma3 = bias
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)

        def get_pk_ln_term(term):
            l, n, m = term
            pk_ln = self.get_pk_ln(l, -1, xi_ln[l, n])  # (nk,)
            return (self._k ** m) * pk_lin * pk_ln

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

        pkmu = jnp.sum(pk_lnm[:, :, None] * coeff_matrix[:, None, :], axis=0)  # (nk, nmu)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop(self, pk_data, f, bias):
        if self.use_hankel:
            pkmu = self.get_pkmu_1loop_hankel(pk_data, f, bias)
        else:
            pkmu = self.get_pkmu_1loop_pld(pk_data, f, bias)
        return pkmu
    
    def get_pkmu_irres_LO_NLO(self, pk_nw, pk_w, damp_fac, pk_data, pk_nw_data, f, bias):
        # LO term
        b1 = bias[0]
        Z1 = b1 + f * self._mu**2
        pkmu_irres_tree = (Z1**2)[None, :] * (pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None] * (1 + damp_fac))
        
        # NLO term
        pkmu_1loop = self.get_pkmu_1loop(pk_data, f, bias)
        pkmu_1loop_nw = self.get_pkmu_1loop(pk_nw_data, f, bias)
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
        Sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.rbao, self.ks)
        dSigma2 = ir_resum.get_dSigma2(pk_nw_data, self.rbao, self.ks)
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
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        P_shot, a0, a2 = params.stoch
        k_nl = params.k_nl
        ndens = params.ndens

        pkmu = P_shot \
            + a0 * jnp.outer((k / k_nl)**2, mu**0) \
            + a2 * jnp.outer((k / k_nl)**2, mu**2)
        pkmu = (1. / ndens) * pkmu

        return pkmu