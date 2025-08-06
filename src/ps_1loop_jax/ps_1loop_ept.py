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

from .hankel import Hankel
from .power_law_decomp import get_decomp_data

from . import pt_coeff
from . import pt_matrix
from . import utils_loop
from .utils_loop import get_pk, get_pk_int, get_pk_int2
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
                 use_hankel=False,
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

        self._initialize_loop_matrix()
        self._initialize_loop_coeff()

        self.use_hankel = use_hankel

    def _set_matrix(self, names=[]):
        for name in names:
            mat_file = glob.glob(os.path.dirname(__file__)+'/pt_matrix/*/*/%s.txt' % (name))[0]
            if '22' in name or 'I' in name or '12' in name:
                self.mat[name] = pt_matrix.PTMatrix22(mat_file)
            elif '13' in name or 'F' in name:
                self.mat[name] = pt_matrix.PTMatrix13(mat_file)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))
        
        # precompute the PT matrices for appropriate FFT settings.
        for name in names:
            if name in utils_loop.kernel_to_decomp_dict.keys():
                decomp_info = utils_loop.kernel_to_decomp_dict[name]
            elif name in self.pkmu_term_names_22:
                decomp_info = utils_loop.kernel_to_decomp_dict['22']
            elif name in self.pkmu_term_names_13:
                decomp_info = utils_loop.kernel_to_decomp_dict['13']

            eta_m = 2 * jnp.pi / (self._nfft * jnp.log(self._k[1] / self._k[0])) * (jnp.arange(self._nfft) - self._nfft // 2)

            if '22' in name or 'I' in name or '12' in name:
                nu1 = decomp_info[0][1]
                nu2 = decomp_info[1][1]
                nu_m1 = -0.5 * (nu1 + eta_m * 1j)
                nu_m2 = -0.5 * (nu2 + eta_m * 1j)
                nu_m1, nu_m2 = jnp.meshgrid(nu_m1, nu_m2)
                self.matrix[name] = self.mat[name](nu_m1, nu_m2).T

            elif '13' in name or 'F' in name:
                nu1 = decomp_info[0][1]
                nu_m1 = -0.5 * (nu1 + eta_m * 1j)
                self.matrix[name] = self.mat[name](nu_m1)

    def _initialize_loop_matrix(self):
        # store the names of 1-loop terms calculated with the FFTLog-based method
        self.pk_term_names = ['22_dd', '13_dd', 'I_d2', 'I_G2', 'I_d2_d2', 'I_G2_G2', 'I_d2_G2', 'F_G2']

        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/22*.txt')
        self.pkmu_term_names_22 = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/13*.txt')
        self.pkmu_term_names_13 = [re.split('/', fname)[-1][:-4] for fname in fnames]

        # precompute the PT matrices
        self.mat = {}
        self.matrix = {}
        self._set_matrix(self.pk_term_names + self.pkmu_term_names_22 + self.pkmu_term_names_13)

        self.matrices_22_real = jnp.array([self.matrix[name] for name in ['22_dd', 'I_d2', 'I_G2', 'I_d2_d2', 'I_G2_G2', 'I_d2_G2']])
        self.matrices_13_real = jnp.array([self.matrix[name] for name in ['13_dd', 'F_G2']])

        ## create arrays of matrices and degrees for 22 and 13, according to the degrees of mu

        self.name_to_index_22 = {name: i for i, name in enumerate(self.pkmu_term_names_22)}
        self.name_to_index_13 = {name: i for i, name in enumerate(self.pkmu_term_names_13)}
        self.matrices_22 = jnp.array([self.matrix[name] for name in self.pkmu_term_names_22])
        self.matrices_13 = jnp.array([self.matrix[name] for name in self.pkmu_term_names_13])

        def get_degree_vector(name):
            d = utils_loop.get_degree_dict(name)
            if name in self.pkmu_term_names_22:
                return [d['mu'], d['f'], d['b1'], d['b2'], d['bG2']]
            else:
                return [d['mu'], d['f'], d['b1'], d['bG2'], d['bGamma3']]
            
        self.degrees_22 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_22])
        self.degrees_13 = jnp.array([get_degree_vector(name) for name in self.pkmu_term_names_13])

        self.nmu_22 = jnp.array([0, 2, 4, 6, 8])
        self.nmu_13 = jnp.array([0, 2, 4, 6])

        term_indices_22 = [[] for _ in range(len(self.nmu_22))]
        term_indices_13 = [[] for _ in range(len(self.nmu_13))]

        for name in self.pkmu_term_names_22:
            idx = self.name_to_index_22[name]
            nmu = self.degrees_22[idx, 0]
            term_indices_22[int(nmu / 2)].append(int(idx))

        for name in self.pkmu_term_names_13:
            idx = self.name_to_index_13[name]
            nmu = self.degrees_13[idx, 0]
            term_indices_13[int(nmu / 2)].append(int(idx))

        # padding with -1 and converting to JAX array
        def pad_to_array(list_of_lists):
            max_len = max(len(lst) for lst in list_of_lists)
            padded = [jnp.array(lst + [-1] * (max_len - len(lst))) for lst in list_of_lists]
            return jnp.stack(padded), jnp.array([len(lst) for lst in list_of_lists])

        # term_indices_22: shape (nmu, max_len), term_lengths_22: shape (nmu,)
        self.term_indices_22, self.term_lengths_22 = pad_to_array(term_indices_22)
        self.term_indices_13, self.term_lengths_13 = pad_to_array(term_indices_13)
    
    def _initialize_loop_coeff(self):
        # store the names of 1-loop terms calculated with the FFTLog-based method
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_coeff/22*.txt')
        self.pkmu_coeff_names_22 = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_coeff/13*.txt')
        self.pkmu_coeff_names_13 = [re.split('/', fname)[-1][:-4] for fname in fnames]

        ln1n2_list = []
        coeff_info_22 = []
        for name in self.pkmu_coeff_names_22:
            str_list = re.split('_', name)
            l, n1, n2 = int(str_list[1]), int(str_list[2]), int(str_list[3])
            ln1n2_list.append([l, n1, n2])
            coeff_file = glob.glob(os.path.dirname(__file__)+'/pt_coeff/%s.txt' % (name))[0]
            coeff_info_22.append(jnp.array(pt_coeff.get_coeff_info(coeff_file)))
        self.ln1n2_list = jnp.array(ln1n2_list)
        self.coeff_info_22 = coeff_info_22

        lnm_list = []
        coeff_info_13 = []
        for name in self.pkmu_coeff_names_13:
            str_list = re.split('_', name)
            l, n, m = int(str_list[1]), int(str_list[2]), int(str_list[3])
            lnm_list.append([l, n, m])
            coeff_file = glob.glob(os.path.dirname(__file__)+'/pt_coeff/%s.txt' % (name))[0]
            coeff_info_13.append(jnp.array(pt_coeff.get_coeff_info(coeff_file)))
        self.lnm_list = jnp.array(lnm_list)
        self.coeff_info_13 = coeff_info_13
        
        self.ln_list = [(0,0), (0,-2), (0,2), (1,-1), (1,1), (1,-3), (1,3), (2,0), (2,-2), (2,2), (3,-1), (3,1), (4,0)]

        # set the Hankel transforms
        self._q = 1 / self._k[::-1]
        nu = 1.1
        l_list = np.arange(5)
        self.hankel_pk2xi = [Hankel(l, nu, self._k, npad=(self._nfft//2), x_high=(self._kmax/100.), c_window_width=0.25) for l in l_list]
        self.hankel_xi2pk = [Hankel(l, nu, self._q, npad=(self._nfft//2), x_high=None, c_window_width=0.25) for l in l_list]

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
                pk -= self._kmin**3 * jnp.real(p_k0_1 @ matrix @ p_k0_1)
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

        bias = params.bias
        ctr = params.ctr
        stoch = params.stoch

        coeffs = jnp.array([
            bias.b1**2,                    # tree
            bias.b1**2,                    # 22_dd
            bias.b1**2,                    # 13_dd
            bias.b1 * bias.b2,            # I_d2
            2 * bias.b1 * bias.bG2,       # I_G2
            bias.b2**2 / 4,               # I_d2_d2
            bias.bG2**2,                  # I_G2_G2
            bias.b2 * bias.bG2,           # I_d2_G2
            2 * bias.b1 * bias.bG2 + (4/5) * bias.b1 * bias.bGamma3  # F_G2
        ])

        pk_terms = self.get_pk_terms(pk_data)

        # tree + 1-loop
        pk_sum = jnp.tensordot(coeffs, pk_terms, axes=1)  # shape: (nk,)

        # counterterm
        pk_ctr = -2.0 * self._k**2 * ctr.c0 * pk_terms[0]

        # interpolation
        pk_interp = jnp.interp(jnp.log(k), jnp.log(self._k), pk_sum + pk_ctr)

        # stochasticity
        pk_stoch = (stoch.P_shot + stoch.a0 * (k / params.k_nl)**2) / params.ndens

        return pk_interp + pk_stoch

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params) -> jnp.ndarray:
        
        f = params.f
        bias = params.bias
        ctr = params.ctr
        stoch = params.stoch
        
        if self.do_irres:
            # tree + 1-loop
            pk_nw_data = ir_resum.get_pk_nw_data(pk_data, params.h, khmin=7e-5, khmax=7.0, 
                                                 kmin_interp=self._kmin, kmax_interp=self._kmax)
            pkmu = self.get_pkmu_irres_LO_NLO(k, mu, pk_data, pk_nw_data, f, bias)

            # counterterm
            pkmu_ctr_k2 = self.get_pkmu_ctr_k2(k, mu, pk_data, pk_nw_data, f, ctr)
            pkmu_ctr_k4 = self.get_pkmu_ctr_k4(k, mu, pk_data, pk_nw_data, f, bias, ctr)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        else:
            # tree + 1-loop
            pkmu_tree = self.get_pkmu_lin(k, mu, pk_data, params, f, bias)
            pkmu_1loop = self.get_pkmu_1loop(k, mu, pk_data, params, f, bias)
            pkmu = pkmu_tree + pkmu_1loop

            # counterterm
            pkmu_ctr_k2 = self.get_pkmu_ctr_k2(k, mu, pk_data, pk_data, f, ctr)
            pkmu_ctr_k4 = self.get_pkmu_ctr_k4(k, mu, pk_data, pk_data, f, bias, ctr)
            pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        
        # NOTE: can be removed
        # stochasticity
        pkmu_stoch = self.get_pkmu_stoch(k, mu, stoch)
        pkmu = pkmu + pkmu_stoch
        
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

        # spline interpolation
        pkmu_grid = self.get_pkmu(self._k, self._mu, pk_data, params)
        mu_tile = jnp.tile(mu_true, (len(k), 1))
        pkmu = interpax.interp2d(jnp.ravel(k_true), jnp.ravel(mu_tile), self._k, self._mu, pkmu_grid, method='cubic2', extrap=True)
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

        Z1 = bias.b1 + f * mu**2
        pk_lin = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pkmu = jnp.outer(pk_lin, Z1**2)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_22_matrix_mu(self, f, bias):
        nmu, max_terms = self.term_indices_22.shape

        def compute_matrix(i, result):
            indices = self.term_indices_22[i]  # shape: (max_terms,)
            valid_mask = indices >= 0  # shape: (max_terms,), bool

            # 無効な -1 を 0 に置き換えておく（あとで mask で消すので OK）
            safe_indices = jnp.where(valid_mask, indices, 0)

            degrees = self.degrees_22[safe_indices]  # shape: (max_terms, 4)
            coeffs = jax.vmap(lambda deg: (f ** deg[0]) * (bias.b1 ** deg[1]) * (bias.b2 ** deg[2]) * (bias.bG2 ** deg[3]))(degrees)
            coeffs = coeffs * valid_mask.astype(coeffs.dtype)  # 無効要素に0をかける

            matrices = self.matrices_22[safe_indices]  # shape: (max_terms, nfft, nfft)
            matrix = jnp.tensordot(coeffs, matrices, axes=1)  # shape: (nfft, nfft)

            return result.at[i].set(matrix)

        result = jax.lax.fori_loop(
            0, nmu, compute_matrix,
            jnp.zeros((nmu, self._nfft, self._nfft), dtype=jnp.complex128)
        )

        return result
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_13_matrix_mu(self, f, bias):
        nmu, max_terms = self.term_indices_13.shape

        def compute_matrix(i, result):
            indices = self.term_indices_13[i]  # shape: (max_terms,)
            valid_mask = indices >= 0  # shape: (max_terms,), bool

            # 無効な -1 を 0 に置き換えておく（あとで mask で消すので OK）
            safe_indices = jnp.where(valid_mask, indices, 0)

            degrees = self.degrees_13[safe_indices]  # shape: (max_terms, 4)
            coeffs = jax.vmap(lambda deg: (f ** deg[0]) * (bias.b1 ** deg[1]) * (bias.bG2 ** deg[2]) * (bias.bGamma3 ** deg[3]))(degrees)
            coeffs = coeffs * valid_mask.astype(coeffs.dtype)  # 無効要素に0をかける

            matrices = self.matrices_13[safe_indices]  # shape: (max_terms, nfft)
            matrix = jnp.tensordot(coeffs, matrices, axes=1)  # shape: (nfft)

            return result.at[i].set(matrix)

        result = jax.lax.fori_loop(
            0, nmu, compute_matrix,
            jnp.zeros((nmu, self._nfft), dtype=jnp.complex128)
        )

        return result
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop_pld(self, pk_data, f, bias):

        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        p_q, p_k, p_k0 = get_decomp_data(-0.7, self._k, pk_lin)
        
        nk, nmu = len(self._k), len(self._mu)
        pkmu = jnp.zeros((nk, nmu))

        ### ---- 22 term ---- ###
        matrix_mu = self.get_pkmu_22_matrix_mu(f, bias)

        def pk_22_single(matrix, nmu_pow):
            pk = self._k**3 * jnp.real(jnp.diag(p_q.T @ matrix @ p_q))  # shape: (nk,)
            if self.subtract_k0_limit:
                pk -= self._kmin**3 * jnp.real(p_k0 @ matrix @ p_k0)
            return jnp.outer(pk, self._mu**nmu_pow)  # shape: (nk, nmu)
        
        # vmap over mu index
        pkmu_22 = jnp.sum(jnp.stack([
            pk_22_single(matrix_mu[i], i) for i in range(nmu)
        ]), axis=0)

        pkmu = pkmu + pkmu_22
        
        ### ---- 13 term ---- ###
        matrix_mu_13 = self.get_pkmu_13_matrix_mu(f, bias)  # shape: (nmu, nfft)

        def pk_13_single(matrix, nmu_pow):
            pk = self._k**3 * p_k * jnp.real(matrix @ p_q)
            return jnp.outer(pk, self._mu**nmu_pow)

        pkmu_13 = jnp.sum(jnp.stack([
            pk_13_single(matrix_mu_13[i], i) for i in range(nmu)
        ]), axis=0)

        pkmu = pkmu + pkmu_13

        ### ---- 13 UV limit ---- ###
        pkmu = pkmu + self.get_pkmu_13_UV(self._k, self._mu, pk_data, f, bias)

        return pkmu  # shape: (nk, nmu)
    
    @partial(jax.jit, static_argnames=['self'])
    def get_pkmu_13_UV(self, k, mu, pk_data, f, bias):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        Z1_g = bias.b1 + f * mu**2
        Z3_g_UV = - 61./315. * bias.b1 - 64./21. * bias.bG2 - 128./105. * bias.bGamma3 \
                + ((- 3./5. + 2./105. * bias.b1) * f + (- 16./35. - 1./3. * bias.b1) * f**2) * mu**2 \
                + ((- 46./105.) * f**2 + (- 1./3.) * f**3) * mu**4
        Z1Z3_UV = Z1_g * Z3_g_UV

        pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_int = get_pk_int(pk_data)

        pkmu_13 = jnp.outer(k**2 * pk * pk_int, Z1Z3_UV)

        return pkmu_13
    
    def get_xi_ln(self, l, n, array):
        _, xi_ln = self.hankel_pk2xi[l](array * self._k**(n+3) / (2 * jnp.pi**2))
        return xi_ln
    
    def get_pk_ln(self, l, n, array):
        _, pk_ln = self.hankel_xi2pk[l](array * self._q**(n+3))
        return pk_ln
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop_hankel(self, pk_data, f, bias):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)

        # generalized correlation functions
        xi_ln = jnp.zeros((5, 5, len(self._k)))
        for (l, n) in self.ln_list:
            xi_ln = xi_ln.at[l, n].set(self.get_xi_ln(l, n, pk_lin))

        pkmu_22 = self.get_pkmu_22_hankel(xi_ln, pk_data, f, bias)
        pkmu_13 = self.get_pkmu_13_hankel(xi_ln, pk_data, f, bias)

        return pkmu_22 + pkmu_13
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_22_hankel(self, xi_ln, pk_data, f, bias):

        pkmu = jnp.zeros((len(self._k), len(self._mu)))

        for (l, n1, n2) in self.ln1n2_list:
            pk_ln1n2 = self.get_pk_ln(0, 0, (-1)**l * 4 * jnp.pi * xi_ln[l, n1] * xi_ln[l, n2])

            res = self.coeff_info_22[l, n1, n2]
            coeff = res[-1] * bias.b1**res[2] * bias.b2**res[3] * bias.bG2**res[4] * f**res[1] * self._mu**res[0]

            pkmu = pkmu + jnp.outer(pk_ln1n2, coeff)

        if self.subtract_k0_limit:
            pkmu_k0 = bias['b2']**2 / 2. * get_pk_int2(pk_data)
            pkmu = pkmu - pkmu_k0

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_13_hankel(self, xi_ln, pk_data, f, bias):

        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        
        pkmu = jnp.zeros((len(self._k), len(self._mu)))

        for (l, n, m) in self.lnm_list:
            pk_ln = self.get_pk_ln(l, -1, xi_ln[l, n])

            res = self.coeff_info_13[l, n, m]
            coeff = res[-1] * bias.b1**res[2] * bias.b2**res[3] * bias.bG2**res[4] * bias.bGamma3**res[5] * f**res[1] * self._mu**res[0]

            pkmu = pkmu + jnp.outer(self._k**m * pk_lin * pk_ln, coeff)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop(self, k, mu, pk_data, f, bias):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        if self.use_hankel:
            pkmu_data = self.get_pkmu_1loop_hankel(pk_data, f, bias)
        else:
            pkmu_data = self.get_pkmu_1loop_pld(pk_data, f, bias)

        # 2D spline interpolation
        k_tile = jnp.tile(k, (len(mu), 1)).T
        mu_tile = jnp.tile(mu, (len(k), 1))
        pkmu = interpax.interp2d(jnp.ravel(k_tile), jnp.ravel(mu_tile), self._k, self._mu, pkmu_data, method='cubic2', extrap=True)
        pkmu = pkmu.reshape(len(k), len(mu))

        return pkmu
    
    @partial(jax.jit, static_argnames=['self'])
    def get_pkmu_irres_LO_NLO(self, k, mu, pk_data, pk_nw_data, f, bias):
        pk_nw, pk_w, damp_fac = self._get_irres_components(k, mu, pk_data, pk_nw_data, f)

        # LO term
        Z1 = bias.b1 + f * mu**2
        pkmu_irres_tree = (Z1**2)[None, :] * (pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None] * (1 + damp_fac))
        
        # NLO term
        pkmu_1loop = self.get_pkmu_1loop(k, mu, pk_data, f, bias)
        pkmu_1loop_nw = self.get_pkmu_1loop(k, mu, pk_nw_data, f, bias)
        pkmu_1loop_w = pkmu_1loop - pkmu_1loop_nw
        pkmu_irres_1loop = pkmu_1loop_nw + jnp.exp(-damp_fac) * pkmu_1loop_w

        pkmu = pkmu_irres_tree + pkmu_irres_1loop

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_ctr_k2(self, k, mu, pk_data, pk_nw_data, f, ctr):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        ctr_k2_mu = ctr.c0 + ctr.c2 * f * mu**2 + ctr.c4 * f**2 * mu**4
        ctr_k2_fac = - 2 * jnp.outer(k**2, ctr_k2_mu)

        if self.do_irres:
            pk = self._get_pk_irres_rsd(k, mu, pk_data, pk_nw_data, f)
            pkmu_ctr_k2 = ctr_k2_fac * pk
        else:
            pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
            pkmu_ctr_k2 = ctr_k2_fac * pk[:, None]

        return pkmu_ctr_k2
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_ctr_k4(self, k, mu, pk_data, pk_nw_data, f, bias, ctr):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        ctr_k4_mu = ctr.cfog * f**4 * mu**4 * (bias.b1 + f * mu**2)**2
        ctr_k4_fac = - jnp.outer(k**4, ctr_k4_mu)

        if self.do_irres:
            pk = self._get_pk_irres_rsd(k, mu, pk_data, pk_nw_data, f)
            pkmu_ctr_k4 = ctr_k4_fac * pk
        else:
            pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
            pkmu_ctr_k4 = ctr_k4_fac * pk[:, None]

        return pkmu_ctr_k4

    @partial(jit, static_argnames=['self'])
    def _get_pk_irres_rsd(self, k, mu, pk_data, pk_nw_data, f):
        pk_nw, pk_w, damp_fac = self._get_irres_components(k, mu, pk_data, pk_nw_data, f)
        pk = pk_nw[:, None] + jnp.exp(-damp_fac) * pk_w[:, None]
        return pk

    @partial(jit, static_argnames=['self'])
    def _get_irres_components(self, k, mu, pk_data, pk_nw_data, f):
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

    # NOTE: can be removed
    @partial(jit, static_argnames=['self'])
    def get_pkmu_stoch(self, k, mu, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        k_nl = params.k_nl
        ndens = params.ndens

        pkmu = params.stoch.P_shot \
            + params.stoch.a0 * jnp.outer((k / k_nl)**2, mu**0) \
            + params.stoch.a2 * jnp.outer((k / k_nl)**2, mu**2)
        pkmu = (1. / ndens) * pkmu

        return pkmu