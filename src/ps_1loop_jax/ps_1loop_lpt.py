import os

import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp
import numpy as np
import interpax

from .hankel import Hankel

from .utils_loop import get_pk, get_pk_int
from .utils_lpt import get_G00, get_Gs


class PowerSpectrum1LoopLPT:

    def __init__(self, 
                 cross=False,
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=256,
                 lmax=5,
                 ngauss=4,
                 use_galileon=False,
                 use_Pzel=True
                 ):
        
        self.cross = cross # flag to enable the calculation of cross power spectra
        self.lmax = lmax
        self.use_galileon = use_galileon
        self.use_Pzel = use_Pzel

        self.term_names = ['ZA', 'A> A>', 'A22', 'A13', 'W112',
                           'U10', 'A> U_lin', 'xi_lin', 'A> xi_lin', 'U11', 'U20', 
                           'A10', 'U_lin U_lin', 'xi_lin xi_lin', 'xi_lin U_lin',
                           'V10', 'V12', 'Upsilon', 'chi', 'zeta'
                           ]
        self.bias_combs = ['1', 'b1', 'b1 b1', 'b2', 'b1 b2', 'b2 b2', 'bs', 'b1 bs', 'b2 bs', 'bs bs']

        # preparation for Gauss-Legendre quadrature
        self._ngauss = ngauss
        mu, self._ws = np.polynomial.legendre.leggauss(2 * self._ngauss)
        self._mu = mu[self._ngauss:]
        self._leg0 = np.polynomial.legendre.Legendre((1))(mu)
        self._leg2 = np.polynomial.legendre.Legendre((0,0,1))(mu)
        self._leg4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(mu)

        # preparation for FFT
        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._q = 1 / self._k[::-1]
        self._initialize_lpt()

    def _initialize_lpt(self):
        # (l, n) for which xi_ln's are computed
        self.ln_list = [(0,0), (0,-2), (0,2), (1,-1), (1,1), (2,0), (2,-2), (2,2), (3,-1), (3,1), (4,0)]

        # set the Hankel transforms
        self.hankel_pk2xi = {}
        self.hankel_xi2pk = {}
        nu = 1.1
        l_list = [i for i in range(max(4, self.lmax) + 1)]
        for l in l_list:
            self.hankel_pk2xi[l] = Hankel(l, nu, self._k, npad=(self._nfft//2), x_high=(jnp.max(self._k)/10), c_window_width=0.25)
            self.hankel_xi2pk[l] = Hankel(l, nu, self._q, npad=(self._nfft//2), x_high=(jnp.max(self._q)/10), c_window_width=0.25)
        
        # load the coefficients of G00
        loaded = jnp.load(os.path.dirname(__file__)+'/lpt_rsd_coeff/G00_coeffs.npz')
        self.G00_coeffs = {int(k): jnp.array(loaded[k]) for k in loaded.files}

    def get_xi_ln(self, l, n, array):
        _, xi_ln = self.hankel_pk2xi[l](array * self._k**(n+3) / (2 * jnp.pi**2))
        return xi_ln
    
    def get_pk_ln(self, l, n, array):
        _, pk_ln = self.hankel_xi2pk[l](array * self._q**(n+3))
        return pk_ln
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_zel_k_mu(self, k, mu, corrs, f):
        corrs_tree = corrs[0]

        X = corrs_tree[0]
        Y = corrs_tree[1]

        Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
        Ksq = (k * Kfac)**2
        c = (1 + f * mu**2) / Kfac
        s = f * mu * jnp.sqrt(1 - mu**2) / Kfac
        
        A = k * self._q * c
        B = - 0.5 * Ksq * Y
        C = k * self._q * s

        base = 4 * jnp.pi * self._q**3 * jnp.exp(- 0.5 * Ksq * (X + Y))
        
        pkmu = 0.
        for l in range(self.lmax + 1):
            integrand = base * (-2 / (k * self._q))**(l) * get_G00(A, B, C, self.G00_coeffs[l])
            _, pk_fft = self.hankel_xi2pk[l](integrand)
            pkmu = pkmu + interpax.interp1d(jnp.log(k), jnp.log(self._k), pk_fft)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_zel(self, k, mu, pk_data, f):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        corrs = self.get_corrs(pk_data)

        vmap_mu = jax.vmap(self.get_pkmu_zel_k_mu, in_axes=(None, 0, None, None))
        vmap_k_mu = jax.vmap(vmap_mu, in_axes=(0, None, None, None))
        pkmu = vmap_k_mu(k, mu, corrs, f)

        return pkmu

    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict_k_mu(self, k, mu, corrs, f):
        corrs_tree = corrs[0]
        corrs_matter_1loop = corrs[1]
        corrs_bias = corrs[2]

        # matter tree-level terms
        X_lin_lt = corrs_tree[2]
        Y_lin_lt = corrs_tree[3]
        X_lin_gt = corrs_tree[4]
        Y_lin_gt = corrs_tree[5]
        
        # matter one-loop terms
        X22 = corrs_matter_1loop[0]
        Y22 = corrs_matter_1loop[1]
        X13 = corrs_matter_1loop[2]
        Y13 = corrs_matter_1loop[3]
        V1 = corrs_matter_1loop[4]
        V3 = corrs_matter_1loop[5]
        T = corrs_matter_1loop[6]

        # LIMD bias terms
        xi_lin = corrs_tree[6]
        U_lin = corrs_tree[7]
        U3 = corrs_bias[0]
        U11 = corrs_bias[1]
        U20 = corrs_bias[2]
        X10 = corrs_bias[3]
        Y10 = corrs_bias[4]

        # 2nd-order shear bias terms
        V10 = corrs_bias[5]
        V12 = corrs_bias[6]
        X_Upsilon = corrs_bias[7]
        Y_Upsilon = corrs_bias[8]
        chi = corrs_bias[9]
        zeta = corrs_bias[10]

        Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
        K = k * Kfac
        Ksq = K**2
        c = (1 + f * mu**2) / Kfac
        s = f * mu * jnp.sqrt(1 - mu**2) / Kfac
        A_mu = (1 + f) * mu / Kfac
        B_mu = jnp.sqrt(1 - mu**2) / Kfac

        A = k * self._q * c
        B = - 0.5 * Ksq * Y_lin_lt
        C = k * self._q * s

        base = 4 * jnp.pi * self._q**3 * jnp.exp(- 0.5 * Ksq * (X_lin_lt + Y_lin_lt))
        Gs = get_Gs(A, B, C, self.G00_coeffs, self.lmax)
        
        pkmu = {name: 0 for name in self.term_names}
        integrand = {name: 0 for name in self.term_names}

        for l in range(self.lmax + 1):
            
            # prepare analytic solution of angular integral for each combination of mu_q & mu_nq
            mq0 = Gs[0,0][l]
            mq1 = - Gs[1,0][l]
            mq2 = - Gs[2,0][l]
            mq3 = Gs[3,0][l]
            mq4 = Gs[4,0][l]
            nq1 = - A_mu * Gs[1,0][l] + B_mu * Gs[0,1][l]
            nq2 = - A_mu**2 * Gs[2,0][l] + 2 * A_mu * B_mu * Gs[1,1][l] - B_mu**2 * Gs[0,2][l]
            mq1_nq1 = - A_mu * Gs[2,0][l] + B_mu * Gs[1,1][l]
            mq2_nq1 = A_mu * Gs[3,0][l] - B_mu * Gs[2,1][l]

            # matter terms
            integrand['ZA'] = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)

            integrand['A> A>'] = Ksq**2 / 8 * (mq0 * X_lin_gt**2 + 2 * mq2 * X_lin_gt * Y_lin_gt + mq4 * Y_lin_gt**2)
    
            integrand['A22'] = -0.5 * k**2 * ((Kfac**2 + 2 * f * (1 + f) * mu**2 + f**2 * mu**2) * mq0 * X22 + \
                                    (Kfac**2 * mq2 + 2 * f * Kfac * mu * mq1_nq1 + f**2 * mu**2 * nq2) * Y22)

            integrand['A13'] = -0.5 * k**2 * (2 * (Kfac**2 + 2 * f * (1 + f) * mu**2) * mq0 * X13 + \
                                    2 * (Kfac**2 * mq2 + 2 * f * Kfac * mu * mq1_nq1) * Y13)

            integrand['W112'] = 0.5 * k**3 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu**2) * mq1 * V1 +  \
                                    Kfac**2 * (Kfac * mq1 + f * mu * nq1) * V3 + \
                                    Kfac**2 * (Kfac * mq3 + f * mu * mq2_nq1) * T)
            integrand['W112'] = -2 * integrand['W112']

            # LIMD bias terms
            integrand['U10'] = -2 * (K * mq1 * (U_lin + U3) + (2 * f * k * mu * nq1) * U3)

            integrand['A> U_lin'] = Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * (K * U_lin)

            integrand['xi_lin'] = mq0 * xi_lin

            integrand['A> xi_lin'] = -0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * xi_lin

            integrand['U11'] = - (K * mq1 + f * k * mu * nq1) * U11

            integrand['U20'] = - (K * mq1 + f * k * mu * nq1) * U20

            integrand['A10'] = - Ksq * (mq0 * X10 + mq2 * Y10) - f * (1 + f) * (k * mu)**2 * mq0 * X10 \
                                - f * k * mu * mq1_nq1 * Y10

            integrand['U_lin U_lin'] = - Ksq * mq2 * U_lin**2

            integrand['xi_lin xi_lin'] = mq0 * xi_lin**2

            integrand['xi_lin U_lin'] = -2 * K * mq1 * xi_lin * U_lin

            # 2nd-order shear bias terms
            integrand['V10'] = -2 * (K * mq1 + f * k * mu * nq1) * V10

            integrand['V12'] = -2 * K * mq1 * V12

            integrand['Upsilon'] = - Ksq * (mq0 * X_Upsilon + mq2 * Y_Upsilon)

            integrand['chi'] = mq0 * chi

            integrand['zeta'] = mq0 * zeta

            # Hankel transforms
            for name in self.term_names:
                _, pk_fft = self.hankel_xi2pk[l](base * (-2 / (k * self._q))**(l) * integrand[name])
                pkmu[name] = pkmu[name] + interpax.interp1d(jnp.log(k), jnp.log(self._k), pk_fft)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict(self, k, mu, pk_data, f):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        corrs = self.get_corrs(pk_data)

        vmap_mu = jax.vmap(self.get_pkmu_dict_k_mu, in_axes=(None, 0, None, None))
        vmap_k_mu = jax.vmap(vmap_mu, in_axes=(0, None, None, None))
        pkmu_dict = vmap_k_mu(k, mu, corrs, f)

        return pkmu_dict

    @partial(jit, static_argnames=['self'])
    def get_pkmu_bias_dict_k_mu(self, k, mu, corrs, f):
        corrs_tree = corrs[0]
        corrs_matter_1loop = corrs[1]
        corrs_bias = corrs[2]

        # matter tree-level terms
        X_lin_lt = corrs_tree[2]
        Y_lin_lt = corrs_tree[3]
        X_lin_gt = corrs_tree[4]
        Y_lin_gt = corrs_tree[5]
        
        # matter one-loop terms
        X22 = corrs_matter_1loop[0]
        Y22 = corrs_matter_1loop[1]
        X13 = corrs_matter_1loop[2]
        Y13 = corrs_matter_1loop[3]
        V1 = corrs_matter_1loop[4]
        V3 = corrs_matter_1loop[5]
        T = corrs_matter_1loop[6]

        # LIMD bias terms
        xi_lin = corrs_tree[6]
        U_lin = corrs_tree[7]
        U3 = corrs_bias[0]
        U11 = corrs_bias[1]
        U20 = corrs_bias[2]
        X10 = corrs_bias[3]
        Y10 = corrs_bias[4]

        # 2nd-order shear bias terms
        V10 = corrs_bias[5]
        V12 = corrs_bias[6]
        X_Upsilon = corrs_bias[7]
        Y_Upsilon = corrs_bias[8]
        chi = corrs_bias[9]
        zeta = corrs_bias[10]

        # 3rd-order bias terms
        Ub3 = corrs_bias[11]
        theta = corrs_bias[12]

        Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
        K = k * Kfac
        Ksq = K**2
        c = (1 + f * mu**2) / Kfac
        s = f * mu * jnp.sqrt(1 - mu**2) / Kfac
        A_mu = (1 + f) * mu / Kfac
        B_mu = jnp.sqrt(1 - mu**2) / Kfac

        A = k * self._q * c
        B = - 0.5 * Ksq * Y_lin_lt
        C = k * self._q * s

        base = 4 * jnp.pi * self._q**3 * jnp.exp(- 0.5 * Ksq * (X_lin_lt + Y_lin_lt))
        Gs = get_Gs(A, B, C, self.G00_coeffs, self.lmax)

        ncomp = 13
        pkmu_terms = jnp.zeros(ncomp)

        for l in range(self.lmax + 1):

            # prepare analytic solution of angular integral for each combination of mu_q & mu_nq
            mq0 = Gs[0,0][l]
            mq1 = - Gs[1,0][l]
            mq2 = - Gs[2,0][l]
            mq3 = Gs[3,0][l]
            mq4 = Gs[4,0][l]
            nq1 = - A_mu * Gs[1,0][l] + B_mu * Gs[0,1][l]
            nq2 = - A_mu**2 * Gs[2,0][l] + 2 * A_mu * B_mu * Gs[1,1][l] - B_mu**2 * Gs[0,2][l]
            mq1_nq1 = - A_mu * Gs[2,0][l] + B_mu * Gs[1,1][l]
            mq2_nq1 = A_mu * Gs[3,0][l] - B_mu * Gs[2,1][l]

            # matter terms
            integrand_1 = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) # ZA
            integrand_1 = integrand_1 + Ksq**2 / 8 * (mq0 * X_lin_gt**2 + 2 * mq2 * X_lin_gt * Y_lin_gt + mq4 * Y_lin_gt**2) # A> A>
            integrand_1 = integrand_1 - 0.5 * k**2 * ((Kfac**2 + 2 * f * (1 + f) * mu**2 + f**2 * mu**2) * mq0 * X22 + \
                                                      (Kfac**2 * mq2 + 2 * f * Kfac * mu * mq1_nq1 + f**2 * mu**2 * nq2) * Y22) # A22
            integrand_1 = integrand_1 - 0.5 * k**2 * (2 * (Kfac**2 + 2 * f * (1 + f) * mu**2) * mq0 * X13 + \
                                                      2 * (Kfac**2 * mq2 + 2 * f * Kfac * mu * mq1_nq1) * Y13) # A13
            integrand_1 = integrand_1 - 2 * 0.5 * k**3 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu**2) * mq1 * V1 +  \
                                                          Kfac**2 * (Kfac * mq1 + f * mu * nq1) * V3 + \
                                                            Kfac**2 * (Kfac * mq3 + f * mu * mq2_nq1) * T) # W112
            
            # LIMD bias terms
            integrand_b1 = -2 * (K * mq1 * (U_lin + U3) + (2 * f * k * mu * nq1) * U3) # U10
            integrand_b1 = integrand_b1 + Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * (K * U_lin) # A> U_lin
            integrand_b1 = integrand_b1 - Ksq * (mq0 * X10 + mq2 * Y10) - f * (1 + f) * (k * mu)**2 * mq0 * X10 \
                - f * k * mu * mq1_nq1 * Y10 # A10
            
            integrand_b1_b1 = mq0 * xi_lin # xi_lin
            integrand_b1_b1 = integrand_b1_b1 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * xi_lin # A> xi_lin
            integrand_b1_b1 = integrand_b1_b1 - (K * mq1 + f * k * mu * nq1) * U11 # U11
            integrand_b1_b1 = integrand_b1_b1 - Ksq * mq2 * U_lin**2 # U_lin U_lin

            integrand_b2 = - (K * mq1 + f * k * mu * nq1) * U20 # U20
            integrand_b2 = integrand_b2 - Ksq * mq2 * U_lin**2 # U_lin U_lin

            integrand_b1_b2 = -2 * K * mq1 * xi_lin * U_lin # xi_lin U_lin

            integrand_b2_b2 = 0.5 * mq0 * xi_lin**2 # xi_lin xi_lin

            # 2nd-order shear bias terms
            integrand_bs = -2 * (K * mq1 + f * k * mu * nq1) * V10 # V10 
            integrand_bs = integrand_bs - Ksq * (mq0 * X_Upsilon + mq2 * Y_Upsilon) # Upsilon

            integrand_b1_bs = -2 * K * mq1 * V12 # V12

            integrand_b2_bs = mq0 * chi # chi
            
            integrand_bs_bs = mq0 * zeta # zeta

            # 3rd-order bias terms
            integrand_b3 = mq0 * Ub3 # Ub3
            integrand_b1_b3 = mq0 * theta # theta

            # counterterm
            if self.use_Pzel:
                integrand_ctr = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) # ZA
            else:
                integrand_ctr = mq0 * xi_lin # xi_lin

            integrands = jnp.stack([integrand_1, integrand_b1, integrand_b1_b1, 
                                    integrand_b2, integrand_b1_b2, integrand_b2_b2,
                                    integrand_bs, integrand_b1_bs, integrand_b2_bs, integrand_bs_bs, 
                                    integrand_b3, integrand_b1_b3, integrand_ctr
                                    ])

            # Hankel transforms
            for i in range(ncomp):
                _, pk_fft = self.hankel_xi2pk[l](base * (-2 / (k * self._q))**(l) * integrands[i])
                pkmu_terms = pkmu_terms.at[i].add(interpax.interp1d(jnp.log(k), jnp.log(self._k), pk_fft))

        return pkmu_terms

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, k_IR=0.2):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        corrs = self.get_corrs(pk_data, k_IR)

        vmap_mu = jax.vmap(self.get_pkmu_bias_dict_k_mu, in_axes=(None, 0, None, None))
        vmap_k_mu = jax.vmap(vmap_mu, in_axes=(0, None, None, None))
        pkmu_terms = vmap_k_mu(k, mu, corrs, params['f'])

        b1 = params['bias']['b1']
        b2 = params['bias']['b2']
        bs = params['bias']['bs']
        bias_facs = jnp.array([1, b1, b1**2, b2, b1 * b2, b2**2, bs, b1 * bs, b2 * bs, bs**2])
        
        # collect all bias terms
        pkmu = jnp.sum(jnp.array([bias_facs[i] * pkmu_terms[i] for i in self.bias_combs]), axis=0)
        
        # counterterm
        ctr = params['ctr']
        ctr_mu = ctr['alpha0'] + ctr['alpha2'] * mu**2 + ctr['alpha4'] * mu**4 + ctr['alpha6'] * mu**6
        pkmu_ctr = jnp.kron(k**2, ctr_mu).reshape(len(k), len(mu)) * pkmu_terms[-1]
        pkmu = pkmu + pkmu_ctr

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pk_ells(self, k, pk_data, params, k_IR=0.2):
        k = jnp.atleast_1d(k).astype(float)

        pkmu = self.get_pkmu(k, self._mu, pk_data, params, k_IR).T
        pkmu = jnp.vstack([jnp.flip(pkmu, axis=0), pkmu])

        pk0 = 0.5 * jnp.sum((self._ws * self._leg0)[:, None] * pkmu, axis=0)
        pk2 = 2.5 * jnp.sum((self._ws * self._leg2)[:, None] * pkmu, axis=0)
        pk4 = 4.5 * jnp.sum((self._ws * self._leg4)[:, None] * pkmu, axis=0)
        
        return pk0, pk2, pk4
    
    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data, k_IR=0.2):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_lin_lt = pk_lin * jnp.exp(-0.5 * (self._k / k_IR)**2)

        # generalized correlation functions
        xi_ln = jnp.zeros((5, 5, len(self._k)))
        xi_ln_lt = jnp.zeros((5, 5, len(self._k)))
        for (l, n) in self.ln_list:
            xi_ln = xi_ln.at[l, n].set(self.get_xi_ln(l, n, pk_lin))
            xi_ln_lt = xi_ln_lt.at[l, n].set(self.get_xi_ln(l, n, pk_lin_lt))
        
        # tree-level terms
        corrs_tree = self.get_corrs_tree(xi_ln, xi_ln_lt)

        # one-loop terms
        QRs = self.get_QR(xi_ln, pk_lin)
        corrs_matter_1loop = self.get_corrs_matter_1loop(QRs)
        corrs_bias = self.get_corrs_bias(QRs, xi_ln)

        return [corrs_tree, corrs_matter_1loop, corrs_bias]
    
    @partial(jit, static_argnames=['self'])
    def get_corrs_tree(self, xi_ln, xi_ln_lt):
        X_lin = 2/3 * (xi_ln[0,-2][0] - xi_ln[0,-2] - xi_ln[2,-2])
        Y_lin = 2 * xi_ln[2,-2]

        X_lin_lt = 2/3 * (xi_ln_lt[0,-2][0] - xi_ln_lt[0,-2] - xi_ln_lt[2,-2])
        Y_lin_lt = 2 * xi_ln_lt[2,-2]

        X_lin_gt = X_lin - X_lin_lt
        Y_lin_gt = Y_lin - Y_lin_lt
        
        xi_lin = xi_ln[0,0]
        U_lin = - xi_ln[1,-1]

        corrs = jnp.stack([X_lin, Y_lin, X_lin_lt, Y_lin_lt,
                           X_lin_gt, Y_lin_gt, xi_lin, U_lin])
        return corrs
    
    @partial(jit, static_argnames=['self'])
    def get_QR(self, xi_ln, pk_lin):

        integrand_Q1 = 8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2
        integrand_Q5 = 2/3 * xi_ln[0,0]**2 - 2/3 * xi_ln[2,0]**2 \
                    - 2/5 * xi_ln[1,-1] * xi_ln[1,1] + 2/5 * xi_ln[3,-1] * xi_ln[3,1]
        integrand_Q8 = 2/3 * xi_ln[0,0]**2 - 2/3 * xi_ln[2,0]**2

        Q1 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q1)
        Q5 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q5)
        Q8 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q8)

        pk_00 = self.get_pk_ln(0, -1, xi_ln[0,0])
        pk_20 = self.get_pk_ln(2, -1, xi_ln[2,0])
        pk_40 = self.get_pk_ln(4, -1, xi_ln[4,0])
        pk_11 = self.get_pk_ln(1, -1, xi_ln[1,1])
        pk_31 = self.get_pk_ln(3, -1, xi_ln[3,1])

        R1 = self._k**2 * pk_lin * (8/15 * pk_00 - 16/21 * pk_20 + 8/35 * pk_40)
        R3 = 2/3 * self._k**2 * pk_lin * (pk_00 - pk_20) - 2/5 * self._k * pk_lin * (pk_11 - pk_31)

        Q2 = 2 * Q5 - Q1
        R2 = R3 - R1

        QRs = jnp.stack([Q1, Q2, Q5, Q8, R1, R2])
        return QRs
    
    @partial(jit, static_argnames=['self'])
    def get_corrs_matter_1loop(self, QRs):
        Q1 = QRs[0]
        Q2 = QRs[1]
        R1 = QRs[4]
        R2 = QRs[5]

        # X, Y for 1-loop A_{ij}
        xi_ln_22 = jnp.stack([self.get_xi_ln(l, n, 9/98 * Q1) for (l, n) in [(0,-2), (2,-2)]])
        xi_ln_13 = jnp.stack([self.get_xi_ln(l, n, 5/21 * Q2) for (l, n) in [(0,-2), (2,-2)]])

        X22 = 2/3 * (xi_ln_22[0,0] - xi_ln_22[0] - xi_ln_22[1])
        Y22 = 2 * xi_ln_22[1]

        X13 = 2/3 * (xi_ln_13[0,0] - xi_ln_13[0] - xi_ln_13[1])
        Y13 = 2 * xi_ln_13[1]

        # V1, V3, T for W_{ijk}
        T = self.get_xi_ln(3, -1, 3/14 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2))
        V1 = self.get_xi_ln(1, -1, -3/70 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2))
        V3 = self.get_xi_ln(1, -1, 3/70 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2))
        V1 = V1 - 1/5 * T
        V3 = V3 - 1/5 * T

        corrs = jnp.stack([X22, Y22, X13, Y13, V1, V3, T])
        return corrs
    
    @partial(jit, static_argnames=['self'])
    def get_corrs_bias(self, QRs, xi_ln):
        Q1 = QRs[0]
        Q5 = QRs[2]
        Q8 = QRs[3]
        R1 = QRs[4]
        R2 = QRs[5]
        
        # U10, U11, U20
        U3 = self.get_xi_ln(1, -1, -5/21 * R1) # 3rd-order part of U10
        U11 = self.get_xi_ln(1, -1, 3/14 * (R1 + R2)) # U11
        U20 = self.get_xi_ln(1, -1, -3/7 * Q8) # U20
        
        # A10
        xi_ln_A10_q0 = get_pk_int({'k': self._k, 'pk': 2/7 * R1}) # R2 integral is zero
        xi_ln_A10_0m2 = self.get_xi_ln(0, -2, 2/7 * (Q5 + 2 * R2))
        xi_ln_A10_2m2 = self.get_xi_ln(2, -2, 1/7 * (2 * Q5 + 3 * R1 + 4 * R2))
        X10 = xi_ln_A10_q0 - xi_ln_A10_0m2 - xi_ln_A10_2m2
        Y10 = 3 * xi_ln_A10_2m2

        # corrlations from 2nd-order shear
        if self.use_galileon:
            V10 = self.get_xi_ln(1, -1, 3/7 * Q1) # V10 based on G2
            V12 = self.get_xi_ln(1, -1, 2 * Q5) # V12 based on G2
            chi = 2 * (-2/3 * xi_ln[0,0]**2 + 2/3 * xi_ln[2,0]**2) # chi based on G2
            zeta = 2 * (8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2) # zeta based on G2

            # Upsilon based on G2
            X_Upsilon = None
            Y_Upsilon = None
        else:
            V10 = self.get_xi_ln(1, -1, 3/7 * Q1 - 2/7 * Q8) # V10 based on s^2
            V12 = 2 * (4/15 * xi_ln[1,-1] - 2/5 * xi_ln[3,-1]) * xi_ln[2,0] # V12 based on s^2
            chi = 2 * (2/3 * xi_ln[0,0]**2 + xi_ln[2,0]**2) # chi based on s^2
            zeta = 2 * (4/45 * xi_ln[0,0]**2 + 8/63 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2) # zeta based on s^2

            # Upsilon based on s^2
            J2 = 2/15 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
            J3 = -1/5 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
            J4 = xi_ln[3,-1]
            X_Upsilon = 4 * J3**2
            Y_Upsilon = 6 * J2**2 + 8 * J2 * J3 + 4 * J2 * J4 + 4 * J3**2 + 8 * J3 * J4 + 2 * J4**2

        # 3rd-order bias


        corrs = jnp.stack([U3, U11, U20, X10, Y10, V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta])
        return corrs
