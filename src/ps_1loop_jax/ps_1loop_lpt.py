import os

import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp
import numpy as np
import quadax
import interpax

from .hankel import Hankel

from .utils_loop import get_pk, get_pk_int
from .utils_math import legendre
from .utils_lpt import get_G00, get_Gs

class PowerSpectrum1LoopLPT:

    def __init__(self, 
                 cross=False,
                 subtract_k0_limit=True,
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=256,
                 ):
        
        self.cross = cross # flag to enable the calculation of cross power spectra
        self.subtract_k0_limit = subtract_k0_limit # flag to subtract k -> 0 limit from 2-2 terms

        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._q = 1 / self._k[::-1]
        self._mu = jnp.linspace(0., 1., 51)

        self._initialize_loop_coeff()
            
    def _initialize_loop_coeff(self):
        # store the names of 1-loop terms calculated with the FFTLog-based method
        l_list = [0,1,2,3,4]
        
        self.ln_list = [(0,0), (0,-2), (0,2), (1,-1), (1,1), (2,0), (2,-2), (2,2), (3,-1), (3,1), (4,0)]

        # set the Hankel transforms
        self.hankel_pk2xi = {}
        self.hankel_xi2pk = {}
        nu = 1.1
        for l in l_list:
            self.hankel_pk2xi[l] = Hankel(l, nu, self._k, npad=(self._nfft//2), x_high=(jnp.max(self._k)/10), c_window_width=0.2)
            self.hankel_xi2pk[l] = Hankel(l, nu, self._q, npad=(self._nfft//2), x_high=(jnp.max(self._q)/10), c_window_width=0.2)
        
        # load the coefficients of G00
        loaded = jnp.load(os.path.dirname(__file__)+'/lpt_rsd_coeff/G00_coeffs.npz')
        self.G00_coeffs = {int(k): jnp.array(loaded[k]) for k in loaded.files}

    @partial(jit, static_argnames=['self'])
    def get_xi_ln(self, l, n, array):
        _, xi_ln = self.hankel_pk2xi[l](array * self._k**(n+3) / (2 * jnp.pi**2))
        return xi_ln
    
    @partial(jit, static_argnames=['self'])
    def get_pk_ln(self, l, n, array):
        _, pk_ln = self.hankel_xi2pk[l](array * self._q**(n+3))
        return pk_ln
    
    @partial(jit, static_argnames=['self'])
    def get_xi_ln_dict(self, pk_data):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        xi_ln_dict = {(l, n): self.get_xi_ln(l, n, pk_lin) for (l, n) in self.ln_list}
        return xi_ln_dict
    
    @partial(jit, static_argnames=['self'])
    def get_xi_ln_lt_dict(self, pk_data, k_IR=0.2):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax) * jnp.exp(-0.5 * (self._k / k_IR)**2)
        xi_ln_dict = {(l, n): self.get_xi_ln(l, n, pk_lin) for (l, n) in self.ln_list}
        return xi_ln_dict
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_zel(self, k, mu, f, pk_data):
        corrs = self.get_corrs(pk_data)
        
        X = corrs['X_lin']
        Y = corrs['Y_lin']
        
        Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
        Ksq = (k * Kfac)**2
        c = (1 + f * mu**2) / Kfac
        s = f * mu * jnp.sqrt(1 - mu**2) / Kfac
        
        A = k * self._q * c
        B = - 0.5 * Ksq * Y
        C = k * self._q * s

        base = 4 * jnp.pi * self._q**3 * jnp.exp(- 0.5 * Ksq * (X + Y))
        
        pk = 0.
        for l in range(self.lmax + 1):
            integrand = base * (-2 / (k * self._q))**(l) * get_G00(A, B, C, self.G00_coeffs[l])
            _, pk_fft = self.hankel_xi2pk[l](integrand)
            pk = pk + interpax.interp1d(jnp.log(k), jnp.log(self._k), pk_fft)

        return pk

    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict(self, k, mu, f, pk_data):
        corrs = self.get_corrs(pk_data)

        X_lin_lt = corrs['X_lin_lt']
        Y_lin_lt = corrs['Y_lin_lt']
        X_lin_gt = corrs['X_lin_gt']
        Y_lin_gt = corrs['Y_lin_gt']
        
        X22 = corrs['X22']
        Y22 = corrs['Y22']
        X13 = corrs['X13']
        Y13 = corrs['Y13']
        V1 = corrs['V1']
        V3 = corrs['V3']
        T = corrs['T']

        Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
        Ksq = (k * Kfac)**2
        c = (1 + f * mu**2) / Kfac
        s = f * mu * jnp.sqrt(1 - mu**2) / Kfac
        A_mu = (1 + f) * mu / Kfac
        B_mu = jnp.sqrt(1 - mu**2) / Kfac

        A = k * self._q * c
        B = - 0.5 * Ksq * Y_lin_lt
        C = k * self._q * s

        base = 4 * jnp.pi * q_fft**3 * jnp.exp(- 0.5 * Ksq * (X_lin_lt + Y_lin_lt))
        Gs = get_Gs(A, B, C, lmax=self.lmax)

        pkmu = {name: 0 for name in ['za', 'A>', 'A>A>', 'A22', 'A13', 'W112']}
        integrand = {}

        for l in range(self.lmax + 1):
            # G = {(m,n): self.get_G[(m,n)](A, B, C, [self.G00_coeffs[l-i] for i in range(m+n)]) for (m,n) in self.mn_list}
            mu2 = Gs[(2,0)][l]
            mu_nq1 = A_mu * Gs[(2,0)][l] + B_mu * Gs[(1,1)][l]
            nq1 = A_mu * Gs[(1,0)][l] + B_mu * Gs[(0,1)][l]
            nq2 = A_mu**2 * Gs[(2,0)][l] + 2 * A_mu * B_mu * Gs[(1,1)][l] + B_mu**2 * Gs[(0,2)][l]
            mu2_nq1 = A_mu * Gs[(3,0)][l] + B_mu * Gs[(2,1)][l]

            integrand['za'] = Gs[(0,0)][l] # za
            integrand['A>'] = - 0.5 * Ksq * (X_lin_gt * Gs[(0,0)][l] + Y_lin_gt * Gs[(2,0)][l]) # A>
            integrand['A>A>'] = Ksq**2 / 8 * (X_lin_gt**2 * Gs[(0,0)][l] + 2 * X_lin_gt * Y_lin_gt * Gs[(2,0)][l] + Y_lin_gt**2 * Gs[(4,0)][l]) # A>A>
    
            integrand['A22'] = -0.5 * k**2 * ((Kfac**2 + 2*f*(1+f)*mu**2 + f**2*mu**2) * Gs[(0,0)][l] * X22 + \
                                    (Kfac**2 * mu2 + 2 * f * Kfac * mu * mu_nq1 + f**2 * mu**2 * nq2) * Y22) # A22

            integrand['A13'] = -0.5 * k**2 * (2 * (Kfac**2 + 2*f*(1+f)* mu**2) * Gs[(0,0)][l] * X13 + \
                                    2 * (Kfac**2 * mu2 + 2 * f * Kfac * mu * mu_nq1) * Y13 ) # A13

            integrand['W112'] = 0.5 * k**3 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu**2) * Gs[(1,0)][l] * V1 +  \
                                    Kfac**2 * (Kfac * Gs[(1,0)][l] + f * mu * nq1) * V3 + \
                                    Kfac**2 * (Kfac * Gs[(3,0)][l] + f * mu * mu2_nq1) * T) # W112
            integrand['W112'] = -2 * integrand['W112']

            for name in ['za', 'A>', 'A>A>', 'A22', 'A13', 'W112']:
                _, pk_fft = self.hankel_xi2pk[l](base * (-2 / (k * self._q))**(l) * integrand[name])
                pkmu[name] = pkmu[name] + interpax.interp1d(jnp.log(k), jnp.log(self._k), pk_fft)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data, k_IR=0.2):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_lin_lt = pk_lin * jnp.exp(-0.5 * (self._k / k_IR)**2)

        xi_ln = {(l, n): self.get_xi_ln(l, n, pk_lin) for (l, n) in self.ln_list}
        xi_ln_lt = {(l, n): self.get_xi_ln(l, n, pk_lin_lt) for (l, n) in self.ln_list}

        corrs = {}

        corrs['X_lin'] = 2/3 * (xi_ln[(0,-2)][0] - xi_ln[(0,-2)] - xi_ln[(2,-2)])
        corrs['Y_lin'] = 2 * xi_ln[(2,-2)]

        corrs['X_lin_lt'] = 2/3 * (xi_ln_lt[(0,-2)][0] - xi_ln_lt[(0,-2)] - xi_ln_lt[(2,-2)])
        corrs['Y_lin_lt'] = 2 * xi_ln_lt[(2,-2)]

        corrs['X_lin_gt'] = corrs['X_lin'] - corrs['X_lin_lt']
        corrs['Y_lin_gt'] = corrs['Y_lin'] - corrs['Y_lin_lt']
        
        corrs['U_lin'] = - xi_ln[(1,-1)]
        corrs['xi_lin'] = xi_ln[(0,0)]

        # one-loop terms
        QR_dict = self.get_QR(xi_ln, pk_lin)
        corrs_matter_1loop = self.get_corrs_matter_1loop(QR_dict)
        corrs_bias_2nd = self.get_corrs_bias_2nd(QR_dict)
        corrs.update(corrs_matter_1loop)
        corrs.update(corrs_bias_2nd)
        
        return corrs
    
    @partial(jit, static_argnames=['self'])
    def get_QR(self, xi_ln, pk_lin):

        integrand_Q1 = 8/15 * xi_ln[(0,0)]**2 - 16/21 * xi_ln[(2,0)]**2 + 8/35 * xi_ln[(4,0)]**2
        integrand_Q5 = 2/3 * xi_ln[(0,0)]**2 - 2/3 * xi_ln[(2,0)]**2 \
                    - 2/5 * xi_ln[(1,-1)] * xi_ln[(1,1)] + 2/5 * xi_ln[(3,-1)] * xi_ln[(3,1)]
        integrand_Q8 = 2/3 * xi_ln[(0,0)]**2 - 2/3 * xi_ln[(2,0)]**2
        integrand_Q_ex = xi_ln[(0,0)]**2 - xi_ln[(1,-1)] * xi_ln[(1,1)]

        Q1 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q1)
        Q5 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q5)
        Q8 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q8)
        Q_ex = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q_ex)

        coeffs = {(0,0,2): 8/15, (2,0,2): -16/21, (4,0,2): 8/35}
        R1 = 0.
        for (l,n,m) in coeffs.keys():
            pk_ln = self.get_pk_ln(l, -1, xi_ln[(l,n)])
            R1 = R1 + coeffs[(l,n,m)] * pk_ln * self._k**m * pk_lin

        coeffs = {(0,0,2): 2/3, (1,1,1): -2/5, (2,0,2): -2/3, (3,1,1): 2/5}
        R3 = 0.
        for (l,n,m) in coeffs.keys():
            pk_ln = self.get_pk_ln(l, -1, xi_ln[(l,n)])
            R3 = R3 + coeffs[(l,n,m)] * pk_ln * self._k**m * pk_lin

        Q2 = 2 * Q5 - Q1
        Qs2 = 2 * Q8 - 3 * Q1
        R2 = R3 - R1

        QR_dict = {'Q1': Q1, 'Q2': Q2, 'Q5': Q5, 'Q8': Q8, 
                   'Qs2': Qs2, 'Q_ex': Q_ex,
                   'R1': R1, 'R2': R2}
        
        return QR_dict

    @partial(jit, static_argnames=['self'])
    def get_corrs_matter_1loop(self, QR_dict):
        Q1 = QR_dict['Q1']
        Q2 = QR_dict['Q2']
        R1 = QR_dict['R1']
        R2 = QR_dict['R2']

        # X, Y for 1-loop A_{ij}
        xi_ln_22 = {(l, n): self.get_xi_ln(l, n, 9/98 * Q1) for (l, n) in [(0,-2), (2,-2)]}
        xi_ln_13 = {(l, n): self.get_xi_ln(l, n, 5/21 * Q2) for (l, n) in [(0,-2), (2,-2)]}

        X22 = 2/3 * (xi_ln_22[(0,-2)][0] - xi_ln_22[(0,-2)] - xi_ln_22[(2,-2)])
        Y22 = 2 * xi_ln_22[(2,-2)]

        X13 = 2/3 * (xi_ln_13[(0,-2)][0] - xi_ln_13[(0,-2)] - xi_ln_13[(2,-2)])
        Y13 = 2 * xi_ln_13[(2,-2)]

        # V1, V3, T for W_[ijk}
        T = self.get_xi_ln(3, -1, 3/14 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2))
        V1 = self.get_xi_ln(1, -1, -3/70 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2))
        V3 = self.get_xi_ln(1, -1, 3/70 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2))
        V1 = V1 - 1/5 * T
        V3 = V3 - 1/5 * T

        # make a dictionary
        corrs = {'X22': X22, 'Y22': Y22, 'X13': X13, 'Y13': Y13, 'V1': V1, 'V3': V3, 'T': T}
        return corrs
    
    @partial(jit, static_argnames=['self'])
    def get_corrs_bias_2nd(self, QR_dict, xi_ln):
        Q1 = QR_dict['Q1']
        Q5 = QR_dict['Q5']
        Q8 = QR_dict['Q8']
        R1 = QR_dict['R1']
        R2 = QR_dict['R2']

        U3 = self.get_xi_ln(1, -1, -5/21 * R1) # third-order part of U10
        U11 = self.get_xi_ln(1, -1, 3/14 * (R1 + R2)) # U11
        U20 = self.get_xi_ln(1, -1, -3/7 * Q8) # U20
        
        # A10
        xi_ln_A10 = {}
        xi_ln_A10_q0 = get_pk_int({'k': self._k, 'pk': 2/7 * R1}) # R2 integral is zero
        xi_ln_A10[(0,-2)] = self.get_xi_ln(0, -2, 2/7 * (Q5 + 2 * R2))
        xi_ln_A10[(2,-2)] = self.get_xi_ln(2, -2, 1/7 * (2 * Q5 + 3 * R1 + 4 * R2))
        X10 = xi_ln_A10_q0 - xi_ln_A10[(0,-2)] - xi_ln_A10[(2,-2)]
        Y10 = 3 * xi_ln_A10[(2,-2)]

        # corrlations from 2nd-order shear
        V10_G2 = self.get_xi_ln(1, -1, 3/7 * Q1) # V10 based on G2
        V10_s2 = self.get_xi_ln(1, -1, 3/7 * Q1 - 2/7 * Q8) # V10 based on s^2

        V12_G2 = self.get_xi_ln(1, -1, 2 * Q5) # V12 based on G2
        V12_s2 = 2 * (4/15 * xi_ln[(1,-1)] - 2/5 * xi_ln(3,-1)) * xi_ln[(2,0)] # V12 based on s^2

        zeta_G2 = 2 * (8/15 * xi_ln[(0,0)]**2 - 16/21 * xi_ln[(2,0)]**2 + 8/35 * xi_ln[(4,0)]**2) # zeta based on G2
        zeta_s2 = 2 * (4/45 * xi_ln[(0,0)]**2 + 8/63 * xi_ln[(2,0)]**2 + 8/35 * xi_ln[(4,0)]**2) # zeta based on s^2

        chi_G2 = 2 * (-2/3 * xi_ln[(0,0)]**2 + 2/3 * xi_ln[(2,0)]**2) # chi based on G2
        chi_s2 = 2 * (2/3 * xi_ln[(0,0)]**2 + xi_ln[(2,0)]**2) # chi based on s^2

        # make a dictionary
        corrs = {'U3': U3, 'U11': U11, 'U20': U20, 'X10': X10, 'Y10': Y10, 
                 'V10_G2': V10_G2, 'V10_s2': V10_s2, 'V12_G2': V12_G2, 'V12_s2': V12_s2,
                 'zeta_G2': zeta_G2, 'zeta_s2': zeta_s2, 'chi_G2': chi_G2, 'chi_s2': chi_s2
                 }
        return corrs

    @partial(jit, static_argnames=['self'])
    def get_corrs_bias_3rd(self, QR_dict):
        raise NotImplementedError