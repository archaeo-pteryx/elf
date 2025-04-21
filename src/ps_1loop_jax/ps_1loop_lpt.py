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

from .utils_loop import get_pk, get_pk_int, get_pk_int2
from .utils_math import legendre
from .utils_lpt import get_G00, get_G01, get_G02, get_G10, get_G20, get_G21, get_G30, get_G40


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
            self.hankel_pk2xi[l] = Hankel(l, nu, self._k, npad=(self._nfft//2), x_high=(self._kmax/10.), c_window_width=0.2)
            self.hankel_xi2pk[l] = Hankel(l, nu, self._q, npad=(self._nfft//2), x_high=1e4, c_window_width=0.2)
        
        # load the coefficients of G00
        loaded = jnp.load(os.path.dirname(__file__)+'/lpt_rsd_coeff/G00_coeffs.npz')
        self.G00_coeffs = {int(k): jnp.array(loaded[k]) for k in loaded.files}

        self.get_G = {(0,0): get_G00, (0,1): get_G01, (0,2): get_G02, (1,0): get_G10, \
                      (2,0): get_G20, (2,1): get_G21, (3,0): get_G30, (4,0): get_G40}

    @partial(jit, static_argnames=['self'])
    def get_xi_ln(self, pk_data):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)

        xi_ln = {}
        for (l, n) in self.ln_list:
            _, xi = self.hankel_pk2xi[l](pk_lin * self._k**(n+3))
            xi_ln[(l, n)] = xi / (2 * jnp.pi**2)
        
        return xi_ln
    
    @partial(jit, static_argnames=['self'])
    def get_xi_ln_lt(self, pk_data):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)

        xi_ln = {}
        for (l, n) in self.ln_list:
            _, xi = self.hankel_pk2xi[l](pk_lin * self._k**(n+3))
            xi_ln[(l, n)] = xi / (2 * jnp.pi**2)
        
        return xi_ln
    
    @partial(jit, static_argnames=['self'])
    def get_QR(self, xi_ln, pk_data):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)

        integrand_Q1 = 8/15 * xi_ln[(0,0)]**2 - 16/21 * xi_ln[(2,0)]**2 + 8/35 * xi_ln[(4,0)]**2
        _, Q1 = self.hankel_xi2pk[0](4 * jnp.pi * self._q**3 * integrand_Q1)

        integrand_R1 = 8/15 * xi_ln[(0,0)] - 16/21 * xi_ln[(2,0)] + 8/35 * xi_ln[(4,0)]
        _ R1 = self.hankel_xi2pk[0](4 * jnp.pi * self._q**3 * integrand_R1)
        R1 = R1 * self._k**2 * pk_lin

        # for (l, n1, n2) in self.ln1n2_list['Q1']:
        #     integrand = integrand + coeffs['Q1'][(l, n1, n2)] * xi_ln[(l, n1)] * xi_ln[(l, n2)]
        #     _, pk = self.hankel_xi2pk[0](integrand)
        #     QR_dict['Q1'] = pk

        # for (l, n, m) in self.lmn_list['R1']:
        #     integrand = integrand + coeffs['R1'][(l, n, m))] * xi_ln
        #     self.hankel_xi2pk[0](integrand)
    
        return Q1, R1
    
    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data):
        xi_ln = self.get_xi_ln(pk_data)
        xi_ln_lt = self.get_xi_ln_lt(pk_data)

        corrs = {}

        corrs['X_lin'] = 2./3 * (xi_ln[(0,-2)][0] - xi_ln[(0,-2)] - xi_ln[(2,-2)])
        corrs['Y_lin'] = 2 * xi_ln[(2,-2)]

        Xlin_lt = 2./3 * (xi_ln_lt[(0,-2)][0] - xi_ln_lt[(0,-2)] - xi_ln_lt[(2,-2)])
        Ylin_lt = 2 * xi_ln_lt[(2,-2)]

        corrs['X_lin_gt'] = corrs['X_lin'] - Xlin_lt
        corrs['Y_lin_gt'] = corrs['Y_lin'] - Ylin_lt
        
        corrs['U_lin'] = - xi_ln[(1,-1)]
        corrs['xi_lin'] = xi_ln[(0,0)]

        if self._compute_1loop:
            Q, R = self.get_QR(xi_ln)

            for name in 
            for (l, n) in coeffs[]
                integrand = coeff[] * 
                _, xi = self.hankel_pk2xi[l](integrand)

        return corrs
    
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
            integrand = base * (-2 / (k * self._q))**(l) * self.get_G[(0,0)](A, B, C, [self.G00_coeffs[l]])
            k_fft, pk_fft = self.hankel_xi2pk[l](integrand)
            pk = pk + interpax.interp1d(jnp.log(k), jnp.log(k_fft), pk_fft)

        return pk

    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict(self, k, mu, f, pk_data):
        corrs = self.get_corrs(pk_data)

        Xlin = corrs['X_lin']
        Ylin = corrs['Y_lin']
        Xlin_gt = corrs['Xlin_gt']
        Ylin_gt = corrs['Ylin_gt']
        X22 = corrs['X22']
        Y22 = corrs['Y22']
        X13 = corrs['X13']
        Y13 = corrs['Y13']

        Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
        Ksq = (k * Kfac)**2
        c = (1 + f * mu**2) / Kfac
        s = f * mu * jnp.sqrt(1 - mu**2) / Kfac
        A_mu = (1 + f) * mu / Kfac
        B_mu = jnp.sqrt(1 - mu**2) / Kfac

        A = k * self._q * c
        B = - 0.5 * Ksq * Ylin
        C = k * self._q * s

        base = 4 * jnp.pi * self._q**3 * jnp.exp(- 0.5 * Ksq * (Xlin + Ylin))

        integrands = {}

        for l in range(self.lmax + 1):

            G = {(m,n): self.get_G[(m,n)](A, B, C, [self.G00_coeffs[l-i] for i in range(m+n)]) for (m,n) in self.mn_list}

            integrands['zeldovich'] = G[(0,0)] - 0.5 * Ksq * (Xlin_gt * G[(0,0)] + Ylin_gt * G[(2,0)])
            
            integrands['A22'] = -0.5 * k**2 * ((Kfac**2 + 2 * f * (1 + f) * mu**2 + f**2 * mu**2) * G[(0,0)] * X22 \
                                                + (Kfac**2 * G[(2,0)] + 2 * f * Kfac * mu * mu_nq1 + f**2 * mu**2 * nq2) * Y22)
            
            integrands['A13'] = -0.5 * k**2 * (2 * (Kfac**2 + 2 * f * (1 + f) * mu**2) * G[(0,0)] * X13 \
                                               + 2 * (Kfac**2 * G[(2,0)] + 2 * f * Kfac * mu * mu_nq1) * Y13)
            
            integrands['A>A>'] = k**2 * Ksq**2 / 8 * (Xlin_gt**2 * G[(0,0)] + 2 * Xlin_gt * Ylin_gt * G[(2,0)] + Ylin_gt**2 * mu4)

            integrands['W'] = 

            k_fft, pk_fft = self.hankel_xi2pk[l](integrands)
            pk = pk + interpax.interp1d(jnp.log(k), jnp.log(k_fft), pk_fft)

        return pk
    
    @partial(jit, static_argnames=['self'])
    def get_QR(self, pk_data):
        xi_ln = self.get_xi_ln(pk_data)
        pk_lin = get_pk(k_fft, pk_data, kmin=kmin, kmax=kmax)

        integrand_Q1 = 8/15 * xi_ln[(0,0)]**2 - 16/21 * xi_ln[(2,0)]**2 + 8/35 * xi_ln[(4,0)]**2
        _, Q1 = self.hankel_xi2pk[0](4 * jnp.pi * q_fft**3 * integrand_Q1)

        coeffs = {(0,0,2): 8/15, (2,0,2): -16/21, (4,0,2): 8/35}
        R1 = 0.
        for (l,n,m) in coeffs.keys():
            _, pk_ln = self.hankel_xi2pk[l](q_fft**2 * xi_ln[(l,n)])
            R1 = R1 + coeffs[(l,n,m)] * pk_ln * k_fft**m * pk_lin

        integrand_Q2 = 4/5 * xi_ln[(0,0)]**2 - 4/7 * xi_ln[(2,0)]**2 - 8/35 * xi_ln[(4,0)]**2 \
                    - 4/5 * xi_ln[(1,-1)] * xi_ln[(1,1)] + 4/5 * xi_ln[(3,-1)] * xi_ln[(3,1)]
        _, Q2 = self.hankel_xi2pk[0](4 * jnp.pi * q_fft**3 * integrand_Q2)

        coeffs = {(0,0,2): -2/15, (1,-1,3): 2/5, (2,0,2): -2/21, (3,-1,3): -2/5, (4,0,2): 8/35}
        R2 = 0.
        for (l,n,m) in coeffs.keys():
            _, pk_ln = self.hankel_xi2pk[l](q_fft**2 * xi_ln[(l,n)])
            R2 = R2 + coeffs[(l,n,m)] * pk_ln * k_fft**m * pk_lin

        QR_dict = {'Q1': Q1, 'R1': R1, 'Q2': Q2, 'R2': R2}
        
        return QR_dict

    @partial(jit, static_argnames=['self'])
    def get_XY_1loop(self, pk_data):
        QR_dict = self.get_QR(pk_data)

        xi_ln_22 = {}
        for (l, n) in [(0,-2), (2,-2)]:
            _, xi = self.hankel_pk2xi[l](9/98 * QR_dict['Q1'] * k_fft**(n+3))
            xi_ln_22[(l, n)] = xi / (2 * jnp.pi**2)

        xi_ln_13 = {}
        for (l, n) in [(0,-2), (2,-2)]:
            _, xi = self.hankel_pk2xi[l](5/21 * QR_dict['R1'] * k_fft**(n+3))
            xi_ln_13[(l, n)] = xi / (2 * jnp.pi**2)

        X22 = 2./3 * (xi_ln_22[(0,-2)][0] - xi_ln_22[(0,-2)] - xi_ln_22[(2,-2)])
        Y22 = 2 * xi_ln_22[(2,-2)]

        X13 = 2./3 * (xi_ln_13[(0,-2)][0] - xi_ln_13[(0,-2)] - xi_ln_13[(2,-2)])
        Y13 = 2 * xi_ln_13[(2,-2)]

        return X22, Y22, X13, Y13

    @partial(jit, static_argnames=['self'])
    def get_VT(self, pk_data):
        QR_dict = self.get_QR(pk_data)
        Q1 = QR_dict['Q1']
        Q2 = QR_dict['Q2']
        R1 = QR_dict['R1']
        R2 = QR_dict['R2']
        
        _, T = self.hankel_pk2xi[3](3/14 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2))
        
        _, V1 = self.hankel_pk2xi[1](-3/70 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2))
        V1 = V1 - 1/5 * T

        _, V3 = self.hankel_pk2xi[1](3/70 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2))
        V3 = V3 - 1/5 * T

        VT_dict = {'V1': V1, 'V3': V3, 'T': T}
        return VT_dict