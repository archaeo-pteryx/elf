import os

import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp
import interpax

from .hankel import Hankel

from .utils_loop import get_pk, get_pk_int
from .utils_lpt import get_G00, get_Gs


class PowerSpectrum1LoopLPT:

    def __init__(self, 
                 cross=False,
                 subtract_k0_limit=True,
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=256,
                 use_galileon=True,
                 ):
        
        self.cross = cross # flag to enable the calculation of cross power spectra
        self.subtract_k0_limit = subtract_k0_limit # flag to subtract k -> 0 limit from 2-2 terms
        self.use_galileon = use_galileon

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

    def get_xi_ln(self, l, n, array):
        _, xi_ln = self.hankel_pk2xi[l](array * self._k**(n+3) / (2 * jnp.pi**2))
        return xi_ln
    
    def get_pk_ln(self, l, n, array):
        _, pk_ln = self.hankel_xi2pk[l](array * self._q**(n+3))
        return pk_ln
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_zel(self, k, mu, pk_data, f):
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
    def get_pkmu_dict(self, k, mu, pk_data, f):
        corrs = self.get_corrs(pk_data)

        X_lin_lt = corrs['X_lin_lt']
        Y_lin_lt = corrs['Y_lin_lt']
        X_lin_gt = corrs['X_lin_gt']
        Y_lin_gt = corrs['Y_lin_gt']
        
        # matter one-loop terms
        X22 = corrs['X22']
        Y22 = corrs['Y22']
        X13 = corrs['X13']
        Y13 = corrs['Y13']
        V1 = corrs['V1']
        V3 = corrs['V3']
        T = corrs['T']

        # LIMD bias terms
        xi_lin = corrs['xi_lin']
        U_lin = corrs['U_lin']
        U3 = corrs['U3']
        U11 = corrs['U11']
        U20 = corrs['U20']
        X10 = corrs['X10']
        Y10 = corrs['Y10']

        # 2nd-order shear bias terms
        V10 = corrs['V10']
        V12 = corrs['V12']
        X_Upsilon = corrs['X_Upsilon']
        Y_Upsilon = corrs['Y_Upsilon']
        chi = corrs['chi']
        zeta = corrs['zeta']

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
        Gs = get_Gs(A, B, C, lmax=self.lmax)

        pkmu = {name: 0 for name in ['za', 'A>', 'A>A>', 'A22', 'A13', 'W112']}
        integrand = {}

        for l in range(self.lmax + 1):

            mq0 = Gs[(0,0)][l]
            mq1 = Gs[(1,0)][l]
            mq2 = Gs[(2,0)][l]
            mq3 = Gs[(3,0)][l]
            mq4 = Gs[(4,0)][l]
            nq1 = A_mu * Gs[(1,0)][l] + B_mu * Gs[(0,1)][l]
            nq2 = A_mu**2 * Gs[(2,0)][l] + 2 * A_mu * B_mu * Gs[(1,1)][l] + B_mu**2 * Gs[(0,2)][l]
            mq1_nq1 = A_mu * Gs[(2,0)][l] + B_mu * Gs[(1,1)][l]
            mq2_nq1 = A_mu * Gs[(3,0)][l] + B_mu * Gs[(2,1)][l]

            # matter terms
            integrand['za'] = mq0
            integrand['A>'] = - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)
            integrand['A> A>'] = Ksq**2 / 8 * (mq0 * X_lin_gt**2 + 2 * mq2 * X_lin_gt * Y_lin_gt + mq4 * Y_lin_gt**2)
    
            integrand['A22'] = -0.5 * k**2 * ((Kfac**2 + 2*f*(1+f)*mu**2 + f**2*mu**2) * mq0 * X22 + \
                                    (Kfac**2 * mq2 + 2 * f * Kfac * mu * mq1_nq1 + f**2 * mu**2 * nq2) * Y22)

            integrand['A13'] = -0.5 * k**2 * (2 * (Kfac**2 + 2*f*(1+f)* mu**2) * mq0 * X13 + \
                                    2 * (Kfac**2 * mq2 + 2 * f * Kfac * mu * mq1_nq1) * Y13)

            integrand['W112'] = 0.5 * k**3 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu**2) * mq1 * V1 +  \
                                    Kfac**2 * (Kfac * mq1 + f * mu * nq1) * V3 + \
                                    Kfac**2 * (Kfac * mq3 + f * mu * mq2_nq1) * T)
            integrand['W112'] = -2 * integrand['W112']

            # LIMD bias terms
            integrand['U10'] = K * mq1 * U_lin + (K * mq1 + 2 * f * k * mu * nq1) * U3

            integrand['A> U_lin'] = - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)

            integrand['xi_lin'] = mq0 * xi_lin

            integrand['A> xi_lin'] = - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * xi_lin

            integrand['U11'] = (K * mq1 + f * k * mu * nq1) * U11

            integrand['U20'] = (K * mq1 + f * k * mu * nq1) * U20

            integrand['A10'] = Ksq * (mq0 * X10 + mq2 * Y10) + f * k**2 * mu * mu**2 * (1+f) * mq0 * X10 \
                                + f * k * mu * mq1_nq1 * Y10

            integrand['U_lin U_lin'] = Ksq * mq2 * U_lin**2

            integrand['xi_lin xi_lin'] = mq0 * xi_lin**2

            integrand['xi_lin U_lin'] = K * mq1 * xi_lin * U_lin

            # 2nd-order shear bias terms
            integrand['V10'] = (K * mq1 + f * k * mu * nq1) * V10

            integrand['V12'] = K * mq1 * V12

            integrand['Upsilon'] = Ksq * (mq0 * X_Upsilon + mq2 * Y_Upsilon)

            integrand['chi'] = mq0 * chi

            integrand['zeta'] = mq0 * zeta

            for name in ['za', 'A>', 'A> A>', 'A22', 'A13', 'W112']:
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
        corrs_bias_2nd = self.get_corrs_bias_2nd(QR_dict, xi_ln)
        corrs.update(corrs_matter_1loop)
        corrs.update(corrs_bias_2nd)
        
        return corrs
    
    @partial(jit, static_argnames=['self'])
    def get_QR(self, xi_ln, pk_lin):

        integrand_Q1 = 8/15 * xi_ln[(0,0)]**2 - 16/21 * xi_ln[(2,0)]**2 + 8/35 * xi_ln[(4,0)]**2
        integrand_Q5 = 2/3 * xi_ln[(0,0)]**2 - 2/3 * xi_ln[(2,0)]**2 \
                    - 2/5 * xi_ln[(1,-1)] * xi_ln[(1,1)] + 2/5 * xi_ln[(3,-1)] * xi_ln[(3,1)]
        integrand_Q8 = 2/3 * xi_ln[(0,0)]**2 - 2/3 * xi_ln[(2,0)]**2

        Q1 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q1)
        Q5 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q5)
        Q8 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q8)

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
        R2 = R3 - R1

        QR_dict = {'Q1': Q1, 'Q2': Q2, 'Q5': Q5, 'Q8': Q8, 
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

        U3 = self.get_xi_ln(1, -1, -5/21 * R1) # 3rd-order part of U10
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
        if self.use_galileon:
            V10 = self.get_xi_ln(1, -1, 3/7 * Q1) # V10 based on G2
            V12 = self.get_xi_ln(1, -1, 2 * Q5) # V12 based on G2
            chi = 2 * (-2/3 * xi_ln[(0,0)]**2 + 2/3 * xi_ln[(2,0)]**2) # chi based on G2
            zeta = 2 * (8/15 * xi_ln[(0,0)]**2 - 16/21 * xi_ln[(2,0)]**2 + 8/35 * xi_ln[(4,0)]**2) # zeta based on G2
        else:
            V10 = self.get_xi_ln(1, -1, 3/7 * Q1 - 2/7 * Q8) # V10 based on s^2
            V12 = 2 * (4/15 * xi_ln[(1,-1)] - 2/5 * xi_ln(3,-1)) * xi_ln[(2,0)] # V12 based on s^2
            chi = 2 * (2/3 * xi_ln[(0,0)]**2 + xi_ln[(2,0)]**2) # chi based on s^2
            zeta = 2 * (4/45 * xi_ln[(0,0)]**2 + 8/63 * xi_ln[(2,0)]**2 + 8/35 * xi_ln[(4,0)]**2) # zeta based on s^2
        
        # Upsilon based on s^2
        J2 = 2/15 * xi_ln[(1,-1)] - 0.2 * xi_ln[(3,-1)]
        J3 = -0.2 * xi_ln[(1,-1)] - 0.2 * xi_ln[(3,-1)]
        J4 = xi_ln[(3,-1)]
        
        V = 4 * J2 * xi_ln[(2,0)]
        X_Upsilon = 4 * J3**2
        Y_Upsilon = 6 * J2**2 + 8 * J2 * J3 + 4 * J2 * J4 + 4 * J3**2 + 8 * J3 * J4 + 2 * J4**2

        # make a dictionary
        corrs = {'U3': U3, 'U11': U11, 'U20': U20, 'X10': X10, 'Y10': Y10, 
                 'V10': V10, 'V12': V12, 'chi': chi, 'zeta': zeta, 
                 'X_Upsilon': X_Upsilon, 'Y_Upsilon': Y_Upsilon
                 }
        return corrs
