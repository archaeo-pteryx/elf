import os

import jax
# jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial
from jax import lax

import jax.numpy as jnp
import numpy as np
import interpax

from . import hankel

from .utils_loop import get_pk, get_pk_int
from .utils_lpt import get_G00s, get_Gs


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
        # load the coefficients of G00
        loaded = jnp.load(os.path.dirname(__file__)+'/lpt_rsd_coeff/G00_coeffs.npz')
        G00_coeffs = [jnp.array(loaded[k], dtype=jnp.float32) for k in loaded.files]
        L = self.lmax + 1
        coeffs_pad = jnp.zeros((L, L, L))  # axes: [l, k, i]
        for l in range(L):
            coeffs_pad = coeffs_pad.at[l, :l+1, :l+1].set(G00_coeffs[l])
        self.G00_coeffs = coeffs_pad

        # (l, n) for which xi_ln's are computed
        self.ln_list = jnp.array([[0,0], [0,-2], [0,2], [1,-1], [1,1], [2,0], [2,-2], [2,2], [3,-1], [3,1], [4,0]])
        lmax = max(jnp.max(self.ln_list[:, 0]), self.lmax)
        self._set_hankel(lmax)

    def _set_hankel(self, lmax):
        l_list = jnp.arange(lmax + 1)

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
        self._q_high = jnp.max(self._q) / 10
        
        c_window_width = 0.2
        self._w_m_k = hankel.c_window(jnp.arange(nfft_k//2+1), int(c_window_width * (nfft_k//2+1)))
        self._w_m_q = hankel.c_window(jnp.arange(nfft_q//2+1), int(c_window_width * (nfft_q//2+1)))
    
    @partial(jit, static_argnames=['self'])
    def get_xi_ln(self, l, n, array):
        fx = array * self._k**(n + 3) / (2 * jnp.pi**2)
        xi_ln = hankel.get_hankel(self._nu_hankel, fx, self._k_padded, self._y_k, self._u_m_k[l], self._npad, self._k_high, self._w_m_k)
        return xi_ln
    
    @partial(jit, static_argnames=['self'])
    def get_pk_ln(self, l, n, array):
        fx = array * self._q**(n + 3)
        pk_ln = hankel.get_hankel(self._nu_hankel, fx, self._q_padded, self._y_q, self._u_m_q[l], self._npad, self._q_high, self._w_m_q)
        return pk_ln
    
    @partial(jit, static_argnames=['self'])
    def get_xi_ln_array(self, array):
        def compute_ln(ln):
            return self.get_xi_ln(ln[0], ln[1], array)
        xis = jax.vmap(compute_ln)(self.ln_list)
        xi_ln = jnp.zeros((5, 5, len(self._q)))
        ls = self.ln_list[:, 0]
        ns = self.ln_list[:, 1]
        xi_ln = xi_ln.at[ls, ns].set(xis)
        return xi_ln
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_zel_k_mu(self, k, mu, X, Y, f):

        Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
        K    = k * Kfac
        Ksq  = K**2
        c    = (1 + f * mu**2) / Kfac
        s    = f * mu * jnp.sqrt(1 - mu**2) / Kfac

        q    = self._q                             # (nq,)
        A    = k * q * c                           # (nq,)
        B    = -0.5 * Ksq * Y                      # (nq,)
        C    = k * q * s                           # (nq,)

        base = 4 * jnp.pi * q**3 * jnp.exp(-0.5 * Ksq * (X + Y))  # (nq,)
        
        G00s = get_G00s(A, B, C, self.G00_coeffs, self.lmax) # (L, nq)
        L = self.lmax + 1

        inv = -2.0 / (k * q)                       # (nq,)
        ones = jnp.ones_like(inv)[None, :]
        reps = jnp.repeat(inv[None, :], L, axis=0)
        weights = jnp.cumprod(jnp.concatenate([ones, reps[:-1]], axis=0), axis=0)  # (L, nq)
        integrands = base[None, :] * weights * G00s

        logk = jnp.log(k)
        logk_fft = jnp.log(self._k)

        def per_l(l, g):  # g: (nk,)
            pk_fft = self.get_pk_ln(l, -3, g)          # (nk,)
            return jnp.interp(logk, logk_fft, pk_fft)  # (nk,)
        
        terms = jax.vmap(per_l, in_axes=(0, 0))(jnp.arange(L), integrands)  # (L, nk)

        return jnp.sum(terms, axis=0)
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_zel(self, k, mu, pk_data, f):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        
        xi_0m2 = self.get_xi_ln(0, -2, pk_lin)
        xi_2m2 = self.get_xi_ln(2, -2, pk_lin)
        X_lin = 2/3 * (xi_0m2[0] - xi_0m2 - xi_2m2)
        Y_lin = 2 * xi_2m2

        def per_mu(mu_j):
            return jax.vmap(lambda k_i: self.get_pkmu_zel_k_mu(k_i, mu_j, X_lin, Y_lin, f))(k)  # (nk,)
        pkmu = jax.vmap(per_mu)(mu)      # (nmu, nk)
        pkmu = jnp.transpose(pkmu, (1, 0))   # (nk, nmu)

        return pkmu

    @partial(jit, static_argnames=('self',))
    def get_pkmu_terms_k_mu(self, k, mu, corrs, f):

        corrs_tree, corrs_matter_1loop, corrs_bias = corrs

        # matter tree-level
        X_lin_lt, Y_lin_lt = corrs_tree[2], corrs_tree[3]
        X_lin_gt, Y_lin_gt = corrs_tree[4], corrs_tree[5]

        # matter one-loop
        X22, Y22, X13, Y13, V1, V3, T = corrs_matter_1loop

        # LIMD bias
        xi_lin, U_lin = corrs_tree[6], corrs_tree[7]
        U3, U11, U20, X10, Y10 = corrs_bias[0:5]

        # 2nd-order shear bias & 3rd-order bias
        V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta = corrs_bias[5:]

        Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
        K    = k * Kfac
        Ksq  = K**2
        c    = (1 + f * mu**2) / Kfac
        s    = f * mu * jnp.sqrt(1 - mu**2) / Kfac
        A_mu = (1 + f) * mu / Kfac
        B_mu = jnp.sqrt(1 - mu**2) / Kfac

        q    = self._q                             # (nq,)
        A    = k * q * c                           # (nq,)
        B    = -0.5 * Ksq * Y_lin_lt               # (nq,)
        C    = k * q * s                           # (nq,)

        base = 4 * jnp.pi * q**3 * jnp.exp(-0.5 * Ksq * (X_lin_lt + Y_lin_lt))  # (nq,)

        L = self.lmax + 1
        Gs = get_Gs(A, B, C, self.G00_coeffs, self.lmax)

        mq0 =  Gs[0,0]          # (L, nq)
        mq1 = -Gs[1,0]
        mq2 = -Gs[2,0]
        mq3 =  Gs[3,0]
        mq4 =  Gs[4,0]
        nq1 = -A_mu * Gs[1,0] + B_mu * Gs[0,1]
        nq2 = -A_mu**2 * Gs[2,0] + 2*A_mu*B_mu*Gs[1,1] - B_mu**2 * Gs[0,2]
        mq1_nq1 = -A_mu * Gs[2,0] + B_mu * Gs[1,1]
        mq2_nq1 =  A_mu * Gs[3,0] - B_mu * Gs[2,1]

        # --- 13 integrands of (L, nq)---
        # matter
        integrand_1  = mq0 - 0.5 * Ksq * (mq0*X_lin_gt + mq2*Y_lin_gt)
        integrand_1 += (Ksq**2) / 8.0 * (mq0*X_lin_gt**2 + 2*mq2*X_lin_gt*Y_lin_gt + mq4*Y_lin_gt**2)
        integrand_1 += -0.5*k**2 * ((Kfac**2 + 2*f*(1+f)*mu**2 + f**2*mu**2) * mq0 * X22
                                    + (Kfac**2 * mq2 + 2*f*Kfac*mu*mq1_nq1 + f**2*mu**2*nq2) * Y22)
        integrand_1 += -0.5*k**2 * (2*(Kfac**2 + 2*f*(1+f)*mu**2) * mq0 * X13
                                    + 2*(Kfac**2 * mq2 + 2*f*Kfac*mu*mq1_nq1) * Y13)
        integrand_1 += -k**3 * ( 2*Kfac*(Kfac**2 + f*(1+f)*mu**2) * mq1 * V1
                                + Kfac**2 * (Kfac*mq1 + f*mu*nq1) * V3
                                + Kfac**2 * (Kfac*mq3 + f*mu*mq2_nq1) * T )

        # LIMD bias
        integrand_b1  = -2 * (K * mq1 * (U_lin + U3) + (2*f*k*mu*nq1) * U3)
        integrand_b1 +=  Ksq * (mq0*X_lin_gt + mq2*Y_lin_gt) * (K * U_lin)
        integrand_b1 += -Ksq * (mq0*X10 + mq2*Y10) - f*(1+f)*(k*mu)**2 * mq0 * X10 - f*k*mu*mq1_nq1*Y10

        integrand_b1_b1  =  mq0 * xi_lin
        integrand_b1_b1 += -0.5*Ksq * (mq0*X_lin_gt + mq2*Y_lin_gt) * xi_lin
        integrand_b1_b1 += -(K*mq1 + f*k*mu*nq1) * U11
        integrand_b1_b1 += -Ksq * mq2 * U_lin**2

        integrand_b2  = -(K*mq1 + f*k*mu*nq1) * U20
        integrand_b2 += -Ksq * mq2 * U_lin**2

        integrand_b1_b2 = -2 * K * mq1 * xi_lin * U_lin
        integrand_b2_b2 = 0.5 * mq0 * xi_lin**2

        # 2nd-order shear bias
        integrand_bs  = -2 * (K*mq1 + f*k*mu*nq1) * V10
        integrand_bs += -Ksq * (mq0*X_Upsilon + mq2*Y_Upsilon)

        integrand_b1_bs = -2 * K * mq1 * V12
        integrand_b2_bs =  mq0 * chi
        integrand_bs_bs =  mq0 * zeta

        # 3rd-order
        integrand_b3    = -2 * K * mq1 * Ub3
        integrand_b1_b3 = 2 * mq0 * theta

        # counterterm
        integrand_ctr = lax.select(
            self.use_Pzel,
            mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt),
            mq0 * xi_lin
        )

        integrands = jnp.stack([
            integrand_1, integrand_b1, integrand_b1_b1,
            integrand_b2, integrand_b1_b2, integrand_b2_b2,
            integrand_bs, integrand_b1_bs, integrand_b2_bs, integrand_bs_bs,
            integrand_b3, integrand_b1_b3, integrand_ctr
        ], axis=0)   # (ncomp, L, nq)

        inv = -2.0 / (k * q)                     # (nq,)
        ones = jnp.ones_like(inv)[None, :]
        reps = jnp.repeat(inv[None, :], L, axis=0)
        weights = jnp.cumprod(jnp.concatenate([ones, reps[:-1]], axis=0), axis=0)  # (L, nq)

        logk = jnp.log(k)
        logk_fft = jnp.log(self._k)

        g = integrands * (base[None, None, :] * weights[None, :, :]) # (ncomp, L, nq)

        # l を回す vmap（g_l: (nq,) → scalar）
        def per_l(l, g_l):
            pk_fft = self.get_pk_ln(l, -3, g_l)          # (n_fft,)
            return jnp.interp(logk, logk_fft, pk_fft)

        # 成分ごとに「全Lの和」を出す
        def per_comp(g_comp):                             # g_comp: (L, nq)
            terms_L = jax.vmap(per_l, in_axes=(0, 0))(jnp.arange(L), g_comp)  # (L,)
            return jnp.sum(terms_L, axis=0)                                   # ()

        pkmu_terms = jax.vmap(per_comp, in_axes=0)(g)

        return pkmu_terms  # (ncomp,)

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, k_IR=0.2):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        f = params.f
        b1, b2, bs, b3 = params.bias      # (4,)
        alpha0, alpha2, alpha4, alpha6 = params.ctr  # (4,)

        corrs = self.get_corrs(pk_data, k_IR)

        def per_k(k_i):
            return jax.vmap(lambda mu_j: self.get_pkmu_terms_k_mu(k_i, mu_j, corrs, f))(mu)  # (nmu, ncomp)

        pkmu_terms_kmu = jax.vmap(per_k)(k)                 # (nk, nmu, ncomp)
        pkmu_terms = jnp.transpose(pkmu_terms_kmu, (2, 0, 1))  # (ncomp, nk, nmu)

        bias_facs = jnp.array([
            1.,            # 1
            b1,            # b1
            b1**2,         # b1^2
            b2,            # b2
            b1*b2,         # b1*b2
            b2**2,         # b2^2
            bs,            # bs
            b1*bs,         # b1*bs
            b2*bs,         # b2*bs
            bs**2,         # bs^2
            b3,            # b3
            b1*b3          # b1*b3
        ])  # (12,)

        pkmu = jnp.tensordot(bias_facs, pkmu_terms[:-1], axes=(0, 0))   # (nk, nmu)

        # counterterm
        ctr_mu = alpha0 + alpha2 * mu**2 + alpha4 * mu**4 + alpha6 * mu**6   # (nmu,)
        pkmu_ctr = jnp.outer(k**2, ctr_mu) * pkmu_terms[-1]   # (nk, nmu)
        pkmu = pkmu + pkmu_ctr

        return pkmu  # (nk, nmu)
    
    @partial(jit, static_argnames=['self'])
    def get_pk_ells(self, k, pk_data, params, k_IR=0.2):
        k = jnp.atleast_1d(k)

        pkmu = self.get_pkmu(k, self._mu, pk_data, params, k_IR).T
        pkmu = jnp.concatenate([jnp.flip(pkmu, axis=0), pkmu], axis=0)

        pk0 = 0.5 * jnp.sum((self._ws * self._leg0)[:, None] * pkmu, axis=0)
        pk2 = 2.5 * jnp.sum((self._ws * self._leg2)[:, None] * pkmu, axis=0)
        pk4 = 4.5 * jnp.sum((self._ws * self._leg4)[:, None] * pkmu, axis=0)
        
        return pk0, pk2, pk4
    
    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data, k_IR=0.2):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_lin_lt = pk_lin * jnp.exp(-0.5 * (self._k / k_IR)**2)

        # generalized correlation functions
        xi_ln = self.get_xi_ln_array(pk_lin)
        xi_ln_lt = self.get_xi_ln_array(pk_lin_lt)
        
        # tree-level terms
        corrs_tree = self.get_corrs_tree(xi_ln, xi_ln_lt)

        # one-loop terms
        Qs = self.get_Qs(xi_ln)
        Rs = self.get_Rs(xi_ln, pk_lin)
        corrs_matter_1loop = self.get_corrs_matter_1loop(Qs, Rs)
        corrs_bias = self.get_corrs_bias(Qs, Rs, xi_ln)

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
                           X_lin_gt, Y_lin_gt, xi_lin, U_lin], axis=0)
        return corrs
    
    @partial(jit, static_argnames=['self'])
    def get_Qs(self, xi_ln):

        integrand_Q1 = 8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2
        integrand_Q5 = 2/3 * xi_ln[0,0]**2 - 2/3 * xi_ln[2,0]**2 \
                    - 2/5 * xi_ln[1,-1] * xi_ln[1,1] + 2/5 * xi_ln[3,-1] * xi_ln[3,1]
        integrand_Q8 = 2/3 * xi_ln[0,0]**2 - 2/3 * xi_ln[2,0]**2
        integrand_Q_Up_G2 = 2/5 * xi_ln[1,-1]**2 - 2/5 * xi_ln[3,-1]**2

        Q1 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q1)
        Q5 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q5)
        Q8 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q8)
        Q_Up_G2 = self.get_pk_ln(0, 0, 4 * jnp.pi * integrand_Q_Up_G2)

        Q2 = 2 * Q5 - Q1

        Qs = jnp.stack([Q1, Q2, Q5, Q8, Q_Up_G2], axis=0)
        return Qs

    @partial(jit, static_argnames=['self'])
    def get_Rs(self, xi_ln, pk_lin):
        pk_00 = self.get_pk_ln(0, -1, xi_ln[0,0]) * pk_lin
        pk_20 = self.get_pk_ln(2, -1, xi_ln[2,0]) * pk_lin
        pk_40 = self.get_pk_ln(4, -1, xi_ln[4,0]) * pk_lin
        pk_11 = self.get_pk_ln(1, -1, xi_ln[1,1]) * pk_lin
        pk_31 = self.get_pk_ln(3, -1, xi_ln[3,1]) * pk_lin
        pk_22 = self.get_pk_ln(2, -1, xi_ln[2,2]) * pk_lin
        pk_1m1 = self.get_pk_ln(1, -1, xi_ln[1,-1]) * pk_lin
        pk_3m1 = self.get_pk_ln(3, -1, xi_ln[3,-1]) * pk_lin

        R1 = self._k**2 * (8/15 * pk_00 - 16/21 * pk_20 + 8/35 * pk_40)
        R3 = 2/3 * self._k**2 * (pk_00 - pk_20) - 2/5 * self._k * (pk_11 - pk_31)
        R2 = R3 - R1

        F_G2 = self._k**2 * (- 72/35 * pk_00 + 88/49 * pk_20 + 64/245 * pk_40) \
            + self._k * 4/5 * (pk_11 - pk_31) + self._k**3 * 4/5 * (pk_1m1 - pk_3m1)
        Rb3 = self._k**2 * (32/105 * pk_00 - 80/441 * pk_20 + 32/245 * pk_40) \
            - self._k * (64/315 * pk_11 + 32/105 * pk_31) + 16/63 * pk_22

        Rs = jnp.stack([R1, R2, F_G2, Rb3], axis=0)
        return Rs
    
    # @partial(jit, static_argnames=['self'])
    # def get_Rs(self, xi_ln, pk_lin):
    #     ells = jnp.array([0, 2, 4, 1, 3, 2, 1, 3], dtype=jnp.int32)
    #     xis = jnp.stack([
    #         xi_ln[0,  0],
    #         xi_ln[2,  0],
    #         xi_ln[4,  0],
    #         xi_ln[1,  1],
    #         xi_ln[3,  1],
    #         xi_ln[2,  2],
    #         xi_ln[1, -1],
    #         xi_ln[3, -1],
    #     ], axis=0)

    #     def one(l, g):
    #         return self.get_pk_ln(l, -1, g) * pk_lin

    #     pk_list = jax.vmap(one, in_axes=(0, 0))(ells, xis)  # (8, nk)
    #     pk_00, pk_20, pk_40, pk_11, pk_31, pk_22, pk_1m1, pk_3m1 = pk_list

    #     k = self._k
    #     R1 = k**2 * (8/15 * pk_00 - 16/21 * pk_20 + 8/35 * pk_40)
    #     R3 = 2/3 * k**2 * (pk_00 - pk_20) - 2/5 * k * (pk_11 - pk_31)
    #     R2 = R3 - R1

    #     F_G2 = k**2 * (-72/35 * pk_00 + 88/49 * pk_20 + 64/245 * pk_40) \
    #         + k * 4/5 * (pk_11 - pk_31) + k**3 * 4/5 * (pk_1m1 - pk_3m1)
    #     Rb3  = k**2 * (32/105 * pk_00 - 80/441 * pk_20 + 32/245 * pk_40) \
    #         - k * (64/315 * pk_11 + 32/105 * pk_31) + 16/63 * pk_22

    #     return jnp.stack([R1, R2, F_G2, Rb3], axis=0)
    
    @partial(jit, static_argnames=['self'])
    def get_corrs_matter_1loop(self, Qs, Rs):
        Q1 = Qs[0]
        Q2 = Qs[1]
        R1 = Rs[0]
        R2 = Rs[1]

        # X, Y for 1-loop A_{ij}
        xi_ln_22_0m2 = self.get_xi_ln(0, -2, 9/98 * Q1)
        xi_ln_22_2m2 = self.get_xi_ln(2, -2, 9/98 * Q1)
        xi_ln_13_0m2 = self.get_xi_ln(0, -2, 5/21 * Q2)
        xi_ln_13_2m2 = self.get_xi_ln(2, -2, 5/21 * Q2)

        X22 = 2/3 * (xi_ln_22_0m2[0] - xi_ln_22_0m2 - xi_ln_22_2m2)
        Y22 = 2 * xi_ln_22_2m2

        X13 = 2/3 * (xi_ln_13_0m2[0] - xi_ln_13_0m2 - xi_ln_13_2m2)
        Y13 = 2 * xi_ln_13_2m2

        # V1, V3, T for W_{ijk}
        T = self.get_xi_ln(3, -1, 3/14 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2))
        V1 = self.get_xi_ln(1, -1, -3/70 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2))
        V3 = self.get_xi_ln(1, -1, 3/70 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2))
        V1 = V1 - 1/5 * T
        V3 = V3 - 1/5 * T

        corrs = jnp.stack([X22, Y22, X13, Y13, V1, V3, T], axis=0)
        return corrs
    
    @partial(jit, static_argnames=['self'])
    def get_corrs_bias(self, Qs, Rs, xi_ln):
        Q1 = Qs[0]
        Q2 = Qs[1]
        Q5 = Qs[2]
        Q8 = Qs[3]
        Q_Up_G2 = Qs[4]
        R1 = Rs[0]
        R2 = Rs[1]
        F_G2 = Rs[2]
        Rb3 = Rs[3]
        
        # U10, U11, U20
        U3 = self.get_xi_ln(1, -1, -5/21 * R1) # 3rd-order part of U10
        U11 = self.get_xi_ln(1, -1, 3/14 * (R1 + R2)) # U11
        U20 = self.get_xi_ln(1, -1, -3/7 * Q8) # U20
        
        # A10
        pk_data = jnp.stack([self._k, 2/7 * R1], axis=0)
        xi_ln_A10_q0 = get_pk_int(pk_data) # R2 integral is zero
        xi_ln_A10_0m2 = self.get_xi_ln(0, -2, 2/7 * (Q5 + 2 * R2))
        xi_ln_A10_2m2 = self.get_xi_ln(2, -2, 1/7 * (2 * Q5 + 3 * R1 + 4 * R2))
        X10 = xi_ln_A10_q0 - xi_ln_A10_0m2 - xi_ln_A10_2m2
        Y10 = 3 * xi_ln_A10_2m2

        if self.use_galileon:
            V10 = self.get_xi_ln(1, -1, 3/7 * Q1) # V10 based on G2
            V12 = self.get_xi_ln(1, -1, 2 * Q5) # V12 based on G2
            chi = 2 * (-2/3 * xi_ln[0,0]**2 + 2/3 * xi_ln[2,0]**2) # chi based on G2
            zeta = 2 * (8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2) # zeta based on G2

            # Upsilon based on G2
            Up1 = self.get_xi_ln(0, 0, Q_Up_G2)
            Up2 = self.get_xi_ln(0, -2, Q1 - Q2) + self.get_xi_ln(2, -2, Q1 + 2 * Q2)
            X_Upsilon = 1/3 * Up1 - Up2
            Y_Upsilon = Up2 - Up1

            # 3rd-order bias
            Ub3 = self.get_xi_ln(1, -1, -6/5 * F_G2) # Ub3 based on Gamma3
            theta = self.get_xi_ln(0, 0, 6/5 * F_G2) # theta based on Gamma3
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
            Ub3 = self.get_xi_ln(1, -1, - Rb3)
            theta = self.get_xi_ln(0, 0, Rb3)

        corrs = jnp.stack([U3, U11, U20, X10, Y10, V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta], axis=0)
        return corrs
