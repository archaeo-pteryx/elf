import os

import jax
from jax import jit
from functools import partial
from jax import lax

import jax.numpy as jnp

from . import hankel

from .utils_loop import get_pk, get_pk_int
from .utils_lpt import get_G00s, get_Gs
from .multipole import prepare_mu_gauleg, get_legendre_multipoles, get_k_mu_true_for_ap


class PowerSpectrum1LoopLPT:

    def __init__(self, 
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=256,
                 lmax=5,
                 ngauss=4,
                 use_galileon=False,
                 use_Pzel=True
                 ):
        
        self.lmax = lmax
        self.use_galileon = use_galileon
        self.use_Pzel = use_Pzel

        # preparation for Gauss-Legendre quadrature
        self._mu_quad, self._legendre_weights = prepare_mu_gauleg(ngauss)

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

        self._nu_hankel = 1.1
        self._q = 1 / self._k[::-1]

        self._npad = self._nfft // 2
        self._npad = int(jnp.ceil(self._nfft / 3))
        self._k_padded = hankel.get_log_extrap(self._k, self._npad, self._npad)
        self._q_padded = 1 / self._k_padded[::-1]

        n_window = self._nfft // 4
        self._w_m = hankel.get_window(n_window, self._k_padded)

        nfft = len(self._k_padded)
        dln = jnp.log(self._k_padded[1] / self._k_padded[0])
        eta_m = 2 * jnp.pi * jnp.fft.rfftfreq(nfft, d=1.0) / dln
        
        g_l = jnp.array([hankel.get_g_l(l, self._nu_hankel + 1j * eta_m) for l in l_list])
        self._u_m = jnp.array([(self._k_padded[0] * self._q_padded[0])**(-1j * eta_m) * g_l[l] for l in l_list])

        self._k_high = self._kmax
        self._q_high = jnp.max(self._q)
        
        c_window_width = 0.25
        self._w_m_freq = hankel.c_window(jnp.arange(nfft//2+1), int(c_window_width * (nfft//2+1)))

    def get_xi_ln(self, l, n, array):
        fx = array * self._k**(n + 3) / (2 * jnp.pi**2)
        xi_ln = hankel.get_hankel(self._nu_hankel, fx, self._k_padded, self._q_padded, self._u_m[l], self._npad, self._k_high, self._w_m, self._w_m_freq)
        return xi_ln
    
    def get_pk_ln(self, l, n, array):
        fx = array * self._q**(n + 3)
        pk_ln = hankel.get_hankel(self._nu_hankel, fx, self._q_padded, self._k_padded, self._u_m[l], self._npad, self._q_high, self._w_m, self._w_m_freq)
        return pk_ln
    
    def get_pk_batched(self, arrays, u_m):
        pks = hankel.get_hankel_batched(self._nu_hankel, arrays, self._q_padded, self._k_padded, u_m, self._npad, self._q_high, self._w_m, self._w_m_freq)
        return pks
    
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
    def get_pkmu_zel(self, k, mu, pk_data, f):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)
        q  = self._q

        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_int = get_pk_int(pk_data)

        xi_0m2 = self.get_xi_ln(0, -2, pk_lin)
        xi_2m2 = self.get_xi_ln(2, -2, pk_lin)
        # X_lin  = 2/3 * (xi_0m2[0] - xi_0m2 - xi_2m2)   # (nq,)
        X_lin  = 2/3 * (pk_int - xi_0m2 - xi_2m2)      # (nq,)
        Y_lin  = 2 * xi_2m2                            # (nq,)

        L      = self.lmax + 1
        logk_fft = jnp.log(self._k)

        def per_k(k_i):
            inv     = -2.0 / (k_i * q)                                # (nq,)
            weights = jnp.power(inv[None, :], jnp.arange(L)[:, None]) # (L, nq)

            logk   = jnp.log(k_i)
            # preparation for linear interpolation
            i0     = jnp.searchsorted(logk_fft, logk, side='right') - 1
            i0     = jnp.clip(i0, 0, logk_fft.size - 2)
            t      = (logk - logk_fft[i0]) / (logk_fft[i0+1] - logk_fft[i0])

            def per_mu(mu_j):
                Kfac = jnp.sqrt(1 + f * (2 + f) * mu_j**2)
                K    = k_i * Kfac
                Ksq  = K**2
                c    = (1 + f * mu_j**2) / Kfac
                s    = f * mu_j * jnp.sqrt(1 - mu_j**2) / Kfac

                A = k_i * q * c                         # (nq,)
                B = -0.5 * Ksq * Y_lin                  # (nq,)
                C = k_i * q * s                         # (nq,)

                base = jnp.exp(-0.5 * Ksq * (X_lin + Y_lin))  # (nq,)
                G00s = get_G00s(B, s**2, self.G00_coeffs, self.lmax)  # (L, nq)

                g = weights * G00s * base[None, :]    # (L, nq)
                g = g.at[0,:].subtract(g[0,-1])
                g = (4 * jnp.pi * q**3)[None, :] * g

                pk_ffts = self.get_pk_batched(g, self._u_m[:L]) # (L, nfft)
                vals = (1.0 - t) * pk_ffts[:, i0] + t * pk_ffts[:, i0+1] # linear interpolation

                return jnp.sum(vals)  # P(k_i, μ_j)
            
            return jax.vmap(per_mu)(mu)  # (nmu,)
        
        pkmu = jax.vmap(per_k)(k)  # (nk, nmu)
        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_components(self, k, mu, pk_data, f, k_IR=0.2):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)
        q  = self._q

        corrs = self.get_corrs(pk_data, k_IR)
        corrs_tree, corrs_matter_1loop, corrs_bias = corrs[:8], corrs[8:15], corrs[15:]

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

        L        = self.lmax + 1
        logk_fft = jnp.log(self._k)

        def per_k(k_i):
            inv     = -2.0 / (k_i * q)                                # (nq,)
            weights = jnp.power(inv[None, :], jnp.arange(L)[:, None]) # (L, nq)

            logk = jnp.log(k_i)
            # preparation for linear interpolation
            i0   = jnp.searchsorted(logk_fft, logk, side='right') - 1
            i0   = jnp.clip(i0, 0, logk_fft.size - 2)
            t    = (logk - logk_fft[i0]) / (logk_fft[i0+1] - logk_fft[i0])

            def per_mu(mu_j):
                Kfac = jnp.sqrt(1 + f * (2 + f) * mu_j**2)
                K    = k_i * Kfac
                Ksq  = K**2
                c    = (1 + f * mu_j**2) / Kfac
                s    = f * mu_j * jnp.sqrt(1 - mu_j**2) / Kfac

                A_mu = (1 + f) * mu_j / Kfac
                B_mu = jnp.sqrt(1 - mu_j**2) / Kfac

                A = k_i * q * c                         # (nq,)
                B = -0.5 * Ksq * Y_lin_lt               # (nq,)
                C = k_i * q * s                         # (nq,)

                base = jnp.exp(-0.5 * Ksq * (X_lin_lt + Y_lin_lt))  # (nq,)
                Gs = get_Gs(A, B, C, c**2, s**2, self.G00_coeffs, self.lmax)

                mq0 =  Gs[0,0]          # (L, nq)
                mq1 = -Gs[1,0]
                mq2 = -Gs[2,0]
                mq3 =  Gs[3,0]
                mq4 =  Gs[4,0]
                nq1 = -A_mu * Gs[1,0] + B_mu * Gs[0,1]
                nq2 = -A_mu**2 * Gs[2,0] + 2*A_mu*B_mu*Gs[1,1] - B_mu**2 * Gs[0,2]
                mq1_nq1 = -A_mu * Gs[2,0] + B_mu * Gs[1,1]
                mq2_nq1 =  A_mu * Gs[3,0] - B_mu * Gs[2,1]

                # --- 23 integrands of (L, nq)---
                # matter
                integrand_ZA   = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)
                integrand_AA   = Ksq**2 / 8.0 * (mq0 * X_lin_gt**2 + 2 * mq2 * X_lin_gt * Y_lin_gt + mq4 * Y_lin_gt**2)
                integrand_A22  = -0.5 * k_i**2 * ((Kfac**2 + 2 * f * (1 + f) * mu_j**2 + f**2 * mu_j**2) * mq0 * X22
                                            + (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1 + f**2 * mu_j**2 * nq2) * Y22)
                integrand_A13  = -0.5 * k_i**2 * (2 * (Kfac**2 + 2 * f * (1 + f) * mu_j**2) * mq0 * X13
                                            + 2 * (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1) * Y13)
                integrand_W112 = 0.5 * k_i**3 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu_j**2) * mq1 * V1 \
                                        + Kfac**2 * (Kfac * mq1 + f * mu_j * nq1) * V3 \
                                        + Kfac**2 * (Kfac * mq3 + f * mu_j * mq2_nq1) * T)

                # LIMD bias
                integrand_U10 = -2 * (K * mq1 * (U_lin + U3) + (2 * f * k_i * mu_j * nq1) * U3)
                integrand_A_U = Ksq * (mq1 * X_lin_gt + mq3 * Y_lin_gt) * (K * U_lin)
                # integrand_A10 = -Ksq * (mq0 * X10 + mq2 * Y10) - f * (1 + f) * (k_i * mu_j)**2 * mq0 * X10 - f * k_i * mu_j * mq1_nq1 * Y10
                integrand_A10 = -Ksq * (X10 * mq0 + Y10 * mq2) - f * k_i**2 * mu_j * ((1 + f) * mu_j * mq0 * X10 + Kfac * mq1_nq1 * Y10) # A10

                integrand_xi   = mq0 * xi_lin
                integrand_A_xi = -0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * xi_lin
                integrand_U11  = -(K * mq1 + f * k_i * mu_j * nq1) * U11
                integrand_U_U  = -Ksq * mq2 * U_lin**2

                integrand_U20  = -(K * mq1 + f * k_i * mu_j * nq1) * U20

                integrand_xi_U  = -2 * K * mq1 * xi_lin * U_lin
                integrand_xi_xi = 0.5 * mq0 * xi_lin**2

                # 2nd-order shear bias
                integrand_Upsilon = -Ksq * (mq0 * X_Upsilon + mq2 * Y_Upsilon)
                integrand_V10     = -2 * (K * mq1 + f * k_i * mu_j * nq1) * V10
                integrand_V12     = -2 * K * mq1 * V12
                integrand_chi     = mq0 * chi
                integrand_zeta    = mq0 * zeta

                # 3rd-order bias
                integrand_Ub3   = -2 * K * mq1 * Ub3
                integrand_theta = 2 * mq0 * theta

                # counterterm
                integrand_ctr = lax.select(
                    self.use_Pzel,
                    mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt),
                    mq0 * xi_lin
                )

                integrands = jnp.stack([
                    integrand_ZA, integrand_AA, integrand_A22, integrand_A13, integrand_W112,
                    integrand_U10, integrand_A_U, integrand_A10, integrand_xi, integrand_A_xi,
                    integrand_U_U, integrand_U11, integrand_U20, integrand_xi_U, integrand_xi_xi,
                    integrand_Upsilon, integrand_V10, integrand_V12, integrand_chi, integrand_zeta, 
                    integrand_Ub3, integrand_theta, integrand_ctr
                ], axis=0)   # (ncomp, L, nq)

                g = integrands * (base[None, None, :] * weights[None, :, :]) # (ncomp, L, nq)
                g = g.at[:,0,:].subtract(g[:,0,-1][:,None])
                g = (4 * jnp.pi * q**3)[None, None, :] * g

                pk_ffts = self.get_pk_batched(g, self._u_m[None, :L, :]) # (ncomp, L, nfft)
                vals = (1.0 - t) * pk_ffts[:, :, i0] + t * pk_ffts[:, :, i0+1] # (ncomp, L), linear interpolation

                return jnp.sum(vals, axis=1)  # (ncomp,)
            
            return jax.vmap(per_mu)(mu)  # (nmu, ncomp)

        pkmu_terms = jax.vmap(per_k)(k)                    # (nk, nmu, ncomp)
        pkmu_terms = jnp.transpose(pkmu_terms, (2, 0, 1))  # (ncomp, nk, nmu)

        return pkmu_terms
    
    def get_pkmu_terms(self, k, mu, pk_data, f, k_IR=0.2):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)
        q  = self._q

        corrs = self.get_corrs(pk_data, k_IR)
        corrs_tree, corrs_matter_1loop, corrs_bias = corrs[:8], corrs[8:15], corrs[15:] 

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

        L        = self.lmax + 1
        logk_fft = jnp.log(self._k)

        def per_point(k_i, mu_j):
            inv     = -2.0 / (k_i * q)                                # (nq,)
            weights = jnp.power(inv[None, :], jnp.arange(L)[:, None]) # (L, nq)

            logk = jnp.log(k_i)
            # preparation for linear interpolation
            i0   = jnp.searchsorted(logk_fft, logk, side='right') - 1
            i0   = jnp.clip(i0, 0, logk_fft.size - 2)
            t    = (logk - logk_fft[i0]) / (logk_fft[i0+1] - logk_fft[i0])

            Kfac = jnp.sqrt(1 + f * (2 + f) * mu_j**2)
            K    = k_i * Kfac
            Ksq  = K**2
            c    = (1 + f * mu_j**2) / Kfac
            s    = f * mu_j * jnp.sqrt(1 - mu_j**2) / Kfac

            A_mu = (1 + f) * mu_j / Kfac
            B_mu = jnp.sqrt(1 - mu_j**2) / Kfac

            A = k_i * q * c                         # (nq,)
            B = -0.5 * Ksq * Y_lin_lt               # (nq,)
            C = k_i * q * s                         # (nq,)

            base = jnp.exp(-0.5 * Ksq * (X_lin_lt + Y_lin_lt))  # (nq,)
            Gs = get_Gs(A, B, C, c**2, s**2, self.G00_coeffs, self.lmax)

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
            integrand_1  = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)
            integrand_1 += (Ksq**2) / 8.0 * (mq0 * X_lin_gt**2 + 2 * mq2 * X_lin_gt * Y_lin_gt + mq4 * Y_lin_gt**2)
            integrand_1 += -0.5 * k_i**2 * ((Kfac**2 + 2 * f * (1 + f) * mu_j**2 + f**2 * mu_j**2) * mq0 * X22
                                        + (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1 + f**2 * mu_j**2 * nq2) * Y22)
            integrand_1 += -0.5 * k_i**2 * (2 * (Kfac**2 + 2 * f * (1 + f) * mu_j**2) * mq0 * X13
                                        + 2 * (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1) * Y13)
            integrand_1 += 0.5 * k_i**3 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu_j**2) * mq1 * V1
                                    + Kfac**2 * (Kfac * mq1 + f * mu_j * nq1) * V3
                                    + Kfac**2 * (Kfac * mq3 + f * mu_j * mq2_nq1) * T)

            # LIMD bias
            integrand_b1  = -2 * (K * mq1 * (U_lin + U3) + (2 * f * k_i * mu_j * nq1) * U3) # U10
            integrand_b1 += Ksq * (mq1 * X_lin_gt + mq3 * Y_lin_gt) * (K * U_lin) # A> U_lin
            integrand_b1 += -Ksq * (X10 * mq0 + Y10 * mq2) - f * k_i**2 * mu_j * ((1 + f) * mu_j * mq0 * X10 + Kfac * mq1_nq1 * Y10) # A10

            integrand_b1_b1  = mq0 * xi_lin # xi_lin
            integrand_b1_b1 += -0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * xi_lin # A> xi_lin
            integrand_b1_b1 += -(K * mq1 + f * k_i * mu_j * nq1) * U11 # U11
            integrand_b1_b1 += -Ksq * mq2 * U_lin**2 # U_lin^2

            integrand_b2  = -(K * mq1 + f * k_i * mu_j * nq1) * U20 # U20
            integrand_b2 += -Ksq * mq2 * U_lin**2 # U_lin^2

            integrand_b1_b2 = -2 * K * mq1 * xi_lin * U_lin # xi_lin U_lin
            integrand_b2_b2 = 0.5 * mq0 * xi_lin**2 # xi_lin^2

            # 2nd-order shear bias
            integrand_bs  = -Ksq * (mq0 * X_Upsilon + mq2 * Y_Upsilon)
            integrand_bs += -2 * (K * mq1 + f * k_i * mu_j * nq1) * V10

            integrand_b1_bs = -2 * K * mq1 * V12
            integrand_b2_bs = mq0 * chi
            integrand_bs_bs = mq0 * zeta

            # 3rd-order bias
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

            g = integrands * (base[None, None, :] * weights[None, :, :]) # (ncomp, L, nq)
            g = g.at[:,0,:].subtract(g[:,0,-1][:,None])
            g = (4 * jnp.pi * q**3)[None, None, :] * g
            
            pk_ffts = self.get_pk_batched(g, self._u_m[None, :L, :]) # (ncomp, L, nfft)
            vals = (1.0 - t) * pk_ffts[:, :, i0] + t * pk_ffts[:, :, i0+1] # (ncomp, L), linear interpolation

            return jnp.sum(vals, axis=1)  # (ncomp,)

        pkmu_terms = jax.vmap(per_point)(k, mu).T  # (ncomp, n_kmu)
        return pkmu_terms
    
    def get_pkmu_terms_on_grid(self, k, mu, pk_data, f, k_IR=0.2):
        k_grid, mu_grid = jnp.meshgrid(k, mu, indexing='ij')  # (nk, nmu)
        k_flat  = k_grid.reshape(-1)
        mu_flat = mu_grid.reshape(-1)

        pkmu_terms = self.get_pkmu_terms(k_flat, mu_flat, pk_data, f, k_IR)

        nk, nmu = k_grid.shape
        ncomp   = pkmu_terms.shape[0]
        pkmu_terms = pkmu_terms.reshape(ncomp, nk, nmu)

        return pkmu_terms

    # @partial(jit, static_argnames=['self'])
    # def get_pkmu(self, k, mu, pk_data, params, k_IR=0.2):
    #     k  = jnp.atleast_1d(k)
    #     mu = jnp.atleast_1d(mu)

    #     f = params.f
    #     b1, b2, bs, b3 = params.bias
    #     alpha0, alpha2, alpha4, alpha6 = params.ctr
    #     R_h3, sigma2, sigma4 = params.stoch

    #     bias_facs = jnp.array([
    #         1.,            # 1
    #         b1,            # b1
    #         b1**2,         # b1^2
    #         b2,            # b2
    #         b1*b2,         # b1*b2
    #         b2**2,         # b2^2
    #         bs,            # bs
    #         b1*bs,         # b1*bs
    #         b2*bs,         # b2*bs
    #         bs**2,         # bs^2
    #         b3,            # b3
    #         b1*b3          # b1*b3
    #     ])  # (12,)

    #     pkmu_terms = self.get_pkmu_terms_on_grid(k, mu, pk_data, f, k_IR)

    #     pkmu = jnp.tensordot(bias_facs, pkmu_terms[:-1], axes=(0, 0))   # (nk, nmu)

    #     # counterterm
    #     ctr_mu = alpha0 + alpha2 * mu**2 + alpha4 * mu**4 + alpha6 * mu**6   # (nmu,)
    #     pkmu_ctr = jnp.outer(k**2, ctr_mu) * pkmu_terms[-1]   # (nk, nmu)
    #     pkmu = pkmu + pkmu_ctr

    #     # stochasticity
    #     pkmu_stoch = R_h3 * (1 + sigma2 * jnp.outer(k**2, mu**2) + sigma4 * jnp.outer(k**4, mu**4))
    #     pkmu = pkmu + pkmu_stoch

    #     return pkmu  # (nk, nmu)
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=0.2):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        f = params.f
        b1, b2, bs, b3 = params.bias
        alpha0, alpha2, alpha4, alpha6 = params.ctr
        R_h3, sigma2, sigma4 = params.stoch

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

        # mapping of (k, mu)
        k_true, mu_true = get_k_mu_true_for_ap(k, mu, alpha_perp, alpha_para)
        mu_true = jnp.broadcast_to(mu_true, k_true.shape)  # (nk, nmu)

        k_flat  = k_true.reshape(-1)
        mu_flat = mu_true.reshape(-1)
        pkmu_terms = self.get_pkmu_terms(k_flat, mu_flat, pk_data, f, k_IR)

        nk, nmu = k_true.shape
        ncomp   = pkmu_terms.shape[0]
        pkmu_terms = pkmu_terms.reshape(ncomp, nk, nmu)

        pkmu = jnp.tensordot(bias_facs, pkmu_terms[:-1], axes=(0, 0))   # (nk, nmu)

        # counterterm
        ctr_mu = alpha0 + alpha2 * mu_true**2 + alpha4 * mu_true**4 + alpha6 * mu_true**6   # (nmu,)
        pkmu_ctr = k_true**2 * ctr_mu * pkmu_terms[-1]   # (nk, nmu)
        pkmu = pkmu + pkmu_ctr

        # stochasticity
        kmu2 = (k_true * mu_true)**2
        kmu4 = kmu2**2
        pkmu_stoch = R_h3 * (1 + sigma2 * kmu2 + sigma4 * kmu4)
        pkmu = pkmu + pkmu_stoch

        pkmu = pkmu / (alpha_perp**2 * alpha_para)
        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pk_ells(self, k, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=0.2):
        pkmu = self.get_pkmu(k, self._mu_quad, pk_data, params, alpha_perp, alpha_para, k_IR=k_IR)
        pk_ells = get_legendre_multipoles(pkmu, self._legendre_weights)  # (3, nk)
        return pk_ells

    def get_corrs(self, pk_data, k_IR=0.2, k_cut=10.0):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_lin = pk_lin * jnp.exp(-(self._k / k_cut)**2)
        pk_lin_lt = pk_lin * jnp.exp(-(self._k / k_IR)**2)

        # integrals of pk
        pk_int = get_pk_int(pk_data)
        pk_data_lt = jnp.stack([self._k, pk_lin_lt], axis=0)
        pk_int_lt = get_pk_int(pk_data_lt)

        # generalized correlation functions
        xi_ln = self.get_xi_ln_array(pk_lin)
        xi_ln_lt = self.get_xi_ln_array(pk_lin_lt)
        
        # tree-level terms
        corrs_tree = self.get_corrs_tree(xi_ln, xi_ln_lt, pk_int, pk_int_lt)

        # one-loop terms
        Qs = self.get_Qs(xi_ln)
        Rs = self.get_Rs(xi_ln, pk_lin)
        corrs_matter_1loop = self.get_corrs_matter_1loop(Qs, Rs)
        corrs_bias = self.get_corrs_bias(Qs, Rs, xi_ln)
        
        corrs = jnp.concatenate([corrs_tree, corrs_matter_1loop, corrs_bias], axis=0)
        return corrs
    
    def get_corrs_tree(self, xi_ln, xi_ln_lt, pk_int, pk_int_lt):
        # X_lin = 2/3 * (xi_ln[0,-2][0] - xi_ln[0,-2] - xi_ln[2,-2])
        X_lin = 2/3 * (pk_int - xi_ln[0,-2] - xi_ln[2,-2])
        Y_lin = 2 * xi_ln[2,-2]

        # X_lin_lt = 2/3 * (xi_ln_lt[0,-2][0] - xi_ln_lt[0,-2] - xi_ln_lt[2,-2])
        X_lin_lt = 2/3 * (pk_int_lt - xi_ln_lt[0,-2] - xi_ln_lt[2,-2])
        Y_lin_lt = 2 * xi_ln_lt[2,-2]

        X_lin_gt = X_lin - X_lin_lt
        Y_lin_gt = Y_lin - Y_lin_lt
        
        xi_lin = xi_ln[0,0]
        U_lin = - xi_ln[1,-1]

        corrs = jnp.stack([X_lin, Y_lin, X_lin_lt, Y_lin_lt,
                           X_lin_gt, Y_lin_gt, xi_lin, U_lin], axis=0)
        return corrs
    
    def get_Qs(self, xi_ln):

        integrand_Q1 = 8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2
        integrand_Q5 = 2/3 * xi_ln[0,0]**2 - 2/3 * xi_ln[2,0]**2 \
                    - 2/5 * xi_ln[1,-1] * xi_ln[1,1] + 2/5 * xi_ln[3,-1] * xi_ln[3,1]
        integrand_Q8 = 2/3 * xi_ln[0,0]**2 - 2/3 * xi_ln[2,0]**2
        integrand_Q_G2 = 2/5 * xi_ln[1,-1]**2 - 2/5 * xi_ln[3,-1]**2

        integrand_Qs = jnp.stack([integrand_Q1, integrand_Q5, integrand_Q8, integrand_Q_G2], axis=0)
        Qs = self.get_pk_batched(4 * jnp.pi * self._q**3 * integrand_Qs, self._u_m[0])
        Q1, Q5, Q8, Q_G2 = Qs[0], Qs[1], Qs[2], Qs[3]

        Q2 = 2 * Q5 - Q1

        Qs = jnp.stack([Q1, Q2, Q5, Q8, Q_G2], axis=0)
        return Qs
    
    def get_Rs(self, xi_ln, pk_lin):
        ells = jnp.array([0, 2, 4, 1, 3, 2], dtype=jnp.int32)
        xis = jnp.stack([
            xi_ln[0,  0],
            xi_ln[2,  0],
            xi_ln[4,  0],
            xi_ln[1,  1],
            xi_ln[3,  1],
            xi_ln[2,  2],
        ], axis=0)

        pk_list = self.get_pk_batched(self._q**2 * xis, self._u_m[ells]) * pk_lin

        pk_00, pk_20, pk_40, pk_11, pk_31, pk_22 = pk_list

        k = self._k
        R1 = k**2 * (8/15 * pk_00 - 16/21 * pk_20 + 8/35 * pk_40)
        R3 = 2/3 * k**2 * (pk_00 - pk_20) - 2/5 * k * (pk_11 - pk_31)
        R2 = R3 - R1

        F_G2 = -32/21 * k**2 * (pk_00 - 10/7 * pk_20 + 3/7 * pk_40)
        Rb3  = k**2 * (32/105 * pk_00 - 80/441 * pk_20 + 32/245 * pk_40) \
            - k * (64/315 * pk_11 + 32/105 * pk_31) + 16/63 * pk_22

        return jnp.stack([R1, R2, F_G2, Rb3], axis=0)
    
    def get_corrs_matter_1loop(self, Qs, Rs):
        Q1 = Qs[0]
        Q2 = Qs[1]
        R1 = Rs[0]
        R2 = Rs[1]

        # X, Y for 1-loop A_{ij}
        xi_ln_22_0m2 = self.get_xi_ln(0, -2, 9/98 * Q1)
        xi_ln_22_2m2 = self.get_xi_ln(2, -2, 9/98 * Q1)
        xi_ln_13_0m2 = self.get_xi_ln(0, -2, 5/21 * R1)
        xi_ln_13_2m2 = self.get_xi_ln(2, -2, 5/21 * R1)

        pk_data = jnp.stack([self._k, 9/98 * Q1], axis=0)
        xi_ln_22_q0 = get_pk_int(pk_data)
        # X22 = 2/3 * (xi_ln_22_0m2[0] - xi_ln_22_0m2 - xi_ln_22_2m2)
        X22 = 2/3 * (xi_ln_22_q0 - xi_ln_22_0m2 - xi_ln_22_2m2)
        Y22 = 2 * xi_ln_22_2m2

        pk_data = jnp.stack([self._k, 5/21 * R1], axis=0)
        xi_ln_13_q0 = get_pk_int(pk_data)
        # X13 = 2/3 * (xi_ln_13_0m2[0] - xi_ln_13_0m2 - xi_ln_13_2m2)
        X13 = 2/3 * (xi_ln_13_q0 - xi_ln_13_0m2 - xi_ln_13_2m2)
        Y13 = 2 * xi_ln_13_2m2

        # V1, V3, T for W_{ijk}
        T = self.get_xi_ln(3, -3, -3/7 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2))
        V1 = self.get_xi_ln(1, -3, 3/35 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2)) - 0.2 * T
        V3 = self.get_xi_ln(1, -3, -3/35 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2)) - 0.2 * T

        corrs = jnp.stack([X22, Y22, X13, Y13, V1, V3, T], axis=0)
        return corrs
    
    def get_corrs_bias(self, Qs, Rs, xi_ln):
        Q1 = Qs[0]
        Q2 = Qs[1]
        Q5 = Qs[2]
        Q8 = Qs[3]
        Q_G2 = Qs[4]
        R1 = Rs[0]
        R2 = Rs[1]
        F_G2 = Rs[2]
        Rb3 = Rs[3]
        
        # U10, U11, U20
        U3 = self.get_xi_ln(1, -1, -5/21 * R1) # 3rd-order part of U10
        U11 = self.get_xi_ln(1, -1, -6/7 * (R1 + R2)) # U11
        U20 = self.get_xi_ln(1, -1, -3/7 * Q8) # U20
        
        # A10
        pk_data = jnp.stack([self._k, 2/7 * R1], axis=0)
        xi_ln_A10_q0 = get_pk_int(pk_data) # R2 integral is zero
        xi_ln_A10_0m2 = self.get_xi_ln(0, -2, 2/7 * (Q5 + 2 * R2))
        xi_ln_A10_2m2 = self.get_xi_ln(2, -2, 1/7 * (2 * Q5 + 3 * R1 + 4 * R2))
        X10 = xi_ln_A10_q0 - xi_ln_A10_0m2 - xi_ln_A10_2m2
        Y10 = 3 * xi_ln_A10_2m2

        if self.use_galileon:
            # Upsilon based on G2
            Up1 = - self.get_xi_ln(0, 0, Q_G2)
            Up2 = - self.get_xi_ln(0, -2, Q1 - Q2) - self.get_xi_ln(2, -2, Q1 + 2 * Q2)
            X_Upsilon = 1/3 * Up1 - Up2
            Y_Upsilon = Up2 - Up1

            V10 = self.get_xi_ln(1, -1, 3/7 * Q1) # V10 based on G2
            V12 = self.get_xi_ln(1, -1, 2 * Q5) # V12 based on G2
            chi = 4/3 * (xi_ln[2,0]**2 - xi_ln[0,0]**2) # chi based on G2
            zeta = 2 * (8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2) # zeta based on G2

            # 3rd-order bias
            Ub3 = self.get_xi_ln(1, -1, -2/5 * F_G2) # Ub3 based on Gamma3
            theta = self.get_xi_ln(0, 0, 2/5 * F_G2) # theta based on Gamma3
        else:
            # Upsilon based on s^2
            J2 = 2/15 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
            J3 = -1/5 * xi_ln[1,-1] - 1/5 * xi_ln[3,-1]
            J4 = xi_ln[3,-1]
            X_Upsilon = 4 * J3**2
            Y_Upsilon = 6 * J2**2 + 8 * J2 * J3 + 4 * J2 * J4 + 4 * J3**2 + 8 * J3 * J4 + 2 * J4**2

            V10 = self.get_xi_ln(1, -1, 3/7 * Q1 - 2/7 * Q8) # V10 based on s^2
            V12 = 2 * (4/15 * xi_ln[1,-1] - 2/5 * xi_ln[3,-1]) * xi_ln[2,0] # V12 based on s^2
            chi = 4/3 * xi_ln[2,0]**2 # chi based on s^2
            zeta = 2 * (4/45 * xi_ln[0,0]**2 + 8/63 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2) # zeta based on s^2
            
            # 3rd-order bias
            Ub3 = self.get_xi_ln(1, -1, -Rb3)
            theta = self.get_xi_ln(0, 0, Rb3)

        corrs = jnp.stack([U3, U11, U20, X10, Y10, V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta], axis=0)
        return corrs
