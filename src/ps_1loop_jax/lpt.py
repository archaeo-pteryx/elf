import os

import jax
from jax import jit
from functools import partial
from jax import lax

import jax.numpy as jnp

from . import hankel
from . import spline

from .utils_loop import get_pk, get_pk_int
from .utils_lpt import get_G00s, get_Gs
from .multipole import prepare_mu_gauleg, get_legendre_multipoles, get_k_mu_true_for_ap


class LPT:

    def __init__(self,
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=512,
                 hankel_nu=1.1,
                 hankel_k_damp=None,
                 hankel_q_damp=0.5,
                 hankel_c_window_width=0.0,
                 hankel_npad_factor=0.5,
                 hankel_pad_mode='zero-pad',
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
        self._nu_hankel = hankel_nu
        self._hankel_k_damp = hankel_k_damp
        self._hankel_q_damp = hankel_q_damp
        self._hankel_c_window_width = hankel_c_window_width
        self._hankel_npad_factor = hankel_npad_factor
        self._hankel_pad_mode = hankel_pad_mode
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
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
        self._ln_list = jnp.array([[0,0], [0,-2], [1,-1], [2,0], [2,-2], [3,-1], [4,0]])
        lmax = max(jnp.max(self._ln_list[:, 0]), self.lmax)
        self._set_hankel(lmax)

    def _set_hankel(self, lmax):
        l_list = jnp.arange(lmax + 1)

        self._npad = max(1, int(self._hankel_npad_factor * self._nfft))
        self._k_padded = hankel.get_log_extrap(self._k, self._npad, self._npad)

        n_window = self._nfft // 4
        self._w_m = hankel.get_window(n_window, self._k_padded)

        nfft = len(self._k_padded)
        dln = jnp.log(self._k_padded[1] / self._k_padded[0])
        eta_m = 2 * jnp.pi * jnp.fft.rfftfreq(nfft, d=1.0) / dln

        # Compute each xi_l^n on its own low-ringing output grid, then
        # resample to the common q grid before LPT combines different ell.
        lnxy = jnp.array([dln * jnp.angle(hankel.get_g_l(l, self._nu_hankel + 1j * jnp.pi / dln)) / jnp.pi for l in l_list])
        self._q_padded = jnp.array([jnp.exp(lnxy[l] - dln) / self._k_padded[::-1] for l in l_list])
        self._q = jnp.array([self._q_padded[l][self._npad:-self._npad] for l in l_list])
        
        g_l = jnp.array([hankel.get_g_l(l, self._nu_hankel + 1j * eta_m) for l in l_list])
        self._u_m = jnp.array([jnp.exp(lnxy[l])**(-1j * eta_m) * g_l[l] for l in l_list])

        # Forward damping: Gaussian exp(-(k/k_high)^2). k_high=inf -> no damping.
        self._k_high = self._hankel_k_damp if self._hankel_k_damp is not None else jnp.inf
        self._q_high = jnp.max(self._q) * self._hankel_q_damp
        
        c_window_width = self._hankel_c_window_width
        self._w_m_freq = hankel.c_window(jnp.arange(nfft//2+1), int(c_window_width * (nfft//2+1)))

    def get_xi_ln(self, l, n, array):
        fx = array * self._k**(n + 3) / (2 * jnp.pi**2)
        xi_ln = hankel.get_hankel(
            self._nu_hankel,
            fx,
            self._k_padded,
            self._q_padded[l],
            self._u_m[l],
            self._npad,
            self._k_high,
            self._w_m,
            self._w_m_freq,
            self._hankel_pad_mode,
        )
        xi_ln = spline.interp1d(jnp.log(self._q), jnp.log(self._q[l]), xi_ln)
        return xi_ln

    def get_xi_ln_batched(self, ells, ns, arrays):
        ells = jnp.asarray(ells, dtype=jnp.int32)
        ns = jnp.asarray(ns)
        arrays = jnp.asarray(arrays)

        fx = arrays * self._k[None, :] ** (ns[:, None] + 3) / (2 * jnp.pi**2)
        xi_ln = hankel.get_hankel_batched(
            self._nu_hankel,
            fx,
            self._k_padded,
            self._q_padded[ells],
            self._u_m[ells],
            self._npad,
            self._k_high,
            self._w_m,
            self._w_m_freq,
            self._hankel_pad_mode,
        )
        def interp_to_common_q(ell, xi):
            return spline.interp1d(jnp.log(self._q[0]), jnp.log(self._q[ell]), xi)
        
        return jax.vmap(interp_to_common_q)(ells, xi_ln)
    
    def get_pk_ln(self, l, n, array):
        fx = array * self._q**(n + 3)
        pk_ln = hankel.get_hankel(self._nu_hankel, fx, self._q_padded[l], self._k_padded, self._u_m[l], self._npad, self._q_high, self._w_m, self._w_m_freq, self._hankel_pad_mode)
        return pk_ln
    
    def get_pk_batched(self, arrays, q_padded, u_m):
        pks = hankel.get_hankel_batched(self._nu_hankel, arrays, q_padded, self._k_padded, u_m, self._npad, self._q_high, self._w_m, self._w_m_freq, self._hankel_pad_mode)
        return pks
    
    def get_xi_ln_array(self, array):
        ls = self._ln_list[:, 0]
        ns = self._ln_list[:, 1]
        arrays = jnp.broadcast_to(array, (self._ln_list.shape[0], array.shape[0]))
        xis = self.get_xi_ln_batched(ls, ns, arrays)
        xi_ln = jnp.zeros((5, 3, self._nfft))
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
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_terms_from_corrs(self, k, mu, corrs, f):
        k  = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)
        q  = self._q

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

    def get_pkmu_terms(self, k, mu, pk_data, f, k_IR=0.2):
        corrs = self.get_corrs(pk_data, k_IR)
        return self.get_pkmu_terms_from_corrs(k, mu, corrs, f)

    @partial(jit, static_argnames=['self'])
    def get_pkmu_terms_on_grid_from_corrs(self, k, mu, corrs, f, alpha_perp=1.0, alpha_para=1.0):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)
        q = self._q

        corrs_tree, corrs_matter_1loop, corrs_bias = corrs[:8], corrs[8:15], corrs[15:]

        X_lin_lt, Y_lin_lt = corrs_tree[2], corrs_tree[3]
        X_lin_gt, Y_lin_gt = corrs_tree[4], corrs_tree[5]
        X22, Y22, X13, Y13, V1, V3, T = corrs_matter_1loop
        xi_lin, U_lin = corrs_tree[6], corrs_tree[7]
        U3, U11, U20, X10, Y10 = corrs_bias[0:5]
        V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta = corrs_bias[5:]

        L = self.lmax + 1
        logk_fft = jnp.log(self._k)

        k_true, mu_true = get_k_mu_true_for_ap(k, mu, alpha_perp, alpha_para)
        mu_true = jnp.broadcast_to(mu_true, k_true.shape)  # (nk, nmu)

        def per_k(k_true_row, mu_true_row):
            # k_true_row: (nmu,), mu_true_row: (nmu,)

            def per_mu(k_i, mu_j):
                inv = -2.0 / (k_i * q)
                weights = jnp.power(inv[None, :], jnp.arange(L)[:, None])

                logk = jnp.log(k_i)
                i0 = jnp.searchsorted(logk_fft, logk, side='right') - 1
                i0 = jnp.clip(i0, 0, logk_fft.size - 2)
                t = (logk - logk_fft[i0]) / (logk_fft[i0+1] - logk_fft[i0])

                Kfac = jnp.sqrt(1 + f * (2 + f) * mu_j**2)
                K = k_i * Kfac
                Ksq = K**2
                c = (1 + f * mu_j**2) / Kfac
                s = f * mu_j * jnp.sqrt(1 - mu_j**2) / Kfac

                A_mu = (1 + f) * mu_j / Kfac
                B_mu = jnp.sqrt(1 - mu_j**2) / Kfac

                A = k_i * q * c
                B = -0.5 * Ksq * Y_lin_lt
                C = k_i * q * s

                base = jnp.exp(-0.5 * Ksq * (X_lin_lt + Y_lin_lt))
                Gs = get_Gs(A, B, C, c**2, s**2, self.G00_coeffs, self.lmax)

                mq0 =  Gs[0,0]
                mq1 = -Gs[1,0]
                mq2 = -Gs[2,0]
                mq3 =  Gs[3,0]
                mq4 =  Gs[4,0]
                nq1 = -A_mu * Gs[1,0] + B_mu * Gs[0,1]
                nq2 = -A_mu**2 * Gs[2,0] + 2*A_mu*B_mu*Gs[1,1] - B_mu**2 * Gs[0,2]
                mq1_nq1 = -A_mu * Gs[2,0] + B_mu * Gs[1,1]
                mq2_nq1 =  A_mu * Gs[3,0] - B_mu * Gs[2,1]

                integrand_1  = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)
                integrand_1 += (Ksq**2) / 8.0 * (mq0 * X_lin_gt**2 + 2 * mq2 * X_lin_gt * Y_lin_gt + mq4 * Y_lin_gt**2)
                integrand_1 += -0.5 * k_i**2 * ((Kfac**2 + 2 * f * (1 + f) * mu_j**2 + f**2 * mu_j**2) * mq0 * X22
                                            + (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1 + f**2 * mu_j**2 * nq2) * Y22)
                integrand_1 += -0.5 * k_i**2 * (2 * (Kfac**2 + 2 * f * (1 + f) * mu_j**2) * mq0 * X13
                                            + 2 * (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1) * Y13)
                integrand_1 += 0.5 * k_i**3 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu_j**2) * mq1 * V1
                                        + Kfac**2 * (Kfac * mq1 + f * mu_j * nq1) * V3
                                        + Kfac**2 * (Kfac * mq3 + f * mu_j * mq2_nq1) * T)

                integrand_b1  = -2 * (K * mq1 * (U_lin + U3) + (2 * f * k_i * mu_j * nq1) * U3)
                integrand_b1 += Ksq * (mq1 * X_lin_gt + mq3 * Y_lin_gt) * (K * U_lin)
                integrand_b1 += -Ksq * (X10 * mq0 + Y10 * mq2) - f * k_i**2 * mu_j * ((1 + f) * mu_j * mq0 * X10 + Kfac * mq1_nq1 * Y10)

                integrand_b1_b1  = mq0 * xi_lin
                integrand_b1_b1 += -0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * xi_lin
                integrand_b1_b1 += -(K * mq1 + f * k_i * mu_j * nq1) * U11
                integrand_b1_b1 += -Ksq * mq2 * U_lin**2

                integrand_b2  = -(K * mq1 + f * k_i * mu_j * nq1) * U20
                integrand_b2 += -Ksq * mq2 * U_lin**2

                integrand_b1_b2 = -2 * K * mq1 * xi_lin * U_lin
                integrand_b2_b2 = 0.5 * mq0 * xi_lin**2

                integrand_bs  = -Ksq * (mq0 * X_Upsilon + mq2 * Y_Upsilon)
                integrand_bs += -2 * (K * mq1 + f * k_i * mu_j * nq1) * V10

                integrand_b1_bs = -2 * K * mq1 * V12
                integrand_b2_bs = mq0 * chi
                integrand_bs_bs = mq0 * zeta

                integrand_b3    = -2 * K * mq1 * Ub3
                integrand_b1_b3 = 2 * mq0 * theta

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

                g = integrands * (base[None, None, :] * weights[None, :, :])
                g = g.at[:, 0, :].subtract(g[:, 0, -1][:, None])
                g = (4 * jnp.pi * q**3)[None, None, :] * g
                return g, i0, t

            # g: (nmu, ncomp, L, nq); one batched Hankel over all mu×comp at once
            gs, i0s, ts = jax.vmap(per_mu)(k_true_row, mu_true_row)
            nmu = gs.shape[0]
            ncomp = gs.shape[1]
            pk_ffts = self.get_pk_batched(
                gs.reshape(nmu * ncomp, L, -1),
                self._u_m[None, :L, :]
            ).reshape(nmu, ncomp, L, -1)  # (nmu, ncomp, L, nfft)

            def interp_mu(pk_fft, i0, t):
                vals = (1.0 - t) * pk_fft[:, :, i0] + t * pk_fft[:, :, i0+1]  # (ncomp, L)
                return jnp.sum(vals, axis=1)  # (ncomp,)

            return jax.vmap(interp_mu)(pk_ffts, i0s, ts)  # (nmu, ncomp)

        pkmu_terms = jax.vmap(per_k)(k_true, mu_true)           # (nk, nmu, ncomp)
        return jnp.transpose(pkmu_terms, (2, 0, 1))             # (ncomp, nk, nmu)

    def get_pkmu_terms_on_grid(self, k, mu, pk_data, f, k_IR=0.2):
        corrs = self.get_corrs(pk_data, k_IR)
        return self.get_pkmu_terms_on_grid_from_corrs(k, mu, corrs, f)

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
    def combine_pkmu_terms(self, k, mu, pkmu_terms, params, alpha_perp=1.0, alpha_para=1.0):
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
    def get_pkmu_from_corrs(self, k, mu, corrs, params, alpha_perp=1.0, alpha_para=1.0):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)
        q = self._q

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
        ])

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

        L = self.lmax + 1
        logk_fft = jnp.log(self._k)

        k_true, mu_true = get_k_mu_true_for_ap(k, mu, alpha_perp, alpha_para)
        mu_true = jnp.broadcast_to(mu_true, k_true.shape)  # (nk, nmu)

        def per_k(k_true_row, mu_true_row):
            # k_true_row: (nmu,), mu_true_row: (nmu,)
            # Compute per-mu integrands, then do a single batched Hankel over all mu.

            def per_mu(k_i, mu_j):
                inv = -2.0 / (k_i * q)
                weights = jnp.power(inv[None, :], jnp.arange(L)[:, None])

                logk = jnp.log(k_i)
                i0 = jnp.searchsorted(logk_fft, logk, side='right') - 1
                i0 = jnp.clip(i0, 0, logk_fft.size - 2)
                t = (logk - logk_fft[i0]) / (logk_fft[i0+1] - logk_fft[i0])

                Kfac = jnp.sqrt(1 + f * (2 + f) * mu_j**2)
                K = k_i * Kfac
                Ksq = K**2
                c = (1 + f * mu_j**2) / Kfac
                s = f * mu_j * jnp.sqrt(1 - mu_j**2) / Kfac

                A_mu = (1 + f) * mu_j / Kfac
                B_mu = jnp.sqrt(1 - mu_j**2) / Kfac

                A = k_i * q * c
                B = -0.5 * Ksq * Y_lin_lt
                C = k_i * q * s

                base = jnp.exp(-0.5 * Ksq * (X_lin_lt + Y_lin_lt))
                Gs = get_Gs(A, B, C, c**2, s**2, self.G00_coeffs, self.lmax)

                mq0 = Gs[0, 0]
                mq1 = -Gs[1, 0]
                mq2 = -Gs[2, 0]
                mq3 = Gs[3, 0]
                mq4 = Gs[4, 0]
                nq1 = -A_mu * Gs[1, 0] + B_mu * Gs[0, 1]
                nq2 = -A_mu**2 * Gs[2, 0] + 2*A_mu*B_mu*Gs[1, 1] - B_mu**2 * Gs[0, 2]
                mq1_nq1 = -A_mu * Gs[2, 0] + B_mu * Gs[1, 1]
                mq2_nq1 = A_mu * Gs[3, 0] - B_mu * Gs[2, 1]

                integrand_1 = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)
                integrand_1 += (Ksq**2) / 8.0 * (mq0 * X_lin_gt**2 + 2 * mq2 * X_lin_gt * Y_lin_gt + mq4 * Y_lin_gt**2)
                integrand_1 += -0.5 * k_i**2 * ((Kfac**2 + 2 * f * (1 + f) * mu_j**2 + f**2 * mu_j**2) * mq0 * X22
                                            + (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1 + f**2 * mu_j**2 * nq2) * Y22)
                integrand_1 += -0.5 * k_i**2 * (2 * (Kfac**2 + 2 * f * (1 + f) * mu_j**2) * mq0 * X13
                                            + 2 * (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1) * Y13)
                integrand_1 += 0.5 * k_i**3 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu_j**2) * mq1 * V1
                                        + Kfac**2 * (Kfac * mq1 + f * mu_j * nq1) * V3
                                        + Kfac**2 * (Kfac * mq3 + f * mu_j * mq2_nq1) * T)

                integrand_b1 = -2 * (K * mq1 * (U_lin + U3) + (2 * f * k_i * mu_j * nq1) * U3)
                integrand_b1 += Ksq * (mq1 * X_lin_gt + mq3 * Y_lin_gt) * (K * U_lin)
                integrand_b1 += -Ksq * (X10 * mq0 + Y10 * mq2) - f * k_i**2 * mu_j * ((1 + f) * mu_j * mq0 * X10 + Kfac * mq1_nq1 * Y10)

                integrand_b1_b1 = mq0 * xi_lin
                integrand_b1_b1 += -0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * xi_lin
                integrand_b1_b1 += -(K * mq1 + f * k_i * mu_j * nq1) * U11
                integrand_b1_b1 += -Ksq * mq2 * U_lin**2

                integrand_b2 = -(K * mq1 + f * k_i * mu_j * nq1) * U20
                integrand_b2 += -Ksq * mq2 * U_lin**2

                integrand_b1_b2 = -2 * K * mq1 * xi_lin * U_lin
                integrand_b2_b2 = 0.5 * mq0 * xi_lin**2

                integrand_bs = -Ksq * (mq0 * X_Upsilon + mq2 * Y_Upsilon)
                integrand_bs += -2 * (K * mq1 + f * k_i * mu_j * nq1) * V10

                integrand_b1_bs = -2 * K * mq1 * V12
                integrand_b2_bs = mq0 * chi
                integrand_bs_bs = mq0 * zeta

                integrand_b3 = -2 * K * mq1 * Ub3
                integrand_b1_b3 = 2 * mq0 * theta

                integrand_ctr = lax.select(
                    self.use_Pzel,
                    mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt),
                    mq0 * xi_lin
                )

                ctr_mu = alpha0 + alpha2 * mu_j**2 + alpha4 * mu_j**4 + alpha6 * mu_j**6
                integrand = (
                    bias_facs[0] * integrand_1
                    + bias_facs[1] * integrand_b1
                    + bias_facs[2] * integrand_b1_b1
                    + bias_facs[3] * integrand_b2
                    + bias_facs[4] * integrand_b1_b2
                    + bias_facs[5] * integrand_b2_b2
                    + bias_facs[6] * integrand_bs
                    + bias_facs[7] * integrand_b1_bs
                    + bias_facs[8] * integrand_b2_bs
                    + bias_facs[9] * integrand_bs_bs
                    + bias_facs[10] * integrand_b3
                    + bias_facs[11] * integrand_b1_b3
                )
                integrand = integrand + k_i**2 * ctr_mu * integrand_ctr

                g = integrand * (base[None, :] * weights)
                g = g.at[0, :].subtract(g[0, -1])
                g = (4 * jnp.pi * q**3)[None, :] * g

                kmu2 = (k_i * mu_j)**2
                stoch = R_h3 * (1 + sigma2 * kmu2 + sigma4 * kmu2**2)
                return g, i0, t, stoch

            # g: (nmu, L, nq); one batched Hankel over all mu at once
            gs, i0s, ts, stochs = jax.vmap(per_mu)(k_true_row, mu_true_row)
            pk_ffts = self.get_pk_batched(gs, self._u_m[None, :L, :])  # (nmu, L, nfft)

            def interp_mu(pk_fft, i0, t):
                return jnp.sum((1.0 - t) * pk_fft[:, i0] + t * pk_fft[:, i0+1])

            return jax.vmap(interp_mu)(pk_ffts, i0s, ts) + stochs  # (nmu,)

        pkmu = jax.vmap(per_k)(k_true, mu_true)  # (nk, nmu)
        pkmu = pkmu / (alpha_perp**2 * alpha_para)
        return pkmu

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=0.2):
        corrs = self.get_corrs(pk_data, k_IR)
        return self.get_pkmu_from_corrs(k, mu, corrs, params, alpha_perp, alpha_para)
    
    @partial(jit, static_argnames=['self'])
    def get_pk_ells_from_corrs(self, k, corrs, params, alpha_perp=1.0, alpha_para=1.0):
        pkmu = self.get_pkmu_from_corrs(k, self._mu_quad, corrs, params, alpha_perp, alpha_para)
        pk_ells = get_legendre_multipoles(pkmu, self._legendre_weights)  # (3, nk)
        return pk_ells

    @partial(jit, static_argnames=['self'])
    def get_pk_ells_from_corrs_no_ap(self, k, corrs, params):
        k = jnp.atleast_1d(k)
        q = self._q

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
        ])

        corrs_tree, corrs_matter_1loop, corrs_bias = corrs[:8], corrs[8:15], corrs[15:]

        X_lin_lt, Y_lin_lt = corrs_tree[2], corrs_tree[3]
        X_lin_gt, Y_lin_gt = corrs_tree[4], corrs_tree[5]
        X22, Y22, X13, Y13, V1, V3, T = corrs_matter_1loop
        xi_lin, U_lin = corrs_tree[6], corrs_tree[7]
        U3, U11, U20, X10, Y10 = corrs_bias[0:5]
        V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta = corrs_bias[5:]

        L = self.lmax + 1
        logk_fft = jnp.log(self._k)
        nmu = self._mu_quad.shape[0]
        leg_weights_pos = self._legendre_weights[:, nmu:] + jnp.flip(self._legendre_weights[:, :nmu], axis=1)

        def per_k(k_i):
            inv = -2.0 / (k_i * q)
            weights = jnp.power(inv[None, :], jnp.arange(L)[:, None])

            logk = jnp.log(k_i)
            i0 = jnp.searchsorted(logk_fft, logk, side='right') - 1
            i0 = jnp.clip(i0, 0, logk_fft.size - 2)
            t = (logk - logk_fft[i0]) / (logk_fft[i0+1] - logk_fft[i0])

            def per_mu(mu_j):
                Kfac = jnp.sqrt(1 + f * (2 + f) * mu_j**2)
                K = k_i * Kfac
                Ksq = K**2
                c = (1 + f * mu_j**2) / Kfac
                s = f * mu_j * jnp.sqrt(1 - mu_j**2) / Kfac

                A_mu = (1 + f) * mu_j / Kfac
                B_mu = jnp.sqrt(1 - mu_j**2) / Kfac

                A = k_i * q * c
                B = -0.5 * Ksq * Y_lin_lt
                C = k_i * q * s

                base = jnp.exp(-0.5 * Ksq * (X_lin_lt + Y_lin_lt))
                Gs = get_Gs(A, B, C, c**2, s**2, self.G00_coeffs, self.lmax)

                mq0 = Gs[0, 0]
                mq1 = -Gs[1, 0]
                mq2 = -Gs[2, 0]
                mq3 = Gs[3, 0]
                mq4 = Gs[4, 0]
                nq1 = -A_mu * Gs[1, 0] + B_mu * Gs[0, 1]
                nq2 = -A_mu**2 * Gs[2, 0] + 2*A_mu*B_mu*Gs[1, 1] - B_mu**2 * Gs[0, 2]
                mq1_nq1 = -A_mu * Gs[2, 0] + B_mu * Gs[1, 1]
                mq2_nq1 = A_mu * Gs[3, 0] - B_mu * Gs[2, 1]

                integrand_1 = mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt)
                integrand_1 += (Ksq**2) / 8.0 * (mq0 * X_lin_gt**2 + 2 * mq2 * X_lin_gt * Y_lin_gt + mq4 * Y_lin_gt**2)
                integrand_1 += -0.5 * k_i**2 * ((Kfac**2 + 2 * f * (1 + f) * mu_j**2 + f**2 * mu_j**2) * mq0 * X22
                                            + (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1 + f**2 * mu_j**2 * nq2) * Y22)
                integrand_1 += -0.5 * k_i**2 * (2 * (Kfac**2 + 2 * f * (1 + f) * mu_j**2) * mq0 * X13
                                            + 2 * (Kfac**2 * mq2 + 2 * f * Kfac * mu_j * mq1_nq1) * Y13)
                integrand_1 += 0.5 * k_i**3 * (2 * Kfac * (Kfac**2 + f * (1 + f) * mu_j**2) * mq1 * V1
                                        + Kfac**2 * (Kfac * mq1 + f * mu_j * nq1) * V3
                                        + Kfac**2 * (Kfac * mq3 + f * mu_j * mq2_nq1) * T)

                integrand_b1 = -2 * (K * mq1 * (U_lin + U3) + (2 * f * k_i * mu_j * nq1) * U3)
                integrand_b1 += Ksq * (mq1 * X_lin_gt + mq3 * Y_lin_gt) * (K * U_lin)
                integrand_b1 += -Ksq * (X10 * mq0 + Y10 * mq2) - f * k_i**2 * mu_j * ((1 + f) * mu_j * mq0 * X10 + Kfac * mq1_nq1 * Y10)

                integrand_b1_b1 = mq0 * xi_lin
                integrand_b1_b1 += -0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt) * xi_lin
                integrand_b1_b1 += -(K * mq1 + f * k_i * mu_j * nq1) * U11
                integrand_b1_b1 += -Ksq * mq2 * U_lin**2

                integrand_b2 = -(K * mq1 + f * k_i * mu_j * nq1) * U20
                integrand_b2 += -Ksq * mq2 * U_lin**2

                integrand_b1_b2 = -2 * K * mq1 * xi_lin * U_lin
                integrand_b2_b2 = 0.5 * mq0 * xi_lin**2

                integrand_bs = -Ksq * (mq0 * X_Upsilon + mq2 * Y_Upsilon)
                integrand_bs += -2 * (K * mq1 + f * k_i * mu_j * nq1) * V10

                integrand_b1_bs = -2 * K * mq1 * V12
                integrand_b2_bs = mq0 * chi
                integrand_bs_bs = mq0 * zeta

                integrand_b3 = -2 * K * mq1 * Ub3
                integrand_b1_b3 = 2 * mq0 * theta

                integrand_ctr = lax.select(
                    self.use_Pzel,
                    mq0 - 0.5 * Ksq * (mq0 * X_lin_gt + mq2 * Y_lin_gt),
                    mq0 * xi_lin
                )

                ctr_mu = alpha0 + alpha2 * mu_j**2 + alpha4 * mu_j**4 + alpha6 * mu_j**6
                integrand = (
                    bias_facs[0] * integrand_1
                    + bias_facs[1] * integrand_b1
                    + bias_facs[2] * integrand_b1_b1
                    + bias_facs[3] * integrand_b2
                    + bias_facs[4] * integrand_b1_b2
                    + bias_facs[5] * integrand_b2_b2
                    + bias_facs[6] * integrand_bs
                    + bias_facs[7] * integrand_b1_bs
                    + bias_facs[8] * integrand_b2_bs
                    + bias_facs[9] * integrand_bs_bs
                    + bias_facs[10] * integrand_b3
                    + bias_facs[11] * integrand_b1_b3
                )
                integrand = integrand + k_i**2 * ctr_mu * integrand_ctr
                integrand = integrand * (base[None, :] * weights)

                kmu2 = (k_i * mu_j)**2
                stoch = R_h3 * (1 + sigma2 * kmu2 + sigma4 * kmu2**2)
                return integrand, stoch

            integrands_mu, stoch_mu = jax.vmap(per_mu)(self._mu_quad)
            integrands_ell = jnp.einsum('em,mlq->elq', leg_weights_pos, integrands_mu)
            stoch_ell = jnp.einsum('em,m->e', leg_weights_pos, stoch_mu)

            integrands_ell = integrands_ell.at[:, 0, :].subtract(integrands_ell[:, 0, -1][:, None])
            g = (4 * jnp.pi * q**3)[None, None, :] * integrands_ell

            pk_ffts = self.get_pk_batched(g, self._u_m[None, :L, :])
            vals = (1.0 - t) * pk_ffts[:, :, i0] + t * pk_ffts[:, :, i0+1]
            return jnp.sum(vals, axis=1) + stoch_ell

        return jax.vmap(per_k)(k).T

    def get_pk_ells(self, k, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=0.2):
        corrs = self.get_corrs(pk_data, k_IR)
        try:
            use_no_ap = float(alpha_perp) == 1.0 and float(alpha_para) == 1.0
        except Exception:
            use_no_ap = False

        if use_no_ap:
            return self.get_pk_ells_from_corrs_no_ap(k, corrs, params)
        return self.get_pk_ells_from_corrs(k, corrs, params, alpha_perp, alpha_para)

    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data, k_IR=0.2):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)
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
        integrands = jnp.stack([
            8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2,
            xi_ln[1,-1]**2 - xi_ln[3,-1]**2,
            xi_ln[0,0]**2 - xi_ln[2,0]**2,
        ], axis=0)

        res = self.get_pk_batched(4 * jnp.pi * self._q[0]**3 * integrands, self._q_padded[0], self._u_m[0])

        Q1 = res[0]
        Q2 = Q1 - 2/5 * self._k**2 * res[1]
        Q5 = (Q1 + Q2) / 2
        Q8 = 2/3 * res[2]
        Q_G2 = 2/5 * res[1]

        Qs = jnp.stack([Q1, Q2, Q5, Q8, Q_G2], axis=0)
        return Qs
    
    def get_Rs(self, xi_ln, pk_lin):
        ells = jnp.array([0, 2, 4, 1, 3], dtype=jnp.int32)
        xis = jnp.stack([xi_ln[0,0], xi_ln[2,0], xi_ln[4,0], xi_ln[1,-1], xi_ln[3,-1]], axis=0)

        pk_list = self.get_pk_batched(self._q[0]**2 * xis, self._q_padded[0], self._u_m[ells]) * pk_lin
        pk_00, pk_20, pk_40, pk_1m1, pk_3m1 = pk_list

        k = self._k
        R1 = k**2 * (8/15 * pk_00 - 16/21 * pk_20 + 8/35 * pk_40)
        R2 = k**2 * (-2/15 * pk_00 - 2/21 * pk_20 + 8/35 * pk_40 + 2/5 * k * (pk_1m1 - pk_3m1)) 
        F_G2 = -32/21 * k**2 * (pk_00 - 10/7 * pk_20 + 3/7 * pk_40)

        return jnp.stack([R1, R2, F_G2], axis=0)
    
    def get_corrs_matter_1loop(self, Qs, Rs):
        Q1 = Qs[0]
        Q2 = Qs[1]
        R1 = Rs[0]
        R2 = Rs[1]

        # X, Y for 1-loop A_{ij}
        xi_A = self.get_xi_ln_batched(
            jnp.array([0, 2, 0, 2]),
            jnp.array([-2, -2, -2, -2]),
            jnp.stack([9/98 * Q1, 9/98 * Q1, 5/21 * R1, 5/21 * R1], axis=0),
        )
        xi_ln_22_0m2, xi_ln_22_2m2, xi_ln_13_0m2, xi_ln_13_2m2 = xi_A

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
        xi_W = self.get_xi_ln_batched(
            jnp.array([3, 1, 1]),
            jnp.array([-3, -3, -3]),
            jnp.stack([
                -3/7 * (Q1 + 2 * Q2 + 2 * R1 + 4 * R2),
                3/35 * (Q1 + 2 * Q2 - 3 * R1 + 4 * R2),
                -3/35 * (4 * Q1 - 2 * Q2 - 2 * R1 - 4 * R2),
            ], axis=0),
        )
        T = xi_W[0]
        V1 = xi_W[1] - 0.2 * T
        V3 = xi_W[2] - 0.2 * T

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
        
        # A10
        pk_data = jnp.stack([self._k, 2/7 * R1], axis=0)
        xi_ln_A10_q0 = get_pk_int(pk_data) # R2 integral is zero

        xi_bias = self.get_xi_ln_batched(
            jnp.array([1, 1, 1, 0, 2, 0, 0, 2, 1, 1, 1, 0]),
            jnp.array([-1, -1, -1, -2, -2, 0, -2, -2, -1, -1, -1, 0]),
            jnp.stack([
                -5/21 * R1,
                -6/7 * (R1 + R2),
                -3/7 * Q8,
                2/7 * (Q5 + 2 * R2),
                1/7 * (2 * Q5 + 3 * R1 + 4 * R2),
                Q_G2,
                Q1 - Q2,
                Q1 + 2 * Q2,
                3/7 * Q1,
                2 * Q5,
                -2/5 * F_G2,
                2/5 * F_G2,
            ], axis=0),
        )
        U3, U11, U20 = xi_bias[0], xi_bias[1], xi_bias[2]
        xi_ln_A10_0m2, xi_ln_A10_2m2 = xi_bias[3], xi_bias[4]
        X10 = xi_ln_A10_q0 - xi_ln_A10_0m2 - xi_ln_A10_2m2
        Y10 = 3 * xi_ln_A10_2m2

        # Upsilon based on G2
        Up1 = -xi_bias[5]
        Up2 = -xi_bias[6] - xi_bias[7]
        X_Upsilon = 1/3 * Up1 - Up2
        Y_Upsilon = Up2 - Up1

        V10 = xi_bias[8] # V10 based on G2
        V12 = xi_bias[9] # V12 based on G2
        chi = 4/3 * (xi_ln[2,0]**2 - xi_ln[0,0]**2) # chi based on G2
        zeta = 2 * (8/15 * xi_ln[0,0]**2 - 16/21 * xi_ln[2,0]**2 + 8/35 * xi_ln[4,0]**2) # zeta based on G2

        # 3rd-order bias
        Ub3 = xi_bias[10] # Ub3 based on Gamma3
        theta = xi_bias[11] # theta based on Gamma3

        corrs = jnp.stack([U3, U11, U20, X10, Y10, V10, V12, X_Upsilon, Y_Upsilon, chi, zeta, Ub3, theta], axis=0)
        return corrs

    @partial(jit, static_argnames=['self'])
    def get_xi_ells(self, r, pk_data, params, alpha_perp=1.0, alpha_para=1.0, k_IR=0.2):
        r = jnp.atleast_1d(r)

        k = jnp.geomspace(1e-3, 1, 128) # ad-hoc down-sampling of k
        pk_ells = self.get_pk_ells(k, pk_data, params, alpha_perp, alpha_para, k_IR)

        pk0 = get_pk(self._k, jnp.stack([k, pk_ells[0]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk2 = get_pk(self._k, jnp.stack([k, pk_ells[1]], axis=0), kmin=self._kmin, kmax=self._kmax)
        pk4 = get_pk(self._k, jnp.stack([k, pk_ells[2]], axis=0), kmin=self._kmin, kmax=self._kmax)

        xi0 = spline.interp1d(jnp.log(r), jnp.log(self._q[0]), self.get_xi_ln(0, 0, pk0))
        xi2 = spline.interp1d(jnp.log(r), jnp.log(self._q[2]), -self.get_xi_ln(2, 0, pk2))
        xi4 = spline.interp1d(jnp.log(r), jnp.log(self._q[4]), self.get_xi_ln(4, 0, pk4))

        xi_ells = jnp.stack([xi0, xi2, xi4], axis=0)
        return xi_ells
