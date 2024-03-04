import jax
import jax.numpy as jnp
import quadax
import interpax
from .utils_math import spherical_jn


class IRResum:

    def __init__(self, pk_lin, hubble, rbao=110, khmin=7e-5, khmax=7, n_min=120, n_max=240,
                kmin_interp=1e-7, kmax_interp=1e+7, kwarg={}):
        self.rbao = rbao
        kh = jnp.linspace(khmin, khmax, 2**16) # in unit of 1/Mpc
        plin = pk_lin(kh / hubble, **kwarg) * hubble**(-3) # in unit of Mpc^3
        plin_nw = self.remove_wiggle(kh, plin, n_min, n_max) # in unit of Mpc^3
        
        # ad-hoc adjustment at high k for extrapolation
        plin_nw = plin_nw.at[-10:].set(plin[-10:])

        # extrapolation
        k_low = jnp.geomspace(kmin_interp, kh[0] / hubble, 100)[:-1]
        k_high = jnp.geomspace(kh[-1] / hubble, kmax_interp, 100)[1:]
        k_extrap = jnp.hstack((k_low, kh / hubble, k_high)) # in unit of h/Mpc
        plin_nw_extrap = jnp.hstack((pk_lin(k_low, **kwarg), plin_nw * hubble**3, pk_lin(k_high, **kwarg))) # in unit of (Mpc/h)^3

        # spline interpolation
        self.pk_nw_interp = interpax.Interpolator1D(jnp.log(k_extrap), jnp.log(plin_nw_extrap))
        
    def remove_wiggle(self, kh, plin, n_min, n_max):
        # wiggly-non-wiggly splitting of linear power spectrum using DST (Sec. 4.2 of arXiv:2004.10607)

        # harms = dst(jnp.log(kh * plin))
        signs = (-1)**jnp.arange(0, len(plin))
        harms = jax.scipy.fft.dct(jnp.log(kh * plin) * signs)[::-1]

        n = jnp.arange(1,len(harms)+1)
        i_odd = jnp.arange(0,len(harms)-1,2)
        i_even = jnp.arange(1,len(harms),2)

        n_odd = n[i_odd]
        n_even = n[i_even]
        harms_odd = harms[i_odd]
        harms_even = harms[i_even]

        n = n[:int(len(harms)/2)]
        n_sd = jnp.hstack((n[n <= n_min], n[n >= n_max]))
        harms_odd_sd = jnp.hstack((harms_odd[n <= n_min], harms_odd[n >= n_max]))
        harms_even_sd = jnp.hstack((harms_even[n <= n_min], harms_even[n >= n_max]))

        harms_odd_s = interpax.interp1d(n, n_sd, harms_odd_sd, method="cubic")
        harms_even_s = interpax.interp1d(n, n_sd, harms_even_sd, method="cubic")

        i_rec = jnp.argsort(jnp.hstack((n_odd, n_even)))
        harms_s = jnp.hstack((harms_odd_s, harms_even_s))[i_rec]
        # plin_nw = jnp.exp(idst(harms_s)) / kh # in unit of Mpc^3
        plin_nw = jnp.exp(jax.scipy.fft.idct(harms_s[::-1]) * signs) / kh # in unit of Mpc^3

        return plin_nw

    def get_pk_nw(self, k):
        return jnp.exp(self.pk_nw_interp(jnp.log(k)))

    def get_Sigma2(self, ks, kmin=1e-7, epsrel=1e-6):
        func = lambda q: self.get_pk_nw(q) * (1 - spherical_jn(0,self.rbao*q) + 2 * spherical_jn(2,self.rbao*q))
        res = quadax.quadgk(func, [kmin, ks], epsrel=epsrel)
        return res[0] / (6 * jnp.pi**2)

    def get_dSigma2(self, ks, kmin=1e-7, epsrel=1e-6):
        func = lambda q: self.get_pk_nw(q) * spherical_jn(2,self.rbao*q)
        res = quadax.quadgk(func, [kmin, ks], epsrel=epsrel)
        return res[0] / (2 * jnp.pi**2)

    def get_sigmav2(self, kmin=1e-7, kmax=1e+7, epsrel=1e-6):
        func = lambda q: self.get_pk_nw(q)
        res = quadax.quadgk(func, [kmin, kmax], epsrel=epsrel)
        return res[0] / (6 * jnp.pi**2)

    def get_Sigma2_rsd(self, fgrowth, mu, ks=0.2):
        Sigma2_1 = (1 + mu**2 * fgrowth * (2 + fgrowth)) * self.get_Sigma2(ks)
        Sigma2_2 = fgrowth**2 * mu**2 * (mu**2 - 1) * self.get_dSigma2(ks)
        return Sigma2_1 + Sigma2_2
