import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp
import quadax
import interpax
from .utils_loop import get_log_extrap
from .utils_math import spherical_jn

@jit
def get_pk_nw_data(pk_data, h, khmin=7e-5, khmax=7., kmin_interp=1e-7, kmax_interp=1e7):
    k_extrap, pk_extrap = get_log_extrap(pk_data['k'], pk_data['pk'], kmin_interp, kmax_interp)
    pk_spl = interpax.Interpolator1D(jnp.log(k_extrap), jnp.log(pk_extrap))

    kh = jnp.linspace(khmin, khmax, 2**16) # in unit of 1/Mpc
    pk = jnp.exp(pk_spl(jnp.log(kh / h))) * h**(-3) # in unit of Mpc^3
    pk_nw = remove_wiggle(kh, pk) # in unit of Mpc^3
    
    # ad-hoc adjustment at high k for extrapolation
    pk_nw = pk_nw.at[-10:].set(pk[-10:])

    # extrapolation
    k_low = jnp.geomspace(kmin_interp, kh[0] / h, 100)[:-1]
    k_high = jnp.geomspace(kh[-1] / h, kmax_interp, 100)[1:]
    k_extrap = jnp.hstack((k_low, kh / h, k_high)) # in unit of h/Mpc

    pk_low = jnp.exp(pk_spl(jnp.log(k_low)))
    pk_high = jnp.exp(pk_spl(jnp.log(k_high)))
    pk_nw_extrap = jnp.hstack((pk_low, pk_nw * h**3, pk_high)) # in unit of (Mpc/h)^3

    pk_data = {'k': k_extrap, 'pk': pk_nw_extrap}
    return pk_data
    
@partial(jit, static_argnames=['n_min', 'n_max'])
def remove_wiggle(kh, pk, n_min=140, n_max=210):
    # wiggly-non-wiggly splitting of linear power spectrum using DST (Sec. 4.2 of arXiv:2004.10607)

    signs = (-1)**jnp.arange(0, len(pk))
    harms = jax.scipy.fft.dct(jnp.log(kh * pk) * signs)[::-1]

    n = jnp.arange(1,len(harms)+1)
    i_odd = jnp.arange(0,len(harms)-1,2)
    i_even = jnp.arange(1,len(harms),2)

    n_odd = n[i_odd]
    n_even = n[i_even]
    harms_odd = harms[i_odd]
    harms_even = harms[i_even]

    n = n[:int(len(harms)/2)]
    n_sd = jnp.hstack((n[:n_min], n[n_max:]))
    harms_odd_sd = jnp.hstack((harms_odd[:n_min], harms_odd[n_max:]))
    harms_even_sd = jnp.hstack((harms_even[:n_min], harms_even[n_max:]))

    harms_odd_s = interpax.interp1d(n, n_sd, harms_odd_sd, method='cubic')
    harms_even_s = interpax.interp1d(n, n_sd, harms_even_sd, method='cubic')

    i_rec = jnp.argsort(jnp.hstack((n_odd, n_even)))
    harms_s = jnp.hstack((harms_odd_s, harms_even_s))[i_rec]

    pk_nw = jnp.exp(jax.scipy.fft.idct(harms_s[::-1]) * signs) / kh # in unit of Mpc^3

    return pk_nw

@jit
def get_pk_nw(k, pk_data):
    pk_nw = jnp.exp(interpax.interp1d(jnp.log(k), jnp.log(pk_data['k']), jnp.log(pk_data['pk']), method='cubic'))
    return pk_nw
    
@partial(jit, static_argnames=['num'])
def get_Sigma2(pk_data, rbao, ks, kmin=1e-4, num=1000):
    q = jnp.linspace(kmin, ks, num)
    integrand = get_pk_nw(q, pk_data) * (1 - spherical_jn(0, rbao * q) + 2 * spherical_jn(2, rbao * q))
    res = quadax.simpson(integrand, x=q) / (6 * jnp.pi**2)
    return res
    
@partial(jit, static_argnames=['num'])
def get_dSigma2(pk_data, rbao, ks, kmin=1e-4, num=1000):
    q = jnp.linspace(kmin, ks, num)
    integrand = get_pk_nw(q, pk_data) * spherical_jn(2, rbao * q)
    res = quadax.simpson(integrand, x=q) / (2 * jnp.pi**2)
    return res
