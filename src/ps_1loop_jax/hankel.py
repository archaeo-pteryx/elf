####
## This code is based on FFTLog-and-Beyond developed by Xiao Fang
####

import jax
# jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma
from functools import partial


# @partial(jit, static_argnames=['npad'])
def get_hankel(nu, fx, x, y, u_m, npad, x_high, w_m):
    # zero padding
    fx = jnp.concatenate([jnp.zeros(npad), fx, jnp.zeros(npad)], axis=0)

    # damp high-x end
    fx = fx * jnp.exp(-(x / x_high)**2)

    # FFT on biased data
    c_m = jnp.fft.rfft(fx * x**(-nu))

    # apply the smoothing window on Fourier components
    c_m = c_m * w_m

    # Inverse FFT to get the Hankel transform
    res = jnp.fft.irfft(jnp.conj(c_m * u_m))
    res = res * y**(-nu) * jnp.sqrt(jnp.pi) / 4.

    # unpad
    res = res[npad:len(fx)-npad]
    
    return res

# @partial(jit, static_argnames=['npad'])
def get_hankel_batched(nu, fx, x, y, u_m, npad, x_high, w_m):
    # zero padding
    fx = jnp.pad(fx, ((0, 0), (npad, npad))) # (ncomp, nfft)

    # damp high-x end
    fx = fx * jnp.exp(-(x / x_high)**2)[None, :] # (ncomp, nfft)

    # FFT on biased data
    c_m = jnp.fft.rfft(fx * (x**(-nu))[None, :]) # (ncomp, nfft)

    # apply the smoothing window on Fourier components
    c_m = c_m * w_m[None, :] # (ncomp, nfft)

    # Inverse FFT to get the Hankel transform
    res = jnp.fft.irfft(jnp.conj(c_m * u_m[None, :])) # (ncomp, nfft)
    res = res * (y**(-nu))[None, :] * jnp.sqrt(jnp.pi) / 4. # (ncomp, nfft)

    # unpad
    res = res[:, npad:fx.shape[1]-npad] # (ncomp, nfft)
    
    return res

def get_g_l(l, z):
    return 2.**z * get_g_base(l+0.5, z-1.5)

def get_g_base(mu, x):
    g_base = np.zeros(x.size, dtype=complex)
    
    cut = 200
    
    # normal formula
    sel = (np.abs(np.imag(x)) + np.abs(mu) <= cut) & (x != mu + 1 + 0.0j)
    g_base[sel] = gamma((mu + 1 + x[sel]) / 2.) / gamma((mu + 1 - x[sel]) / 2.)
    
    # asymptotic formula
    sel = np.abs(np.imag(x)) + np.abs(mu) > cut
    asym_plus = (mu + 1 + x[sel]) / 2.
    asym_minus = (mu + 1 - x[sel]) / 2.
    g_base[sel] = np.exp((asym_plus - 0.5) * np.log(asym_plus) - (asym_minus - 0.5) * np.log(asym_minus) - x[sel] \
        + 1./12. * (1./asym_plus - 1./asym_minus) + 1./360. * (1./asym_minus**3 - 1./asym_plus**3) + 1./1260 * (1./asym_plus**5 - 1./asym_minus**5))
    
    g_base[np.where(x == mu + 1 + 0.0j)[0]] = 0. + 0.0j
    
    return g_base

def c_window(n, n_cut):
    n_right = n[-1] - n_cut
    n_r = n[n[:] > n_right]
    theta_right = (n[-1] - n_r) / (n[-1] - n_right - 1)
    W = jnp.ones(n.size)
    W = W.at[n[:] > n_right].set(theta_right - 1 / (2 * jnp.pi) * jnp.sin(2 * jnp.pi * theta_right))
    return W

def get_log_extrap(x, num_low, num_high):
    dlnx = jnp.log(x[1] / x[0])
    x_low = x[0] * jnp.exp(dlnx * jnp.arange(-num_low, 0))
    x_high = x[-1] * jnp.exp(dlnx * jnp.arange(1, num_high+1))
    return jnp.concatenate([x_low, x, x_high], axis=0)