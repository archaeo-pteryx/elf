####
## This code is based on FFTLog-and-Beyond developed by Xiao Fang
####

import jax.numpy as jnp
import numpy as np
from scipy.special import gamma

def get_log_extrap(array_x, num_low, num_high):
    ratio = array_x[1] / array_x[0]
    exp   = jnp.arange(-num_low, 0)
    _array_x = array_x[0] * (ratio ** exp)
    ratio    = array_x[-1] / array_x[-2]
    exp      = jnp.arange(1, num_high + 1)
    array_x_ = array_x[-1] * (ratio ** exp)
    return jnp.concatenate([_array_x, array_x, array_x_], axis=0)

def pad(array_y, n_pad, mode='power-law'):
    if mode == 'power-law':
        return get_log_extrap(array_y, n_pad, n_pad)
    elif mode == 'zero-pad':
        _array_y = jnp.zeros(n_pad)
        array_y_ = jnp.zeros(n_pad)
        return jnp.concatenate([_array_y, array_y, array_y_], axis=0)

def func_window(x: jnp.ndarray) -> jnp.ndarray:
    return x - jnp.sin(2.0*jnp.pi*x)/(2.0*jnp.pi)

def get_window(n_window, array_x) -> jnp.ndarray:
    window = jnp.ones_like(array_x)
    i_left = jnp.arange(n_window, dtype=jnp.float64)
    x_left = (i_left + 1) / (n_window + 1)
    window = window.at[:n_window].set( func_window(x_left) )

    i_right = jnp.arange(n_window, dtype=jnp.float64)
    x_right = (i_right + 1) / (n_window + 1)
    window = window.at[-n_window:].set( func_window(x_right[::-1]) )
    return window

#def get_hankel(nu, fx, x, y, u_m, npad, x_high, w_m):
def get_hankel(nu, fx, x, y, u_m, n_pad, x_high, window, window_freq):
    ### extrapolate fx. Note that x is already extrapolated.
    fx_pad = pad(fx, n_pad, mode='zero-pad')

    # damp high-x end
    fx_pad = fx_pad * jnp.exp(-(x / x_high)**4)

    # FFT on biased data
    c_m = jnp.fft.rfft(fx_pad * x**(-nu) * window)

    c_m = c_m * window_freq

    # Inverse FFT to get the Hankel transform
    res = jnp.fft.irfft(jnp.conj(c_m * u_m)) * y**(-nu)

    # unpad
    res = res[n_pad:-n_pad]
    
    return res    

def get_hankel_batched(nu, fx, x, y, u_m, npad, x_high, w_m):
    *batch, _ = fx.shape
    ndim = len(batch)

    # zero padding
    fx = jnp.pad(fx, [(0, 0) for _ in range(ndim)] + [(npad, npad)])

    # damp high-x end
    fx = fx * jnp.exp(-(x / x_high)**2)

    # FFT on biased data
    c_m = jnp.fft.rfft(fx * (x**(-nu)))

    # apply the smoothing window on Fourier components
    c_m = c_m * w_m

    # Inverse FFT to get the Hankel transform
    res = jnp.fft.irfft(jnp.conj(c_m * u_m)) * y**(-nu)

    # unpad
    res = res[..., npad:-npad]
    
    return res

# Mellin transform of spherical Bessel function
def get_g_l(ell, z, eps=1e-15):
    z = np.asarray(z, dtype=np.complex128)
    z = np.where(np.abs(z.imag) < eps, z + 1j*eps, z)
    g_l = jnp.sqrt(jnp.pi) * 2.**(z-2) * gamma((ell+z)*0.5) / gamma((3+ell-z)*0.5)
    return g_l

def c_window(n, n_cut):
    n_right = n[-1] - n_cut
    n_r = n[n[:] > n_right]
    theta_right = (n[-1] - n_r) / (n[-1] - n_right - 1)
    W = jnp.ones(n.size)
    W = W.at[n[:] > n_right].set(theta_right - 1 / (2 * jnp.pi) * jnp.sin(2 * jnp.pi * theta_right))
    return W
