import jax.numpy as jnp
import numpy as np
from scipy.special import loggamma

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
        tiny = jnp.finfo(array_y.dtype).tiny
        ratio_low = jnp.where(jnp.abs(array_y[..., 0]) > tiny, array_y[..., 1] / array_y[..., 0], 1.0)
        ratio_high = jnp.where(jnp.abs(array_y[..., -2]) > tiny, array_y[..., -1] / array_y[..., -2], 1.0)

        exp_low = jnp.arange(-n_pad, 0)
        exp_high = jnp.arange(1, n_pad + 1)
        expand = (None,) * (array_y.ndim - 1)
        array_low = array_y[..., :1] * ratio_low[..., None] ** exp_low[expand]
        array_high = array_y[..., -1:] * ratio_high[..., None] ** exp_high[expand]
        return jnp.concatenate([array_low, array_y, array_high], axis=-1)
    elif mode == 'zero-pad':
        pad_width = [(0, 0)] * (array_y.ndim - 1) + [(n_pad, n_pad)]
        return jnp.pad(array_y, pad_width)
    elif mode == 'smooth-zero-pad':
        x = (jnp.arange(n_pad, dtype=array_y.dtype) + 1.0) / (n_pad + 1.0)
        taper = func_window(x)
        expand = (None,) * (array_y.ndim - 1)
        array_low = array_y[..., :1] * taper[expand]
        array_high = array_y[..., -1:] * taper[::-1][expand]
        return jnp.concatenate([array_low, array_y, array_high], axis=-1)
    else:
        raise ValueError(f"unknown pad mode: {mode!r}")

def func_window(x: jnp.ndarray) -> jnp.ndarray:
    return x - jnp.sin(2.0*jnp.pi*x)/(2.0*jnp.pi)

def get_window(n_window, array_x) -> jnp.ndarray:
    if n_window <= 0:
        return jnp.ones_like(array_x)
    window = jnp.ones_like(array_x)
    i_left = jnp.arange(n_window, dtype=jnp.float64)
    x_left = (i_left + 1) / (n_window + 1)
    window = window.at[:n_window].set( func_window(x_left) )

    i_right = jnp.arange(n_window, dtype=jnp.float64)
    x_right = (i_right + 1) / (n_window + 1)
    window = window.at[-n_window:].set( func_window(x_right[::-1]) )
    return window

def get_hankel(nu, fx, x, y, u_m, n_pad, window, pad_mode='zero-pad'):
    fx_pad = pad(fx, n_pad, mode=pad_mode)
    c_m = jnp.fft.rfft(fx_pad * x**(-nu) * window)
    res = jnp.fft.irfft(jnp.conj(c_m * u_m)) * y**(-nu)
    return res[n_pad:-n_pad]

def get_hankel_batched(nu, fx, x, y, u_m, n_pad, window, pad_mode='zero-pad'):
    fx = pad(fx, n_pad, mode=pad_mode)
    c_m = jnp.fft.rfft(fx * (x**(-nu)) * window)
    res = jnp.fft.irfft(jnp.conj(c_m * u_m)) * y**(-nu)
    return res[..., n_pad:-n_pad]

# Mellin transform of spherical Bessel function
def get_g_l(ell, z, eps=1e-15):
    z = np.asarray(z, dtype=np.complex128)
    z = np.where(np.abs(z.imag) < eps, z + 1j*eps, z)
    g_l = np.sqrt(np.pi) * np.exp( np.log(2.0) * (z-2) + loggamma(0.5*(ell+z)) - loggamma(0.5*(3+ell-z)) )
    return g_l