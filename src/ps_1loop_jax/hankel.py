####
## This code is based on FFTLog-and-Beyond developed by Xiao Fang
####

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
    elif mode == 'edge-pad':
        array_low = jnp.broadcast_to(array_y[..., :1], array_y.shape[:-1] + (n_pad,))
        array_high = jnp.broadcast_to(array_y[..., -1:], array_y.shape[:-1] + (n_pad,))
        return jnp.concatenate([array_low, array_y, array_high], axis=-1)
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

def get_high_x_damp(x, x_high, kind='exp', power=6.0):
    if kind == 'none':
        return jnp.ones_like(x)
    if kind == 'exp':
        return jnp.exp(-(x / x_high)**power)
    if kind == 'cosine':
        tiny = jnp.finfo(x.dtype).tiny
        denom = jnp.maximum(jnp.log(x[-1]) - jnp.log(x_high), tiny)
        t = jnp.clip((jnp.log(x) - jnp.log(x_high)) / denom, 0.0, 1.0)
        return 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    raise ValueError(f"unknown high-x damping kind: {kind!r}")

#def get_hankel(nu, fx, x, y, u_m, npad, x_high, w_m):
def get_hankel(nu, fx, x, y, u_m, n_pad, x_high, window, window_freq, pad_mode='zero-pad', damp_kind='exp', damp_power=6.0):
    ### extrapolate fx. Note that x is already extrapolated.
    fx_pad = pad(fx, n_pad, mode=pad_mode)

    # damp high-x end
    fx_pad = fx_pad * get_high_x_damp(x, x_high, damp_kind, damp_power)

    # FFT on biased data
    c_m = jnp.fft.rfft(fx_pad * x**(-nu) * window)

    c_m = c_m * window_freq

    # Inverse FFT to get the Hankel transform
    res = jnp.fft.irfft(jnp.conj(c_m * u_m)) * y**(-nu)

    # unpad
    res = res[n_pad:-n_pad]
    
    return res    

def get_hankel_batched(nu, fx, x, y, u_m, n_pad, x_high, window, window_freq, pad_mode='zero-pad', damp_kind='exp', damp_power=6.0):
    *batch, _ = fx.shape
    ndim = len(batch)

    fx = pad(fx, n_pad, mode=pad_mode)

    # damp high-x end
    fx = fx * get_high_x_damp(x, x_high, damp_kind, damp_power)

    # FFT on biased data
    c_m = jnp.fft.rfft(fx * (x**(-nu)) * window)

    # apply the smoothing window on Fourier components
    c_m = c_m * window_freq

    # Inverse FFT to get the Hankel transform
    res = jnp.fft.irfft(jnp.conj(c_m * u_m)) * y**(-nu)

    # unpad
    res = res[..., n_pad:-n_pad]
    
    return res

# Mellin transform of spherical Bessel function
def get_g_l(ell, z, eps=1e-15):
    z = np.asarray(z, dtype=np.complex128)
    z = np.where(np.abs(z.imag) < eps, z + 1j*eps, z)
    g_l = np.sqrt(np.pi) * np.exp( np.log(2.0) * (z-2) + loggamma(0.5*(ell+z)) - loggamma(0.5*(3+ell-z)) )
    return g_l

def get_hankel_pld_backward(nu_back, fx, x, y, u_m_pld, n_pad, x_high, window, window_freq,
                             pad_mode='zero-pad', damp_kind='exp', damp_power=6.0):
    """Backward Hankel (l=0) with PLD decomposition of the integrand F(q).

    Decomposes F(q) in power-law modes q^{\nu_back+i*\eta}; the q^3 integration-measure
    factor is absorbed analytically into u_m_pld = exp(lnxy)^{-i*\eta} * g_0(\nu_back+3+i*\eta).
    Convergence strip: -3 < \nu_back < -1.
    """
    fx_pad = pad(fx, n_pad, mode=pad_mode)
    fx_pad = fx_pad * get_high_x_damp(x, x_high, damp_kind, damp_power)
    d_m = jnp.fft.rfft(fx_pad * x**(-nu_back) * window)
    d_m = d_m * window_freq
    res = jnp.fft.irfft(jnp.conj(d_m * u_m_pld)) * y**(-(nu_back + 3))
    return res[n_pad:-n_pad]

def get_hankel_pld_backward_batched(nu_back, fx, x, y, u_m_pld, n_pad, x_high, window, window_freq,
                                     pad_mode='zero-pad', damp_kind='exp', damp_power=6.0):
    """Batched version of get_hankel_pld_backward.  fx has shape (*batch, nfft); all batch
    elements share the same x, y, u_m_pld, window — only the input data fx differs."""
    fx = pad(fx, n_pad, mode=pad_mode)
    fx = fx * get_high_x_damp(x, x_high, damp_kind, damp_power)
    d_m = jnp.fft.rfft(fx * x**(-nu_back) * window)
    d_m = d_m * window_freq
    res = jnp.fft.irfft(jnp.conj(d_m * u_m_pld)) * y**(-(nu_back + 3))
    return res[..., n_pad:-n_pad]

def c_window(n, n_cut):
    if n_cut <= 0:
        return jnp.ones(n.size)
    if n_cut == 1:
        W = jnp.ones(n.size)
        return W.at[-1].set(0.0)
    n_right = n[-1] - n_cut
    n_r = n[n[:] > n_right]
    theta_right = (n[-1] - n_r) / (n[-1] - n_right - 1)
    W = jnp.ones(n.size)
    W = W.at[n[:] > n_right].set(theta_right - 1 / (2 * jnp.pi) * jnp.sin(2 * jnp.pi * theta_right))
    return W
