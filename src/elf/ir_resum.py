from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from .utils import get_pk
from . import spline


@dataclass(frozen=True)
class DSTSettings:
    """Settings of :func:`get_pk_nw` (module instance ``DST_SETTINGS``; ``kh_*``, ``n_*`` DST only)."""

    kmin_ext: float = 1e-6
    """Lower end (h/Mpc) of the returned no-wiggle table."""
    kmax_ext: float = 1e3
    """Upper end (h/Mpc) of the returned no-wiggle table."""
    kh_min: float = 7e-5
    """DST: lower end of the linear k grid, in 1/Mpc."""
    kh_max: float = 7.0
    """DST: upper end of the linear k grid, in 1/Mpc."""
    n_grid: int = 2**15
    """DST: number of nodes of the linear k grid."""
    n_min: int = 140
    """DST: first harmonic of the removed (BAO) band."""
    n_max: int = 200
    """DST: end (exclusive) of the removed harmonic band."""
    n_keep_high: int = 100
    """DST: high-k nodes where the input replaces the smoothed spectrum."""
    n_mid: int = 200
    """Number of log-spaced nodes of the smoothed range in the returned table."""
    n_ext: int = 40
    """Log-spaced nodes per side of the unsmoothed extension to ``[kmin_ext, kmax_ext]``."""


DST_SETTINGS = DSTSettings()


def _pk_below(pk_data, lambda_ir, kmin, num):
    """``(q, P(q))`` on ``num`` linear nodes of ``[kmin, lambda_ir]`` (cubic in log-log)."""
    q = jnp.linspace(kmin, lambda_ir, num)
    return q, jnp.exp(spline.interp1d(jnp.log(q), jnp.log(pk_data[0]), jnp.log(pk_data[1])))


def get_Sigma2(pk_data, r_bao, lambda_ir, kmin=1e-4, num=1000):
    """BAO displacement dispersion ``Sigma^2``, integrated over ``k <= lambda_ir``."""
    q, pk = _pk_below(pk_data, lambda_ir, kmin, num)
    integrand = pk * (1 - spherical_jn(0, r_bao * q) + 2 * spherical_jn(2, r_bao * q))
    res = jnp.trapezoid(integrand, x=q) / (6 * jnp.pi**2)
    return res

def get_dSigma2(pk_data, r_bao, lambda_ir, kmin=1e-4, num=1000):
    """Anisotropic part ``delta Sigma^2``, integrated over ``k <= lambda_ir``."""
    q, pk = _pk_below(pk_data, lambda_ir, kmin, num)
    integrand = pk * spherical_jn(2, r_bao * q)
    res = jnp.trapezoid(integrand, x=q) / (2 * jnp.pi**2)
    return res

def damping_exponent(k, mu, f, Sigma2, dSigma2):
    """``k^2 Sigma^2_s``, ``Sigma^2_s = (1 + mu^2 f (2 + f)) Sigma^2 + f^2 mu^2 (mu^2 - 1) delta Sigma^2``."""
    Sigma2_s = (1 + mu**2 * f * (2 + f)) * Sigma2 + f**2 * mu**2 * (mu**2 - 1) * dSigma2
    return k**2 * Sigma2_s


def spherical_jn(n, x):
    x = jnp.atleast_1d(x)
    res = jax.lax.cond(n == 0, 
                       lambda: jnp.sin(x) / x, 
                       lambda: jax.lax.cond(n == 1, 
                                            lambda: (jnp.sin(x) - x * jnp.cos(x)) / x**2, 
                                            lambda: ((3 - x**2) * jnp.sin(x) - 3 * x * jnp.cos(x)) / x**3))
    return res

def get_pk_nw(pk_data, h, method='DST'):
    s = DST_SETTINGS
    kmin_ext, kmax_ext = s.kmin_ext, s.kmax_ext

    def pk_at(k):
        # the input continued by its end slopes to [kmin_ext, kmax_ext]
        return get_pk(k, pk_data, kmin=kmin_ext, kmax=kmax_ext)

    if method == 'DST':
        kh = jnp.linspace(s.kh_min, s.kh_max, s.n_grid) # 1/Mpc
        k = kh / h
        kmin, kmax = k[0], k[-1]
        pk = pk_at(k)

        pk_nw = _remove_wiggle_dst(kh, pk, s.n_min, s.n_max)

        # ad-hoc adjustment at high k for extrapolation
        pk_nw = pk_nw.at[-s.n_keep_high:].set(pk[-s.n_keep_high:])

    elif method == 'SG':
        kmin, kmax, num = 1e-4, 1e1, 256
        window_length = 50
        poly_degree = 3

        k = jnp.geomspace(kmin, kmax, num)
        pk = pk_at(k)

        coeffs = _savgol_coeffs(window_length, poly_degree)

        m = (len(coeffs) - 1) // 2
        y = jnp.log(pk)
        y_pad = jnp.pad(y, (m, m), mode='reflect')
        z = jnp.convolve(y_pad, coeffs, mode='valid')  # same shape as y
        pk_nw = jnp.exp(z)

        # ad-hoc adjustment at low & high k for extrapolation
        pk_nw = pk_nw.at[:int(num/5)].set(pk[:int(num/5)])
        pk_nw = pk_nw.at[-int(num/5):].set(pk[-int(num/5):])

    elif method == 'WH':
        kmin, kmax, num = 1e-4, 1e1, 256
        lam, p = 1e3, 2

        k = jnp.geomspace(kmin, kmax, num)
        pk = pk_at(k)

        n = len(pk)
        D = _diff_mat(n, p)               # (n-p, n)
        A = jnp.eye(n) + lam * (D.T @ D)
        y = jnp.log(pk)
        z = jax.scipy.linalg.solve(A, y, assume_a='pos')
        pk_nw = jnp.exp(z)

        # ad-hoc adjustment at high k for extrapolation
        pk_nw = pk_nw.at[-int(num/5):].set(pk[-int(num/5):])

    k_mid = jnp.geomspace(kmin, kmax, s.n_mid) # h/Mpc
    pk_nw = jnp.exp(spline.interp1d(jnp.log(k_mid), jnp.log(k), jnp.log(pk_nw)))

    # unsmoothed input outside the smoothed range
    k_low   = jnp.geomspace(kmin_ext, kmin, s.n_ext)[:-1]
    k_high  = jnp.geomspace(kmax, kmax_ext, s.n_ext)[1:]
    pk_low  = pk_at(k_low)
    pk_high = pk_at(k_high)

    k_extrap     = jnp.concatenate([k_low, k_mid, k_high], axis=0) # h/Mpc
    pk_nw_extrap = jnp.concatenate([pk_low, pk_nw, pk_high], axis=0)
    
    pk_nw_data = jnp.stack([k_extrap, pk_nw_extrap], axis=0)

    return pk_nw_data

@lru_cache(maxsize=None)
def _dct_tables(N):
    """``cos``, ``sin`` of ``pi k / 2N`` and the orthonormal scale ``1/sqrt(f_k N)`` (``f_0 = 4``, else 2)."""
    theta = np.pi * np.arange(N) / (2 * N)
    scale = 1.0 / np.sqrt(np.where(np.arange(N) == 0, 4.0, 2.0) * N)
    return np.cos(theta), np.sin(theta), scale


def _dct2_ortho(x):
    """``jax.scipy.fft.dct(x, norm='ortho')`` by one rfft (Makhoul's reordering; even length)."""
    N = x.shape[-1]
    if N % 2:
        return jax.scipy.fft.dct(x, norm='ortho')
    cos, sin, scale = _dct_tables(N)
    v = jnp.concatenate([x[0::2], x[1::2][::-1]])
    h = N // 2
    V = jnp.fft.rfft(v)                                           # k = 0 .. N/2
    lo = (V.real * cos[:h + 1] + V.imag * sin[:h + 1]) * (2.0 * scale[:h + 1])
    # k = N - j, j = 1 .. N/2 - 1: V_{N-j} = conj(V_j), theta_{N-j} = pi/2 - theta_j
    hi = (V.real[1:h] * sin[1:h] - V.imag[1:h] * cos[1:h]) * (2.0 * scale[1:h])
    return jnp.concatenate([lo, hi[::-1]])


def _idct2_ortho(y):
    """``jax.scipy.fft.idct(y, norm='ortho')``, the inverse of :func:`_dct2_ortho`, by one irfft."""
    N = y.shape[-1]
    if N % 2:
        return jax.scipy.fft.idct(y, norm='ortho')
    cos, sin, scale = _dct_tables(N)
    c = y * (2.0 * N * scale)
    h = N // 2
    c_mk = c[h:][::-1]                                            # c_{N-k}, k = 1 .. N/2
    # Hermitian half H_k = (c_k e^{i theta_k} + c_{N-k} e^{-i theta_{N-k}}) / 2, H_0 = c_0
    re = 0.5 * (c[1:h + 1] * cos[1:h + 1] + c_mk * sin[1:h + 1])
    im = 0.5 * (c[1:h + 1] * sin[1:h + 1] - c_mk * cos[1:h + 1])
    H = jax.lax.complex(jnp.concatenate([c[:1], re]), jnp.concatenate([jnp.zeros_like(c[:1]), im]))
    v = jnp.fft.irfft(H, n=N)
    return jnp.stack([v[:h], v[h:][::-1]], axis=-1).reshape(N)


def _remove_wiggle_dst(kh, pk, n_min, n_max):
    signs = (-1)**jnp.arange(0, len(pk))
    harms = _dct2_ortho(jnp.log(kh * pk) * signs)[::-1]

    n_half = int(len(harms) / 2)
    n = jnp.arange(1, n_half + 1)
    harms_odd = harms[0::2]
    harms_even = harms[1::2]

    n_sd = jnp.concatenate([n[:n_min], n[n_max:]], axis=0)
    harms_odd_sd  = jnp.concatenate([harms_odd[:n_min], harms_odd[n_max:]], axis=0)
    harms_even_sd = jnp.concatenate([harms_even[:n_min], harms_even[n_max:]], axis=0)

    # interpolate only inside the removed band; outside it the original coefficients are kept exactly
    n_gap = n[n_min:n_max]
    harms_odd_gap = spline.interp1d(n_gap, n_sd, harms_odd_sd)
    harms_even_gap = spline.interp1d(n_gap, n_sd, harms_even_sd)
    harms_odd_s = harms_odd.at[n_min:n_max].set(harms_odd_gap)
    harms_even_s = harms_even.at[n_min:n_max].set(harms_even_gap)

    harms_s = jnp.empty_like(harms)
    harms_s = harms_s.at[0::2].set(harms_odd_s)
    harms_s = harms_s.at[1::2].set(harms_even_s)

    pk_nw = jnp.exp(_idct2_ortho(harms_s[::-1]) * signs) / kh
    return pk_nw

def _savgol_coeffs(window_length, poly_degree, dtype=jnp.float64):
    m = (window_length - 1) // 2
    x = jnp.arange(-m, m + 1, dtype=dtype)        # [-m, ..., m]
    
    A = x[:, None] ** jnp.arange(poly_degree + 1, dtype=dtype)[None, :]
    
    H = jax.scipy.linalg.solve(A.T @ A, A.T)      # (poly_degree+1, window_length)
    coeffs = H[0]
    
    return coeffs[::-1]

def _diff_mat(n, p):
    D = jnp.eye(n)
    for _ in range(p):
        D = D[1:] - D[:-1]
    return D
