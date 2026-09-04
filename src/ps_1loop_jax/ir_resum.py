import jax
import jax.numpy as jnp
from .utils import get_log_extrap
from . import spline


def get_Sigma2(pk_data, r_bao, k_IR, kmin=1e-4, num=1000):
    q = jnp.linspace(kmin, k_IR, num)
    pk = jnp.exp(spline.interp1d(jnp.log(q), jnp.log(pk_data[0]), jnp.log(pk_data[1])))
    integrand = pk * (1 - spherical_jn(0, r_bao * q) + 2 * spherical_jn(2, r_bao * q))
    res = jnp.trapezoid(integrand, x=q) / (6 * jnp.pi**2)
    return res

def get_dSigma2(pk_data, r_bao, k_IR, kmin=1e-4, num=1000):
    q = jnp.linspace(kmin, k_IR, num)
    pk = jnp.exp(spline.interp1d(jnp.log(q), jnp.log(pk_data[0]), jnp.log(pk_data[1])))
    integrand = pk * spherical_jn(2, r_bao * q)
    res = jnp.trapezoid(integrand, x=q) / (2 * jnp.pi**2)
    return res

def spherical_jn(n, x):
    x = jnp.atleast_1d(x)
    res = jax.lax.cond(n == 0, 
                       lambda: jnp.sin(x) / x, 
                       lambda: jax.lax.cond(n == 1, 
                                            lambda: (jnp.sin(x) - x * jnp.cos(x)) / x**2, 
                                            lambda: ((3 - x**2) * jnp.sin(x) - 3 * x * jnp.cos(x)) / x**3))
    return res

def get_pk_nw(pk_data, h, kmin_ext=1e-6, kmax_ext=1e3, method='DST'):

    k_grid, pk_grid = get_log_extrap(pk_data[0], pk_data[1], kmin_ext, kmax_ext)

    if method == 'DST':
        khmin, khmax, num = 7e-5, 7.0, 2**15
        n_min, n_max = 140, 200

        kh = jnp.linspace(khmin, khmax, num) # 1/Mpc
        k = kh / h
        kmin, kmax = k[0], k[-1]
        pk = spline.interp1d(jnp.log(k), jnp.log(k_grid), pk_grid)

        # remove the BAO using DST
        pk_nw = _remove_wiggle_dst(kh, pk, n_min, n_max)

        # ad-hoc adjustment at high k for extrapolation
        pk_nw = pk_nw.at[-100:].set(pk[-100:])

    elif method == 'SG':
        kmin, kmax, num = 1e-4, 1e1, 256
        window_length = 50
        poly_degree = 3

        k = jnp.geomspace(kmin, kmax, num)
        pk = spline.interp1d(jnp.log(k), jnp.log(k_grid), pk_grid)

        coeffs = _savgol_coeffs(window_length, poly_degree)

        # Savitzky-Golay filtering
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
        pk = spline.interp1d(jnp.log(k), jnp.log(k_grid), pk_grid)

        # Whittaker-Henderson smoothing
        n = len(pk)
        D = _diff_mat(n, p)               # (n-p, n)
        A = jnp.eye(n) + lam * (D.T @ D)
        y = jnp.log(pk)
        z = jax.scipy.linalg.solve(A, y, assume_a='pos')
        pk_nw = jnp.exp(z)

        # ad-hoc adjustment at high k for extrapolation
        pk_nw = pk_nw.at[-int(num/5):].set(pk[-int(num/5):])

    # redefine the intermediate k grids for a roughly equidistant logarithmic binning
    k_mid = jnp.geomspace(kmin, kmax, 200) # h/Mpc
    pk_nw = jnp.exp(spline.interp1d(jnp.log(k_mid), jnp.log(k), jnp.log(pk_nw)))

    # extrapolation with the un-smoothed linear power spectrum
    k_low   = jnp.geomspace(kmin_ext, kmin, 40)[:-1]
    k_high  = jnp.geomspace(kmax, kmax_ext, 40)[1:]
    pk_low  = spline.interp1d(jnp.log(k_low), jnp.log(k_grid), pk_grid)
    pk_high = spline.interp1d(jnp.log(k_high), jnp.log(k_grid), pk_grid)

    k_extrap     = jnp.concatenate([k_low, k_mid, k_high], axis=0) # h/Mpc
    pk_nw_extrap = jnp.concatenate([pk_low, pk_nw, pk_high], axis=0)
    
    pk_nw_data = jnp.stack([k_extrap, pk_nw_extrap], axis=0)

    return pk_nw_data

def _remove_wiggle_dst(kh, pk, n_min=140, n_max=200):
    # wiggly-non-wiggly splitting using DST-II

    signs = (-1)**jnp.arange(0, len(pk))
    harms = jax.scipy.fft.dct(jnp.log(kh * pk) * signs, norm='ortho')[::-1]

    n_half = int(len(harms) / 2)
    n = jnp.arange(1, n_half + 1)
    harms_odd = harms[0::2]
    harms_even = harms[1::2]

    n_sd = jnp.concatenate([n[:n_min], n[n_max:]], axis=0)
    harms_odd_sd  = jnp.concatenate([harms_odd[:n_min], harms_odd[n_max:]], axis=0)
    harms_even_sd = jnp.concatenate([harms_even[:n_min], harms_even[n_max:]], axis=0)

    # Only the removed BAO band needs interpolation; 
    # outside it, evaluating the spline on original knots would reproduce the original coefficients.
    n_gap = n[n_min:n_max]
    harms_odd_gap = spline.interp1d(n_gap, n_sd, harms_odd_sd)
    harms_even_gap = spline.interp1d(n_gap, n_sd, harms_even_sd)
    harms_odd_s = harms_odd.at[n_min:n_max].set(harms_odd_gap)
    harms_even_s = harms_even.at[n_min:n_max].set(harms_even_gap)

    harms_s = jnp.empty_like(harms)
    harms_s = harms_s.at[0::2].set(harms_odd_s)
    harms_s = harms_s.at[1::2].set(harms_even_s)

    pk_nw = jnp.exp(jax.scipy.fft.idct(harms_s[::-1], norm='ortho') * signs) / kh
    return pk_nw

def _savgol_coeffs(window_length, poly_degree, dtype=jnp.float64):
    m = (window_length - 1) // 2
    x = jnp.arange(-m, m + 1, dtype=dtype)        # [-m, ..., m]
    
    # design matrix A[i, j] = x[i]^j
    A = x[:, None] ** jnp.arange(poly_degree + 1, dtype=dtype)[None, :]
    
    H = jax.scipy.linalg.solve(A.T @ A, A.T)      # (poly_degree+1, window_length)
    coeffs = H[0]
    
    return coeffs[::-1]

def _diff_mat(n, p):
    D = jnp.eye(n)
    for _ in range(p):
        D = D[1:] - D[:-1]
    return D
