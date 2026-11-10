from dataclasses import dataclass
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from .utils import eval_pk, prepare_pk
from . import spline


@dataclass(frozen=True)
class DSTSettings:
    """Settings of :func:`get_pk_nw` (module instance ``DST_SETTINGS``).

    ``kmin_ext``, ``kmax_ext``, ``n_mid`` and ``n_ext`` are used by all three no-wiggle
    methods ('DST', 'SG', 'WH'), the other fields only by 'DST'.  The filter of 'DST' is fixed
    by the basis (``k_mpc_min``, ``k_mpc_max``, ``n_basis``) and the removed band (``n_min``,
    ``n_max``); ``n_samples`` sets only the precision of the DST coefficients.
    """

    kmin_ext: float = 1e-6
    """Lower end (h/Mpc) of the returned no-wiggle table."""
    kmax_ext: float = 1e3
    """Upper end (h/Mpc) of the returned no-wiggle table."""
    k_mpc_min: float = 7e-5
    """DST: lower end of the smoothed k range, in 1/Mpc."""
    k_mpc_max: float = 7.0
    """DST: upper end of the smoothed k range, in 1/Mpc."""
    n_basis: int = 2**15
    """DST: the DST-II basis sin(pi n (k - a) / (b - a)) is that of the linear grid of ``n_basis`` nodes
    of [k_mpc_min, k_mpc_max] (spacing dk), [a, b] = [k_mpc_min - dk/2, k_mpc_max + dk/2]."""
    n_samples: int = 2**13
    """DST: samples of ln(k P), the midpoints of ``n_samples`` equal cells of [a, b] (even).  A
    precision knob, not the filter: P_nw is converged to ~1e-8 from 2^13 on.  The first sample
    a + (b - a) / (2 n_samples) must be positive, which holds up to about 2.9 n_basis with the
    default range."""
    n_min: int = 140
    """DST: first removed row i of the odd (harmonic 2i + 1) and of the even (2i + 2) series."""
    n_max: int = 200
    """DST: end (exclusive) of the removed rows, so the harmonics 2 n_min + 1 .. 2 n_max (the BAO)."""
    n_keep_high: int = 100
    """DST: the input replaces the smoothed spectrum on the last ``n_keep_high`` nodes of the basis
    grid, k_mpc >= k_mpc_max - (n_keep_high - 1) dk."""
    n_mid: int = 200
    """Number of log-spaced nodes of the smoothed range in the returned table."""
    n_ext: int = 40
    """Log-spaced nodes per side of the unsmoothed extension to ``[kmin_ext, kmax_ext]``."""


DST_SETTINGS = DSTSettings()


def _pk_below(pk_data, lambda_ir, kmin, num):
    """``(q, P(q))`` on ``num`` linear nodes of ``[kmin, lambda_ir]`` (cubic in log-log) of the ``(2, n)`` table
    ``pk_data``, k in row 0 and P in row 1."""
    k, pk = pk_data[0], pk_data[1]
    q = jnp.linspace(kmin, lambda_ir, num)
    return q, jnp.exp(spline.interp1d(jnp.log(q), jnp.log(k), jnp.log(pk)))


def sigma_integrals(pk_nw_data, r_bao, lambda_ir, kmin=1e-4, num=1000):
    """``(Sigma2, dSigma2)``, the BAO displacement dispersion ``Sigma^2`` and its anisotropic part ``delta Sigma^2``.

    Both are integrated over ``k <= lambda_ir`` from one sample of ``pk_nw_data`` (the ``(2, n)`` no-wiggle table, k in
    row 0 and P_nw in row 1).
    """
    q, pk = _pk_below(pk_nw_data, lambda_ir, kmin, num)
    Sigma2 = jnp.trapezoid(pk * (1 - _j0(r_bao * q) + 2 * _j2(r_bao * q)), x=q) / (6 * jnp.pi**2)
    dSigma2 = jnp.trapezoid(pk * _j2(r_bao * q), x=q) / (2 * jnp.pi**2)
    return Sigma2, dSigma2


def get_Sigma2(pk_data, r_bao, lambda_ir, kmin=1e-4, num=1000):
    """BAO displacement dispersion ``Sigma^2`` of :func:`sigma_integrals`."""
    return sigma_integrals(pk_data, r_bao, lambda_ir, kmin, num)[0]


def get_dSigma2(pk_data, r_bao, lambda_ir, kmin=1e-4, num=1000):
    """Anisotropic part ``delta Sigma^2`` of :func:`sigma_integrals`."""
    return sigma_integrals(pk_data, r_bao, lambda_ir, kmin, num)[1]

def damping_exponent(k, mu, f, Sigma2, dSigma2):
    """``k^2 Sigma^2_s``, ``Sigma^2_s = (1 + mu^2 f (2 + f)) Sigma^2 + f^2 mu^2 (mu^2 - 1) delta Sigma^2``."""
    Sigma2_s = (1 + mu**2 * f * (2 + f)) * Sigma2 + f**2 * mu**2 * (mu**2 - 1) * dSigma2
    return k**2 * Sigma2_s


def _j0(x):
    """Spherical Bessel function ``j_0(x) = sin(x) / x`` (x > 0)."""
    return jnp.sin(x) / x


def _j2(x):
    """Spherical Bessel function ``j_2(x) = ((3 - x^2) sin(x) - 3 x cos(x)) / x^3`` (x > 0)."""
    return ((3 - x**2) * jnp.sin(x) - 3 * x * jnp.cos(x)) / x**3

def get_pk_nw(pk_data, h, method='DST'):
    """The ``(2, n)`` no-wiggle table, k in row 0 and P_nw in row 1, of ``pk_data`` (the ``(2, n)`` array of the input
    table, k [h/Mpc] in row 0 and P_lin in row 1) by ``method``.

    'DST' (a band of harmonics of the discrete sine transform of ``ln(k P)`` in k [1/Mpc, hence ``h``]
    removed: ``P_nw = P e^{-W}`` at the table nodes, :func:`_removed_band`),
    'SG' (Savitzky-Golay filter of ``ln P``) or 'WH' (Whittaker-Henderson smoothing of ``ln P``).
    The input P must be positive on the table nodes: every method takes its logarithm.
    """
    s = DST_SETTINGS
    kmin_ext, kmax_ext = s.kmin_ext, s.kmax_ext
    # the input continued by its end slopes to [kmin_ext, kmax_ext], prepared once for every evaluation below
    table = prepare_pk(pk_data, kmin=kmin_ext, kmax=kmax_ext)

    if method == 'DST':
        kmin, kmax = s.k_mpc_min / h, s.k_mpc_max / h
        k_mid = jnp.geomspace(kmin, kmax, s.n_mid) # h/Mpc
        k_mpc, basis = _dst_constants(s)
        # W, the part of ln(k P) the filter removes, summed at the table nodes (the samples k_mpc / h ascend)
        w = basis @ _removed_band(jnp.log(k_mpc * eval_pk(table, k_mpc / h, ascending=True)), s.n_min, s.n_max)
        pk_nw = eval_pk(table, k_mid) * jnp.exp(-w)

    elif method == 'SG':
        kmin, kmax, num = 1e-4, 1e1, 256
        window_length = 50
        poly_degree = 3

        k = jnp.geomspace(kmin, kmax, num)
        pk = eval_pk(table, k)

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
        pk = eval_pk(table, k)

        n = len(pk)
        D = _diff_mat(n, p)               # (n-p, n)
        A = jnp.eye(n) + lam * (D.T @ D)
        y = jnp.log(pk)
        z = jax.scipy.linalg.solve(A, y, assume_a='pos')
        pk_nw = jnp.exp(z)

        # ad-hoc adjustment at high k for extrapolation
        pk_nw = pk_nw.at[-int(num/5):].set(pk[-int(num/5):])

    if method != 'DST':
        # 'SG' and 'WH' smooth on their own grid k, resampled here onto the table nodes
        k_mid = jnp.geomspace(kmin, kmax, s.n_mid) # h/Mpc
        pk_nw = jnp.exp(spline.interp1d(jnp.log(k_mid), jnp.log(k), jnp.log(pk_nw)))

    # unsmoothed input outside the smoothed range
    k_low   = jnp.geomspace(kmin_ext, kmin, s.n_ext)[:-1]
    k_high  = jnp.geomspace(kmax, kmax_ext, s.n_ext)[1:]
    pk_low  = eval_pk(table, k_low)
    pk_high = eval_pk(table, k_high)

    k_extrap     = jnp.concatenate([k_low, k_mid, k_high], axis=0) # h/Mpc
    pk_nw_extrap = jnp.concatenate([pk_low, pk_nw, pk_high], axis=0)
    return jnp.stack([k_extrap, pk_nw_extrap], axis=0)

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


@lru_cache(maxsize=None)
def _dst_constants(s):
    """``(k_mpc, basis)`` of 'DST' with the settings ``s`` (numpy).

    ``k_mpc`` (1/Mpc): the ``n_samples`` midpoints of equal cells of the basis interval [a, b] (for
    ``n_samples = n_basis`` the nodes of the basis grid).  ``basis`` ``(n_mid, 2 (n_max - n_min))``: the
    orthonormal DST-II basis functions sqrt(2 / n_samples) sin(pi n (k - a) / (b - a)) of the removed harmonics n
    (in the order of :func:`_removed_band`) at the table nodes ``k_mid h``; zero rows where the input is kept.
    """
    dk = (s.k_mpc_max - s.k_mpc_min) / (s.n_basis - 1)
    a, b = s.k_mpc_min - dk / 2, s.k_mpc_max + dk / 2
    k_mpc = a + (np.arange(s.n_samples) + 0.5) * ((b - a) / s.n_samples)
    if k_mpc[0] <= 0:
        raise ValueError(f'DSTSettings.n_samples = {s.n_samples} puts the first sample at k <= 0 (the basis interval '
                         f'starts at {a:.3g} / Mpc)')
    rows = np.arange(s.n_min, s.n_max)
    harmonics = np.concatenate([2 * rows + 1, 2 * rows + 2])
    k_mid_mpc = np.geomspace(s.k_mpc_min, s.k_mpc_max, s.n_mid)
    basis = np.sqrt(2.0 / s.n_samples) * np.sin(np.pi * np.outer((k_mid_mpc - a) / (b - a), harmonics))
    basis[k_mid_mpc >= s.k_mpc_max - (s.n_keep_high - 1) * dk] = 0.0
    return k_mpc, basis


def _removed_band(y, n_min, n_max):
    """The coefficients Delta of the removed band of the samples ``y`` (``ln(k P)`` at ``k_mpc``).

    The orthonormal DST-II coefficients of ``y`` split into the odd (harmonic 2i + 1) and the even (2i + 2)
    series of the rows i; in each series Delta is the coefficients of the rows ``n_min <= i < n_max`` minus the
    cubic across that gap through the other rows (odd series first), the cubic of ``spline.interp1d`` as the fixed
    weights of ``spline.stencil``, which reach the two kept rows on each side of the gap.  The removed part of ``y`` is
    ``W(k) = sum_n Delta_n sqrt(2 / N) sin(pi n (k - a) / (b - a))``, ``ln P_nw = ln P - W``.
    """
    # the DST-II coefficient of harmonic m + 1 is the DCT-II coefficient N - 1 - m of y (-1)^j
    harms = _dct2_ortho(y * (-1)**jnp.arange(len(y)))[::-1]
    n = np.arange(1, len(harms) // 2 + 1)
    kept = np.concatenate([np.arange(n_min), np.arange(n_max, len(n))])     # the rows outside the gap
    # the cubic of spline.interp1d as fixed weights of the four kept rows next to the gap (spline.stencil)
    index, weight = spline.stencil(n[n_min:n_max], n[kept])
    delta = [series[n_min:n_max] - jnp.sum(weight * series[kept[index]], axis=-1)
             for series in (harms[0::2], harms[1::2])]
    return jnp.concatenate(delta)


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
