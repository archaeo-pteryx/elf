import jax.numpy as jnp

from . import spline

def get_pk(k, pk_data, kmin=1e-4, kmax=1e4):
    x, y = pk_data[0], pk_data[1]
    x_low, x_high = _log_extrap_nodes(x, kmin, kmax)
    y_low, y_high = get_log_extrap(x, y, x_low, x_high)
    k_extrap = jnp.concatenate([x_low, x, x_high], axis=0)
    pk_extrap = jnp.concatenate([y_low, y, y_high], axis=0)
    pk = spline.interp1d(jnp.log(k), jnp.log(k_extrap), pk_extrap)
    return pk

def get_log_extrap(x, y, x_low, x_high):
    """``(y_low, y_high)``: ``y`` continued to ``x_low``/``x_high`` by its end log-slopes.

    ``y(x) = y_end (x / x_end)^s``, ``s = ln(y_hi / y_lo) / ln(x_hi / x_lo)``; zero unless both end values are positive.
    """
    def continuation(y_lo, y_hi, y_end, steps):
        den_ok = y_lo > 0.0
        # inner where: never divide by a non-positive (possibly subnormal) node
        raw = y_hi / jnp.where(den_ok, y_lo, 1.0)
        ok = den_ok & (raw > 0.0)
        # outer where: discard the replaced branch, value and gradient alike
        ratio = jnp.where(ok, raw, 1.0)
        amp = jnp.where(ok, y_end, 0.0)
        return amp * jnp.exp(jnp.log(ratio) * steps)

    dlnx_low = jnp.log(x[1] / x[0])
    dlnx_high = jnp.log(x[-1] / x[-2])

    y_dtype = y.dtype
    steps_low = (jnp.log(x_low / x[0]) / dlnx_low).astype(y_dtype)
    steps_high = (jnp.log(x_high / x[-1]) / dlnx_high).astype(y_dtype)

    expand = (None,) * (y.ndim - 1)
    y_lo, y_hi = y[..., 0:1], y[..., 1:2]
    y_low = continuation(y_lo, y_hi, y_lo, steps_low[expand])
    y_lo, y_hi = y[..., -2:-1], y[..., -1:]
    y_high = continuation(y_lo, y_hi, y_hi, steps_high[expand])
    return y_low, y_high

def _log_extrap_nodes(x, xmin, xmax, num=10):
    """``num`` log-spaced nodes per side reaching ``xmin``/``xmax`` (at least one native spacing out).

    They move continuously with ``x``, so results stay smooth in ``h`` when ``x = k/h``.
    """
    if num < 1:
        raise ValueError("num must be a positive integer")

    dtype = x.dtype
    xmin = jnp.asarray(xmin, dtype)
    xmax = jnp.asarray(xmax, dtype)

    dlnx_low  = jnp.log(x[1] / x[0])
    dlnx_high = jnp.log(x[-1] / x[-2])

    # clamp to one native spacing: the nodes must stay sorted for searchsorted
    s_low  = jnp.maximum(jnp.log(x[0] / xmin),  dlnx_low)
    s_high = jnp.maximum(jnp.log(xmax / x[-1]), dlnx_high)

    t = jnp.arange(1, num + 1, dtype=dtype) / jnp.asarray(num, dtype)

    x_low  = (x[0]  * jnp.exp(-s_low * t))[::-1]
    x_high = x[-1] * jnp.exp(s_high * t)
    return x_low, x_high

def cross_bias_factor(bias_pow, bias_a, bias_b):
    """Cross-symmetrised bias monomials ``(max_len,)`` of powers ``bias_pow`` (total degree <= 2)."""
    deg = jnp.sum(bias_pow, axis=1)  # (max_len,)

    fac0 = jnp.ones_like(deg, dtype=bias_a.dtype)

    # degree 1: b_i -> (b_i^A + b_i^B)/2
    lin_a = jnp.sum(bias_pow * bias_a[None, :], axis=1)
    lin_b = jnp.sum(bias_pow * bias_b[None, :], axis=1)
    fac1 = 0.5 * (lin_a + lin_b)

    # degree 2: b_i^2 -> b_i^A b_i^B
    same = jnp.sum((bias_pow == 2) * (bias_a * bias_b)[None, :], axis=1)

    # case b_i b_j -> 1/2 (b_i^A b_j^B + b_i^B b_j^A)
    outer_ab = bias_a[:, None] * bias_b[None, :]
    outer_ba = bias_b[:, None] * bias_a[None, :]
    sym_outer = 0.5 * (outer_ab + outer_ba)

    pair_mask = (bias_pow[:, :, None] == 1) & (bias_pow[:, None, :] == 1)
    upper = jnp.triu(jnp.ones((bias_a.size, bias_a.size), dtype=bool), k=1)
    mixed = jnp.sum(pair_mask * upper[None, :, :] * sym_outer[None, :, :], axis=(1, 2))

    fac2 = same + mixed

    return jnp.where(deg == 0, fac0, jnp.where(deg == 1, fac1, fac2))

def eval_power_coeffs(degrees, f, bias_a, bias_b):
    """``(mu_pow, f^p x bias factor)`` of rows of powers of (mu, f, b1, b2, bG2, bGamma3)."""
    mu_pow   = degrees[:, 0].astype(jnp.int32)
    f_pow    = degrees[:, 1].astype(jnp.int32)
    bias_pow = degrees[:, 2:].astype(jnp.int32)

    coeffs = (f ** f_pow) * cross_bias_factor(bias_pow, bias_a, bias_b)
    return mu_pow, coeffs
