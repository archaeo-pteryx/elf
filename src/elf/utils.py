import jax
import jax.numpy as jnp
import re

from . import spline
from .fftlog import get_log_extrap  # noqa: F401 (re-exported: ir_resum.py and tests import it from here)

_NU_GROUP_MATTER_TAG = -0.3
_NU_GROUP_BIAS_TAG = -1.6

def get_nu_group_tag_from_name(name):
    """Return the matrix-path grouping tag for a PT kernel name.

    This value is used to assign terms to matter-like and bias-like matrix groups. 
    It is not necessarily the FFTlog bias used to build the matrix;
    EPT P22 bias groups are currently recomputed with a nu_override.
    """
    degree_dict = get_degree_dict(name)
    if 'b2' in degree_dict.keys():
        if degree_dict['b2'] > 0: return _NU_GROUP_BIAS_TAG
    if 'bG2' in degree_dict.keys():
        if degree_dict['bG2'] > 0: return _NU_GROUP_BIAS_TAG
    if 'bGamma3' in degree_dict.keys():
        if degree_dict['bGamma3'] > 0: return _NU_GROUP_BIAS_TAG
    return _NU_GROUP_MATTER_TAG

def get_degree_dict(name):
    degree_name = re.split('=', name)[-1]
    str_list = re.split('_', degree_name)
    degree_dict = {}
    for s in str_list:
        string = re.split('-', s)
        key = string[0]
        val = int(string[1])
        degree_dict[key] = val
    return degree_dict

def get_pk(k, pk_data, kmin=1e-4, kmax=1e4):
    x, y = pk_data[0], pk_data[1]
    x_low, x_high = _log_extrap_nodes(x, kmin, kmax)
    y_low, y_high = get_log_extrap(x, y, x_low, x_high)
    k_extrap = jnp.concatenate([x_low, x, x_high], axis=0)
    pk_extrap = jnp.concatenate([y_low, y, y_high], axis=0)
    pk = spline.interp1d(jnp.log(k), jnp.log(k_extrap), pk_extrap)
    return pk

def _log_extrap_nodes(x, xmin, xmax, num=10):
    """Ascending extrapolation nodes below ``x[0]`` and above ``x[-1]``.

    The returned arrays always contain ``num`` points on either side of the
    input.  Keeping this size static is important when the function is traced
    by JAX.  ``xmin`` and ``xmax`` specify how far the padding should reach
    when they lie outside the input interval; when a requested bound is
    already inside the interval, one native endpoint spacing is used instead.

    The padding spans are continuous functions of the endpoints: the low side
    covers ``log(x[0]/xmin)`` in ``num`` equal logarithmic steps (and likewise
    ``log(xmax/x[-1])`` on the high side), rather than a spacing quantised to
    an integer count of native steps.  The padded grid -- and any function
    evaluated on it, and their derivatives -- therefore vary smoothly with the
    input grid, e.g. with the ``modes/h`` grid an emulator hands over, where
    ``h`` shifts every node continuously.  The only switch left is the ``max``
    below, which engages just when a requested bound lies within one native
    spacing of the data.

    ``x`` must be positive and strictly increasing.  Shared by
    :func:`get_pk` and ``ir_resum.get_pk_nw``, the two callers that need this
    smoothly-varying node placement (as opposed to ``fftlog.pad``, whose
    padded nodes sit at a fixed integer number of native log-spacings from the
    grid edge).
    """
    if num < 1:
        raise ValueError("num must be a positive integer")

    dtype = x.dtype
    xmin = jnp.asarray(xmin, dtype)
    xmax = jnp.asarray(xmax, dtype)

    dlnx_low  = jnp.log(x[1] / x[0])
    dlnx_high = jnp.log(x[-1] / x[-2])

    # The output shape is deliberately static, so padding is also added when the requested interval is narrower than the data interval.
    # In that case the raw span below is zero or negative.
    # Clamp it to one native log-spacing;
    # otherwise x_low/x_high would run *into* the input interval (and x_high would be descending), violating the sorted-grid contract of spline.interp1d/searchsorted.
    s_low  = jnp.maximum(jnp.log(x[0] / xmin),  dlnx_low)
    s_high = jnp.maximum(jnp.log(xmax / x[-1]), dlnx_high)

    # Equal logarithmic steps across the requested span, so the padded nodes move
    # continuously with the endpoints instead of jumping when an integer step count changes.
    t = jnp.arange(1, num + 1, dtype=dtype) / jnp.asarray(num, dtype)

    x_low  = (x[0]  * jnp.exp(-s_low * t))[::-1]   # ascending; x_low[0] sits at xmin (or one native spacing below x[0])
    x_high = x[-1] * jnp.exp(s_high * t)
    return x_low, x_high

def cross_bias_factor(bias_pow, bias_a, bias_b):
    """
    bias_pow: (max_len, nbias)
    bias_a, bias_b: (nbias,)
    assumes total bias degree <= 2
    """
    deg = jnp.sum(bias_pow, axis=1)  # (max_len,)

    # degree 0
    fac0 = jnp.ones_like(deg, dtype=bias_a.dtype)

    # degree 1: b_i -> (b_i^A + b_i^B)/2
    lin_a = jnp.sum(bias_pow * bias_a[None, :], axis=1)
    lin_b = jnp.sum(bias_pow * bias_b[None, :], axis=1)
    fac1 = 0.5 * (lin_a + lin_b)

    # degree 2
    # case b_i^2 -> b_i^A b_i^B
    same = jnp.sum((bias_pow == 2) * (bias_a * bias_b)[None, :], axis=1)

    # case b_i b_j -> 1/2 (b_i^A b_j^B + b_i^B b_j^A)
    outer_ab = bias_a[:, None] * bias_b[None, :]
    outer_ba = bias_b[:, None] * bias_a[None, :]
    sym_outer = 0.5 * (outer_ab + outer_ba)

    # mask selects pairs i<j with pow_i=pow_j=1
    pair_mask = (bias_pow[:, :, None] == 1) & (bias_pow[:, None, :] == 1)
    upper = jnp.triu(jnp.ones((bias_a.size, bias_a.size), dtype=bool), k=1)
    mixed = jnp.sum(pair_mask * upper[None, :, :] * sym_outer[None, :, :], axis=(1, 2))

    fac2 = same + mixed

    return jnp.where(deg == 0, fac0, jnp.where(deg == 1, fac1, fac2))

def eval_power_coeffs(degrees, f, bias_a, bias_b):
    """
    degrees columns:
    0: mu power
    1: f power
    2: b1 power
    3: b2 power
    4: bG2 power
    5: bGamma3 power
    """
    mu_pow   = degrees[:, 0].astype(jnp.int32)
    f_pow    = degrees[:, 1].astype(jnp.int32)
    bias_pow = degrees[:, 2:].astype(jnp.int32)

    coeffs = (f ** f_pow) * cross_bias_factor(bias_pow, bias_a, bias_b)
    return mu_pow, coeffs
