import jax
# jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import re

from . import spline

_NU_GROUP_MATTER_TAG = -0.3
_NU_GROUP_BIAS_TAG = -1.6

def get_nu_group_tag_from_name(name):
    """Return the matrix-path grouping tag for a PT kernel name.

    This value is used to assign terms to matter-like and bias-like matrix
    groups. It is not necessarily the FFTLog bias used to build the matrix;
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
    k_extrap, pk_extrap = get_log_extrap(pk_data[0], pk_data[1], kmin, kmax)
    pk = spline.interp1d(jnp.log(k), jnp.log(k_extrap), pk_extrap)
    return pk

def get_pk_int(pk_data, kmin=1e-4, kmax=1e4, num=1000):
    q = jnp.geomspace(kmin, kmax, num)
    integrand = q * get_pk(q, pk_data, kmin * 0.1, kmax * 10.)
    res = jnp.trapezoid(integrand, x=jnp.log(q)) / (2 * jnp.pi**2)
    return res

def get_pk_int2(pk_data, kmin=1e-4, kmax=1e4, num=1000):
    q = jnp.geomspace(kmin, kmax, num)
    integrand = q**3 * get_pk(q, pk_data, kmin * 0.1, kmax * 10.)**2
    res = jnp.trapezoid(integrand, x=jnp.log(q)) / (2 * jnp.pi**2)
    return res

def get_log_extrap(x, y, xmin, xmax, num_extrap=10):
    x_dtype = x.dtype
    y_dtype = y.dtype

    xmin = jnp.asarray(xmin, x_dtype)
    xmax = jnp.asarray(xmax, x_dtype)

    dlnx_low  = jnp.log(x[1] / x[0])
    dlnx_high = jnp.log(x[-1] / x[-2])

    num_low  = (jnp.log(x[0] / xmin) / dlnx_low).astype(jnp.int32) + 1
    num_high = (jnp.log(xmax / x[-1]) / dlnx_high).astype(jnp.int32) + 1

    fac_low  = num_low.astype(x_dtype)  / jnp.asarray(num_extrap, x_dtype)
    fac_high = num_high.astype(x_dtype) / jnp.asarray(num_extrap, x_dtype)

    t_low  = jnp.arange(-num_extrap, 0, dtype=x_dtype)
    t_high = jnp.arange(1, num_extrap + 1, dtype=x_dtype)

    x_low  = x[0]  * jnp.exp(dlnx_low  * fac_low  * t_low)
    x_high = x[-1] * jnp.exp(dlnx_high * fac_high * t_high)

    def _low_true(_):
        den   = jnp.where(y[0] == 0, jnp.inf, y[0])
        ratio = y[1] / den
        ratio = jnp.where(ratio <= 0, jnp.asarray(1.0, y_dtype), ratio)  # log(1)=0
        growth = jnp.exp(jnp.log(ratio) * (num_low.astype(y_dtype) / jnp.asarray(num_extrap, y_dtype)) * t_low.astype(y_dtype))
        return y[0] * growth

    def _low_false(_):
        return jnp.zeros((num_extrap,), dtype=y_dtype)

    y_low = jax.lax.cond(y[0] > 0, _low_true, _low_false, operand=None)

    def _high_true(_):
        den   = jnp.where(y[-2] == 0, jnp.inf, y[-2])
        ratio = y[-1] / den
        ratio = jnp.where(ratio <= 0, jnp.asarray(1.0, y_dtype), ratio)
        growth = jnp.exp(jnp.log(ratio) * (num_high.astype(y_dtype) / jnp.asarray(num_extrap, y_dtype)) * t_high.astype(y_dtype))
        return y[-1] * growth

    def _high_false(_):
        return jnp.zeros((num_extrap,), dtype=y_dtype)

    y_high = jax.lax.cond(y[-1] > 0, _high_true, _high_false, operand=None)

    x_extrap = jnp.concatenate([x_low, x, x_high], axis=0)
    y_extrap = jnp.concatenate([y_low, y, y_high], axis=0)
    return x_extrap, y_extrap

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
