import jax
# jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
import re

from . import spline

kernel_to_decomp_dict = {
    # matter
    '22_dd': -0.3,
    '13_dd': -0.3,
    # biased tracer
    'I_d2': -1.6,
    'I_G2': -1.6,
    'I_d2_d2': -1.6,
    'I_d2_G2': -1.6,
    'I_G2_G2': -1.6,
    'F_G2': -1.6,
}

def get_nu_from_name(name):
    if name in kernel_to_decomp_dict.keys():
        return kernel_to_decomp_dict[name]
    else:
        degree_dict = get_degree_dict(name)
        if 'b2' in degree_dict.keys():
            if degree_dict['b2'] > 0: return -1.6
        if 'bG2' in degree_dict.keys():
            if degree_dict['bG2'] > 0: return -1.6
        if 'bGamma3' in degree_dict.keys():
            if degree_dict['bGamma3'] > 0: return -1.6
        return -0.3

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
