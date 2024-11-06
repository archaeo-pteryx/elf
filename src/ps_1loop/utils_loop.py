import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp
import quadax
import interpax
import re

kernel_to_decomp_dict = {
    # matter
    '22_dd': ['pk_lin nu=-0.3','pk_lin nu=-0.3'],
    '13_dd': ['pk_lin nu=-0.3','pk_lin nu=-0.3'],
    '22_dv': ['pk_lin nu=-0.3','pk_lin nu=-0.3'],
    '13_dv': ['pk_lin nu=-0.3','pk_lin nu=-0.3'],
    '22_vv': ['pk_lin nu=-0.3','pk_lin nu=-0.3'],
    '13_vv': ['pk_lin nu=-0.3','pk_lin nu=-0.3'],
    # biased tracer (Gaussian)
    'I_d2': ['pk_lin nu=-1.6','pk_lin nu=-1.6'],
    'I_G2': ['pk_lin nu=-1.6','pk_lin nu=-1.6'],
    'I_d2_d2': ['pk_lin nu=-1.6','pk_lin nu=-1.6'],
    'I_d2_G2': ['pk_lin nu=-1.6','pk_lin nu=-1.6'],
    'I_G2_G2': ['pk_lin nu=-1.6','pk_lin nu=-1.6'],
    'F_G2': ['pk_lin nu=-1.6','pk_lin nu=-1.6'],
    'I_d2_v': ['pk_lin nu=-1.6','pk_lin nu=-1.6'],
    'I_s2_v': ['pk_lin nu=-1.6','pk_lin nu=-1.6'],
    # local PNG
    'I_phi': ['pk_1phi nu=-1.6','pk_lin nu=-0.3'],
    'I_phi-d': ['pk_1phi nu=-1.6','pk_lin nu=-1.6'],
    'I_d2_phi': ['pk_1phi nu=-1.6','pk_lin nu=-1.6'],
    'I_G2_phi': ['pk_1phi nu=-1.6','pk_lin nu=-1.6'],
    'I_d2_phi-d': ['pk_1phi nu=-1.6','pk_lin nu=-1.6'],
    'I_G2_phi-d': ['pk_1phi nu=-1.6','pk_lin nu=-1.6'],
    'F_phi': ['pk_1phi nu=-1.6','pk_lin nu=-1.6'],
    'F_G2_LPNG': ['pk_lin nu=-1.6','pk_1phi nu=-1.6'],
    'I_phi_tilde': ['pk_1phi nu=-2.1','pk_1phi nu=-2.1'],
    'I_phi_tilde_d2': ['pk_1phi nu=-2.1','pk_1phi nu=-2.1'],
    'I_phi_tilde_G2': ['pk_1phi nu=-2.1','pk_1phi nu=-2.1'],
    # biased tracer in redshift space (Gaussian)
    '22': ['pk_lin nu=-0.7','pk_lin nu=-0.7'],
    '13': ['pk_lin nu=-0.7','pk_lin nu=-0.7'],
    # biased tracer in redshift space (local PNG contribution)
    '12_lpng1': ['pk_1phi nu=-1.6','Mk nu=0.2'],
    '12_lpng2': ['pk_1phi nu=-2.1','pk_1phi nu=-2.1'],
    '22_lpng': ['pk_lin nu=-0.7','pk_1phi nu=-0.9'],
    '13_lpng1': ['pk_lin nu=-0.7','pk_1phi nu=-0.9'],
    '13_lpng3': ['pk_1phi nu=-1.6','pk_lin nu=-1.6'],
}

def get_degree_info(name):
    degree_name = re.split('=', name)[-1]
    str_list = re.split('_', degree_name)
    degree_dict = {}
    for s in str_list:
        string = re.split('-', s)
        key = string[0]
        val = int(string[1])
        if (not key in ['f','mu']) and (val == 0):
            continue
        degree_dict[key] = val
    nf = degree_dict['f']
    nmu = degree_dict['mu']
    _ = degree_dict.pop('f')
    _ = degree_dict.pop('mu')
    return nf, nmu, degree_dict

def get_bias_factor(degree_dict, bias1, bias2):
    keys = list(degree_dict.keys())

    if len(degree_dict) == 0:
        bias_factor = 1.
    elif len(degree_dict) == 1:
        key = keys[0]
        if degree_dict[key] == 1:
            bias_factor = (bias1[key] + bias2[key]) / 2.
        elif degree_dict[key] == 2:
            bias_factor = bias1[key] * bias2[key]
    elif len(degree_dict) == 2:
        key1 = keys[0]
        key2 = keys[1]
        bias_factor = (bias1[key1] * bias2[key2] + bias2[key1] * bias1[key2]) / 2.
    else:
        raise ValueError('Invalid numbers of bias parameters.')
    
    return bias_factor

@jit
def get_pk(k, pk_data, kmin=1e-4, kmax=1e4):
    k_extrap, pk_extrap = get_log_extrap(pk_data['k'], pk_data['pk'], kmin, kmax)
    pk = jnp.exp(interpax.interp1d(jnp.log(k), jnp.log(k_extrap), jnp.log(pk_extrap), method='cubic'))
    return pk

@partial(jit, static_argnames=['num'])
def get_pk_int(pk_data, kmin=1e-4, kmax=1e4, num=1000):
    q = jnp.geomspace(kmin, kmax, num)
    res = quadax.simpson(q * get_pk(q, pk_data, kmin, kmax), x=jnp.log(q)) / (2 * jnp.pi**2)
    return res

@partial(jit, static_argnames=['num_extrap'])
def get_log_extrap(x, y, xmin, xmax, num_extrap=10):

    dlnx_low = jnp.log(x[1] / x[0])
    dlny_low = jnp.log(y[1] / y[0])
    num_low = (jnp.log(x[0] / xmin) / dlnx_low).astype(int) + 1

    x_low = x[0] * jnp.exp(dlnx_low * num_low / num_extrap * jnp.arange(-num_extrap, 0))
    y_low = y[0] * jnp.exp(dlny_low * num_low / num_extrap * jnp.arange(-num_extrap, 0))

    dlnx_high = jnp.log(x[-1] / x[-2])
    dlny_high = jnp.log(y[-1] / y[-2])
    num_high = (jnp.log(xmax / x[-1]) / dlnx_high).astype(int) + 1

    x_high = x[-1] * jnp.exp(dlnx_high * num_high / num_extrap * jnp.arange(1, num_extrap+1))
    y_high = y[-1] * jnp.exp(dlny_high * num_high / num_extrap * jnp.arange(1, num_extrap+1))

    x_extrap = jnp.hstack((x_low, x, x_high))
    y_extrap = jnp.hstack((y_low, y, y_high))
    
    return x_extrap, y_extrap
