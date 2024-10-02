import jax
import jax.numpy as jnp
import re

kernel_to_decomp_dict = {
    # matter
    '22_dd': ['plin nu=-0.3','plin nu=-0.3'],
    '13_dd': ['plin nu=-0.3','plin nu=-0.3'],
    '22_dv': ['plin nu=-0.3','plin nu=-0.3'],
    '13_dv': ['plin nu=-0.3','plin nu=-0.3'],
    '22_vv': ['plin nu=-0.3','plin nu=-0.3'],
    '13_vv': ['plin nu=-0.3','plin nu=-0.3'],
    # biased tracer (Gaussian)
    'I_d2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_d2_d2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_d2_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_G2_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'F_G2': ['plin nu=-1.6','plin nu=-1.6'],
    'I_d2_v': ['plin nu=-1.6','plin nu=-1.6'],
    'I_s2_v': ['plin nu=-1.6','plin nu=-1.6'],
    # local PNG
    'I_phi': ['p1phi nu=-1.6','plin nu=-0.3'],
    'I_phi-d': ['p1phi nu=-1.6','plin nu=-1.6'],
    'I_d2_phi': ['p1phi nu=-1.6','plin nu=-1.6'],
    'I_G2_phi': ['p1phi nu=-1.6','plin nu=-1.6'],
    'I_d2_phi-d': ['p1phi nu=-1.6','plin nu=-1.6'],
    'I_G2_phi-d': ['p1phi nu=-1.6','plin nu=-1.6'],
    'F_phi': ['p1phi nu=-1.6','plin nu=-1.6'],
    'F_G2_LPNG': ['plin nu=-1.6','p1phi nu=-1.6'],
    'I_phi_tilde': ['p1phi nu=-2.1','p1phi nu=-2.1'],
    'I_phi_tilde_d2': ['p1phi nu=-2.1','p1phi nu=-2.1'],
    'I_phi_tilde_G2': ['p1phi nu=-2.1','p1phi nu=-2.1'],
    # biased tracer in redshift space (Gaussian)
    '22_gg': ['plin nu=-0.7','plin nu=-0.7'],
    '13_gg': ['plin nu=-0.7','plin nu=-0.7'],
    # biased tracer in redshift space (local PNG contribution)
    '22_gg_lpng': ['plin nu=-0.7','p1phi nu=-0.9'],
    '13_gg_lpng1': ['plin nu=-0.7','p1phi nu=-0.9'],
    '13_gg_lpng3': ['p1phi nu=-1.6','plin nu=-1.6'],
    '12_gg_lpng1': ['p1phi nu=-1.6','M nu=0.2'],
    '12_gg_lpng2': ['p1phi nu=-2.1','p1phi nu=-2.1']
}

def get_deg_info(name):
    deg_name = re.split('=', name)[-1]
    str_list = re.split('_', deg_name)
    deg_dict = {}
    for s in str_list:
        string = re.split('-', s)
        key = string[0]
        val = int(string[1])
        if (not key in ['f','mu']) and (val == 0):
            continue
        deg_dict[key] = val
    nf = deg_dict['f']
    nmu = deg_dict['mu']
    _ = deg_dict.pop('f')
    _ = deg_dict.pop('mu')
    return nf, nmu, deg_dict

@jax.jit
def get_log_extrap(x, y, xmin, xmax):
    num_extrap = 10 # JIT compilation requires an array to have a fixed size.
    
    dlnx_low = jnp.log(x[1] / x[0])
    dlny_low = jnp.log(y[1] / y[0])
    num_low = (jnp.log(x[0] / xmin) / dlnx_low).astype(int)
    num_low = jax.lax.cond(num_low <= 0, lambda x: 1, lambda x: x, num_low)

    x_low = x[0] * jnp.exp(dlnx_low * num_low / num_extrap * jnp.arange(-num_extrap, 0))
    y_low = y[0] * jnp.exp(dlny_low * num_low / num_extrap * jnp.arange(-num_extrap, 0))

    dlnx_high= jnp.log(x[-1] / x[-2])
    dlny_high = jnp.log(y[-1] / y[-2])
    num_high = (jnp.log(xmax / x[-1]) / dlnx_high).astype(int)
    num_high = jax.lax.cond(num_high <= 0, lambda x: 1, lambda x: x, num_high)

    x_high = x[-1] * jnp.exp(dlnx_high * num_high / num_extrap * jnp.arange(1, num_extrap+1))
    y_high = y[-1] * jnp.exp(dlny_high * num_high / num_extrap * jnp.arange(1, num_extrap+1))

    x_extrap = jnp.hstack((x_low, x, x_high))
    y_extrap = jnp.hstack((y_low, y, y_high))
    
    return x_extrap, y_extrap
