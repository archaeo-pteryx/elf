import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
from jax.scipy.special import gamma
from functools import partial

@jit
def get_hypergerom_2F1(m, n, z, coeff):
    res = jnp.array([coeff[m][n, i] * z**i for i in range(m+1)])
    return jnp.sum(res, axis=0)

@partial(jit, static_argnames=['nmax'])
def get_G_0_m(m, nmax, A, B, C, coeff):
    n_list = jnp.arange(m, nmax+1)
    f_nm = gamma(m + n_list + 1/2) / (gamma(m + 1) * gamma(n_list + 1/2) * gamma(1 - m + n_list))
    
    rho2 = A**2 + C**2
    z = rho2 / C**2
    x = rho2 / A**2
    hypergerom = jnp.array([(1 - x)**n * get_hypergerom_2F1(m, n, z, coeff) for n in n_list])
    res = jnp.array([f_nm[n] * (B * A**2 / rho2)**n * hypergerom[n] for n in n_list])

    return jnp.sum(res, axis=0)

