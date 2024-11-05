import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp

@jit
def legendre(n, x):
    x = jnp.atleast_1d(x).astype(float)
    res = jax.lax.cond(n == 0, 
                       lambda: jnp.ones(x.shape), 
                       lambda: jax.lax.cond(n == 2, 
                                            lambda: 1.5 * x**2 - 0.5, 
                                            lambda: 4.375 * x**4 - 3.75 * x**2 + 0.375))
    return res

@jit
def spherical_jn(n, x):
    x = jnp.atleast_1d(x).astype(float)
    res = jax.lax.cond(n == 0, 
                       lambda: jnp.sin(x) / x, 
                       lambda: jax.lax.cond(n == 1, 
                                            lambda: (jnp.sin(x) - x * jnp.cos(x)) / x**2, 
                                            lambda: ((3 - x**2) * jnp.sin(x) - 3 * x * jnp.cos(x)) / x**3))
    return res