import jax.numpy as jnp
from jax.scipy.special import lpmn

def get_legendre(n, x):
    x = jnp.atleast_1d(x)
    return lpmn(n, n, x)[0][0, -1]

def spherical_jn(n, x):
    x = jnp.atleast_1d(x)
    if n == 0:
        return jnp.sin(x) / x
    elif n == 1:
        return jnp.sin(x) / x**2 - jnp.cos(x) / x
    elif n == 2:
        return (3 / x**3 - 1 / x) * jnp.sin(x) - (3 / x**2) * jnp.cos(x)
    else:
        raise NotImplementedError
