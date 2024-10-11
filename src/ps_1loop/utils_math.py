import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=0)
def get_legendre(n, x):
    x = jnp.atleast_1d(x).astype(float)
    if n == 0:
        return jnp.ones(x.shape)
    elif n == 1:
        return x
    else:
        l1 = get_legendre(n - 1, x)
        l2 = get_legendre(n - 2, x)
        return ((2 * n - 1) * x * l1 - (n - 1) * l2) / n

@partial(jax.jit, static_argnums=0)
def spherical_jn(n, x):
    x = jnp.atleast_1d(x).astype(float)
    if n == 0:
        return jnp.sin(x) / x
    elif n == 1:
        return (jnp.sin(x) / (x**2)) - (jnp.cos(x) / x)
    else:
        jn1 = spherical_jn(n - 1, x)
        jn2 = spherical_jn(n - 2, x)
        return ((2 * n - 1) / x) * jn1 - jn2
