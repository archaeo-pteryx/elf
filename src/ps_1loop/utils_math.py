import jax.numpy as jnp

def eval_legendre(n, x):
    x = jnp.atleast_1d(x)
    if n == 0:
        return jnp.ones(len(x))
    elif n == 1:
        return x
    elif n == 2:
        return (3 * x**2 - 1) / 2
    elif n == 3:
        return (5 * x**3 - 3 * x) / 2
    elif n == 4:
        return (35 * x**4 - 30 * x**2 + 3) / 8
    else:
        raise NotImplementedError

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
