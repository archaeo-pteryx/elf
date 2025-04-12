import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
from functools import partial

@jit
def get_G00(A, B, C, coeff):
    x = - B
    y = C**2 / (A**2 + C**2)
    
    res = 0
    l = coeff.shape[0] - 1
    for i in range(l+1):
        for k in range(l+1):
            res = res + coeff[k, i] * x**(l + i) * y**(l + i - k)
    res = res * jnp.exp(x * y)
    
    return res

@jit
def get_G10(A, B, C, coeffs):
    return get_dG00dA(A, B, C, coeffs[0]) + 0.5 * A * get_G00(A, B, C, coeffs[1])

@jit
def get_G20(A, B, C, coeffs):
    return get_d2G00dA2(A, B, C, coeffs[0]) + A * get_dG00dA(A, B, C, coeffs[1]) + 0.5 * get_G00(A, B, C, coeffs[1]) + 0.25 * A**2 * get_G00(A, B, C, coeffs[2])

@jit
def get_G30(A, B, C, coeffs):
    return get_d3G00dA3(A, B, C, coeffs[0]) + 1.5 * A * get_d2G00dA2(A, B, C, coeffs[1]) + 1.5 * get_dG00dA(A, B, C, coeffs[1]) \
            + 0.75 * A**2 * get_dG00dA(A, B, C, coeffs[2]) + 0.75 * A * get_G00(A, B, C, coeffs[2]) + 0.125 * A**3 * get_G00(A, B, C, coeffs[3])

def nth_derivative(f, n, argnums=0):
    df = f
    for _ in range(n):
        df = jax.grad(df, argnums)
    return df
    
get_dG00dA = jit(jax.vmap(nth_derivative(get_G00, 1, argnums=0), in_axes=(0, 0, 0, None)))
get_dG00dC = jit(jax.vmap(nth_derivative(get_G00, 1, argnums=2), in_axes=(0, 0, 0, None)))

get_d2G00dA2 = jit(jax.vmap(nth_derivative(get_G00, 2, argnums=0), in_axes=(0, 0, 0, None)))
get_d2G00dC2 = jit(jax.vmap(nth_derivative(get_G00, 2, argnums=2), in_axes=(0, 0, 0, None)))

get_d3G00dA3 = jit(jax.vmap(nth_derivative(get_G00, 3, argnums=0), in_axes=(0, 0, 0, None)))

get_d4G00dA4 = jit(jax.vmap(nth_derivative(get_G00, 4, argnums=0), in_axes=(0, 0, 0, None)))

@jit
def get_G_0_0(x, y, coeff):
    res = 0
    m = len(coeff) - 1
    for i in range(m+1):
        for k in range(m+1):
            res = res + coeff[k, i] * x**(m + i) * y**(m + i - k)
    res = res * jnp.exp(x * y)
    return res

@jit
def get_cs_mu(f, mu):
    denom = jnp.sqrt(1 + f * (2 + f) * mu**2)
    c = (1 + f * mu**2) / denom
    s = f * mu * jnp.sqrt(1 - mu**2) / denom
    return c, s
