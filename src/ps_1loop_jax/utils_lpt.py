import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp

@jit
def get_G00(A, B, C, coeffs):
    coeff = coeffs[0]

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
    return get_dG00dA(A, B, C, [coeffs[0]]) + 0.5 * A * get_G00(A, B, C, [coeffs[1]])

@jit
def get_G20(A, B, C, coeffs):
    return get_d2G00dA2(A, B, C, [coeffs[0]]) + A * get_dG00dA(A, B, C, [coeffs[1]]) \
        + 0.5 * get_G00(A, B, C, [coeffs[1]]) + 0.25 * A**2 * get_G00(A, B, C, [coeffs[2]])

@jit
def get_G30(A, B, C, coeffs):
    return get_d3G00dA3(A, B, C, [coeffs[0]]) + 1.5 * A * get_d2G00dA2(A, B, C, [coeffs[1]]) \
        + 1.5 * get_dG00dA(A, B, C, [coeffs[1]]) + 0.75 * A**2 * get_dG00dA(A, B, C, [coeffs[2]]) \
        + 0.75 * A * get_G00(A, B, C, [coeffs[2]]) + 0.125 * A**3 * get_G00(A, B, C, [coeffs[3]])

@jit
def get_G40(A, B, C, coeffs):
    return get_d4G00dA4(A, B, C, [coeffs[0]]) + 2 * A * get_d3G00dA3(A, B, C, [coeffs[1]]) \
        + 3 * get_d2G00dA2(A, B, C, [coeffs[1]]) + 1.5 * A**2 * get_d2G00dA2(A, B, C, [coeffs[2]]) \
        + 3 * A * get_dG00dA(A, B, C, [coeffs[2]]) + 0.5 * A**3 * get_dG00dA(A, B, C, [coeffs[3]]) \
        + 0.75 * get_G00(A, B, C, [coeffs[2]]) + 0.75 * A**2 * get_G00(A, B, C, [coeffs[3]]) \
        + 1/16 * A**4 * get_G00(A, B, C, [coeffs[4]])

@jit
def get_G01(A, B, C, coeffs):
    return get_dG00dC(A, B, C, [coeffs[0]]) + 0.5 * C * get_G00(A, B, C, [coeffs[1]])

@jit
def get_G02(A, B, C, coeffs):
    return get_d2G00dC2(A, B, C, [coeffs[0]]) + C * get_dG00dC(A, B, C, [coeffs[1]]) \
        + 0.5 * get_G00(A, B, C, [coeffs[1]]) + 0.25 * C**2 * get_G00(A, B, C, [coeffs[2]])

@jit
def get_G11(A, B, C, coeffs):
    return get_d2G00dAdC(A, B, C, [coeffs[0]]) + 0.5 * C * get_dG00dA(A, B, C, [coeffs[1]]) \
        + 0.5 * A * get_dG00dC(A, B, C, [coeffs[1]]) + 0.25 * A * C * get_G00(A, B, C, [coeffs[2]])

@jit
def get_G21(A, B, C, coeffs):
    return get_d3G00dA2dC(A, B, C, [coeffs[0]]) + 0.5 * C * get_d2G00dA2(A, B, C, [coeffs[1]]) \
        + A * get_d2G00dAdC(A, B, C, [coeffs[1]]) + 0.5 * get_dG00dC(A, B, C, [coeffs[1]]) \
        + 0.5 * A * C * get_dG00dA(A, B, C, [coeffs[2]]) + 0.25 * A**2 * get_dG00dC(A, B, C, [coeffs[2]]) \
        + 0.25 * C * get_G00(A, B, C, [coeffs[2]]) + 0.125 * A**2 * C * get_G00(A, B, C, [coeffs[3]])

def nth_derivative(f, n, argnums=0):
    df = f
    for _ in range(n):
        df = jax.grad(df, argnums)
    return df

get_dG00dA = jit(jax.vmap(nth_derivative(get_G00, 1, argnums=0), in_axes=(0, 0, 0, None)))
get_dG00dC = jit(jax.vmap(nth_derivative(get_G00, 1, argnums=2), in_axes=(0, 0, 0, None)))

get_d2G00dA2 = jit(jax.vmap(nth_derivative(get_G00, 2, argnums=0), in_axes=(0, 0, 0, None)))
get_d2G00dC2 = jit(jax.vmap(nth_derivative(get_G00, 2, argnums=2), in_axes=(0, 0, 0, None)))
get_d2G00dAdC = jit(jax.vmap(jax.grad(jax.grad(get_G00, argnums=2), argnums=0), in_axes=(0, 0, 0, None)))

get_d3G00dA3 = jit(jax.vmap(nth_derivative(get_G00, 3, argnums=0), in_axes=(0, 0, 0, None)))
get_d3G00dA2dC = jit(jax.vmap(jax.grad(jax.grad(jax.grad(get_G00, argnums=2), argnums=0), argnums=0), in_axes=(0, 0, 0, None)))

get_d4G00dA4 = jit(jax.vmap(nth_derivative(get_G00, 4, argnums=0), in_axes=(0, 0, 0, None)))
