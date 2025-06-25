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

def nth_derivative(f, n, argnums=0):
    df = f
    for _ in range(n):
        df = jax.grad(df, argnums)
    return df

get_dGdA = jit(jax.vmap(nth_derivative(get_G00, 1, argnums=0), in_axes=(0, 0, 0, None)))
get_dGdC = jit(jax.vmap(nth_derivative(get_G00, 1, argnums=2), in_axes=(0, 0, 0, None)))

get_d2GdA2 = jit(jax.vmap(nth_derivative(get_G00, 2, argnums=0), in_axes=(0, 0, 0, None)))
get_d2GdC2 = jit(jax.vmap(nth_derivative(get_G00, 2, argnums=2), in_axes=(0, 0, 0, None)))
get_d2GdAdC = jit(jax.vmap(jax.grad(jax.grad(get_G00, argnums=2), argnums=0), in_axes=(0, 0, 0, None)))

get_d3GdA3 = jit(jax.vmap(nth_derivative(get_G00, 3, argnums=0), in_axes=(0, 0, 0, None)))
get_d3GdA2dC = jit(jax.vmap(jax.grad(jax.grad(jax.grad(get_G00, argnums=2), argnums=0), argnums=0), in_axes=(0, 0, 0, None)))

get_d4GdA4 = jit(jax.vmap(nth_derivative(get_G00, 4, argnums=0), in_axes=(0, 0, 0, None)))

@partial(jit, static_argnames=['lmax'])
def get_Gs(A, B, C, G00_coeffs, lmax=10):
    
    G00s = [get_G00(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)] + [0.,0.,0.,0.]
    
    dGdAs = [get_dGdA(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)] + [0.,0.,0.]
    dGdCs = [get_dGdC(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)] + [0.,0.,0.]
    
    d2GdA2s = [get_d2GdA2(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)] + [0.,0.]
    d2GdC2s = [get_d2GdC2(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)] + [0.,0.]
    d2GdAdCs = [get_d2GdAdC(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)] + [0.,0.]

    d3GdA3s = [get_d3GdA3(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)] + [0.]
    d3GdA2dCs = [get_d3GdA2dC(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)] + [0.]

    d4GdA4s = [get_d4GdA4(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)]

    Gs = {}
    
    Gs[(0,0)] = [G00s[l] for l in range(lmax + 1)]
    
    Gs[(1,0)] = [-(dGdAs[l] + 0.5 * A * G00s[l-1]) for l in range(lmax + 1)]
    Gs[(2,0)] = [-(d2GdA2s[l] + A * dGdAs[l-1] + 0.5 * G00s[l-1] + 0.25 * A**2 * G00s[l-2]) for l in range(lmax + 1)]
    Gs[(3,0)] = [d3GdA3s[l] + 1.5 * A * d2GdA2s[l-1] + 1.5 * dGdAs[l-1] \
                + 0.75 * A**2 * dGdAs[l-2] + 0.75 * A * G00s[l-2] + A**3/8. * G00s[l-3] for l in range(lmax + 1)]
    Gs[(4,0)] = [d4GdA4s[l] + 2 * A * d3GdA3s[l-1] + 3 * d2GdA2s[l-1] \
                + 1.5 * A**2 * d2GdA2s[l-2] + 3 * A * dGdAs[l-2] + 0.75 * G00s[l-2] \
                + 0.5 * A**3 * dGdAs[l-3] + 0.75 * A**2 * G00s[l-3] \
                + A**4 / 16. * G00s[l-4] for l in range(lmax + 1)]
    
    Gs[(0,1)] = [dGdCs[l] + 0.5 * C * G00s[l-1] for l in range(lmax + 1)]
    Gs[(0,2)] = [-(d2GdC2s[l] + C * dGdCs[l-1] + 0.5 * G00s[l-1] + 0.25 * C**2 *G00s[l-2]) for l in range(lmax + 1)]
    Gs[(1,1)] = [d2GdAdCs[l] + 0.5 * C * dGdAs[l-1] + 0.5 * A * dGdCs[l-1] + 0.25 * A * C * G00s[l-2] for l in range(lmax + 1)]
    Gs[(2,1)] = [-(d3GdA2dCs[l] + 0.5 * C * d2GdA2s[l-1] + A * d2GdAdCs[l-1] + 0.5 * dGdCs[l-1] \
                + 0.5 * A * C * dGdAs[l-2] + 0.25 * A**2 * dGdCs[l-2] + 0.25 * C * G00s[l-2] + A**2 * C / 8 * G00s[l-3])  for l in range(lmax + 1)]
    
    return Gs