import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
from jax.scipy.special import gamma
from functools import partial

@jit
def get_coeff(m, n, k):
    res = (-1)**(k+m) * gamma(n+1) * gamma(1/2+k-n) / (gamma(m+1) * gamma(k+1) * gamma(1/2+k-m-n) * gamma(1-k+n) * gamma(1-m+n))
    return res

@partial(jit, static_argnames=['nmax'])
def get_G_0_0_m(m, A, B, C, nmax=10):
    rho2 = A**2 + C**2
    x = B
    y = A**2 / rho2
    
    res = jnp.zeros(A.shape)
    for n in range(nmax+1):
        for k in range(n+1):
            res = res + jnp.heaviside(n-m, 1.) * get_coeff(m, n, k) * x**n * y**(n-k)

    return res

@jit
def get_G_0_0(A, B, C, coeff):
    x = C**2 / (A**2 + C**2)
    res = jnp.zeros(A.shape)
    m = len(coeff)
    for i in range(m):
        for k in range(m):
            res = res + coeff[k, i] * (- B)**(m + i) * x**(m + i - k)
    res = res * jnp.exp(- B * x)
    return res

@jit
def get_cs_mu(f, mu):
    denom = jnp.sqrt(1 + f * (2 + f) * mu**2)
    c = (1 + f * mu**2) / denom
    s = f * mu * jnp.sqrt(1 - mu**2) / denom
    return c, s
