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
def get_dGs_l(A, B, C, coeff):
    rho2 = A**2 + C**2
    c2 = A**2 / rho2
    s2 = C**2 / rho2
    Bs2 = B * s2

    G00 = 0
    dGdA = 0
    dGdC = 0
    dG2dA2 = 0
    d2GdC2 = 0
    d2GdAdC = 0
    dG3dA3 = 0
    d3GdA2dC = 0
    d4GdA4 = 0
    
    l = coeff.shape[0] - 1
    for i in range(l+1):
        for k in range(l+1):

            n = l + i - k
            term = coeff[k, i] * (- B)**(l + i) * s2**(n)

            G00 = G00 + term

            factor = Bs2 - n
            dGdA = dGdA + factor * term

            factor = -Bs2 + n
            dGdC = dGdC + factor * term

            factor = -n + 2 * c2 * n * (1 + n) + Bs2 - 4 * Bs2 * c2 * (1 + n) + 2 * Bs2*2 * c2
            dG2dA2 = dG2dA2 + factor * term

            factor = n * (-3 + 2 * c2 * (1 + n)) + Bs2 * (3 - 4 * c2 * (1 + n)) + 2 * Bs2**2 * c2
            d2GdC2 = d2GdC2 + factor * term

            factor = n * (-1 + c2 * (1 + n)) + Bs2 * (1 - 2 * c2 * (1 + n)) + Bs2**2 * c2
            d2GdAdC = d2GdAdC + factor * term

            factor = n * (1 + n) * (-3 + 2 * c2 * (2 + n)) - 6 * Bs2 * (1 + n) * (-1 + c2 * (2 + n)) \
                    + 3 * Bs2**2 * (-1 + 2 * c2 * (2 + n)) - 2 * Bs2**3 * c2
            dG3dA3 = dG3dA3 + factor * term

            factor = n + c2 * n * (1 + n) * (-5 + 2 * c2 * (2 + n)) \
                    - Bs2 * (1 + 2 * c2 * (1 + n) * (-5 + 3 * c2 * (2 + n))) \
                    + Bs2**2 * c2 * (-5 + 6 * c2 * (2 + n)) \
                    - 2 * Bs2**3 * c2**2
            d3GdA2dC = d3GdA2dC + factor * term
            
            factor = n * (1 + n) * (3 + 4 * c2 * (2 + n) * (-3 + c2 * (3 + n))) \
                    - 2 * Bs2 * (1 + n) * (3 + 2 * c2 * (2 + n) * (-9 + 4 * c2 * (3 + n))) \
                    + 3 * Bs2**2 * (1 + 4 * c2 * (2 + n) * (-3 + 2 * c2 * (3 + n))) \
                    - 4 * Bs2**3 * c2 * (-3 + 4 * c2 * (3 + n)) \
                    + 4 * Bs2**4 * c2**2
            d4GdA4 = d4GdA4 + factor * term

    G00 = G00 * jnp.exp(- Bs2)
    
    dGdA = dGdA * jnp.exp(- Bs2) * (2 * A / rho2)
    dGdC = dGdC * jnp.exp(- Bs2) * (2 * c2 / jnp.sqrt(s2 * rho2))
    
    dG2dA2 = dG2dA2 * jnp.exp(- Bs2) * (2 / rho2)
    d2GdC2 = d2GdC2 * jnp.exp(- Bs2) * (2 * c2 / (s2 * rho2))
    d2GdAdC = d2GdAdC * jnp.exp(- Bs2) * (-4 * jnp.sqrt(c2 / s2) / rho2)
    
    dG3dA3 = dG3dA3 * jnp.exp(- Bs2) * (- 4 * jnp.sqrt(c2) / rho2**1.5)
    d3GdA2dC = d3GdA2dC * jnp.exp(- Bs2) * (4 / jnp.sqrt(s2) / rho2**1.5)

    d4GdA4 = d4GdA4 * jnp.exp(- Bs2) * (4 / rho2**2)

    dGs_l = jnp.vstack([G00, dGdA, dGdC, dG2dA2, d2GdC2, d2GdAdC, dG3dA3, d3GdA2dC, d4GdA4])
    return dGs_l

@partial(jit, static_argnames=['lmax'])
def get_Gs(A, B, C, G00_coeffs, lmax=10):
    dGs = jnp.array([get_dGs_l(A, B, C, G00_coeffs[l]) for l in range(lmax + 1)])
    nq = dGs.shape[2]
    
    G00s = dGs[:,0]
    dGdAs = dGs[:,1]
    dGdCs = dGs[:,2]
    d2GdA2s = dGs[:,3]
    d2GdC2s = dGs[:,4]
    d2GdAdCs = dGs[:,5]
    d3GdA3s = dGs[:,6]
    d3GdA2dCs = dGs[:,7]
    d4GdA4s = dGs[:,8]

    zeros = jnp.zeros((1, nq))
    
    G00s_lm1 = jnp.vstack([zeros, G00s[:-1]])
    G00s_lm2 = jnp.vstack([zeros, G00s_lm1[:-1]])
    G00s_lm3 = jnp.vstack([zeros, G00s_lm2[:-1]])
    G00s_lm4 = jnp.vstack([zeros, G00s_lm3[:-1]])
    
    dGdAs_lm1 = jnp.vstack([zeros, dGdAs[:-1]])
    dGdAs_lm2 = jnp.vstack([zeros, dGdAs_lm1[:-1]])
    dGdAs_lm3 = jnp.vstack([zeros, dGdAs_lm2[:-1]])

    dGdCs_lm1 = jnp.vstack([zeros, dGdCs[:-1]])
    dGdCs_lm2 = jnp.vstack([zeros, dGdCs_lm1[:-1]])

    d2GdAdCs_lm1 = jnp.vstack([zeros, d2GdAdCs[:-1]])

    d2GdA2s_lm1 = jnp.vstack([zeros, d2GdA2s[:-1]])
    d2GdA2s_lm2 = jnp.vstack([zeros, d2GdA2s_lm1[:-1]])

    d3GdA3s_lm1 = jnp.vstack([zeros, d3GdA3s[:-1]])

    Gs = {}
    
    Gs[(0,0)] = G00s

    Gs[(1,0)] = -(dGdAs + 0.5 * A * G00s_lm1)
    Gs[(0,1)] = dGdCs + 0.5 * C * G00s_lm1

    Gs[(2,0)] = -(d2GdA2s + A * dGdAs_lm1 + 0.5 * G00s_lm1 + 0.25 * A**2 * G00s_lm2)
    Gs[(0,2)] = -(d2GdC2s + C * dGdCs_lm1 + 0.5 * G00s_lm1 + 0.25 * C**2 * G00s_lm2)
    Gs[(1,1)] = d2GdAdCs + 0.5 * C * dGdAs_lm1 + 0.5 * A * dGdCs_lm1 + 0.25 * A * C * G00s_lm2

    Gs[(3,0)] = d3GdA3s + 1.5 * A * d2GdA2s_lm1 + 1.5 * dGdAs_lm1 \
                + 0.75 * A**2 * dGdAs_lm2 + 0.75 * A * G00s_lm2 + A**3 / 8 * G00s_lm3
    Gs[(2,1)] = -(d3GdA2dCs + 0.5 * C * d2GdA2s_lm1 + A * d2GdAdCs_lm1 + 0.5 * dGdCs_lm1 \
                  + 0.5 * A * C * dGdAs_lm2 + 0.25 * A**2 * dGdCs_lm2 + 0.25 * C * G00s_lm2 + A**2 * C / 8 * G00s_lm3)

    Gs[(4,0)] = d4GdA4s + 2 * A * d3GdA3s_lm1 + 3 * d2GdA2s_lm1 \
                + 1.5 * A**2 * d2GdA2s_lm2 + 3 * A * dGdAs_lm2 + 0.75 * G00s_lm2 \
                + 0.5 * A**3 * dGdAs_lm3 + 0.75 * A**2 * G00s_lm3 \
                + A**4 / 16 * G00s_lm4
    
    return Gs
