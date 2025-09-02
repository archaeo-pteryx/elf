import jax
# jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
from functools import partial

@jit
def get_G00(A, B, C, coeff):
    x = -B
    y = C**2 / (A**2 + C**2)

    l = coeff.shape[0] - 1
    i = jnp.arange(l + 1)
    k = jnp.arange(l + 1)

    v = jnp.power(y[..., None], l - k)

    M = jnp.tensordot(v, coeff, axes=((-1,), (0,)))

    xpow = jnp.power(x[..., None], l + i)
    ypow = jnp.power(y[..., None], i)

    res = jnp.sum(M * xpow * ypow, axis=-1) * jnp.exp(x * y)

    return res

def get_dGs(A, B, C, coeffs, lmax):
    L  = lmax + 1
    nq = A.shape[0]
    
    rho2 = A**2 + C**2           # (nq,)
    c2   = A**2 / rho2           # (nq,)
    s2   = C**2 / rho2           # (nq,)
    Bs2  = B * s2                # (nq,)
    
    l_idx = jnp.arange(L)        # (L,)
    k_idx = jnp.arange(L)        # (K=L,)
    i_idx = jnp.arange(L)        # (I=L,)

    # --- v_{l,k} = s2^(l-k) : (L,K,1,nq) ---
    exp_lk = (l_idx[:, None, None] - k_idx[None, :, None]).reshape(L, L, 1, 1)     # (L,K,1,1)
    v = jnp.power(s2.reshape(1, 1, 1, nq), exp_lk)                                 # (L,K,1,nq)

    # --- w_{l,i} = (-B)^(l+i) * s2^i : (L,1,I,nq) ---
    powB_l    = jnp.power((-B)[None, :], l_idx[:, None])                          # (L, nq)
    powB_i    = jnp.power((-B)[None, :], i_idx[:, None])                          # (I, nq)
    s2_i      = jnp.power(s2[None, :],   i_idx[:, None])                          # (I, nq)
    powB_i_s2 = powB_i * s2_i                                                     # (I, nq)
    w = powB_l[:, None, None, :] * powB_i_s2[None, None, :, :]                    # (L,1,I,nq)

    # --- term_{l,k,i,q} と term2 = term/s2（s2=0は0扱い）---
    term  = v * w * coeffs[..., None]                                             # (L,K,I,nq)
    inv_s2 = jnp.where(s2 == 0.0, 0.0, 1.0 / s2)                                  # (nq,)
    term2 = term * inv_s2[None, None, None, :]                                    # (L,K,I,nq)

    # --- n = l + i - k を (L,K,I,1) に ---
    N = (l_idx[:, None, None] + i_idx[None, None, :] - k_idx[None, :, None]).reshape(L, L, L, 1)

    # --- スカラーを (1,1,1,nq) に持ち上げ（最後の軸が nq） ---
    Bs2b = Bs2.reshape(1, 1, 1, nq)
    c2b  = c2.reshape(1, 1, 1, nq)
    
    G00_sum  = jnp.sum(term, axis=(1, 2))                                         # (L, nq)

    dGdA_sum = jnp.sum((-N + Bs2b) * term,  axis=(1, 2))                          # (L, nq)
    dGdC_sum = jnp.sum(( N - Bs2b) * term2, axis=(1, 2))                          # (L, nq)

    d2A_sum = jnp.sum(
        (-N + 2*N*(1+N)*c2b + Bs2b - 4*(1+N)*c2b*Bs2b + 2*c2b*Bs2b**2) * term,
        axis=(1, 2)
    )                                                                             # (L, nq)

    d2C_sum = jnp.sum(
        (N*(-3 + 2*(1+N)*c2b) + (3 - 4*c2b*(1+N))*Bs2b + 2*c2b*Bs2b**2) * term2,
        axis=(1, 2)
    )

    dAdC_sum = jnp.sum(
        (N*(-1 + (1+N)*c2b) + (1 - 2*(1+N)*c2b)*Bs2b + c2b*Bs2b**2) * term2,
        axis=(1, 2)
    )

    d3A_sum = jnp.sum(
        ( N*(1+N)*(-3 + 2*(2+N)*c2b)
        - 6*(1+N)*(-1 + (2+N)*c2b)*Bs2b
        + 3*(-1 + 2*(2+N)*c2b)*Bs2b**2
        - 2*c2b*Bs2b**3) * term,
        axis=(1, 2)
    )

    d3A2C_sum = jnp.sum(
        ( N + N*(1+N)*(-5 + 2*(2+N)*c2b)*c2b
        - (1 + 2*(1+N)*(-5 + 3*c2b*(2+N))*c2b)*Bs2b
        + (-5 + 6*(2+N)*c2b)*c2b*Bs2b**2
        - 2*c2b**2 * Bs2b**3) * term2,
        axis=(1, 2)
    )

    d4A_sum = jnp.sum(
        ( N*(1+N)*(3 + 4*(2+N)*(-3 + (3+N)*c2b)*c2b)
        - 2*(1+N)*(3 + 2*c2b*(2+N)*(-9 + 4*c2b*(3+N)))*Bs2b
        + 3*(1 + 4*(2+N)*(-3 + 2*(3+N)*c2b)*c2b)*Bs2b**2
        - 4*(-3 + 4*(3+N)*c2b)*c2b*Bs2b**3
        + 4*c2b**2 * Bs2b**4) * term,
        axis=(1, 2)
    )
    
    e2 = jnp.exp(-Bs2)                          # (nq,)
    G00   = G00_sum * e2                        # (L, nq)

    dGdA  = dGdA_sum  * e2 * (2 * A / rho2)
    dGdC  = dGdC_sum  * e2 * (2 * c2 * jnp.sqrt(s2 / rho2))

    d2A   = d2A_sum   * e2 * (2 / rho2)
    d2C   = d2C_sum   * e2 * (2 * c2 / rho2)
    dAdC  = dAdC_sum  * e2 * (-4 * jnp.sqrt(c2 * s2) / rho2)

    d3A   = d3A_sum   * e2 * (-4 * jnp.sqrt(c2) / (rho2**1.5))
    d3A2C = d3A2C_sum * e2 * ( 4 * jnp.sqrt(s2) / (rho2**1.5))

    d4A   = d4A_sum   * e2 * (4 / (rho2**2))
    
    out = jnp.stack([G00, dGdA, dGdC, d2A, d2C, dAdC, d3A, d3A2C, d4A], axis=1) # (L,9,nq)
    return out

def _shift_down(x, s, L):
    # x: (L, nq) -> 上に s 行ゼロを足して長さ L に戻す
    return jnp.pad(x, ((s, 0), (0, 0)))[:L]

@partial(jit, static_argnames=['lmax'])
def get_Gs(A, B, C, coeffs, lmax=10):
    L = lmax + 1
    # (L, 9, nq)
    dGs = get_dGs(A, B, C, coeffs, lmax)
    nq = dGs.shape[-1]

    G00s, dGdAs, dGdCs, d2GdA2s, d2GdC2s, d2GdAdCs, d3GdA3s, d3GdA2dCs, d4GdA4s = (dGs[:, i] for i in range(9))

    G00s_lm1, G00s_lm2, G00s_lm3, G00s_lm4 = (_shift_down(G00s, s, L) for s in (1,2,3,4))
    dGdAs_lm1, dGdAs_lm2, dGdAs_lm3        = (_shift_down(dGdAs, s, L) for s in (1,2,3))
    dGdCs_lm1, dGdCs_lm2                   = (_shift_down(dGdCs, s, L) for s in (1,2))
    d2GdAdCs_lm1                           = _shift_down(d2GdAdCs, 1, L)
    d2GdA2s_lm1, d2GdA2s_lm2               = (_shift_down(d2GdA2s, s, L) for s in (1,2))
    d3GdA3s_lm1                            = _shift_down(d3GdA3s, 1, L)

    Acol, Ccol = A[None, :], C[None, :]  # (1,nq) for broadcasting with (L,nq)
    
    Gs = jnp.zeros((5, 3, L, nq), dtype=A.dtype)

    Gs = Gs.at[0,0].set(G00s)

    Gs = Gs.at[1,0].set(dGdAs + 0.5 * Acol * G00s_lm1)
    Gs = Gs.at[0,1].set(dGdCs + 0.5 * Ccol * G00s_lm1)

    Gs = Gs.at[2,0].set(d2GdA2s + Acol * dGdAs_lm1 + 0.5 * G00s_lm1 + 0.25 * (Acol**2) * G00s_lm2)
    Gs = Gs.at[0,2].set(d2GdC2s + Ccol * dGdCs_lm1 + 0.5 * G00s_lm1 + 0.25 * (Ccol**2) * G00s_lm2)
    Gs = Gs.at[1,1].set(d2GdAdCs + 0.5 * Ccol * dGdAs_lm1 + 0.5 * Acol * dGdCs_lm1
                         + 0.25 * (Acol * Ccol) * G00s_lm2)

    Gs = Gs.at[3,0].set(d3GdA3s + 1.5 * Acol * d2GdA2s_lm1 + 1.5 * dGdAs_lm1
                         + 0.75 * (Acol**2) * dGdAs_lm2 + 0.75 * Acol * G00s_lm2
                         + (Acol**3) / 8 * G00s_lm3)

    Gs = Gs.at[2,1].set(d3GdA2dCs + 0.5 * Ccol * d2GdA2s_lm1 + Acol * d2GdAdCs_lm1 + 0.5 * dGdCs_lm1
                         + 0.5 * Acol * Ccol * dGdAs_lm2 + 0.25 * (Acol**2) * dGdCs_lm2
                         + 0.25 * Ccol * G00s_lm2 + (Acol**2) * Ccol / 8 * G00s_lm3)

    Gs = Gs.at[4,0].set(d4GdA4s + 2 * Acol * d3GdA3s_lm1 + 3 * d2GdA2s_lm1
                         + 1.5 * (Acol**2) * d2GdA2s_lm2 + 3 * Acol * dGdAs_lm2 + 0.75 * G00s_lm2
                         + 0.5 * (Acol**3) * dGdAs_lm3 + 0.75 * (Acol**2) * G00s_lm3
                         + (Acol**4) / 16 * G00s_lm4)

    return Gs
