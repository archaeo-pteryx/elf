import jax.numpy as jnp

def _pow_table_vec(x, L):
    x = jnp.array(x)
    one = jnp.ones((1, x.shape[0]), dtype=x.dtype)  # (1,nq)
    if L == 1:
        return one
    xs   = jnp.repeat(x[None, :], L-1, axis=0)     # (L-1,nq)
    base = jnp.concatenate([one, xs], axis=0)      # (L,nq)
    return jnp.cumprod(base, axis=0)               # (L,nq)

def get_G00s(B, s2, coeffs, lmax):
    L   = lmax + 1
    Bs2 = B * s2  # (nq,)

    # N = l + i - k
    li = jnp.arange(L)[:, None, None]
    ki = jnp.arange(L)[None, :, None]
    ii = jnp.arange(L)[None, None, :]
    N  = li + ii - ki                         # (L,K,I)
    mask = (N >= 0)

    # (-B)^l, (-B)^i
    B_pows = _pow_table_vec(-B, L)  # (L,nq)

    # s2^(l+i-k)
    s2_powN = jnp.where(mask, jnp.power(s2, N), 0.0)  # (L,K,I)

    # core[l,k,i] = coeff[l,k,i] * s2^(l+i-k)
    core = coeffs * mask.astype(coeffs.dtype) * s2_powN  # (L,K,I)

    # contraction：Σ_{k,i} core[l,k,i] * (-B)^l(q) * (-B)^i(q)
    S0 = jnp.einsum('lki,lq,iq->lq', core, B_pows, B_pows)  # (L,nq)

    G00 = S0 * jnp.exp(-Bs2)[None, :]  # (L,nq)
    return G00

def get_dGs(A, B, C, c2, s2, coeffs, lmax):
    L = lmax + 1

    rho2 = A*A + C*C         # (nq,)
    Bs2  = B * s2            # (nq,)
    
    # N = l + i - k
    li = jnp.arange(L)[:, None, None]
    ki = jnp.arange(L)[None, :, None]
    ii = jnp.arange(L)[None, None, :]
    N  = li + ii - ki                         # (L,K,I)
    mask = (N >= 0)

    # (-B)^l, (-B)^i
    B_pows = _pow_table_vec(-B, L)  # (L,nq)

    # s2^(l+i-k)
    s2_powN = jnp.where(mask, jnp.power(s2, N), 0.0)  # (L,K,I)

    # core[l,k,i] = coeff[l,k,i] * s2^(l+i-k)
    core = coeffs * mask.astype(coeffs.dtype) * s2_powN  # (L,K,I)

    # S_d = Σ coeffs·N^d·term
    One = jnp.ones_like(N, dtype=A.dtype) # (L,K,I)
    Nd  = N.astype(A.dtype)               # (L,K,I)
    N2  = Nd*Nd                           # (L,K,I)
    N3  = N2*Nd                           # (L,K,I)
    N4  = N2*N2                           # (L,K,I)
    poly = jnp.stack([One, Nd, N2, N3, N4], axis=0)   # (5,L,K,I)

    S = jnp.einsum('dlki,lki,lq,iq->dlq', poly, core, B_pows, B_pows) # (5,L,nq)
    S0, S1, S2, S3, S4 = S

    # term2 = term / s2 → T_d = (1/s2) * S_d
    tiny  = jnp.finfo(A.dtype).tiny
    zero  = jnp.array(0.0, dtype=A.dtype)
    invs2 = jnp.where(s2 <= tiny, zero, 1.0 / s2)
    T0, T1, T2, T3 = (invs2*S0, invs2*S1, invs2*S2, invs2*S3)

    # overall factors
    e2   = jnp.exp(-Bs2)[None, :]                               # (1,nq)
    Afac = (2 * A / rho2)[None, :]
    Cfac = (2 * c2 * jnp.sqrt(s2 / rho2))[None, :]
    A2f  = (2 / rho2)[None, :]
    C2f  = (2 * c2 / rho2)[None, :]
    ACf  = (-4 * jnp.sqrt(c2 * s2) / rho2)[None, :]
    A3f  = (-4 * jnp.sqrt(c2) / rho2**1.5)[None, :]
    A2Cf = ( 4 * jnp.sqrt(s2) / rho2**1.5)[None, :]
    A4f  = ( 4 / rho2**2)[None, :]

    # G00
    G00   = S0 * e2

    # dGdA_sum = -S1 + Bs2*S0
    dGdA  = (-S1 + Bs2[None,:]*S0) * e2 * Afac

    # dGdC_sum =  T1 - Bs2*T0
    dGdC  = ( T1 - Bs2[None,:]*T0) * e2 * Cfac

    # d2A_sum = -S1 + 2 c2 (S1+S2) + Bs2 S0 - 4 c2 Bs2 (S0+S1) + 2 c2 Bs2^2 S0
    d2A   = (- S1
             + 2.0*c2*(S1 + S2)
             + Bs2[None,:]*S0
             - 4.0*c2*Bs2[None,:]*(S0 + S1)
             + 2.0*c2*(Bs2[None,:]**2)*S0) * e2 * A2f

    # d2C_sum = -3 T1 + 2 c2 (T1+T2) + Bs2 (3 T0 - 4 c2 (T0+T1)) + 2 c2 Bs2^2 T0
    d2C   = (-3.0*T1
             + 2.0*c2*(T1 + T2)
             + Bs2[None,:]*(3.0*T0 - 4.0*c2*(T0 + T1))
             + 2.0*c2*(Bs2[None,:]**2)*T0) * e2 * C2f

    # dAdC_sum = -T1 + c2 (T1+T2) + Bs2 (T0 - 2 c2 (T0+T1)) + c2 Bs2^2 T0
    dAdC  = (- T1
             + c2*(T1 + T2)
             + Bs2[None,:]*(T0 - 2.0*c2*(T0 + T1))
             + c2*(Bs2[None,:]**2)*T0) * e2 * ACf

    # d3A_sum：
    #   P1 = N(1+N)(-3 + 2(2+N)c2) = (-3)(S1+S2) + 2 c2 (S3 + 3S2 + 2S1)
    P1 = (-3.0)*(S1 + S2) + 2.0*c2*(S3 + 3.0*S2 + 2.0*S1)
    #   P2 = +6(1+N)Bs2 - 6 c2 (N^2+3N+2)Bs2
    P2 = Bs2[None,:]*( 6.0*(S0 + S1) - 6.0*c2*(S2 + 3.0*S1 + 2.0*S0) )
    #   P3 = +3(-1 + 4 c2)Bs2^2*S0 + 3*(2 c2)Bs2^2*S1
    P3 = 3.0 * ( (-1.0 + 4.0*c2) * S0 + 2.0*c2 * S1 ) * (Bs2[None,:]**2)
    #   P4 = - 2 c2 Bs2^3 S0
    P4 = -2.0 * c2 * (Bs2[None,:]**3) * S0

    d3A   = (P1 + P2 + P3 + P4) * e2 * A3f

    # d3A2C_sum：
    Q1 = T1
    Q2 = c2*( -5.0*(T1 + T2) + 2.0*c2*(T3 + 3.0*T2 + 2.0*T1) )
    Q3 = (- Bs2[None,:]*T0
          + 2.0*c2*Bs2[None,:]*( 5.0*(T0+T1) - 6.0*c2*(T0+T1) - 3.0*c2*(T1+T2) ))
    Q4 = c2*(Bs2[None,:]**2)*( -5.0*T0 + 12.0*c2*T0 + 6.0*c2*T1 )
    Q5 = -2.0 * (c2**2) * (Bs2[None,:]**3) * T0

    d3A2C = (Q1 + Q2 + Q3 + Q4 + Q5) * e2 * A2Cf

    # d4A_sum
    a0 = 3.0 - 24.0*c2 + 24.0*(c2**2)
    a1 = -12.0*c2 + 20.0*(c2**2)
    a2 = 4.0*(c2**2)
    term1 = a0*S1 + (a0+a1)*S2 + (a1+a2)*S3 + a2*S4

    b0 = 3.0 + (-36.0*c2 + 48.0*(c2**2))
    b1 = (3.0 - 18.0*c2 + 40.0*(c2**2))
    b2 = (8.0*(c2**2))
    c0, c1, c2c, c3 = b0, (b0+b1), (b1+b2), b2
    term2 = -2.0 * Bs2[None,:] * ( c0*S0 + c1*S1 + c2c*S2 + c3*S3 )

    d0 = 1.0 - 24.0*c2 + 48.0*(c2**2)
    d1 = -12.0*c2 + 40.0*(c2**2)
    d2 = 8.0*(c2**2)
    term3 = 3.0 * (Bs2[None,:]**2) * ( d0*S0 + d1*S1 + d2*S2 )

    e0 = 12.0*c2 - 48.0*(c2**2)
    e1 = -16.0*(c2**2)
    term4 = (Bs2[None,:]**3) * ( e0*S0 + e1*S1 )

    term5 = 4.0 * (c2**2) * (Bs2[None,:]**4) * S0

    d4A = (term1 + term2 + term3 + term4 + term5) * e2 * A4f

    out = jnp.stack([G00, dGdA, dGdC, d2A, d2C, dAdC, d3A, d3A2C, d4A], axis=0)  # (9,L,nq)
    return out

def _shift_down(x, s, L):
    return jnp.pad(x, ((s, 0), (0, 0)))[:L]

def get_Gs(A, B, C, c2, s2, coeffs, lmax=10):
    L = lmax + 1
    
    dGs = get_dGs(A, B, C, c2, s2, coeffs, lmax) # (9, L, nq)
    nq = dGs.shape[-1]

    G00s, dGdAs, dGdCs, d2GdA2s, d2GdC2s, d2GdAdCs, d3GdA3s, d3GdA2dCs, d4GdA4s = (dGs[i] for i in range(9))

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

    return Gs # (5, 3, L, nq)
