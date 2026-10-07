import jax
import jax.numpy as jnp
import numpy as np

from .lpt_coefficients import generate_g00_coefficient, validate_lmax

def make_G00_coeffs(lmax, dtype=jnp.float64):
    """The ``[l, k, i]`` table of the angular coefficients up to ``lmax`` (zero for i > l)."""
    lmax = validate_lmax(lmax)
    L = lmax + 1
    coeffs = np.zeros((L, L, L), dtype=np.float64)  # axes: [l, k, i]
    for l in range(L):
        coeffs[l, :l + 1, :l + 1] = generate_g00_coefficient(l)
    return jnp.asarray(coeffs, dtype=dtype)

def _pow_table_vec(x, L):
    """Rows ``x**n``, n = 0..L-1; the explicit constant row keeps derivatives finite at x = 0."""
    x = jnp.asarray(x)
    return jnp.stack([jnp.ones_like(x)] + [x**n for n in range(1, L)])

def compute_V_mu(s2, coeffs, lmax):
    """``V[d, l, i] = sum_k N^d coeffs[l, k, i] s2^N``, ``N = l + i - k``; shape ``(5, L, L)``."""
    L = lmax + 1
    li = jnp.arange(L)[:, None, None]
    ki = jnp.arange(L)[None, :, None]
    ii = jnp.arange(L)[None, None, :]
    N = li + ii - ki
    mask = (N >= 0)
    # no negative power even in the masked branch: its derivative would be inf/NaN at s2 = 0
    s2_powN = jnp.where(mask, jnp.power(s2, jnp.maximum(N, 0)), 0.0)
    core = coeffs * mask.astype(s2_powN.dtype) * s2_powN  # (L, L, L)
    Nd = N.astype(jnp.float64)
    N2, N3, N4 = Nd * Nd, Nd * Nd * Nd, Nd * Nd * Nd * Nd
    poly = jnp.stack(
        [jnp.ones(N.shape, dtype=jnp.float64), Nd, N2, N3, N4], axis=0
    )  # (5, L, L, L)
    return jnp.einsum('dlki,lki->dli', poly, core.astype(jnp.float64))  # (5, L, L)


def _shift_down(x, s, L):
    return jnp.pad(x, ((s, 0), (0, 0)))[:L]

def _assemble_lpt_moments_from_derivatives(
    A,
    C,
    A_mu,
    B_mu,
    G00s,
    dGdAs,
    dGdCs,
    d2GdA2s,
    d2GdC2s,
    d2GdAdCs,
    d3GdA3s,
    d3GdA2dCs,
    d4GdA4s,
):
    L = G00s.shape[0]

    G00s_lm1, G00s_lm2, G00s_lm3, G00s_lm4 = (_shift_down(G00s, s, L) for s in (1,2,3,4))
    dGdAs_lm1, dGdAs_lm2, dGdAs_lm3 = (_shift_down(dGdAs, s, L) for s in (1,2,3))
    dGdCs_lm1, dGdCs_lm2 = (_shift_down(dGdCs, s, L) for s in (1,2))
    d2GdAdCs_lm1 = _shift_down(d2GdAdCs, 1, L)
    d2GdA2s_lm1, d2GdA2s_lm2 = (_shift_down(d2GdA2s, s, L) for s in (1,2))
    d3GdA3s_lm1 = _shift_down(d3GdA3s, 1, L)

    Acol, Ccol = A[None, :], C[None, :]

    G10 = dGdAs + 0.5 * Acol * G00s_lm1
    G01 = dGdCs + 0.5 * Ccol * G00s_lm1
    G20 = d2GdA2s + Acol * dGdAs_lm1 + 0.5 * G00s_lm1 + 0.25 * (Acol**2) * G00s_lm2
    G02 = d2GdC2s + Ccol * dGdCs_lm1 + 0.5 * G00s_lm1 + 0.25 * (Ccol**2) * G00s_lm2
    G11 = (d2GdAdCs + 0.5 * Ccol * dGdAs_lm1 + 0.5 * Acol * dGdCs_lm1
           + 0.25 * (Acol * Ccol) * G00s_lm2)
    G30 = (d3GdA3s + 1.5 * Acol * d2GdA2s_lm1 + 1.5 * dGdAs_lm1
           + 0.75 * (Acol**2) * dGdAs_lm2 + 0.75 * Acol * G00s_lm2
           + (Acol**3) / 8 * G00s_lm3)
    G21 = (d3GdA2dCs + 0.5 * Ccol * d2GdA2s_lm1 + Acol * d2GdAdCs_lm1 + 0.5 * dGdCs_lm1
           + 0.5 * Acol * Ccol * dGdAs_lm2 + 0.25 * (Acol**2) * dGdCs_lm2
           + 0.25 * Ccol * G00s_lm2 + (Acol**2) * Ccol / 8 * G00s_lm3)
    G40 = (d4GdA4s + 2 * Acol * d3GdA3s_lm1 + 3 * d2GdA2s_lm1
           + 1.5 * (Acol**2) * d2GdA2s_lm2 + 3 * Acol * dGdAs_lm2 + 0.75 * G00s_lm2
           + 0.5 * (Acol**3) * dGdAs_lm3 + 0.75 * (Acol**2) * G00s_lm3
           + (Acol**4) / 16 * G00s_lm4)

    mq0 = G00s
    mq1 = -G10
    mq2 = -G20
    mq3 = G30
    mq4 = G40
    nq1 = -A_mu * G10 + B_mu * G01
    nq2 = -A_mu**2 * G20 + 2 * A_mu * B_mu * G11 - B_mu**2 * G02
    mq1_nq1 = -A_mu * G20 + B_mu * G11
    mq2_nq1 = A_mu * G30 - B_mu * G21

    return mq0, mq1, mq2, mq3, mq4, nq1, nq2, mq1_nq1, mq2_nq1

def get_lpt_moments(A, B, C, c2, s2, A_mu, B_mu, V, *, c, s):
    """The nine LPT moment arrays ``(L, nq)``, without the factor ``exp(-B s^2)``.

    The angular integral is taken with the polar axis along K = k + f (k.n) n.
    ``A = k q c``, ``B = -(1/2) K^2 Y_lin_lt(q)`` and ``C = k q s`` are ``(nq,)`` arrays,
    ``c`` and ``s`` the cosine and sine of the angle between K and k, ``c2`` and ``s2``
    their squares.  ``A_mu = (1 + f) mu / Kfac`` and ``B_mu = sqrt(1 - mu^2) / Kfac``
    give the line of sight in that frame.  ``V`` ``(5, L, L)`` is the angular table
    :func:`compute_V_mu` at ``s2``, with ``L = lmax + 1``.

    The caller folds ``exp(-B s^2)`` into ``exp(-K^2 (X + c^2 Y) / 2)``, because the two
    factors computed separately give 0 * inf.
    """
    L = V.shape[1]

    rho2 = A*A + C*C
    Bs2  = B * s2

    B_pows = _pow_table_vec(-B, L)

    # S[d,l,q] = (-B)^l sum_i V[d,l,i] (-B)^i, by Horner through degree l
    z = -B
    coefficients = V.astype(B_pows.dtype)
    rows = []
    for ell in range(L):
        row = coefficients[:, ell, ell, None] + jnp.zeros_like(z)[None, :]
        for i in range(ell - 1, -1, -1):
            row = row * z + coefficients[:, ell, i, None]
        rows.append(row)
    T = jnp.stack(rows, axis=1)
    S = T * B_pows[None]  # multiply by (-B)^l for each l

    S0, S1, S2, S3, S4 = S

    # double where: finite derivative of 1/s^2 at s2 = 0 (T_d = 0 there; nq2 alone is then not its mu -> 0 limit)
    tiny = jnp.finfo(A.dtype).tiny
    safe = s2 > tiny
    s2_safe = jnp.where(safe, s2, jnp.ones_like(s2))
    invs2 = jnp.where(safe, 1.0 / s2_safe, jnp.zeros_like(s2))
    T0, T1, T2, T3 = (invs2*S0, invs2*S1, invs2*S2, invs2*S3)

    # powers of 1/rho from one reciprocal square root
    inv_rho = jax.lax.rsqrt(rho2)
    inv_rho2 = inv_rho * inv_rho
    inv_rho3 = inv_rho2 * inv_rho
    Afac = (2 * A * inv_rho2)[None, :]
    Cfac = (2 * c2 * s * inv_rho)[None, :]
    A2f = (2 * inv_rho2)[None, :]
    C2f = (2 * c2 * inv_rho2)[None, :]
    ACf = (-4 * c * s * inv_rho2)[None, :]
    A3f = (-4 * c * inv_rho3)[None, :]
    A2Cf = (4 * s * inv_rho3)[None, :]
    A4f = (4 * inv_rho2 * inv_rho2)[None, :]

    G00 = S0
    dGdA = (-S1 + Bs2[None, :]*S0) * Afac
    dGdC = (T1 - Bs2[None, :]*T0) * Cfac
    d2A = (-S1
           + 2.0*c2*(S1 + S2)
           + Bs2[None, :]*S0
           - 4.0*c2*Bs2[None, :]*(S0 + S1)
           + 2.0*c2*(Bs2[None, :]**2)*S0) * A2f
    d2C = (-3.0*T1
           + 2.0*c2*(T1 + T2)
           + Bs2[None, :]*(3.0*T0 - 4.0*c2*(T0 + T1))
           + 2.0*c2*(Bs2[None, :]**2)*T0) * C2f
    dAdC = (-T1
            + c2*(T1 + T2)
            + Bs2[None, :]*(T0 - 2.0*c2*(T0 + T1))
            + c2*(Bs2[None, :]**2)*T0) * ACf

    P1 = (-3.0)*(S1 + S2) + 2.0*c2*(S3 + 3.0*S2 + 2.0*S1)
    P2 = Bs2[None, :]*(6.0*(S0 + S1) - 6.0*c2*(S2 + 3.0*S1 + 2.0*S0))
    P3 = 3.0*((-1.0 + 4.0*c2)*S0 + 2.0*c2*S1) * (Bs2[None, :]**2)
    P4 = -2.0*c2*(Bs2[None, :]**3)*S0
    d3A = (P1 + P2 + P3 + P4) * A3f

    Q1 = T1
    Q2 = c2*(-5.0*(T1 + T2) + 2.0*c2*(T3 + 3.0*T2 + 2.0*T1))
    Q3 = (-Bs2[None, :]*T0
          + 2.0*c2*Bs2[None, :]*(5.0*(T0 + T1) - 6.0*c2*(T0 + T1) - 3.0*c2*(T1 + T2)))
    Q4 = c2*(Bs2[None, :]**2)*(-5.0*T0 + 12.0*c2*T0 + 6.0*c2*T1)
    Q5 = -2.0*(c2**2)*(Bs2[None, :]**3)*T0
    d3A2C = (Q1 + Q2 + Q3 + Q4 + Q5) * A2Cf

    a0 = 3.0 - 24.0*c2 + 24.0*(c2**2)
    a1 = -12.0*c2 + 20.0*(c2**2)
    a2 = 4.0*(c2**2)
    term1 = a0*S1 + (a0 + a1)*S2 + (a1 + a2)*S3 + a2*S4

    b0 = 3.0 + (-36.0*c2 + 48.0*(c2**2))
    b1 = (-18.0*c2 + 40.0*(c2**2))
    b2 = (8.0*(c2**2))
    c0, c1, c2c, c3 = b0, (b0 + b1), (b1 + b2), b2
    term2 = -2.0*Bs2[None, :]*(c0*S0 + c1*S1 + c2c*S2 + c3*S3)

    d0 = 1.0 - 24.0*c2 + 48.0*(c2**2)
    d1 = -12.0*c2 + 40.0*(c2**2)
    d2 = 8.0*(c2**2)
    term3 = 3.0*(Bs2[None, :]**2)*(d0*S0 + d1*S1 + d2*S2)

    e0 = 12.0*c2 - 48.0*(c2**2)
    e1 = -16.0*(c2**2)
    term4 = (Bs2[None, :]**3)*(e0*S0 + e1*S1)
    term5 = 4.0*(c2**2)*(Bs2[None, :]**4)*S0
    d4A = (term1 + term2 + term3 + term4 + term5) * A4f

    return _assemble_lpt_moments_from_derivatives(
        A, C, A_mu, B_mu,
        G00, dGdA, dGdC, d2A, d2C, dAdC, d3A, d3A2C, d4A,
    )
