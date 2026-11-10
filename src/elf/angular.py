"""The Legendre representation 'leg' of the angular integrals of the LPT final integral (observable-independent).

The frame is that of ``lpt_defs._RSDGeometry``.  n is the line of sight, mu = khat.n, K = k + f mu k n the
redshift-space wavevector, c and s the cosine and sine of the angle between K and k, and e_x = (khat - c Khat) / s the
unit vector across K in the plane of k and n, so that khat = c Khat + s e_x and n = A_mu Khat - B_mu e_x.  A moment j
is a product of powers of Khat.qhat and n.qhat in the final integrand (``lpt_tables.MOMENTS``).

Per point (k, mu) and per q the final integrand needs, for every moment j, the average over the directions of q of
its angular polynomial Pi_j = sum_t coef_t(A_mu, B_mu) u^a w^b (degree n_j = a + b, ``lpt_tables.MOMENT_POLYNOMIALS``)
against the plane wave and the damping.  With u = mu_K = Khat.qhat, w = mu_x = e_x.qhat, X = khat.qhat = c u + s w,
rho = k q, B = -(1/2) K^2 Y_lin_lt(q) <= 0 (Y_lin_lt the row ``Y_LIN_LT`` of ``lpt.PTTerms``) and <.> the average over
the directions of q, the plane-wave expansion e^{i rho X} = sum_L (2L + 1) i^L j_L(rho) P_L(X) gives

    (-i)^(n_j mod 2) <Pi_j e^{i rho X} e^{B u^2}> = sum_L j_L(rho) C_L^j(B),
    C_L^j(B) = sum_i G^j_{L,i} J_i(B),     J_i(B) = int_{-1}^{1} P_i(u) e^{B u^2} du   (i even),
    G^j_{L,i} = sum_t coef_t sum_r s^r P_L^(r)(c) T^(a_t,b_t,r)_(L,i),

with T the exact table tables/lpt/angular.dat (tools/pt_autogen/lpt_legendre.py, its header states the identity),
P_L^(r) the r-th derivative of the Legendre polynomial P_L and r = b, b - 2, .. >= 0.  The parity leaves the orders
L = n_j mod 2, and T is banded, |i - L| <= n_j.  The coefficients carry no power of 1/rho, so the series is stable at
small rho.  G depends on mu and f only and is built once per mu (``angular_matrix``), J depends on B(q)
(``exp_moments``), and C is a short banded sum per row (``banded_coefficients``).  The B derivative of J is
tridiagonal in J (``lam_matrix``), which the tangent rule (``lpt_rule``) uses.  The moments have no out-of-plane
factor mu_y (b_y = 0), and angular.dat has no such entries.
"""
import os

import numpy as np
import jax
import jax.numpy as jnp

from .table_text import fraction, records

MILLER_START = 28
"""The static start index M of the backward recursion of ``exp_moments`` (the ratios from r_{M+1} = 0)."""
KUMMER_TERMS = 48
"""The terms of the positive series of J_0 (``kummer_J0``)."""
BARRIER_EVERY = 8
"""The recursion of ``exp_moments`` is materialized every BARRIER_EVERY steps and the series of ``kummer_J0`` every
2 BARRIER_EVERY terms (``jax.lax.optimization_barrier``, values unchanged).  As one long fused elementwise chain XLA's
CPU code is ten times slower (45 ms against 3.5 ms for 160 x 512 points)."""


def load_table(path=os.path.join(os.path.dirname(__file__), 'tables', 'lpt', 'angular.dat')):
    """{(a, b, r): {(L, i): T^(a,b,r)_(L,i)}} of angular.dat, the exact rationals as floats."""
    table = {}
    for a, b, r, L, i, coef in records(path, key=lambda record: tuple(record[:5])):
        table.setdefault((int(a), int(b), int(r)), {})[(int(L), int(i))] = float(fraction(coef))
    return table


TABLE = load_table()
"""The table of tables/lpt/angular.dat, {(a, b, r): {(L, i): T}}."""
L_TABLE = max(L for entries in TABLE.values() for L, _ in entries)
"""The highest order L of TABLE."""


class AngularPlan:
    """The static plan of the angular factors C_L^j of the moments ``moment_polynomials`` for the orders L <= ``lmax``.

    ``moment_polynomials`` lists, in the order of the moment index j, the terms (coef, pAm, pBm, a, b, b_y) of each
    moment's polynomial, coef A_mu^pAm B_mu^pBm mu_K^a mu_x^b (``lpt_tables.MOMENT_POLYNOMIALS``).  Built once at
    construction and only read afterwards:

    * ``lmax``, ``degrees[j]`` the degree n_j = a + b of moment j.
    * ``rows`` the pairs (j, L) with L = n_j mod 2 and L <= lmax, one per angular factor C_L^j, by moment and then L.
      ``rows_of_moment[j]`` the indices of the rows of moment j (contiguous), ``rows_of_order[L]`` those of order L,
      ``row_moment``, ``row_order`` the j and L of each row.
    * ``nm`` the number of even Legendre indices i = 2m (m < nm) the rows need, i <= lmax + max n_j.
    * ``band[row]`` the m with a nonzero table entry in that row, ``wide_band[row]`` the band widened by one index on
      each side within 0 .. nm (for the product of G with ``lam_matrix``, which multiplies J of nm + 1 indices).
    * ``types`` the (a, b, r) of the monomials (by a + b, b, a and r), ``rmax`` their highest r and ``T_rows`` the
      ``(n_rows, n_types, nm)`` table values T^(a,b,r)_(L_row, 2m).
    * ``coefficients[j][t]`` the coefficient of the monomial mu_K^a mu_x^b of type t in Pi_j as the terms
      ((coef, pAm, pBm), ..) of coef A_mu^pAm B_mu^pBm, the lines of Pi_j with equal (a, b, pAm, pBm) summed (empty
      when Pi_j has no such monomial).

    ValueError when lmax exceeds the highest order L_TABLE of angular.dat.
    """

    def __init__(self, lmax, moment_polynomials):
        if lmax > L_TABLE:
            raise ValueError(f'lmax = {lmax} exceeds the orders L <= {L_TABLE} of tables/lpt/angular.dat')
        self.lmax = lmax
        self.degrees = tuple({a + b for *_, a, b, _ in terms}.pop() for terms in moment_polynomials)
        self.types = tuple(sorted({(a, b, r) for terms in moment_polynomials for *_, a, b, _ in terms
                                   for r in range(b % 2, b + 1, 2)}, key=lambda t: (t[0] + t[1], t[1], t[0], t[2])))
        self.rmax = max(r for _, _, r in self.types)
        self.rows = tuple((j, L) for j, n in enumerate(self.degrees) for L in range(n % 2, lmax + 1, 2))
        self.row_moment = tuple(j for j, _ in self.rows)
        self.row_order = tuple(L for _, L in self.rows)
        self.rows_of_moment = tuple(tuple(r for r, j in enumerate(self.row_moment) if j == jj)
                                    for jj in range(len(self.degrees)))
        self.rows_of_order = tuple(tuple(r for r, L in enumerate(self.row_order) if L == LL) for LL in range(lmax + 1))
        coefficients = []
        for terms in moment_polynomials:
            summed = {}
            for coef, pAm, pBm, a, b, _ in terms:
                summed[(a, b, pAm, pBm)] = summed.get((a, b, pAm, pBm), 0.0) + float(coef)
            coefficients.append(tuple(tuple((v, pAm, pBm) for (a, b, pAm, pBm), v in summed.items()
                                            if (a, b) == (ta, tb)) for ta, tb, _ in self.types))
        self.coefficients = tuple(coefficients)
        self.nm = (lmax + max(self.degrees)) // 2 + 1
        # row r = (j, L) holds the entries of order L of the types of moment j only, so its band is that of moment j
        T_rows = np.zeros((len(self.rows), len(self.types), self.nm))
        for r, (j, L) in enumerate(self.rows):
            for t, typ in enumerate(self.types):
                if self.coefficients[j][t]:
                    for (LL, i), v in TABLE[typ].items():
                        if LL == L:
                            T_rows[r, t, i // 2] = v
        self.T_rows = T_rows
        self.band = tuple(tuple(int(m) for m in np.flatnonzero(np.any(T_rows[r] != 0.0, axis=0)))
                          for r in range(len(self.rows)))
        self.wide_band = tuple(tuple(range(max(0, b[0] - 1), min(self.nm, b[-1] + 1) + 1)) for b in self.band)


def legendre_derivatives(c, lmax, rmax):
    """``(rmax + 1, lmax + 1)``: P_L^(r)(c), the r-th derivatives of the Legendre polynomials P_L at ``c``, by the
    recurrences (L + 1) P_{L+1} = (2L + 1) c P_L - L P_{L-1} and P^(r)_{L+1} = P^(r)_{L-1} + (2L + 1) P^(r-1)_L,
    which stay stable at c = 1 (P_L^(r)(1) grows like L^(2r))."""
    one, zero = jnp.ones_like(c), jnp.zeros_like(c)
    P = [[one, c]] + [[zero, one] if r == 1 else [zero, zero] for r in range(1, rmax + 1)]
    for L in range(1, lmax):
        P[0].append(((2 * L + 1) * c * P[0][L] - L * P[0][L - 1]) / (L + 1))
        for r in range(1, rmax + 1):
            P[r].append(P[r][L - 1] + (2 * L + 1) * P[r - 1][L])
    return jnp.stack([jnp.stack(P_r[:lmax + 1]) for P_r in P])


def _power(x, p):
    """x^p for a static int p >= 0 by repeated multiplication (ones for p = 0), so that autodiff never forms
    p x^(p-1) at x = 0."""
    out = jnp.ones_like(x)
    for _ in range(p):
        out = out * x
    return out


def angular_matrix(plan, geometry):
    """``(n_rows, nm)``: G^j_{L,2m} on the rows (j, L) of ``plan`` at one mu, from the ``lpt_defs._RSDGeometry``
    ``geometry`` (c, s, A_mu, B_mu)."""
    c, s, A_mu, B_mu = geometry.c, geometry.s, geometry.A_mu, geometry.B_mu
    P = legendre_derivatives(c, plan.lmax, plan.rmax)                           # (rmax + 1, lmax + 1)
    s_pow = jnp.stack([_power(s, r) for r in range(plan.rmax + 1)])
    zero = jnp.zeros_like(c)
    coef = jnp.stack([jnp.stack([sum((v * _power(A_mu, pAm) * _power(B_mu, pBm) for v, pAm, pBm in terms), zero)
                                 for terms in per_type]) for per_type in plan.coefficients])   # (n_moments, n_types)
    r_of_type = np.array([r for _, _, r in plan.types])
    # W[row, t] = coef_t of the row's moment times s^r P_L^(r)(c) of the row's order
    W = coef[np.array(plan.row_moment)] * (s_pow[r_of_type][None, :] * P[r_of_type][:, np.array(plan.row_order)].T)
    return jnp.einsum('rt,rtm->rm', W, jnp.asarray(plan.T_rows))


def kummer_J0(beta, terms=KUMMER_TERMS):
    """J_0 = int_{-1}^{1} e^{-beta u^2} du = 2 e^{-beta} sum_k beta^k / (3/2)_k for beta = -B >= 0 (positive terms,
    branch-free)."""
    term = jnp.ones_like(beta)
    acc = term
    for k in range(1, terms):
        term = term * beta / (k + 0.5)
        acc = acc + term
        if k % (2 * BARRIER_EVERY) == 0:
            term, acc = jax.lax.optimization_barrier((term, acc))
    return 2.0 * jnp.exp(-beta) * acc


def exp_moments(B, nm):
    """``(nm, nq)``: J_{2m}(B) = int_{-1}^{1} P_{2m}(u) e^{B u^2} du, m = 0 .. nm - 1, for B <= 0.

    The three-term recurrence of the scaled Jhat_m = J_{2m} / B^m (regular at B = 0),

        Jhat_{m-1} = a_m Jhat_m + b_m Jhat_{m+1},
        a_m = (4m - 1)/(2(2m - 1)) [4m + 1 + 2B(2m + 1)/(4m + 3) - 4Bm/(4m - 1)],
        b_m = (4m - 1)/(2(2m - 1)) 2(2m + 2)/(4m + 3) B^2,

    has J as its minimal solution, so it runs backward (Miller) in its ratio form r_m = Jhat_m / Jhat_{m-1} =
    1 / (a_m + b_m r_{m+1}) from r_{M+1} = 0, M = MILLER_START, and is normalised by the series J_0 of ``kummer_J0``:
    J_{2m} = J_{2m-2} r_m B.  a_m > 0 and b_m >= 0 for B <= 0, so no denominator vanishes and B -> 0 is exact.  Valid to
    relative 3e-15 for beta = -B <= 12 and m <= 19 (2e-15 for J_0), and to 5e-8 at beta = 20, set by the 48 terms of
    the J_0 series (measured against the exact series).  The final integral also reaches larger beta = K^2 Y_lt / 2
    (above 12 for k sqrt(1 + f (2 + f) mu^2) > 1.1 with a z = 0 spectrum), where J loses accuracy (1e-3 at beta = 30,
    J -> 0 beyond 70).  That does not reach the spectrum, because the same q carry the damping exp(-K^2 X_lt / 2)
    outside, which is at most e^{-beta / 2} since X_lt >= Y_lt / 2 for a positive spectrum (the rows of the model break
    this inequality only by their discretisation error at q < 0.05, where beta is far below 1), and the relative effect
    on P(k, mu) was measured below 5e-9 up to k = 3.  The loops are unrolled, because a lax.scan form is 3-5x slower.
    """
    if MILLER_START < nm:
        raise ValueError(f'exp_moments: the Miller start {MILLER_START} must reach the highest index {nm - 1}')
    B2 = B * B
    r = jnp.zeros_like(B)
    ratios = [None] * nm
    for m in range(MILLER_START, 0, -1):
        a = (4 * m - 1) / (2 * (2 * m - 1))
        r = 1.0 / (a * (4 * m + 1 + (2 * (2 * m + 1) / (4 * m + 3) - 4 * m / (4 * m - 1)) * B)
                   + (a * 2 * (2 * m + 2) / (4 * m + 3)) * B2 * r)
        if m % BARRIER_EVERY == 0:
            r = jax.lax.optimization_barrier(r)
        if m < nm:
            ratios[m] = r
    J = [kummer_J0(-B)]
    for m in range(1, nm):
        J.append(J[-1] * (ratios[m] * B))
    return jnp.stack(J)


def lam_matrix(nm):
    """The static ``(nm, nm + 1)`` matrix Lam with dJ_{2m}/dB = sum_k Lam[m, k] J_{2k}, from
    u^2 P_i = [(i+1)(i+2)/((2i+1)(2i+3))] P_{i+2} + [(2i^2+2i-1)/((2i-1)(2i+3))] P_i + [i(i-1)/((2i-1)(2i+1))] P_{i-2}
    at i = 2m."""
    Lam = np.zeros((nm, nm + 1))
    for m in range(nm):
        i = 2 * m
        Lam[m, m + 1] = (i + 1) * (i + 2) / ((2 * i + 1) * (2 * i + 3))
        Lam[m, m] = (2 * i * i + 2 * i - 1) / ((2 * i - 1) * (2 * i + 3))
        if m > 0:
            Lam[m, m - 1] = i * (i - 1) / ((2 * i - 1) * (2 * i + 1))
    return Lam


def banded_coefficients(plan, G_rows, J):
    """The rows C_r = sum over m in ``plan.band[r]`` of G_rows[r, m] J[m] ((nq,) each, a tuple in row order).  The sums
    are elementwise because a small matrix product per point is slower on XLA CPU (18 ms against 4.5 ms at lmax 5)."""
    out = []
    for r, band in enumerate(plan.band):
        acc = G_rows[r, band[0]] * J[band[0]]
        for m in band[1:]:
            acc = acc + G_rows[r, m] * J[m]
        out.append(acc)
    return tuple(out)
