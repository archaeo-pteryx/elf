"""The tangent rule of the LPT point loop (``lpt.LPT._point_loop``, a ``jax.custom_jvp``): ``point_loop_jvp``.

Per point (k, mu) the forward contracts g_l = D E_l sum_j c_j M_j(l) (D the IR damping, E_l = (-2/k)^l q^-l, M_j the
nine moments, c_j(q) the moment coefficients) with the kernel KW[l, q] = 4 pi q^3 kern_l(q) at ln k:
out[g] = sum_q sum_l KW[l, q] g_l(q) - (sum_q KW[0, q]) g_0(q_max).  As c_j is linear in the basis rows R_t(q)
(``row_basis``) with per-point scalars, out[c M_j] = sum_q c[q] Y_j[q] + y_j sum_q KW[0, q] (c[q] - c[last]) with the
residuals Y_j = sum_l KW_l D E_l M_j(l) - KW_0 y_j, y_j = D E_0 M_j(0) at q_max (split before the weights, exact at
large q).  ``point_residuals`` computes, per point, the residual sets below that the active input directions need,
independent of the weights W and of the rows.  ``point_loop_jvp`` contracts every set with the basis rows on the
(moment, basis row) pairs of the component table (``contract_pairs``, one GEMM per set and moment), adds its q_max
term, applies the per-point chain rule to the contracted sets (the fold below) and the component weights
(``fold_weights``).  Per active direction d it forms the point's whole Jacobian along d in the primal part (the folded
sets plus the unit-direction jvps of the fold weights and of the small terms) and adds t_d times it to the tangent
groups of the rows, W and P_lin.

Residual sets: X_j(l) = D E_l F_j(l), Y_j = sum_l w_l X_j(l) - w_0 y_j, y_j = X_j(0) at q_max, w = KW unless noted
(A = k q c, C = k q s, B = -(1/2) K^2 Y_lt; c, s the cosine and sine of the angle between K and k):
    R0    F = M                         R_B   F = dM/dB                     R_k   F = q (c dM/dA + s dM/dC)
    R_c   F = k q dM/dA + dM/dc         R_s   F = k q dM/dC + dM/ds         R_Am  F = dM/dA_mu     R_Bm  F = dM/dB_mu
    R_kE  F = M, w_l = KW'_l - l KW_l (KW' = dKW/dln k: d(KW_l E_l)/dln k = (KW'_l - l KW_l) E_l), w_0 = KW'_0
With M_j(l) = sum_{(b, s) in PAIRS_OF[j], s <= l} basis_b[l - s] psi_{j,b,s} (``lpt_tables``) every sum over l collapses
onto the pairs, sum_l w_l D E_l F_j(l) = sum_pairs (coefficient) G^w_{b,s} with G^w_{b,s} = sum_{l >= s} w_l D E_l
basis_b[l - s] (``pair_sums``, computed once per point); SET_RULE lists the terms.  At l = 0 only the pair (S0, 0)
contributes (basis_S0[0] = 1, basis_Td[0] = 0, the derivative bases vanish at m = 0): y_j = D(q_max) times the first
coefficient of (S0, 0).
Contracted sets on the pairs (j, t): Z^x = Y^x_j . R_t + y^x_j rho_t with rho = KW_0 . (R - R[last]); Z^0X, Z^0Y, Z^BY
the same of Y^0 and Y^B with R X_lt and R Y_lt (rho with R X_lt, R Y_lt), Z^kE that of R_kE with the q_max value y^0
and KW'_0.  Folded direction d, J^d_x = dx/dd of the scalars of ``rsd_scalars`` (the terms of identically zero J^d_x not
built):
    Z^d = J^d_c [Z^c - Ksq c Z^0Y] + J^d_s Z^s + J^d_Am Z^Am + J^d_Bm Z^Bm
          - (J^d_Ksq / 2) [Z^0X + c^2 Z^0Y + Z^BY] + [d = k] (Z^k + Z^kE / k).

Scope.  The rule is written for the density power spectrum, and these parts of it must be generalised for another
observable.  The damping D = exp(-K^2 (X_lt + c^2 Y_lt) / 2) and its chain rule through B = -(1/2) K^2 Y_lt are written
out (Z^0X, Z^0Y, Z^BY of the fold and the rows tangent of ``point_loop_jvp``).  The linear-part and DC corrections of
``small_terms`` (``LPT._linear_rows``, ``_LINEAR_MATRIX`` and ``_KSPACE_ROWS``) are those of the density components.
``row_basis`` builds each basis row R_t as the constant, a single row or a product of rows.  The moments are expanded in
the Horner basis S0, T1 .. T4 (``BASIS_MMIN``, the families G_B and G_s), and the q_max endpoint values y_j take F_j(0)
from the pair (S0, 0) alone.  Only the numbers of moments and of basis functions are read from the tables
(``len(MOMENTS)``, ``BASIS``).
"""
import numpy as np

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from . import spline
from .lpt_defs import _rsd_geometry
from .lpt_tables import (BASIS, COEFFS, CT_ENTRY_NAMES, MOMENTS, PAIRS, PAIRS_OF, SCALAR_MATRIX,
                         TREE_ROWS, Component, Row, _KSPACE_ROWS, _LINEAR_MATRIX, entry_scalars, psi_family, row_basis,
                         scalar_dmonomials, scalar_monomials)
from .utils_lpt import compute_V_mu, compute_V_mu_prime, compute_V_mu_prime2, lpt_horner_poly

DIRECTIONS = ('k', 'mu', 'sin', 'f')                       # the arguments of rsd_scalars
SCALARS = ('c', 's', 'Kfac', 'K', 'Ksq', 'Am', 'Bm')       # its outputs
# direction -> the scalars whose Jacobian along it is not identically zero (_rsd_geometry: k changes only K, Ksq)
SCALAR_TERMS = {'k': ('Ksq',), 'mu': ('c', 's', 'Am', 'Bm', 'Ksq'), 'sin': ('s', 'Bm'),
                'f': ('c', 's', 'Am', 'Bm', 'Ksq')}
SETS_OF_SCALAR = {'c': ('R_c', 'R0'), 's': ('R_s',), 'Am': ('R_Am',), 'Bm': ('R_Bm',), 'Ksq': ('R0', 'R_B')}
SET_OF_COEFFICIENT = {'s': 'R_s', 'Am': 'R_Am', 'Bm': 'R_Bm'}
K_SETS = ('R_k', 'R_kE')                                    # the extra sets of d = k
Y_SETS = ('R0', 'R_c', 'R_s', 'R_Am', 'R_Bm', 'R_B', 'R_k', 'R_kE')    # the rows of y
# set -> its terms (outer factor, ((factor, name, deriv), ...), G family): the coefficient of the pair key is
# outer * sum factor psi[name]_deriv[key], factors 1, q, k q (kq), c, s; the first term gives y_j.  The G families:
# G (w = KW), G_kE (w_l = KW'_l - l KW_l), G_B (basis -> d basis/dB), G_s (basis -> d basis/ds).
SET_RULE = {
    'R0': ((None, ((1, 'K', 'psi'),), 'G'),),                                          # psi[K] G
    'R_c': ((None, (('kq', 'K', 'dA'), (1, 'Kc', 'psi')), 'G'),),                      # (k q psi_A[K] + psi[K^c]) G
    'R_s': ((None, (('kq', 'K', 'dC'), (1, 'Ks', 'psi')), 'G'),                        # (k q psi_C[K] + psi[K^s]) G
            (None, ((1, 'K', 'psi'),), 'G_s')),                                        #   + psi[K] G^s
    'R_Am': ((None, ((1, 'KAm', 'psi'),), 'G'),),                                      # psi[K^Am] G
    'R_Bm': ((None, ((1, 'KBm', 'psi'),), 'G'),),                                      # psi[K^Bm] G
    'R_k': (('q', (('c', 'K', 'dA'), ('s', 'K', 'dC')), 'G'),),                        # q (c psi_A[K] + s psi_C[K]) G
    'R_kE': ((None, ((1, 'K', 'psi'),), 'G_kE'),),                                     # psi[K] G^kE
    'R_B': ((None, ((1, 'K', 'dB'),), 'G'), (None, ((1, 'K', 'psi'),), 'G_B')),         # psi_B[K] G + psi[K] G^B
}
SET_NEED = {n: tuple(dict.fromkeys((nm, d) for _, terms, _ in rule for _, nm, d in terms))
            for n, rule in SET_RULE.items()}                 # set -> the (name, deriv) of psi_family it reads
COEFFICIENT_VARIABLE = {name: x for name, x in COEFFS.items() if x}   # K^x = scalar_dmonomials(x) @ SCALAR_MATRIX
BASIS_INDEX = {b: n for n, b in enumerate(BASIS)}
BASIS_MMIN = (0,) + (1,) * (len(BASIS) - 1)                # basis_Td[0] = V'[d - 1, 0, 0] = 0 identically
CT_ENTRIES = tuple((int(Component[c]), MOMENTS.index(m), t) for c, m, t in CT_ENTRY_NAMES)   # (component, moment, t)
# the (moment j, basis row t) pairs of the entries, by j and t: the only products of a residual with a basis row that a
# fold weight does not multiply by an exact zero (45 of the 9 x N_CT)
JT_PAIRS = tuple(sorted({(j, t) for _, j, t in CT_ENTRIES}))
JT_J, JT_T = (np.array(v, dtype=np.int32) for v in zip(*JT_PAIRS))
ROWS_OF_MOMENT = tuple(np.array([t for jj, t in JT_PAIRS if jj == j], dtype=np.int32) for j in range(len(MOMENTS)))
ENTRY_COMPONENT = np.array([c for c, _, _ in CT_ENTRIES], dtype=np.int32)
ENTRY_PAIR = np.array([JT_PAIRS.index((j, t)) for _, j, t in CT_ENTRIES], dtype=np.int32)


def contract_pairs(Y, Qs):
    """``[Z (nk, nmu, n_pairs) per Q of Qs]``, Z[p] = Y_j . Q_t for the pairs p = (j, t) of JT_PAIRS: per moment one GEMM
    of Y_j ``(nk, nmu, nq)`` (``Y`` the sequence of the moments) with the rows [Q[T_j] for Q in Qs] (``Qs`` of
    ``(N_CT, nq)``)."""
    parts = [[] for _ in Qs]
    for Y_j, T in zip(Y, ROWS_OF_MOMENT):
        if len(T):
            Z = jnp.einsum('kmq,tq->kmt', Y_j, jnp.concatenate([Q[T] for Q in Qs]) if len(Qs) > 1 else Qs[0][T])
            for i in range(len(Qs)):
                parts[i].append(Z[..., i * len(T):(i + 1) * len(T)])
    return [jnp.concatenate(z, axis=-1) for z in parts]


def on_pairs(y):
    """The per-moment values ``(..., 9)`` on the pairs ``(..., n_pairs)`` (the value of moment j at every pair (j, t))."""
    return y[..., JT_J]


def edge(KW0, Q):
    """``(nk, nmu, n_pairs)``: KW0 . (Q_t - Q_t[last]) on the pairs (j, t), the factor of the q_max values y_j in the
    contraction of the l = 0 integrand with Q_t (KW0 ``(nk, nmu, nq)``, Q ``(N_CT, nq)``)."""
    return jnp.einsum('kmq,tq->kmt', KW0, Q - Q[:, -1:])[..., JT_T]


def contract_weights(W, x):
    """``sum_c W[..., o, c] x[..., c]`` -> ``(nk, nmu, n_out)``; W per point (4-d) or shared (2-d)."""
    return jnp.einsum('oc,kmc->kmo' if W.ndim == 2 else 'kmoc,kmc->kmo', W, x)


def rsd_scalars(k, mu, sin, f):
    """The per-point scalars (c, s, Kfac, K, Ksq, A_mu, B_mu) (SCALARS) of (k, mu, sin, f)."""
    geo = _rsd_geometry(f, mu, sin)
    K = k * geo.Kfac
    return geo.c, geo.s, geo.Kfac, K, K**2, geo.A_mu, geo.B_mu


def jvp_if_active(fn, primals, tangents):
    """``(fn(*primals), tangent)``: jax.jvp of fn w.r.t. the arguments whose tangent is not None (None if none)."""
    act = [n for n, t in enumerate(tangents) if t is not None]
    if not act:
        return fn(*primals), None
    sub = lambda *xs: fn(*[dict(zip(act, xs)).get(n, x) for n, x in enumerate(primals)])
    return jax.jvp(sub, tuple(primals[n] for n in act), tuple(tangents[n] for n in act))


def static_mul(a, b):
    """``a * b``, the Python float 0.0 if either factor is it (an exact zero)."""
    return 0.0 if any(isinstance(x, float) and x == 0.0 for x in (a, b)) else a * b


def static_lin(terms):
    """``sum a * b`` over the terms, skipping exact zeros (the Python float 0.0 on either side); 0.0 if none."""
    acc = None
    for a, b in terms:
        t = static_mul(a, b) if not (isinstance(a, float) and a == 1.0) else b
        if not (isinstance(t, float) and t == 0.0):
            acc = t if acc is None else acc + t
    return 0.0 if acc is None else acc


def static_last(x):
    """The value at q_max of a (nq,) array (a per-point scalar or a Python float is itself)."""
    return x if isinstance(x, float) or jnp.ndim(x) == 0 else x[-1]


def materialized(tree):
    """``tree`` with its array leaves passed through one ``jax.lax.optimization_barrier`` (the Python floats, exact
    zeros, unchanged): XLA computes them once and reads them in every consumer."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    arrays = [i for i, x in enumerate(leaves) if not isinstance(x, float)]
    done = jax.lax.optimization_barrier(tuple(leaves[i] for i in arrays))
    for i, x in zip(arrays, done):
        leaves[i] = x
    return jax.tree_util.tree_unflatten(treedef, leaves)


def pair_sums(wde, rows, pairs):
    """``{(b, s): sum_{l >= s} wde[l] rows[b][l - s]}``: static sums of (nq,) products, exact zeros skipped."""
    return {(b, s): static_lin([(wde[l], rows[BASIS_INDEX[b]][l - s]) for l in range(s, len(wde))]) for b, s in pairs}


def active_sets(dirs, rows_active):
    """The sets the outputs need (SET_RULE order): R0, those of each active direction (K_SETS for k), R_B (rows)."""
    need = {'R0'} | ({'R_B'} if rows_active else set())
    for d in dirs:
        for x in SCALAR_TERMS[d]:
            need.update(SETS_OF_SCALAR[x])
        if d == 'k':
            need.update(K_SETS)
    return tuple(n for n in SET_RULE if n in need)


def small_terms(model, rows, k, mu, sin, f, p, KW):
    """``(small, lin_c)`` per component ``(23,)``, the parts outside the moment contraction: small_c = out[linear rows
    of kind(c)] (weighted -lin_weight, l = 0 as sum_q KW[0, q] (x[q] - x[last])) + lin_kind(c) p + the ``_KSPACE_ROWS``
    values of c + [c = XI_XI] out[-xi^2/2], lin_c = lin_kind(c); the output is sum_c W[o, c] (comp_c + small_c).  The
    linear rows (``LPT._linear_terms``) are contracted as their combinations of the kernel contracted with the tree
    rows (rho, one product of the point's kernel with the shared rows per order: a GEMM over the points whose
    transpose, with the rows tangent, is a GEMM too)."""
    geometry, V = _rsd_geometry(f, mu, sin), model._angular_table(f, mu, sin)
    ft, tree, K = model._final_transform, rows[TREE_ROWS], k * geometry.Kfac
    terms, kaiser = model._linear_terms(mu, f, V, geometry, K, K**2)
    lin_weight = model._linear_weight(K**2, tree)
    rho = {}
    for _, l, inv_q, _ in terms:
        if (l, inv_q) not in rho:
            w = KW[l] * ((-2.0 / k) * ft.q_inv_pows[1]) if inv_q else KW[l]
            rho[(l, inv_q)] = tree @ w - jnp.sum(w) * tree[:, -1] if l == 0 else tree @ w   # l = 0: q_max subtracted
    s_kind = [0.0] * len(kaiser)
    for kind, l, inv_q, coefficients in terms:
        s_kind[kind] = s_kind[kind] + sum(c * rho[(l, inv_q)][r] for r, c in coefficients)
    s_kind = -lin_weight * jnp.stack(s_kind)
    lin_kind, lin_mat = lin_weight * jnp.stack(kaiser), jnp.asarray(_LINEAR_MATRIX)
    small = lin_mat @ (s_kind + lin_kind * p)
    out0 = lambda x: KW[0] @ (x - x[-1])                             # l = 0 with the q_max subtraction
    small = small.at[Component.XI_XI].add(-out0(0.5 * tree[Row.XI_LIN]**2))
    (kspace_rows, kspace_slopes), logk = model._kspace_tables(rows), jnp.log(k)
    for component, row, factor, interpolated, k0_const in _KSPACE_ROWS:
        if not (k0_const and model.subtract_k0_const):
            value = (spline.hermite(logk, model._logk_fft, kspace_rows[row], kspace_slopes[row]) if interpolated
                     else kspace_rows[row])
            small = small.at[component].add(factor * value)
    return small, lin_mat @ lin_kind


def component_weights_jvp(W_fn, k_true, mu_b, W_args, dk, dmu, dW_args):
    """``(W, dW)``: the weights ``(nk, nmu, n_out, 23)`` (or the shared ``(n_out, 23)``) and their tangent (None if
    zero), the jvp of the per-point ``W_fn`` w.r.t. (k, mu, its closed-over values)."""
    nz = lambda t: not isinstance(t, SymbolicZero)
    if W_fn is None:
        return W_args[0], (dW_args[0] if nz(dW_args[0]) else None)
    W_all = lambda kk, mm, *a: jax.vmap(jax.vmap(lambda ki, mj: W_fn(ki, mj, *a)))(kk, mm)
    dmu_b = jnp.broadcast_to(dmu, mu_b.shape) if nz(dmu) else dmu
    primals, tangents = (k_true, mu_b) + tuple(W_args), (dk, dmu_b) + tuple(dW_args)
    if not any(nz(t) for t in tangents):
        return W_all(*primals), None
    zero = lambda x: (jnp.zeros_like(x) if jnp.issubdtype(jnp.result_type(x), jnp.inexact)
                      else np.zeros(jnp.shape(x), jax.dtypes.float0))
    tangents = tuple(t if nz(t) else zero(x) for x, t in zip(primals, tangents))
    return jax.jvp(W_all, primals, tangents)


def mu_tables(model, f, mu_true, sin_true, sets):
    """The per-mu tables, once per mu: V, V', V'' (with R_s) at s2 and the vectors K = scalar_monomials(c, s, A_mu,
    B_mu) @ SCALAR_MATRIX, K^x = scalar_dmonomials(.., x) @ SCALAR_MATRIX the sets need; (mu, vector) is a batch
    dimension of the dot (XLA's OptimizeDotOfConcat splits a 2-D dot with a constant matrix into outer products)."""
    names = [n for n in COEFFS if any(n == nm for s in sets for nm, _ in SET_NEED[s])]

    def one(mu_j, sin_j):
        geo = _rsd_geometry(f, mu_j, sin_j)
        out = {'V': compute_V_mu(geo.s2, model.G00_coeffs, model.lmax),
               'Vp': compute_V_mu_prime(geo.s2, model.G00_coeffs, model.lmax)}
        if 'R_s' in sets:
            out['Vp2'] = compute_V_mu_prime2(geo.s2, model.G00_coeffs, model.lmax)
        args = (geo.c, geo.s, geo.A_mu, geo.B_mu)
        out['mono'] = jnp.stack([scalar_monomials(*args) if n == 'K'
                                 else scalar_dmonomials(*args, COEFFICIENT_VARIABLE[n]) for n in names])
        return out
    tabs = jax.vmap(one)(mu_true, sin_true)
    mono = tabs.pop('mono')                                             # (nmu, n_vec, N_MONO)
    nmu, nv, nm = mono.shape
    Kall = jnp.einsum('um,umt->ut', mono.reshape(nmu * nv, nm),
                      jnp.broadcast_to(SCALAR_MATRIX, (nmu * nv,) + SCALAR_MATRIX.shape))
    Kall = Kall.reshape(nmu, nv, Kall.shape[-1])
    tabs['Kvecs'] = {n: Kall[:, i] for i, n in enumerate(names)}
    return tabs


def fold_weights(k, mu, f, Kfac, W):
    """Wfold[o, p] = sum_e W[o, c_e] s_e [p_e = p] ``(n_out, n_pairs)``: the component weights of the entries of
    CT_ENTRIES times their per-point scalars s_e, summed on their (moment, basis row) pairs (JT_PAIRS); a gather of W
    and a scatter-add (Kfac an explicit argument)."""
    v = jnp.take(W, ENTRY_COMPONENT, axis=-1, mode='clip') * entry_scalars(k, mu, f, Kfac)[None, :]
    return jnp.zeros((W.shape[0], len(JT_PAIRS)), v.dtype).at[:, ENTRY_PAIR].add(v)


def point_kernel(model, k, derivative):
    """``(KW, KW')``: KW ``(L, nq)`` = 4 pi q^3 kern_l(q) at ln k, KW' = dKW/dln k if ``derivative`` (else None), the
    weighted kernel rows of ``LPT._point_kernel`` (the forward's interpolant)."""
    return model._point_kernel(k, derivative, weighted=True)


def point_residuals(model, k, mu, sin, f, corrs_tree, sets, Kvecs, V, Vp, Vp2, KW, KWp):
    """The residual sets of one point, ``(Y, y)``: Y[set][j] ``(nq,)`` the residual of moment j of each set of ``sets``
    (``active_sets``, SET_RULE; zero arrays for the exact zeros) and y ``(len(Y_SETS), 9)`` their q_max values (0 for
    a set not computed).  ``Kvecs``, ``V``, ``Vp``, ``Vp2`` (with R_s) the tables of ``mu_tables``; ``KW``, ``KWp`` the
    kernel of ``point_kernel``."""
    ft, L = model._final_transform, model.lmax + 1
    geo, X_lt, Y_lt = _rsd_geometry(f, mu, sin), corrs_tree[Row.X_LIN_LT], corrs_tree[Row.Y_LIN_LT]
    q, K = model._q, k * geo.Kfac
    Ksq, kq = K**2, k * q
    A, B, C = kq * geo.c, -0.5 * Ksq * Y_lt, kq * geo.s
    ir = jax.lax.rsqrt(A * A + C * C)
    D = jnp.exp(-0.5 * Ksq * (X_lt + geo.c2 * Y_lt))
    ef = (-2.0 / k) ** ft.ell_indices
    DE = [D * (ef[l] * ft.q_inv_pows[l]) for l in range(L)]
    WDE = [KW[l] * DE[l] for l in range(L)]
    # the Horner basis [b][m] and the G families over the pairs (shift s <= lmax)
    coef = jnp.concatenate([V[:1], Vp], axis=0)                          # (5, L, L)
    basis = lpt_horner_poly(B, coef, 0, BASIS_MMIN)
    pairs = [p for p in PAIRS if p[1] < L]
    G = {'G': pair_sums(WDE, basis, pairs)}
    if 'R_kE' in sets:
        G['G_kE'] = pair_sums([KWp[0] * DE[0]] + [(KWp[l] - float(l) * KW[l]) * DE[l] for l in range(1, L)], basis, pairs)
    if 'R_B' in sets:      # d basis_b[m]/dB = -(-B)^(m - 1) sum_i (m + i) coef_b[m, i] (-B)^i
        mfac = -(np.arange(L)[:, None] + np.arange(L)[None, :]).astype(np.float64)
        G['G_B'] = pair_sums(WDE, lpt_horner_poly(B, coef * mfac, -1, 1), pairs)
    if 'R_s' in sets:      # d basis_b[m]/ds = 2 s (-B)^m sum_i (d coef_b[m, i]/d s2) (-B)^i (V'[0] for S0, V'' for T_d)
        coef_s = (2.0 * geo.s) * jnp.concatenate([Vp[:1], Vp2], axis=0)
        G['G_s'] = pair_sums(WDE, lpt_horner_poly(B, coef_s, 0, 1), pairs)
    # materialize the pair sums once per point (XLA otherwise recomputes them, with the Horner basis, in every fusion
    # of the residuals that reads them)
    G = materialized(G)
    psi = psi_family(A, C, B, ir, Kvecs, frozenset(x for n in sets for x in SET_NEED[n]))
    factor = {1: 1.0, 'q': q, 'kq': kq, 'c': geo.c, 's': geo.s}

    def coefficient(outer, terms, key):
        """outer * sum factor psi[name]_deriv[key] of a SET_RULE term (exact zeros skipped)."""
        v = static_lin([(factor[a], psi[(nm, d)].get(key, 0.0)) for a, nm, d in terms])
        return v if outer is None else static_mul(factor[outer], v)

    zero_q, D_last = jnp.zeros_like(q), D[-1]
    res, yv = {}, {}                  # per set: Y^set_j (a (nq,) array or the exact zero 0.0) and y^set_j
    for name in sets:
        rule = SET_RULE[name]
        w0 = KWp[0] if rule[0][2] == 'G_kE' else KW[0]
        res[name], yv[name] = [], []
        for j in range(len(MOMENTS)):
            co = {p: [(coefficient(outer, terms, (j,) + p), G[fam][p]) for outer, terms, fam in rule]
                  for p in PAIRS_OF[j] if p[1] < L}
            Y = static_lin([t for p in co for t in co[p]])
            x0 = 0.0
            c0 = co[('S0', 0)][0][0]                         # F_j(0): the first coefficient of the pair (S0, 0)
            if not (isinstance(c0, float) and c0 == 0.0):
                x0 = D_last * static_last(c0)
                Y = Y - w0 * x0
            res[name].append(Y)
            yv[name].append(x0)

    Y = {name: tuple(zero_q if isinstance(v, float) else v for v in res[name]) for name in sets}
    zero = jnp.zeros((), q.dtype)
    ys = [zero if (name not in yv or isinstance(v, float)) else v
          for name in Y_SETS for v in (yv[name] if name in yv else [0.0] * len(MOMENTS))]
    return Y, jnp.stack(ys).reshape(len(Y_SETS), len(MOMENTS))


def point_loop_jvp(model, W_fn, primals, tangents):
    """The jvp rule of ``LPT._point_loop``: primal = Wfold . Z^0 + sum_c W small_c and the tangent groups (each skipped
    when its input tangents are symbolic zeros):
      d      t_d Jfull[d] per active d, with the point's Jacobian along the unit tangent e_d (primal-only)
               Jfull[d] = Wfold . Z^d + dWfold_d . Z^0 + sum_c W_c dsmall_d,c,
             Z^d the fold of the contracted sets (module docstring), dWfold_d the jvp of ``fold_weights`` along
             (e_d[k], e_d[mu], e_d[f], J^d_Kfac, 0), dsmall_d the jvp of ``small_terms`` (the point's kernel KW fixed)
             along (0, e_d[k], e_d[mu], e_d[sin], e_d[f]) plus, for d = k, (small(KW') - small(0)) / k (small is
             affine in KW and dKW/dk = KW'/k)
      rows   Wfold . [C(Y^0, dR) - (Ksq/2) C(Y^0, R dX_lt) - (Ksq c^2/2) C(Y^0, R dY_lt) - (Ksq/2) C(Y^B, R dY_lt)]
             + sum_c W_c dsmall_rows,c (the jvp of ``small_terms`` in the rows alone), C(Y, Q) = Y_j . Q_t + y_j
             KW_0 . (Q_t - Q_t[last]) on the pairs (j, t)
      W      sum_c dW_c small_c + Wfold(dW) . Z^0 (``fold_weights`` at dW, as the fold weights are linear in W)
      P_lin  sum_c W_c lin_c dP.
    J^d is the jvp of ``rsd_scalars`` along e_d.  Each jvp is linear in its tangent, so the jvp along the actual
    tangents is sum_d t_d times the jvp along e_d.  Of fold_weights, rsd_scalars and small_terms only the rows direction
    of small_terms is linearized with the rule's tangents (the linear part that reverse mode transposes, reading the
    residuals Y^0 and Y^B).
    """
    k_true, mu_true, sin_true, pk_lin_true, rows, f, W_args = primals
    dk, dmu, dsin, dp, drows, df, dW_args = tangents
    nz = lambda t: not isinstance(t, SymbolicZero)
    val = lambda t: t if nz(t) else None
    (nk, nmu), tree = k_true.shape, rows[TREE_ROWS]
    mu_b = jnp.broadcast_to(mu_true, k_true.shape)
    W, dW = component_weights_jvp(W_fn, k_true, mu_b, W_args, dk, dmu, dW_args)
    w_axis, dw_axis = (0 if W.ndim == 4 else None), (None if (dW is None or dW.ndim == 2) else 0)
    t_rows, t_k, t_mu, t_sin, t_f = val(drows), val(dk), val(dmu), val(dsin), val(df)
    t_dir = {'k': t_k, 'mu': t_mu, 'sin': t_sin, 'f': t_f}
    dirs = tuple(d for d in DIRECTIONS if t_dir[d] is not None)          # the active input directions
    sets = active_sets(dirs, t_rows is not None)
    tabs = mu_tables(model, f, mu_true, sin_true, sets)

    def per_mu(k_i, mu_j, sin_j, p_i, W_ij, tW, tab):
        prim = (k_i, mu_j, sin_j, f)  # the unit tangent e_d of each active direction, the Jacobians J^d of the scalars
        unit = {d: tuple(jnp.ones_like(x) if n == DIRECTIONS.index(d) else jnp.zeros_like(x)
                         for n, x in enumerate(prim)) for d in dirs}
        J = {d: jax.jvp(rsd_scalars, prim, unit[d])[1] for d in dirs}
        # the kernel materialized once per point (XLA otherwise recomputes its interpolation in every fusion reading it)
        KW, KWp = materialized(point_kernel(model, k_i, 'R_kE' in sets))
        Y, y = point_residuals(model, k_i, mu_j, sin_j, f, tree, sets, tab['Kvecs'], tab['V'], tab['Vp'],
                               tab.get('Vp2'), KW, KWp)
        r = dict(Y=Y, y=y, KW0=KW[0], J=J) | ({} if KWp is None else {'KWp0': KWp[0]})
        sc = rsd_scalars(k_i, mu_j, sin_j, f)
        # small_terms with the point's kernel KW held fixed; its k dependence through KW added explicitly
        small_fn = lambda rr, kk, mm, ss, ff: small_terms(model, rr, kk, mm, ss, ff, p_i, KW)
        small, lin_c = small_fn(rows, k_i, mu_j, sin_j, f)
        r.update(sc=sc, wfold=fold_weights(k_i, mu_j, f, sc[2], W_ij), small=small, lin_c=lin_c)
        # the jvps along e_d (primal-only), fold_weights with Kfac along J^d_Kfac and W inactive, small_terms with the
        # rows inactive, plus for d = k the kernel's k dependence (small(KW') - small(0)) / k
        for d in dirs:
            u = dict(zip(DIRECTIONS, unit[d]))
            r['dwfold_' + d] = jvp_if_active(fold_weights, (k_i, mu_j, f, sc[2], W_ij),
                                             (u['k'], u['mu'], u['f'], J[d][2], None))[1]
            dsm = jvp_if_active(small_fn, (rows, k_i, mu_j, sin_j, f), (None, u['k'], u['mu'], u['sin'], u['f']))[1][0]
            if d == 'k':
                small_p = small_terms(model, rows, k_i, mu_j, sin_j, f, p_i, KWp)[0]
                small_0 = small_terms(model, rows, k_i, mu_j, sin_j, f, p_i, 0.0 * KW)[0]
                dsm = dsm + (small_p - small_0) / k_i
            r['dsmall_' + d] = dsm
        if t_rows is not None:        # the rows direction of the small terms (in the linear part)
            r['dsmall_rows'] = jvp_if_active(small_fn, (rows, k_i, mu_j, sin_j, f),
                                             (t_rows, None, None, None, None))[1][0]
        if tW is not None:            # the fold weights at dW (linear in W, no jvp)
            r['wfold_dW'] = fold_weights(k_i, mu_j, f, sc[2], tW)
        return r

    def per_k(k_row, p_row, W_row, tW_row):
        return jax.vmap(per_mu, in_axes=(0, 0, 0, 0, w_axis, dw_axis, 0))(
            k_row, mu_true, sin_true, p_row, W_row, tW_row, tabs)
    res = jax.vmap(per_k, in_axes=(0, 0, w_axis, dw_axis))(k_true, pk_lin_true, W, dW)

    Y, KW0, wfold = res['Y'], res['KW0'], res['wfold']   # Y[set][j] (nk, nmu, nq), wfold (nk, nmu, n_out, n_pairs)
    c, s, Kfac, K, Ksq, A_mu, B_mu = res['sc']
    R = row_basis(rows)
    X_lt, Y_lt = rows[Row.X_LIN_LT], rows[Row.Y_LIN_LT]
    y = {name: on_pairs(res['y'][:, :, i]) for i, name in enumerate(Y_SETS)}    # the q_max values on the pairs
    fold = lambda Z: jnp.einsum('kmop,kmp->kmo', wfold, Z)
    e = lambda x: x[..., None]
    # the sets contracted with the basis rows on the pairs, each with its q_max term: Z[set] = Y^set . R + y^set rho;
    # the damping's chain rule needs Y^0 and Y^B with R X_lt and R Y_lt too, and R_kE has the q_max value y^0 of KW'_0
    damping = any(x in ('c', 'Ksq') for d in dirs for x in SCALAR_TERMS[d])
    rho = edge(KW0, R)
    Z0 = contract_pairs(Y['R0'], [R] + ([R * X_lt, R * Y_lt] if damping else []))
    Z = {'R0': Z0[0] + y['R0'] * rho}
    if damping:
        Z['R0X'] = Z0[1] + y['R0'] * edge(KW0, R * X_lt)
        Z['R0Y'] = Z0[2] + y['R0'] * edge(KW0, R * Y_lt)
        Z['RBY'] = contract_pairs(Y['R_B'], [R * Y_lt])[0] + y['R_B'] * edge(KW0, R * Y_lt)
    for name in sets:
        if name in SET_OF_COEFFICIENT.values() or name in ('R_c', 'R_k'):
            Z[name] = contract_pairs(Y[name], [R])[0] + y[name] * rho
    if 'R_kE' in sets:
        Z['R_kE'] = contract_pairs(Y['R_kE'], [R])[0] + y['R0'] * edge(res['KWp0'], R)

    def folded(d):
        """Z^d of the module docstring, ``(nk, nmu, n_pairs)``."""
        Jd = {x: e(v) for x, v in zip(SCALARS, res['J'][d])}
        terms = []
        for x in SCALAR_TERMS[d]:
            if x == 'c':
                terms.append(Jd['c'] * (Z['R_c'] - e(Ksq * c) * Z['R0Y']))
            elif x == 'Ksq':
                terms.append(-0.5 * Jd['Ksq'] * (Z['R0X'] + e(c**2) * Z['R0Y'] + Z['RBY']))
            else:
                terms.append(Jd[x] * Z[SET_OF_COEFFICIENT[x]])
        if d == 'k':
            terms.append(Z['R_k'] + Z['R_kE'] / e(k_true))
        return sum(terms[1:], terms[0])

    primal = fold(Z['R0']) + contract_weights(W, res['small'])
    tangent = jnp.zeros_like(primal)
    for d in dirs:        # the point's Jacobian along e_d (primal-only) times t_d
        Jfull = (fold(folded(d)) + jnp.einsum('kmop,kmp->kmo', res['dwfold_' + d], Z['R0'])
                 + contract_weights(W, res['dsmall_' + d]))
        tangent = tangent + e(jnp.broadcast_to(t_dir[d], (nk, nmu))) * Jfull
    if t_rows is not None:
        dR = jax.jvp(row_basis, (rows,), (drows,))[1]
        RdX, RdY = R * drows[Row.X_LIN_LT], R * drows[Row.Y_LIN_LT]
        # Y^0 with [dR; R dX_lt; R dY_lt] and Y^B with R dY_lt, one GEMM per moment each
        Z1, Z2, Z3 = contract_pairs(Y['R0'], [dR, RdX, RdY])
        ZB = contract_pairs(Y['R_B'], [RdY])[0]
        tangent = tangent + (fold(Z1 + y['R0'] * edge(KW0, dR)) - e(0.5 * Ksq) * fold(Z2 + y['R0'] * edge(KW0, RdX))
                             - e(0.5 * Ksq * c**2) * fold(Z3 + y['R0'] * edge(KW0, RdY))
                             - e(0.5 * Ksq) * fold(ZB + y['R_B'] * edge(KW0, RdY)))
        tangent = tangent + contract_weights(W, res['dsmall_rows'])
    if dW is not None:
        tangent = tangent + contract_weights(dW, res['small']) + jnp.einsum('kmop,kmp->kmo', res['wfold_dW'], Z['R0'])
    if nz(dp):
        tangent = tangent + contract_weights(W, res['lin_c']) * dp[:, :, None]
    return primal, tangent
