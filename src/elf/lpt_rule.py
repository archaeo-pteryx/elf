"""The tangent rule of the LPT point loop (``lpt.LPT._point_loop``, a ``jax.custom_jvp``): ``point_loop_jvp``.

Per point (k, mu) the forward contracts g_l = D E_l sum_j c_j M_j(l) (D the IR damping, E_l = (-2/k)^l q^-l, M_j the
nine moments, c_j(q) the moment coefficients) with the kernel KW[l, q] = 4 pi q^3 kern_l(q) at ln k:
out[g] = sum_q sum_l KW[l, q] g_l(q) - (sum_q KW[0, q]) g_0(q_max).  As c_j is linear in the basis rows R_t(q)
(``row_basis``) with per-point scalars, out[c M_j] = sum_q c[q] Y_j[q] + y_j sum_q KW[0, q] (c[q] - c[last]) with the
residuals Y_j = sum_l KW_l D E_l M_j(l) - KW_0 y_j, y_j = D E_0 M_j(0) at q_max (split before the weights, exact at
large q).  ``point_residuals`` computes this bundle per point, independent of the weights W and of the rows: R0 (the
primal), one folded set D_d per active input direction d of DIRECTIONS (the per-point chain rule through
``rsd_scalars`` applied inside the bundle) and R_B (rows tangent).  ``point_loop_jvp`` contracts it with R in one GEMM
over the points (``basis_contract``) and folds it with the component weights (``fold_weights``).  Per active direction d
it forms the point's whole Jacobian along d in the primal part (the folded set plus the unit-direction jvps of the fold
weights and of the small terms) and adds t_d times it to the tangent groups of the rows, W and P_lin.

Residual sets: X_j(l) = D E_l F_j(l), Y_j = sum_l w_l X_j(l) - w_0 y_j, y_j = X_j(0) at q_max, w = KW unless noted
(A = k q c, C = k q s, B = -(1/2) K^2 Y_lt; c, s the cosine and sine of the angle between K and k):
    R0    F = M                         R_B   F = dM/dB                     R_k   F = q (c dM/dA + s dM/dC)
    R_c   F = k q dM/dA + dM/dc         R_s   F = k q dM/dC + dM/ds         R_Am  F = dM/dA_mu     R_Bm  F = dM/dB_mu
    R_kap F = M, w = KW' = dKW/dln k (the y_j of R0)                       R_E   F = M, w_l = -l KW_l (y_j = 0)
With M_j(l) = sum_{(b, s) in PAIRS_OF[j], s <= l} basis_b[l - s] psi_{j,b,s} (``lpt_tables``) every sum over l collapses
onto the pairs, sum_l w_l D E_l F_j(l) = sum_pairs (coefficient) G^w_{b,s} with G^w_{b,s} = sum_{l >= s} w_l D E_l
basis_b[l - s] (``pair_sums``); SET_RULE lists the terms.  At l = 0 only the pair (S0, 0) contributes (basis_S0[0] = 1,
basis_Td[0] = 0, the derivative bases vanish at m = 0): y_j = D(q_max) times the first coefficient of (S0, 0).
Folded direction d, J^d_x = dx/dd of the scalars of ``rsd_scalars`` (the terms of identically zero J^d_x not built):
    Y^d = J^d_c [Y^c - Ksq c Y_lt Y^0] + J^d_s Y^s + J^d_Am Y^Am + J^d_Bm Y^Bm
          - (J^d_Ksq / 2) [X_lt Y^0 + c^2 Y_lt Y^0 + Y_lt Y^B] + [d = k] (Y^k + (Y^kap + Y^E) / k).

Scope.  The rule is written for the density power spectrum, and these parts of it must be generalised for another
observable.  The damping D = exp(-K^2 (X_lt + c^2 Y_lt) / 2) and its chain rule through B = -(1/2) K^2 Y_lt are written
out (the X_lt and Y_lt terms of Y^d above, of ``point_residuals`` and of the q_max terms of ``point_loop_jvp``).  The
linear-part and DC corrections of ``small_terms`` (``LPT._linear_rows``, ``_LINEAR_MATRIX`` and ``_KSPACE_ROWS``) are
those of the density components.  ``row_basis`` builds each basis row R_t as the constant, a single row or a product of
at most two rows.  The moments are expanded in the Horner basis S0, T1 .. T4 (``BASIS_MMIN``, the families G_B and
G_s), and the q_max endpoint values y_j take F_j(0) from the pair (S0, 0) alone.  Only the numbers of moments and of
basis functions are read from the tables (``len(MOMENTS)``, ``BASIS``).
"""
import numpy as np

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from . import spline
from .lpt_defs import _rsd_geometry
from .lpt_tables import (BASIS, COEFFS, CT_ENTRY_NAMES, MOMENTS, N_COMPONENTS, N_CT, PAIRS, PAIRS_OF, SCALAR_MATRIX,
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
K_SETS = ('R_k', 'R_kap', 'R_E')                            # the extra sets of d = k
Y_SETS = ('R0', 'R_c', 'R_s', 'R_Am', 'R_Bm', 'R_B', 'R_k', 'R_kap')   # the rows of y
# set -> its terms (outer factor, ((factor, name, deriv), ...), G family): the coefficient of the pair key is
# outer * sum factor psi[name]_deriv[key], factors 1, q, k q (kq), c, s; the first term gives y_j.  The G families:
# G (w = KW), G_kap (w = KW'), G_E (w_l = -l KW_l), G_B (basis -> d basis/dB), G_s (basis -> d basis/ds).
SET_RULE = {
    'R0': ((None, ((1, 'K', 'psi'),), 'G'),),                                          # psi[K] G
    'R_c': ((None, (('kq', 'K', 'dA'), (1, 'Kc', 'psi')), 'G'),),                      # (k q psi_A[K] + psi[K^c]) G
    'R_s': ((None, (('kq', 'K', 'dC'), (1, 'Ks', 'psi')), 'G'),                        # (k q psi_C[K] + psi[K^s]) G
            (None, ((1, 'K', 'psi'),), 'G_s')),                                        #   + psi[K] G^s
    'R_Am': ((None, ((1, 'KAm', 'psi'),), 'G'),),                                      # psi[K^Am] G
    'R_Bm': ((None, ((1, 'KBm', 'psi'),), 'G'),),                                      # psi[K^Bm] G
    'R_k': (('q', (('c', 'K', 'dA'), ('s', 'K', 'dC')), 'G'),),                        # q (c psi_A[K] + s psi_C[K]) G
    'R_kap': ((None, ((1, 'K', 'psi'),), 'G_kap'),),                                   # psi[K] G^kap
    'R_E': ((None, ((1, 'K', 'psi'),), 'G_E'),),                                       # psi[K] G^E
    'R_B': ((None, ((1, 'K', 'dB'),), 'G'), (None, ((1, 'K', 'psi'),), 'G_B')),         # psi_B[K] G + psi[K] G^B
}
SET_NEED = {n: tuple(dict.fromkeys((nm, d) for _, terms, _ in rule for _, nm, d in terms))
            for n, rule in SET_RULE.items()}                 # set -> the (name, deriv) of psi_family it reads
COEFFICIENT_VARIABLE = {name: x for name, x in COEFFS.items() if x}   # K^x = scalar_dmonomials(x) @ SCALAR_MATRIX
BASIS_INDEX = {b: n for n, b in enumerate(BASIS)}
BASIS_MMIN = (0,) + (1,) * (len(BASIS) - 1)                # basis_Td[0] = V'[d - 1, 0, 0] = 0 identically
CT_ENTRIES = tuple((int(Component[c]), MOMENTS.index(m), t) for c, m, t in CT_ENTRY_NAMES)   # (component, moment, t)
# COMPONENT_SELECT[c, e] = ENTRY_SCATTER[e, j N_CT + t] = 1 for the entries e = (c, j, t)
COMPONENT_SELECT = np.zeros((N_COMPONENTS, len(CT_ENTRIES)))
ENTRY_SCATTER = np.zeros((len(CT_ENTRIES), len(MOMENTS) * N_CT))
for _e, (_c, _j, _t) in enumerate(CT_ENTRIES):
    COMPONENT_SELECT[_c, _e], ENTRY_SCATTER[_e, _j * N_CT + _t] = 1.0, 1.0


def basis_contract(Y, g0, KW0, Q):
    """``(nk, nmu, ..., 9, n_t)``: Y[j] . Q_t + g0[j] KW0 . (Q_t - Q_t[last]), i.e. out[c G_j] summed for c = Q_t,
    with Y ``(nk, nmu, ..., 9, nq)``, g0 ``(nk, nmu, ..., 9)`` or None, KW0 ``(nk, nmu, nq)``, Q ``(n_t, nq)``."""
    out = jnp.einsum('km...q,tq->km...t', Y, Q)
    if g0 is None:
        return out
    rho = jnp.einsum('kmq,tq->kmt', KW0, Q - Q[:, -1:])
    rho = rho.reshape(rho.shape[:2] + (1,) * (Y.ndim - 3) + rho.shape[2:])
    return out + g0[..., None] * rho


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
    values of c + [c = XI_XI] out[-xi^2/2], lin_c = lin_kind(c); the output is sum_c W[o, c] (comp_c + small_c)."""
    geometry, V = _rsd_geometry(f, mu, sin), model._angular_table(f, mu, sin)
    ft, corrs_tree, K = model._final_transform, rows[TREE_ROWS], k * geometry.Kfac
    ell_factors = ((-2.0 / k) ** ft.ell_indices)[:, None] * ft.q_inv_pows
    lin_weight, za, b1, xi, kaiser = model._linear_rows(mu, f, V, geometry, K, K**2, ell_factors, corrs_tree)
    out0 = lambda x: KW[0] @ (x - x[-1])                             # l = 0 with the q_max subtraction
    s_kind = -lin_weight * jnp.stack([out0(za[0]) + KW[1] @ za[1] + KW[2] @ za[2], KW[1] @ b1, out0(xi)])
    lin_kind, lin_mat = lin_weight * jnp.stack(kaiser), jnp.asarray(_LINEAR_MATRIX)
    small = lin_mat @ (s_kind + lin_kind * p)
    small = small.at[Component.XI_XI].add(-out0(0.5 * corrs_tree[Row.XI_LIN]**2))
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
    """Wfold[o, j, t] = sum_e W[o, c_e] s_e [j_e = j, t_e = t] ``(n_out, 9, N_CT)``: the component weights on the
    (moment, basis row) entries of CT_ENTRIES times their per-point scalars s_e (Kfac an explicit argument)."""
    s_e = entry_scalars(k, mu, f, Kfac)
    v = (W @ jnp.asarray(COMPONENT_SELECT)) * s_e[None, :]
    return (v @ jnp.asarray(ENTRY_SCATTER)).reshape(W.shape[0], len(MOMENTS), N_CT)


def point_kernel(model, k, derivative):
    """``(KW, KW')``: KW ``(L, nq)`` = 4 pi q^3 kern_l(q) at ln k, KW' = dKW/dln k if ``derivative`` (else None), as
    one contraction of spline.hermite's weights with the 4 nodes (value and slope at both ends of the interval) taken
    from the forward's kernel arrays pk_select_kernel / pk_select_slopes ([l, k, q], read in place, no copy)."""
    ft, x, logk = model._final_transform, model._logk_fft, jnp.log(k)
    idx = spline._interval_index(x, logk)
    h_i = (x[1:] - x[:-1])[idx]
    t = (logk - x[idx]) / h_i
    w = jnp.stack(spline.hermite_weights(t, h_i))                        # weights of y0, m0, y1, m1
    node = lambda a, i: jnp.take(a, i, axis=1, mode='clip')              # the (L, q) rows of node i
    nodes = jnp.stack([node(ft.pk_select_kernel, idx), node(ft.pk_select_slopes, idx),
                       node(ft.pk_select_kernel, idx + 1), node(ft.pk_select_slopes, idx + 1)])   # y0, m0, y1, m1
    if not derivative:
        return ft.four_pi_q3 * jnp.einsum('n,nlq->lq', w, nodes), None
    wp = jnp.stack(spline.hermite_weight_derivatives(t, h_i))            # d/d ln k of the weights
    both = jnp.einsum('an,nlq->alq', jnp.stack([w, wp]), nodes)
    return ft.four_pi_q3 * both[0], ft.four_pi_q3 * both[1]


def point_residuals(model, k, mu, sin, f, corrs_tree, outs, J, Kvecs, V, Vp, Vp2, KW, KWp):
    """The residual bundle of one point, ``(Y (n_out, 9, nq), y (8, 9))``: the outputs ``outs`` ('R0', 'D_d' per active
    direction in DIRECTIONS order, 'R_B' last with the rows tangent; the sets of ``active_sets`` are intermediates) and
    the q_max values y of the sets in Y_SETS order (0 for a set not computed).  ``J`` = {d: J^d}; ``Kvecs``, ``V``,
    ``Vp``, ``Vp2`` (with R_s) the tables of ``mu_tables``; ``KW``, ``KWp`` the kernel of ``point_kernel``."""
    ft, L = model._final_transform, model.lmax + 1
    dirs = tuple(o[2:] for o in outs if o.startswith('D_'))
    sets = active_sets(dirs, 'R_B' in outs)
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
    if 'R_kap' in sets:
        G['G_kap'] = pair_sums([KWp[l] * DE[l] for l in range(L)], basis, pairs)
    if 'R_E' in sets:
        G['G_E'] = pair_sums([0.0] + [(-float(l)) * WDE[l] for l in range(1, L)], basis, pairs)
    if 'R_B' in sets:      # d basis_b[m]/dB = -(-B)^(m - 1) sum_i (m + i) coef_b[m, i] (-B)^i
        mfac = -(np.arange(L)[:, None] + np.arange(L)[None, :]).astype(np.float64)
        G['G_B'] = pair_sums(WDE, lpt_horner_poly(B, coef * mfac, -1, 1), pairs)
    if 'R_s' in sets:      # d basis_b[m]/ds = 2 s (-B)^m sum_i (d coef_b[m, i]/d s2) (-B)^i (V'[0] for S0, V'' for T_d)
        coef_s = (2.0 * geo.s) * jnp.concatenate([Vp[:1], Vp2], axis=0)
        G['G_s'] = pair_sums(WDE, lpt_horner_poly(B, coef_s, 0, 1), pairs)
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
        w0 = KWp[0] if rule[0][2] == 'G_kap' else KW[0]
        res[name], yv[name] = [], []
        for j in range(len(MOMENTS)):
            co = {p: [(coefficient(outer, terms, (j,) + p), G[fam][p]) for outer, terms, fam in rule]
                  for p in PAIRS_OF[j] if p[1] < L}
            Y = static_lin([t for p in co for t in co[p]])
            x0 = 0.0
            if rule[0][2] != 'G_E':
                c0 = co[('S0', 0)][0][0]                     # F_j(0): the first coefficient of the pair (S0, 0)
                if not (isinstance(c0, float) and c0 == 0.0):
                    x0 = D_last * static_last(c0)
                    Y = Y - w0 * x0
            res[name].append(Y)
            yv[name].append(x0)

    def folded(d, j):
        """Y^d_j of the module docstring (exact zeros skipped; 0.0 if every term is one)."""
        Jd = dict(zip(SCALARS, J[d]))
        Y0 = res['R0'][j]
        terms = []
        for x in SCALAR_TERMS[d]:
            if x == 'c':
                terms.append((Jd['c'], static_lin([(1.0, res['R_c'][j]), (-(Ksq * geo.c), static_mul(Y_lt, Y0))])))
            elif x == 'Ksq':
                terms.append((-0.5 * Jd['Ksq'], static_lin([(X_lt, Y0), (geo.c2, static_mul(Y_lt, Y0)),
                                                            (Y_lt, res['R_B'][j])])))
            else:
                terms.append((Jd[x], res[SET_OF_COEFFICIENT[x]][j]))
        if d == 'k':
            terms.append((1.0, static_lin([(1.0, res['R_k'][j]),
                                           (1.0 / k, static_lin([(1.0, res['R_kap'][j]), (1.0, res['R_E'][j])]))])))
        return static_lin(terms)

    Ys = [folded(o[2:], j) if o.startswith('D_') else res[o][j] for o in outs for j in range(len(MOMENTS))]
    Y = jnp.stack([zero_q if isinstance(Yo, float) else Yo for Yo in Ys]).reshape(len(outs), len(MOMENTS), q.shape[0])
    zero = jnp.zeros((), q.dtype)
    ys = [zero if (name not in yv or isinstance(v, float)) else v
          for name in Y_SETS for v in (yv[name] if name in yv else [0.0] * len(MOMENTS))]
    return Y, jnp.stack(ys).reshape(len(Y_SETS), len(MOMENTS))


def point_loop_jvp(model, W_fn, primals, tangents):
    """The jvp rule of ``LPT._point_loop``: primal = Wfold . Yt + sum_c W small_c with Yt = BC(Y^0, y^0, KW0, R) (BC =
    ``basis_contract``, R = ``row_basis``, '.' the contraction with Wfold = ``fold_weights``) and the tangent groups
    (each skipped when its input tangents are symbolic zeros):
      d      t_d Jfull[d] per active d, with the point's Jacobian along the unit tangent e_d (primal-only)
               Jfull[d] = Wfold . [Y^d . R + q_max terms] + dWfold_d . Yt + sum_c W_c dsmall_d,c,
             dWfold_d the jvp of ``fold_weights`` along (e_d[k], e_d[mu], e_d[f], J^d_Kfac, 0), dsmall_d the jvp of
             ``small_terms`` (the point's kernel KW fixed) along (0, e_d[k], e_d[mu], e_d[sin], e_d[f]) plus, for d = k,
             (small(KW') - small(0)) / k (small is affine in KW and dKW/dk = KW'/k).  The q_max terms per point
               J^d_c [y^c rho - Ksq c y^0 rho^RY] + J^d_s y^s rho + J^d_Am y^Am rho + J^d_Bm y^Bm rho
               - (J^d_Ksq / 2) [y^0 rho^RX + c^2 y^0 rho^RY + y^B rho^RY] + [d = k] (y^k rho + (1/k) y^0 rho'),
             rho = KW0 . (R - R[last]), rho^RX, rho^RY the same with R X_lt, R Y_lt, rho' = KW'0 . (R - R[last])
      rows   Wfold . [BC(Y^0, y^0, KW0, dR) - (Ksq/2) BC(Y^0, y^0, KW0, R dX_lt)
             - (Ksq c^2/2) BC(Y^0, y^0, KW0, R dY_lt) - (Ksq/2) BC(Y^B, y^B, KW0, R dY_lt)]
             + sum_c W_c dsmall_rows,c (the jvp of ``small_terms`` in the rows alone)
      W      sum_c dW_c small_c + Wfold(dW) . Yt (``fold_weights`` at dW, as the fold weights are linear in W)
      P_lin  sum_c W_c lin_c dP.
    J^d is the jvp of ``rsd_scalars`` along e_d.  Each jvp is linear in its tangent, so the jvp along the actual
    tangents is sum_d t_d times the jvp along e_d.  Of fold_weights, rsd_scalars and small_terms only the rows direction
    of small_terms is linearized with the rule's tangents (the linear part that reverse mode transposes).
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
    # the outputs: R0, the folded directions (contracted with R: a leading slice), R_B (rows tangent) last
    outs = ('R0',) + tuple('D_' + d for d in dirs) + (('R_B',) if t_rows is not None else ())
    sets = active_sets(dirs, t_rows is not None)
    tabs = mu_tables(model, f, mu_true, sin_true, sets)

    def per_mu(k_i, mu_j, sin_j, p_i, W_ij, tW, tab):
        prim = (k_i, mu_j, sin_j, f)  # the unit tangent e_d of each active direction, the Jacobians J^d of the scalars
        unit = {d: tuple(jnp.ones_like(x) if n == DIRECTIONS.index(d) else jnp.zeros_like(x)
                         for n, x in enumerate(prim)) for d in dirs}
        J = {d: jax.jvp(rsd_scalars, prim, unit[d])[1] for d in dirs}
        KW, KWp = point_kernel(model, k_i, 'R_kap' in sets)
        Y, y = point_residuals(model, k_i, mu_j, sin_j, f, tree, outs, J, tab['Kvecs'], tab['V'], tab['Vp'],
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

    Y, y, KW0, wfold = res['Y'], res['y'], res['KW0'], res['wfold']
    c, s, Kfac, K, Ksq, A_mu, B_mu = res['sc']
    R = row_basis(rows)
    X_lt, Y_lt = rows[Row.X_LIN_LT], rows[Row.Y_LIN_LT]
    fold = lambda Z: jnp.einsum('kmojt,kmjt->kmo', wfold, Z)
    ys = lambda name: y[:, :, Y_SETS.index(name)]
    bc = lambda name, Q: basis_contract(Y[:, :, outs.index(name)], ys(name), KW0, Q)
    # the outputs contracted with R (all but R_B, a leading slice): one GEMM, then the q_max terms per output
    nR = len(outs) - (1 if 'R_B' in outs else 0)
    ZR = basis_contract(Y[:, :, :nR], None, KW0, R)                     # (nk, nmu, n_R, 9, N_CT)
    Rd = R - R[:, -1:]
    rho = jnp.einsum('kmq,tq->kmt', KW0, Rd)
    outer = lambda a, b: a[..., :, None] * b[..., None, :]                 # (nk, nmu, 9, N_CT)
    e2 = lambda x: x[..., None, None]
    if any(x in ('c', 'Ksq') for d in dirs for x in SCALAR_TERMS[d]):
        RX, RY = R * X_lt, R * Y_lt
        rho_RX = jnp.einsum('kmq,tq->kmt', KW0, RX - RX[:, -1:])
        rho_RY = jnp.einsum('kmq,tq->kmt', KW0, RY - RY[:, -1:])
    if 'k' in dirs:
        rho_p = jnp.einsum('kmq,tq->kmt', res['KWp0'], Rd)
    qmax = [outer(ys('R0'), rho)]
    for d in dirs:
        Jd = dict(zip(SCALARS, res['J'][d]))
        terms = []
        for x in SCALAR_TERMS[d]:
            if x == 'c':
                terms.append(e2(Jd['c']) * (outer(ys('R_c'), rho) - e2(Ksq * c) * outer(ys('R0'), rho_RY)))
            elif x == 'Ksq':
                terms.append(-e2(0.5 * Jd['Ksq']) * (outer(ys('R0'), rho_RX) + e2(c**2) * outer(ys('R0'), rho_RY)
                                                     + outer(ys('R_B'), rho_RY)))
            else:
                terms.append(e2(Jd[x]) * outer(ys(SET_OF_COEFFICIENT[x]), rho))
        if d == 'k':
            terms.append(outer(ys('R_k'), rho) + e2(1.0 / k_true) * outer(ys('R0'), rho_p))
        qmax.append(sum(terms[1:], terms[0]))
    ZR = ZR + jnp.stack(qmax, axis=2)
    FR = jnp.einsum('kmojt,kmsjt->kmso', wfold, ZR)                        # (nk, nmu, n_R, n_out)
    Yt = ZR[:, :, 0]
    primal = FR[:, :, 0] + contract_weights(W, res['small'])
    # the point's Jacobian along each active direction, primal-only, (nk, nmu, n_out)
    Jfull = {d: FR[:, :, 1 + i] + jnp.einsum('kmojt,kmjt->kmo', res['dwfold_' + d], Yt)
             + contract_weights(W, res['dsmall_' + d]) for i, d in enumerate(dirs)}
    e = lambda x: x[..., None]
    tangent = jnp.zeros_like(primal)
    for d in dirs:
        tangent = tangent + e(jnp.broadcast_to(t_dir[d], (nk, nmu))) * Jfull[d]
    if t_rows is not None:
        dR = jax.jvp(row_basis, (rows,), (drows,))[1]
        dX_lt, dY_lt = drows[Row.X_LIN_LT], drows[Row.Y_LIN_LT]
        tangent = tangent + (fold(bc('R0', dR)) - e(0.5 * Ksq) * fold(bc('R0', R * dX_lt))
                             - e(0.5 * Ksq * c**2) * fold(bc('R0', R * dY_lt))
                             - e(0.5 * Ksq) * fold(bc('R_B', R * dY_lt)))
        tangent = tangent + contract_weights(W, res['dsmall_rows'])
    if dW is not None:
        tangent = tangent + contract_weights(dW, res['small']) + jnp.einsum('kmojt,kmjt->kmo', res['wfold_dW'], Yt)
    if nz(dp):
        tangent = tangent + contract_weights(W, res['lin_c']) * dp[:, :, None]
    return primal, tangent
