"""The tangent rule of the LPT point loop (``lpt.LPT._point_loop``, a ``jax.custom_jvp``): ``point_loop_jvp``.

The forward at one point (k, mu).  The integrand of order L of j_L(kq) is g_L = D_X sum_{j: (j, L) a row} c_j C_L^j,
with D_X = exp(-K^2 X_lt / 2), the angular factors C_L^j = sum_m G[r, m] J_m(B(q)) of the rows r = (j, L) of the
angular plan (G per mu, B = -(1/2) K^2 Y_lt, ``angular``) and the moment coefficients c_j(q).  With the kernel
KW[L, q] = 4 pi q^3 kern_L(q) at k (``LPT._point_kernel``), out = sum_{L, q} KW[L, q] g_L(q) - (sum_q KW[0, q])
g_0(q_max).  The coefficients are linear in the basis rows R_t(q) (``row_basis``) with per-point weights,
c_j = sum_t Wfold[j, t] R_t (``fold_weights``), and only the (moment j, basis row t) pairs of the component table have
a weight (``PAIRS``, 45 pairs; T_j the basis rows t of moment j).  So

    out = sum_{(j, t) in PAIRS} Wfold[j, t] (Y_j . R_t + y_j rho_t),          rho_t = KW_0 . (R_t - R_t[q_max]),
    Y_j = D_X sum_{L: (j, L) a row} KW_L C_L^j - KW_0 y_j,      y_j = D_X(q_max) C_0^j(q_max)   (0 for odd n_j).

Residual sets.  Per point, from one J(B(q)) and the stacked per-mu matrices [G; G Lambda; d_d G] of ``mu_tables``,
``moment_residuals`` makes Y_j and y_j of the sets

    S0     G         C = G J              the primal
    S_B    G Lambda  dC/dB = G Lambda J   Lambda the static tridiagonal dJ/dB (``angular.lam_matrix``)
    S_d    d_d G     dC/dd = (d_d G) J    one set per active d in G_DIRECTIONS (mu, sin, f): G depends on these only
    S_kap  G         C with the kernel's ln k derivative KW' = dKW/dln k in place of KW

in two groups per moment, main_j = [Y^0, Y^B] and extra_j = [Y^{S_d}.., Y^kap], each computed along a broadcast
set axis.

Directions.  C does not depend on k q (the kernel carries j_L(kq)), so k enters through K^2 and the kernel only.
For an active input direction d of DIRECTIONS (k, mu, sin, f), d_d x = dx/dd of the scalars x = (Kfac, Ksq) of
``rsd_scalars``, the derivative of the bracket of a pair (j, t) is

    Z^d = -(d_d Ksq / 2) [Y^0 . R X_lt + Y^B . R Y_lt + y^0 rho^X + y^B rho^Y] + Y^{S_d} . R + y^{S_d} rho
          + [d = k] (Y^kap . R + y^0 rho') / k,

with rho^X, rho^Y and rho' as rho for the rows R X_lt, R Y_lt and the kernel KW' (the d_d Ksq term only for d in
KSQ_DIRECTIONS, the S_d term only for d in G_DIRECTIONS).  The coefficients d_d Ksq / 2 and 1/k are per-point
scalars, so the q contractions come first, per moment, main_j against [R; R X_lt; R Y_lt]_{T_j} and extra_j against
R_{T_j} (``contract_moments``), and the sum over the sets is taken on the pairs.  The rows tangent is the fold of
Y^0 . dR - (K^2/2) Y^0 . R dX_lt - (K^2/2) Y^B . R dY_lt with their q_max terms (``contract_rows``).
``point_loop_jvp`` folds the brackets with Wfold.  Per active direction d it forms the point's whole Jacobian along d
in the primal part (the folded brackets plus the unit-direction jvps of the fold weights and of the small terms) and
adds t_d times it to the tangent groups of the rows, W and P_lin.

Work.  J_m and D_X are materialized once per point (one ``optimization_barrier`` on the tuple of the J_m; without it
XLA repeats the recursion in every fusion that reads J).  The residual groups are one plain loop fusion each, written
in the layout the contraction reads, with no stack over the moments or the sets (a stack is a concatenate fusion that
computes its shared operands again for every output set).  The fold weights are a gather of the component weights and
a static scatter-add onto the pairs.  Against a rule with one stacked bundle, the fold in q space and all 9 x 35
pairs, the chi2 gradient takes 0.67x the time (0.76x with respect to the P values alone, 0.60x for four chains under
vmap) and 36% less scratch memory.

Scope.  The rule is written for the density power spectrum: the damping D_X and B = -(1/2) K^2 Y_lt, the linear-part,
b2^2 and k-space corrections of ``small_terms`` (``LPT._linear_rows``, ``_LINEAR_MATRIX`` and ``_KSPACE_ROWS``) and the
q_max endpoint values are those of the density components.  The moments, their angular plan (``LPT._angular``) and the
component table are read from the tables.
"""
import numpy as np

import jax
import jax.numpy as jnp
from jax.custom_derivatives import SymbolicZero

from . import angular
from .lpt_defs import _rsd_geometry
from .lpt_tables import (CT_ENTRY_NAMES, MOMENTS, TREE_ROWS, Component, LinearKind, Row, _LINEAR_MATRIX, entry_scalars,
                         row_basis)

DIRECTIONS = ('k', 'mu', 'sin', 'f')                       # the arguments of rsd_scalars
SCALARS = ('Kfac', 'Ksq')                                  # its outputs
G_DIRECTIONS = ('mu', 'sin', 'f')                          # the directions the angular matrix G depends on
KSQ_DIRECTIONS = ('k', 'mu', 'f')                          # the directions K^2 depends on (Kfac has no sin)

CT_ENTRIES = tuple((int(Component[c]), MOMENTS.index(m), t) for c, m, t in CT_ENTRY_NAMES)   # (component, moment, t)
PAIRS = tuple(sorted({(j, t) for _, j, t in CT_ENTRIES}))
"""The (moment j, basis row t) pairs of the component table, by moment and row (the pairs of a moment contiguous)."""
PAIR_MOMENT = np.array([j for j, _ in PAIRS], dtype=np.int32)
PAIR_BASIS_ROW = np.array([t for _, t in PAIRS], dtype=np.int32)
BASIS_ROWS_OF_MOMENT = tuple(np.array([t for jj, t in PAIRS if jj == j], dtype=np.int32) for j in range(len(MOMENTS)))
"""T_j, the basis rows of the pairs of moment j."""
ENTRY_COMPONENT = np.array([c for c, _, _ in CT_ENTRIES], dtype=np.int32)
ENTRY_PAIR = np.array([PAIRS.index((j, t)) for _, j, t in CT_ENTRIES], dtype=np.int32)
assert all(len(T) for T in BASIS_ROWS_OF_MOMENT), 'every moment has a pair in the component table'


def contract_weights(W, x):
    """``sum_c W[..., o, c] x[..., c]`` -> ``(nk, nmu, n_out)``; W per point (4-d) or shared (2-d)."""
    return jnp.einsum('oc,kmc->kmo' if W.ndim == 2 else 'kmoc,kmc->kmo', W, x)


def rsd_scalars(k, mu, sin, f):
    """The per-point scalars (Kfac, Ksq) (SCALARS) of (k, mu, sin, f), K = k Kfac."""
    Kfac = _rsd_geometry(f, mu, sin).Kfac
    return Kfac, (k * Kfac)**2


def jvp_if_active(fn, primals, tangents):
    """``(fn(*primals), tangent)``: jax.jvp of fn w.r.t. the arguments whose tangent is not None (None if none)."""
    act = [n for n, t in enumerate(tangents) if t is not None]
    if not act:
        return fn(*primals), None
    sub = lambda *xs: fn(*[dict(zip(act, xs)).get(n, x) for n, x in enumerate(primals)])
    return jax.jvp(sub, tuple(primals[n] for n in act), tuple(tangents[n] for n in act))


def _nonzero(t):
    """Whether the tangent ``t`` is not a symbolic zero."""
    return not isinstance(t, SymbolicZero)


def unit_tangent(primals, index):
    """The tangents of ``primals`` that are one at position ``index`` and zero elsewhere."""
    return tuple(jnp.ones_like(x) if n == index else jnp.zeros_like(x) for n, x in enumerate(primals))


def small_terms(model, rows, k, mu, sin, f, p, KW):
    """``(small, lin_c)``, each ``(23,)`` per component, the parts of the point's output outside the moment contraction.

    ``p`` is P_lin at ``k`` and ``KW`` a weighted kernel ``(lmax + 1, nq)`` (that of ``LPT._point_kernel`` or its ln k
    derivative).  A row x of order L is contracted as <x>_L = KW[L] . x for L > 0 and
    <x>_0 = sum_q KW[0, q] (x[q] - x[q_max]).  small_c is the sum of -lin_weight <x>_L over the linear rows x of the
    kind of c (``LPT._linear_rows``), of lin_c p with lin_c = lin_weight times the Kaiser factor of that kind, of the
    ``_KSPACE_ROWS`` values of c and, for c = XI_XI, of -<xi_lin^2 / 2>_0.  The first two vanish for a component
    without a linear kind.  The output of the point is sum_c W[o, c] (the moment contraction of c + small_c).
    """
    geometry, corrs_tree = _rsd_geometry(f, mu, sin), rows[TREE_ROWS]
    K = k * geometry.Kfac
    lin_weight, linear, kaiser = model._linear_rows(mu, f, geometry, K, K**2, corrs_tree)
    out = lambda L, x: KW[0] @ (x - x[-1]) if L == 0 else KW[L] @ x     # order L, L = 0 with the q_max subtraction
    s_kind = [0.0] * len(LinearKind)
    for kind, L, row in linear:
        s_kind[kind] = s_kind[kind] + out(L, row)
    lin_kind, lin_mat = lin_weight * jnp.stack(kaiser), jnp.asarray(_LINEAR_MATRIX)
    small = lin_mat @ (-lin_weight * jnp.stack(s_kind) + lin_kind * p)
    small = small.at[Component.XI_XI].add(-out(0, 0.5 * corrs_tree[Row.XI_LIN]**2))
    for component, value in model._kspace_terms(jnp.log(k), model._kspace_tables(rows)):
        small = small.at[component].add(value)
    return small, lin_mat @ lin_kind


def component_weights_jvp(W_fn, k_true, mu_b, W_args, dk, dmu, dW_args):
    """``(W, dW)``: the weights ``(nk, nmu, n_out, 23)`` (or the shared ``(n_out, 23)``) and their tangent (None if
    zero), the jvp of the per-point ``W_fn`` w.r.t. (k, mu, its closed-over values)."""
    nz = _nonzero
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


def mu_tables(model, f, mu_true, sin_true, gdirs):
    """The per-mu stacked matrices [G; G Lambda; d_d G for d in ``gdirs``] ``(nmu, 2 + len(gdirs), n_rows, nm + 1)``
    of ``angular.angular_matrix``, G and d_d G padded by a zero column so that every set multiplies the same J of
    nm + 1 indices; d_d G is the jvp of G along the unit tangent of d in G_DIRECTIONS."""
    plan = model._angular
    Lam = jnp.asarray(angular.lam_matrix(plan.nm))

    def G_of(mu_j, sin_j, ff):
        return angular.angular_matrix(plan, _rsd_geometry(ff, mu_j, sin_j))

    def one(mu_j, sin_j):
        prim = (mu_j, sin_j, f)
        G = G_of(*prim)
        pad = lambda M: jnp.pad(M, ((0, 0), (0, 1)))
        mats = [pad(G), G @ Lam]
        for d in gdirs:
            unit = unit_tangent(prim, G_DIRECTIONS.index(d))
            mats.append(pad(jax.jvp(G_of, prim, unit)[1]))
        return jnp.stack(mats)
    return jax.vmap(one)(mu_true, sin_true)


def fold_weights(k, mu, f, Kfac, W):
    """Wfold[o, p] = sum_e W[o, c_e] s_e [p_e = p] ``(n_out, n_pairs)``: the component weights of the entries
    e = (c_e, j, t) of the component table times their per-point scalars s_e (``entry_scalars``, Kfac an explicit
    argument), summed onto the pairs p = (j, t) of PAIRS (a gather and a static scatter-add)."""
    v = jnp.take(W, ENTRY_COMPONENT, axis=-1, mode='clip') * entry_scalars(k, mu, f, Kfac)[None, :]  # (n_out, n_entries)
    return jnp.zeros((W.shape[0], len(PAIRS)), dtype=v.dtype).at[:, ENTRY_PAIR].add(v)


def moment_residuals(plan, Gs, J, D, KW, KWp, n_gdirs, with_B):
    """``(main, extra, y)`` of one point: per moment j (``plan.rows_of_moment``) the residual groups
    main_j = [Y^0, Y^B] ``(n_main, nq)`` (Y^B when ``with_B``) and extra_j = [Y^{S_d} for the n_gdirs sets S_d, Y^kap
    (with ``KWp``)] ``(n_extra, nq)`` (None when empty), and y ``(n_sets, n_moments)`` the q_max values y_j of the sets
    of ``Gs`` (S0, S_B, S_d; zero for S_B without ``with_B``).

    ``Gs`` is one mu of ``mu_tables``, J the nm + 1 exponential moments and D = D_X of the point.  Each group is
    computed along a broadcast set axis: C = Gs[s, r, m] J_m summed over the row's band (widened by one index on each
    side when the group holds G Lambda, ``AngularPlan.wide_band``), Y = D sum_L KW_L C - KW_0 y with the kernel of
    each set (KW', chosen by a static mask, for Y^kap), as elementwise sums: small batched matrix products are slow on
    XLA CPU, the banded sums are fused."""
    main_sets = np.array([0, 1] if with_B else [0])
    extra_sets = np.array([2 + i for i in range(n_gdirs)] + ([0] if KWp is not None else []), dtype=np.int64)
    if KWp is None:
        kernel_extra = lambda L: KW[L][None, :]
    else:
        is_kap = np.array([False] * n_gdirs + [True])[:, None]
        kernel_extra = lambda L: jnp.where(is_kap, KWp[L][None, :], KW[L][None, :])

    def residuals(G, bands, kernel, rows_j):
        """``(Y, y)`` of the sets of ``G`` ``(n_group, n_rows, nm + 1)`` for the angular rows ``rows_j`` of one moment,
        Y = D sum_L kernel(L) C_L - kernel(0) y ``(n_group, nq)`` and y = D(q_max) C_0(q_max) ``(n_group,)`` (zero when
        the moment has no order 0)."""
        acc = y = None
        for r in rows_j:
            L, band = plan.row_order[r], bands[r]
            C = G[:, r, band[0]][:, None] * J[band[0]][None, :]                   # (n_group, nq)
            for m in band[1:]:
                C = C + G[:, r, m][:, None] * J[m][None, :]
            acc = kernel(L) * C if acc is None else acc + kernel(L) * C
            if L == 0:
                y = D[-1] * C[:, -1]
        if y is None:                                                             # an odd moment, no order 0
            y = jnp.zeros(G.shape[0], dtype=D.dtype)
        return D[None, :] * acc - kernel(0) * y[:, None], y

    main, extra, y = [], [], []
    zero = jnp.zeros((), dtype=D.dtype)
    for rows_j in plan.rows_of_moment:
        main_j, y_main = residuals(Gs[main_sets], plan.wide_band if with_B else plan.band, lambda L: KW[L][None, :],
                                   rows_j)
        extra_j, y_extra = (residuals(Gs[extra_sets], plan.band, kernel_extra, rows_j) if len(extra_sets)
                            else (None, None))
        main.append(main_j)
        extra.append(extra_j)
        y.append(jnp.stack([y_main[0], y_main[1] if with_B else zero] + [y_extra[i] for i in range(n_gdirs)]))
    return main, extra, jnp.stack(y, axis=-1)


def contract_moments(main, extra, R, X_lt, Y_lt, with_K2):
    """The q contractions of the residual groups on the pairs, ``(Z0, ZK, Zextra)``.

    Per moment j one product of main_j ``(n_main, nk, nmu, nq)`` with [R; R X_lt; R Y_lt]_{T_j} (R_{T_j} alone without
    ``with_K2``) and one of extra_j with R_{T_j}; the pairs of the moments are concatenated in PAIRS order:
    Z0 = Y^0 . R ``(nk, nmu, n_pairs)``, ZK = Y^0 . R X_lt + Y^B . R Y_lt (None without ``with_K2``) and Zextra
    ``(n_extra, nk, nmu, n_pairs)`` (None without extra groups)."""
    bases = [R] + ([R * X_lt, R * Y_lt] if with_K2 else [])
    Z0, ZK, Zextra = [], [], []
    for main_j, extra_j, T in zip(main, extra, BASIS_ROWS_OF_MOMENT):
        n = len(T)
        Zmain = jnp.einsum('skmq,uq->skmu', main_j, jnp.concatenate([Q[T] for Q in bases]))
        Z0.append(Zmain[0, ..., :n])
        if with_K2:
            ZK.append(Zmain[0, ..., n:2 * n] + Zmain[1, ..., 2 * n:])
        if extra_j is not None:
            Zextra.append(jnp.einsum('skmq,tq->skmt', extra_j, R[T]))
    on_pairs = lambda Z: jnp.concatenate(Z, axis=-1) if Z else None
    return on_pairs(Z0), on_pairs(ZK), on_pairs(Zextra)


def contract_rows(main, Q1, Q2, Q3):
    """The q contractions of the rows tangent on the pairs, ``(Y^0 . Q1, Y^0 . Q2, Y^B . Q3)`` ``(nk, nmu, n_pairs)``
    each (Q1 = dR, Q2 = R dX_lt, Q3 = R dY_lt): per moment one product of main_j = [Y^0, Y^B] with [Q1; Q2; Q3]_{T_j},
    of which the three blocks are kept (main_j read whole, not sliced)."""
    Z1, Z2, Z3 = [], [], []
    for main_j, T in zip(main, BASIS_ROWS_OF_MOMENT):
        n = len(T)
        Z = jnp.einsum('skmq,uq->skmu', main_j, jnp.concatenate([Q1[T], Q2[T], Q3[T]]))
        Z1.append(Z[0, ..., :n])
        Z2.append(Z[0, ..., n:2 * n])
        Z3.append(Z[1, ..., 2 * n:])
    on_pairs = lambda Z: jnp.concatenate(Z, axis=-1)
    return on_pairs(Z1), on_pairs(Z2), on_pairs(Z3)


def point_loop_jvp(model, W_fn, primals, tangents):
    """The jvp rule of ``LPT._point_loop``: primal = Wfold . (Y^0 . R + y^0 rho) + sum_c W small_c on the pairs, and
    the tangent groups (each skipped when its input tangents are symbolic zeros):
      d      t_d dout[d] per active d, with the point's Jacobian along the unit tangent e_d (primal-only)
               dout[d] = Wfold . Z^d + dWfold_d . (Y^0 . R + y^0 rho) + sum_c W_c dsmall_d,c
             (Z^d in the module docstring), dWfold_d the jvp of ``fold_weights`` along (e_d[k], e_d[mu], e_d[f],
             d_d Kfac, 0), dsmall_d the jvp of ``small_terms`` (the point's kernel KW fixed) along
             (0, e_d[k], e_d[mu], e_d[sin], e_d[f]) plus, for d = k, (small(KW') - small(0)) / k (small is affine in KW
             and dKW/dk = KW'/k)
      rows   Wfold . [Y^0 . dR + y^0 rho(dR) - (K^2/2) (Y^0 . R dX_lt + y^0 rho(R dX_lt))
                      - (K^2/2) (Y^B . R dY_lt + y^B rho(R dY_lt))] + sum_c W_c dsmall_rows,c
             (the jvp of ``small_terms`` in the rows alone)
      W      sum_c dW_c small_c + Wfold(dW) . (Y^0 . R + y^0 rho) (``fold_weights`` at dW, linear in W)
      P_lin  sum_c W_c lin_c dP.
    d_d x is the jvp of ``rsd_scalars`` along e_d.  Each jvp is linear in its tangent, so the jvp along the actual
    tangents is sum_d t_d times the jvp along e_d.  Of fold_weights, rsd_scalars and small_terms only the rows direction
    of small_terms is linearized with the rule's tangents (the linear part that reverse mode transposes).
    """
    k_true, mu_true, sin_true, pk_lin_true, rows, f, W_args = primals
    dk, dmu, dsin, dp, drows, df, dW_args = tangents
    nz = _nonzero
    val = lambda t: t if nz(t) else None
    plan = model._angular
    nk, nmu = k_true.shape
    X_lt, Y_lt = rows[Row.X_LIN_LT], rows[Row.Y_LIN_LT]
    mu_b = jnp.broadcast_to(mu_true, k_true.shape)
    W, dW = component_weights_jvp(W_fn, k_true, mu_b, W_args, dk, dmu, dW_args)
    w_axis, dw_axis = (0 if W.ndim == 4 else None), (None if (dW is None or dW.ndim == 2) else 0)
    t_rows = val(drows)
    t_dir = {'k': val(dk), 'mu': val(dmu), 'sin': val(dsin), 'f': val(df)}
    dirs = tuple(d for d in DIRECTIONS if t_dir[d] is not None)          # the active input directions
    gdirs = tuple(d for d in G_DIRECTIONS if d in dirs)
    with_K2 = any(d in KSQ_DIRECTIONS for d in dirs)                      # a direction moves K^2
    with_B = with_K2 or t_rows is not None                               # Y^B: read by the K^2 fold and the rows
    Gs_all = mu_tables(model, f, mu_true, sin_true, gdirs)               # (nmu, 2 + len(gdirs), n_rows, nm + 1)
    i_Kfac, i_Ksq = SCALARS.index('Kfac'), SCALARS.index('Ksq')

    def per_mu(k_i, mu_j, sin_j, p_i, W_ij, tW, Gs):
        # the unit tangent e_d of each active direction, the derivatives d_d of the scalars
        prim = (k_i, mu_j, sin_j, f)
        unit = {d: unit_tangent(prim, DIRECTIONS.index(d)) for d in dirs}
        jac = {d: jax.jvp(rsd_scalars, prim, unit[d])[1] for d in dirs}
        KW, KWp = model._point_kernel(k_i, 'k' in dirs, weighted=True)
        sc = rsd_scalars(*prim)
        D = jnp.exp(-0.5 * sc[i_Ksq] * X_lt)
        J = angular.exp_moments(-0.5 * sc[i_Ksq] * Y_lt, plan.nm + 1)    # (nm + 1, nq)
        *J, D = jax.lax.optimization_barrier(tuple(J[m] for m in range(plan.nm + 1)) + (D,))
        main, extra, y = moment_residuals(plan, Gs, J, D, KW, KWp, len(gdirs), with_B)
        # small_terms with the point's kernel KW held fixed; its k dependence through KW added explicitly
        small_fn = lambda rr, kk, mm, ss, ff: small_terms(model, rr, kk, mm, ss, ff, p_i, KW)
        small, lin_c = small_fn(rows, k_i, mu_j, sin_j, f)
        point = dict(main=main, extra=extra, y=y, KW0=KW[0], jac=jac, Ksq=sc[i_Ksq], small=small, lin_c=lin_c,
                     wfold=fold_weights(k_i, mu_j, f, sc[i_Kfac], W_ij))
        if KWp is not None:
            point['KWp0'] = KWp[0]
        # the jvps along e_d (primal-only), fold_weights with Kfac along d_d Kfac and W inactive, small_terms with the
        # rows inactive, plus for d = k the kernel's k dependence (small(KW') - small(0)) / k
        for d in dirs:
            u = dict(zip(DIRECTIONS, unit[d]))
            point['dwfold_' + d] = jvp_if_active(fold_weights, (k_i, mu_j, f, sc[i_Kfac], W_ij),
                                                 (u['k'], u['mu'], u['f'], jac[d][i_Kfac], None))[1]
            dsm = jvp_if_active(small_fn, (rows, k_i, mu_j, sin_j, f), (None, u['k'], u['mu'], u['sin'], u['f']))[1][0]
            if d == 'k':
                small_p = small_terms(model, rows, k_i, mu_j, sin_j, f, p_i, KWp)[0]
                small_0 = small_terms(model, rows, k_i, mu_j, sin_j, f, p_i, 0.0 * KW)[0]
                dsm = dsm + (small_p - small_0) / k_i
            point['dsmall_' + d] = dsm
        if t_rows is not None:        # the rows direction of the small terms (in the linear part)
            point['dsmall_rows'] = jvp_if_active(small_fn, (rows, k_i, mu_j, sin_j, f),
                                                 (t_rows, None, None, None, None))[1][0]
        if tW is not None:            # the fold weights at dW (linear in W, no jvp)
            point['wfold_dW'] = fold_weights(k_i, mu_j, f, sc[i_Kfac], tW)
        return point

    def per_k(k_row, p_row, W_row, tW_row):
        return jax.vmap(per_mu, in_axes=(0, 0, 0, 0, w_axis, dw_axis, 0))(
            k_row, mu_true, sin_true, p_row, W_row, tW_row, Gs_all)
    res = jax.vmap(per_k, in_axes=(0, 0, w_axis, dw_axis))(k_true, pk_lin_true, W, dW)

    # the residual groups with the set axis first, (n_sets, nk, nmu, nq): the row-major operands of the contractions
    main = [jnp.moveaxis(a, 2, 0) for a in res['main']]
    extra = [None if b is None else jnp.moveaxis(b, 2, 0) for b in res['extra']]
    y, KW0, wfold = res['y'], res['KW0'], res['wfold']                  # y (nk, nmu, n_sets, n_moments)
    R = row_basis(rows)
    fold = lambda Z: jnp.einsum('kmop,kmp->kmo', wfold, Z)
    e = lambda x: x[..., None]
    # rho_of(kernel0, Q) on the pairs, (nk, nmu, n_pairs)
    rho_of = lambda kernel0, Q: jnp.einsum('kmq,tq->kmt', kernel0, Q - Q[:, -1:])[..., PAIR_BASIS_ROW]
    y_of = lambda s: y[:, :, s][..., PAIR_MOMENT]                       # the q_max values of set s on the pairs
    rho = rho_of(KW0, R)
    Z0, ZK, Zextra = contract_moments(main, extra, R, X_lt, Y_lt, with_K2)
    bracket = Z0 + y_of(0) * rho                                         # Y^0 . R + y^0 rho of every pair
    primal = fold(bracket) + contract_weights(W, res['small'])
    tangent = jnp.zeros_like(primal)
    if with_K2:
        q_K2 = y_of(0) * rho_of(KW0, R * X_lt) + y_of(1) * rho_of(KW0, R * Y_lt)
    for d in dirs:                    # Z^d of the module docstring: the q contractions, then the q_max terms
        Zd, qd = [], []
        if d in KSQ_DIRECTIONS:
            half = e(0.5 * res['jac'][d][i_Ksq])
            Zd.append(-half * ZK)
            qd.append(-half * q_K2)
        if d in gdirs:
            Zd.append(Zextra[gdirs.index(d)])
            qd.append(y_of(2 + gdirs.index(d)) * rho)
        if d == 'k':
            Zd.append(Zextra[len(gdirs)] / e(k_true))
            qd.append(y_of(0) * rho_of(res['KWp0'], R) / e(k_true))
        dout = (fold(sum(Zd[1:], Zd[0]) + sum(qd[1:], qd[0]))
                + jnp.einsum('kmop,kmp->kmo', res['dwfold_' + d], bracket) + contract_weights(W, res['dsmall_' + d]))
        tangent = tangent + e(jnp.broadcast_to(t_dir[d], (nk, nmu))) * dout
    if t_rows is not None:
        dR = jax.jvp(row_basis, (rows,), (drows,))[1]
        Q1, Q2, Q3 = dR, R * drows[Row.X_LIN_LT], R * drows[Row.Y_LIN_LT]
        Z1, Z2, Z3 = contract_rows(main, Q1, Q2, Q3)
        half_K2 = e(0.5 * res['Ksq'])
        tangent = tangent + (fold(Z1 + y_of(0) * rho_of(KW0, Q1)) - half_K2 * fold(Z2 + y_of(0) * rho_of(KW0, Q2))
                             - half_K2 * fold(Z3 + y_of(1) * rho_of(KW0, Q3)))
        tangent = tangent + contract_weights(W, res['dsmall_rows'])
    if dW is not None:
        tangent = tangent + contract_weights(dW, res['small']) + jnp.einsum('kmop,kmp->kmo', res['wfold_dW'], bracket)
    if nz(dp):
        tangent = tangent + contract_weights(W, res['lin_c']) * dp[:, :, None]
    return primal, tangent
