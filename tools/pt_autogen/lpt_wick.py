"""Wick contraction of the cumulants of lpt_model.ROW_DICTIONARY into k-space integrands (sympy).

A cumulant is a joint cumulant of fields, Delta^(n) = Psi^(n)(q_2) - Psi^(n)(q_1) (a vector with the kernel (i/n!) L_n of
lpt_kernels) and Lagrangian operators at the point 1 or 2 (lpt_model.OPERATORS).  With q_1 = 0 and q_2 = q every field X
at q_a is int e^{i K_X.q_a} K^X(legs) delta .. delta, so a connected Wick matching of the linear legs gives

* tree      (one pair, crossing)                               int_k e^{ik.q} F(k, mu) P(k),
* 13        (one crossing pair k, one pair p, -p at one point)  int_k e^{ik.q} P(k) int_p F(k, p, x, mu) P(p),
* 22        (two crossing pairs p, k - p)                      int_k e^{ik.q} int_p F(k, p, x, mu) P(p) P(|k-p|),
* zero lag  (no crossing pair), a constant not built here (lpt_reduce takes it from Delta(q = 0) = 0),

where the crossing pairs carry the momentum from the fields at q_2 to those at q_1, mu = khat.qhat and x = khat.phat.
The free indices of the Delta's are contracted with an element of the isotropic basis in qhat (lpt_tensors), a single
index with qhat (pt_autogen.vectors.ndot with n read as qhat, n.k = k mu, n.p = np) and a pair of indices with each other.
The 22 terms are also built with the crossing momenta p, r (``routing='pr'``, F(p, r, z, np, nr), z = phat.rhat) for
the separable product forms.
"""
import itertools
from collections import defaultdict
from typing import NamedTuple

import sympy as sp

from . import lpt_kernels as kn
from . import lpt_model as model
from . import lpt_tensors as te
from .vectors import K_VEC, P_VEC, R_VEC

DELTA = model.DISPLACEMENT


class Field(NamedTuple):
    """A field of a Wick contraction (the Delta's get their point in ``expand_points``, 0 before)."""
    kind: str
    order: int
    point: int


def entry_fields(entry, order_assignment=None):
    """The fields of a dictionary entry, the Delta's in the order of the tensor indices."""
    deltas = [n for kind, n in entry.fields if kind == DELTA]
    if order_assignment is not None:
        deltas = list(order_assignment)
    it = iter(deltas)
    return [Field(DELTA, next(it), 0) if kind == DELTA else Field(kind, model.OPERATORS[kind].order, n)
            for kind, n in entry.fields]


def order_assignments(entry):
    """The assignments of the orders to the Delta's whose sum the entry's rows decompose ('one' the entry's own,
    'all' every distinct permutation)."""
    deltas = tuple(n for kind, n in entry.fields if kind == DELTA)
    if entry.orderings == 'one':
        return [deltas]
    return sorted(set(itertools.permutations(deltas)))


def expand_points(fields):
    """[(sign, fields with points)], every Delta at q_2 (+1) or at q_1 (-1)."""
    idx = [i for i, fl in enumerate(fields) if fl.kind == DELTA]
    out = []
    for pts in itertools.product((2, 1), repeat=len(idx)):
        fs = list(fields)
        sign = 1
        for i, pt in zip(idx, pts):
            fs[i] = Field(DELTA, fs[i].order, pt)
            sign *= 1 if pt == 2 else -1
        out.append((sign, fs))
    return out


def _matchings(items):
    if not items:
        yield ()
        return
    a = items[0]
    for i in range(1, len(items)):
        for rest in _matchings(items[1:i] + items[i + 1:]):
            yield ((a, items[i]),) + rest


def connected_matchings(fields):
    """The perfect matchings of the legs ``(field, leg)`` whose graph on the fields is connected.  None of them may
    contract two distinct operator fields at one point (the self-pairs that the renormalisation removes, asserted)."""
    legs = [(i, j) for i, fl in enumerate(fields) for j in range(fl.order)]
    out = []
    for m in _matchings(legs):
        parent = list(range(len(fields)))

        def root(a):
            while parent[a] != a:
                a = parent[a]
            return a
        selfpair = False
        for (fa, _), (fb, _) in m:
            parent[root(fa)] = root(fb)
            A, B = fields[fa], fields[fb]
            if fa != fb and A.kind != DELTA and B.kind != DELTA and A.point == B.point:
                selfpair = True
        if len({root(a) for a in range(len(fields))}) != 1:
            continue
        assert not selfpair, ('self-pair in a connected matching', fields, m)
        out.append(m)
    return out


def _route(fields, matching, routing):
    """The leg momenta ``{(field, leg): Vec}`` and the kind of the term."""
    cross = [pr for pr in matching if fields[pr[0][0]].point != fields[pr[1][0]].point]
    local = [pr for pr in matching if pr not in cross]
    mom = {}

    def at2(pr):
        a, b = pr
        return (a, b) if fields[a[0]].point == 2 else (b, a)
    if not cross:
        return None, 'zero_lag'
    if len(matching) == 1:
        a2, a1 = at2(cross[0])
        mom[a2], mom[a1] = K_VEC, -K_VEC
        return mom, 'tree'
    if len(cross) == 1:
        a2, a1 = at2(cross[0])
        mom[a2], mom[a1] = K_VEC, -K_VEC
        (u, w), = local
        mom[u], mom[w] = P_VEC, -P_VEC
        return mom, '13'
    (a2, a1), (b2, b1) = at2(cross[0]), at2(cross[1])
    mom[a2], mom[a1] = P_VEC, -P_VEC
    if routing == 'k':
        mom[b2], mom[b1] = K_VEC - P_VEC, P_VEC - K_VEC
    else:
        mom[b2], mom[b1] = R_VEC, -R_VEC
    return mom, '22'


def term_value(fields, sign, matching, contraction, routing='k', transverse=True):
    """``(kind, F)``, the integrand of one matching for the basis element ``contraction`` ((pairs, singles) over the
    Delta index positions), without the P factors and the phase."""
    mom, kind = _route(fields, matching, routing)
    if kind == 'zero_lag':
        return kind, sp.Integer(0)
    value = sp.Integer(sign)
    vecs = []
    for i, fl in enumerate(fields):
        legs = tuple(mom[(i, j)] for j in range(fl.order))
        if fl.kind == DELTA:
            vecs.append(kn.L(*legs, transverse=transverse).scale(model.PSI_PREFACTOR(fl.order)))
        else:
            value *= kn.operator_kernel(fl.kind, legs)
    pairs, singles = contraction
    for s in singles:
        value *= vecs[s].ndot()
    for a, b in pairs:
        value *= vecs[a].dot(vecs[b])
    return kind, value


class EntryIntegrands(NamedTuple):
    name: str
    rank: int
    basis: tuple
    integrands: dict      # {(basis index, kind): F} for the kinds tree, 13, 22 (routing 'k')
    integrands_pr: dict   # {basis index: F} of the 22 kind with the routing 'pr'
    n_matchings: dict     # {kind: number of matchings} (all assignments, all point choices)


def entry_integrands(entry, transverse=True, with_pr=True):
    """The summed integrands of every basis contraction of ``entry`` (the sum over the order assignments of 'all', the
    point choices of the Delta's and the connected matchings)."""
    rank = sum(1 for kind, _ in entry.fields if kind == DELTA)
    basis = te.isotropic_basis(rank)
    acc = defaultdict(lambda: sp.Integer(0))
    acc_pr = defaultdict(lambda: sp.Integer(0))
    counts = defaultdict(int)
    for assignment in order_assignments(entry):
        for sign, fields in expand_points(entry_fields(entry, assignment)):
            for m in connected_matchings(fields):
                kind = _route(fields, m, 'k')[1]
                counts[kind] += 1
                if kind == 'zero_lag':
                    continue
                for t, elem in enumerate(basis):
                    acc[(t, kind)] += term_value(fields, sign, m, elem, 'k', transverse)[1]
                    if kind == '22' and with_pr:
                        acc_pr[t] += term_value(fields, sign, m, elem, 'pr', transverse)[1]
    return EntryIntegrands(entry.name, rank, basis, dict(acc), dict(acc_pr), dict(counts))
