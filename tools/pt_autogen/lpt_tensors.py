"""Isotropic tensors in qhat, the basis, the derived decomposition of the cumulants and their contraction in the frame.

* ``isotropic_basis(rank)``: the rank-r tensors built from qhat and delta (every split of the indices into delta pairs
  and qhat singles), more deltas first and then by the positions of the qhat's.  Rank 2 is (delta_ij, qhat_i qhat_j),
  rank 3 (qhat_i delta_jk, qhat_j delta_ki, qhat_k delta_ij, qhat_i qhat_j qhat_k).  ``gram(rank)`` is its Gram matrix
  (used by the Wick engine to project a contraction onto the basis).
* ``row_map(entry)`` and ``decomposition(name)``: the coefficients of a lpt_model.ROW_DICTIONARY entry's cumulant on
  that basis.  The symmetry under exchanges of identical Delta slots ties coefficients together (W^(112) gives V2 = V1),
  the rows name the orbits in order, and for orderings 'all' the rows decompose the sum over the order assignments so
  that the coefficients of one assignment are solved for (each assignment of A^10 is half of X10 delta + Y10 qq).
* ``contract(symbol)``: a cumulant symbol of lpt_cumulant (each Delta with its vector label) as a polynomial in the
  rows, MQ = Khat.qhat, NQ = n.qhat and (k, mu, f, Kfac), with qhat_i -> v.qhat and delta_ij -> v.w for the vectors in
  the frame (K, n) of lpt_model.FRAME (``VECTORS``, derived from lpt_model.RSD_RULE).
"""
import itertools
from functools import lru_cache

import numpy as np
import sympy as sp

from . import lpt_model as model
from .lpt_cumulant import DELTA, Field, canonical_fields

k, mu, f, Kfac = sp.symbols('k mu f Kfac')
MQ, NQ = sp.symbols('MQ NQ')
ROW_SYMBOLS = {name: sp.Symbol(name) for name in model.ROW_ORDER}
SIGMA2 = sp.Symbol('sigma2')

# ---------------------------------------------------------------------------------------------------------------
# the frame


def _to_frame(a, b):
    """The vector a k + b n in the basis (K, n), with k = K - f (k.n) n = K - f k mu n."""
    return (a, sp.expand(b - a * f * k * mu))


VECTORS = {
    **{f'k{n}': _to_frame(1, n * f * k * mu) for n in model.DISPLACEMENT_ORDERS},    # k_n = k + n f (k.n) n
    'k': _to_frame(1, 0),
    'n': (sp.Integer(0), sp.Integer(1)),
    'K': (sp.Integer(1), sp.Integer(0)),
}
"""The components (K, n) of the vector labels, k_n = K + (n - 1) f k mu n and k = K - f k mu n."""

GRAM_FRAME = {('K', 'K'): k**2 * Kfac**2, ('K', 'n'): k * mu * (1 + f), ('n', 'n'): sp.Integer(1)}
"""K.K = k^2 Kfac^2, K.n = (k + f k mu n).n = k mu (1 + f), n.n = 1."""

QDOT = {'K': k * Kfac * MQ, 'n': NQ}
"""K.qhat = |K| MQ = k Kfac MQ and n.qhat = NQ."""


def qdot(label):
    a, b = VECTORS[label]
    return a * QDOT['K'] + b * QDOT['n']


def vdot(l1, l2):
    a1, b1 = VECTORS[l1]
    a2, b2 = VECTORS[l2]
    return (a1 * a2 * GRAM_FRAME[('K', 'K')] + (a1 * b2 + b1 * a2) * GRAM_FRAME[('K', 'n')]
            + b1 * b2 * GRAM_FRAME[('n', 'n')])


# ---------------------------------------------------------------------------------------------------------------
# the isotropic basis

@lru_cache(maxsize=None)
def isotropic_basis(rank):
    """The basis elements ``(pairs, singles)`` over the index positions 0 .. rank - 1."""
    def splits(free):
        if not free:
            yield (), ()
            return
        a, rest = free[0], free[1:]
        for pairs, singles in splits(rest):
            yield pairs, (a,) + singles
        for i, b in enumerate(rest):
            for pairs, singles in splits(rest[:i] + rest[i + 1:]):
                yield ((a, b),) + pairs, singles
    elems = {(tuple(sorted(tuple(sorted(pr)) for pr in pairs)), tuple(sorted(singles)))
             for pairs, singles in splits(tuple(range(rank)))}
    return tuple(sorted(elems, key=lambda e: (-len(e[0]), e[1], e[0])))


def _tensor(elem, rank):
    pairs, singles = elem
    qhat = np.array([0, 0, 1])
    T = np.zeros((3,) * rank, dtype=object)
    for idx in itertools.product(range(3), repeat=rank):
        v = 1
        for a, b in pairs:
            v *= 1 if idx[a] == idx[b] else 0
        for s in singles:
            v *= qhat[idx[s]]
        T[idx] = v
    return T


@lru_cache(maxsize=None)
def gram(rank):
    """The Gram matrix B_t . B_s of the basis (a sympy Matrix)."""
    tensors = [_tensor(e, rank) for e in isotropic_basis(rank)]
    n = len(tensors)
    return sp.Matrix(n, n, lambda i, j: sp.Integer(int(np.sum(tensors[i] * tensors[j]))))


def _relabel(elem, mapping):
    pairs, singles = elem
    return (tuple(sorted(tuple(sorted((mapping[a], mapping[b]))) for a, b in pairs)),
            tuple(sorted(mapping[s] for s in singles)))


def _orbits(basis, group):
    """The orbits of the basis elements under the permutations ``group`` of the index positions, in basis order."""
    orbit_of, orbits = {}, []
    for e in basis:
        if e in orbit_of:
            continue
        orb = sorted({_relabel(e, g) for g in group}, key=basis.index)
        for y in orb:
            orbit_of[y] = len(orbits)
        orbits.append(orb)
    return orbits, orbit_of


def delta_orders(entry):
    return tuple(n for kind, n in entry.fields if kind == DELTA)


def _symmetry_group(entry):
    """The exchanges of index positions that leave the entry's rows fixed (identical orders, or all for 'all')."""
    orders = delta_orders(entry)
    perms = list(itertools.permutations(range(len(orders))))
    if entry.orderings == 'all':
        return perms
    return [g for g in perms if all(orders[g[i]] == orders[i] for i in range(len(orders)))]


def row_map(entry):
    """``(row index of each basis element, number of rows)`` of the entry's rank (orbits of ``_symmetry_group``)."""
    basis = isotropic_basis(len(delta_orders(entry)))
    orbits, orbit_of = _orbits(basis, _symmetry_group(entry))
    return [orbit_of[e] for e in basis], len(orbits)


ENTRIES = {e.name: e for e in model.ROW_DICTIONARY}


@lru_cache(maxsize=None)
def decomposition(name, part=''):
    """``(basis, coefficients)`` of the cumulant of the entry ``name`` (one order assignment, the entry's index order),
    the coefficients in the row symbols of ``part`` ('' or the IR parts '<' and '>' of a split entry)."""
    entry = ENTRIES[name]
    rows = entry.split[part] if entry.split else entry.rows
    assert bool(part) == bool(entry.split), (name, part)
    orders = delta_orders(entry)
    r = len(orders)
    basis = isotropic_basis(r)
    perms = list(itertools.permutations(range(r)))
    group_S = [g for g in perms if all(orders[g[i]] == orders[i] for i in range(r))]
    orbits_S, orbit_of_S = _orbits(basis, group_S)
    s = sp.symbols(f's0:{len(orbits_S)}')
    S = {e: s[orbit_of_S[e]] for e in basis}
    if entry.orderings == 'one':
        N = S
    else:
        N = {e: sp.Integer(0) for e in basis}
        for assign in sorted(set(itertools.permutations(orders))):
            # the actual position i holds the order assign[i], i.e. the declared position tau[i] with that order
            tau, used = [], set()
            for i in range(r):
                d = next(d for d in range(r) if d not in used and orders[d] == assign[i])
                used.add(d)
                tau.append(d)
            inverse = {d: i for i, d in enumerate(tau)}
            for e, c in S.items():
                N[_relabel(e, inverse)] += c
    rmap, n_rows = row_map(entry)
    assert n_rows == len(rows), (name, n_rows, rows)
    equations = [sp.Eq(N[e], ROW_SYMBOLS[rows[rmap[i]]]) for i, e in enumerate(basis)]
    solution = sp.solve(equations, s, dict=True)
    assert len(solution) == 1 and set(solution[0]) == set(s), (name, solution)
    # the solution of the exact linear system is linear in the row symbols with rational coefficients: sp.expand is
    # its exact canonical form (no numeric identification of the coefficients)
    return basis, tuple(sp.expand(S[e].subs(solution[0])) for e in basis)


def entry_fields(entry):
    """The fields of a dictionary entry, ('Delta', n) -> Delta^(n) and (operator, point) -> the operator."""
    return [Field(DELTA, 0, n) if kind == DELTA else Field(kind, n, model.OPERATORS[kind].order)
            for kind, n in entry.fields]


_LOOKUP = {}
for _e in model.ROW_DICTIONARY:
    _key = canonical_fields(entry_fields(_e))
    assert _key not in _LOOKUP, _e.name
    _LOOKUP[_key] = _e.name


def entry_of(symbol):
    """The dictionary entry name of a cumulant symbol (None if the dictionary has no row for it)."""
    return _LOOKUP.get(tuple(slot.field for slot in symbol.slots))


def align(entry, slots):
    """The vector labels of the symbol's Delta slots at the entry's index positions (matched by order)."""
    deltas = [s for s in slots if s.field.kind == DELTA]
    labels, used = [], set()
    for n in delta_orders(entry):
        i = next(i for i, s in enumerate(deltas) if i not in used and s.field.order == n)
        used.add(i)
        labels.append(deltas[i].vec)
    assert len(used) == len(deltas)
    return labels


def contract(symbol):
    """The value of a cumulant symbol in the frame, a polynomial in the rows, MQ, NQ, k, mu, f, Kfac (and sigma2)."""
    if symbol.part == 'sigma2':
        return SIGMA2
    name = entry_of(symbol)
    if name is None:
        raise KeyError(f'no row for the cumulant {symbol}')
    basis, coefs = decomposition(name, symbol.part)
    labels = align(ENTRIES[name], symbol.slots)
    total = sp.Integer(0)
    for (pairs, singles), c in zip(basis, coefs):
        if c == 0:
            continue
        term = c
        for i in singles:
            term *= qdot(labels[i])
        for i1, i2 in pairs:
            term *= vdot(labels[i1], labels[i2])
        total += term
    return sp.expand(total)
