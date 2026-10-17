"""The cumulant expansion of the LPT generating function (sympy), driven by the declarations of lpt_model.

The exponent is E = sum over the displacement terms c Delta^(n).v of lpt_model.SOURCES + sum_{O, a} J_{O,a} O_a, and

    log Z = sum_m kappa_m(E) / m! = sum over multisets M of the elementary terms e of E of
            [prod_e c_e^{m_e} / m_e!] J^{M} kappa(M),

with kappa(M) the joint cumulant of the fields of M and every Delta^(n) contracted with its vector v (a label, k1 = k_1,
k2, ..).  A cumulant is an abstract symbol, the sorted tuple of its slots (field (kind, point, order), vector), in a
canonical form under the swap of the two Lagrangian points (``canonical``).

Rules (lpt_model).  Fields of total order m <= MAX_ORDER with m even are kept.  The connected Wick diagrams of the
linear legs decide whether a cumulant vanishes, is kept, or is a self-pair cumulant of the renormalisation
(``classify``).  The cumulant A^lin = <Delta^(1) Delta^(1)> is split into A^< (exponent) and A^> (expanded).  Then
Z' = exp(log Z - A^< part) is expanded with the order truncation, D_1 D_2 is applied and J = 0 (``apply_D``).  The
result of ``derive`` is the exponent and the term list {(bias monomial, product of symbols): coefficient}.

The kernel facts behind the declared self-contraction properties and the zero lag of the 'explicit' renormalisation
mode come from the EPT generator's kernels (``self_contraction``, ``delta_G2_zero_lag``).
"""
import itertools
import math
from collections import Counter, defaultdict
from typing import NamedTuple, Tuple

import sympy as sp

from . import lpt_model as model
from . import model as ept_model
from .vectors import K_VEC, P_VEC, k, p, x


class Field(NamedTuple):
    kind: str          # lpt_model.DISPLACEMENT or an operator of lpt_model.OPERATORS
    point: int         # 0 for Delta (both points), 1 or 2 for an operator
    order: int


class Slot(NamedTuple):
    field: Field
    vec: str           # the contraction vector of a Delta, '' for an operator


class Symbol(NamedTuple):
    """A cumulant, ``part`` '' (plain), '<' or '>' (the IR parts of A^lin) or 'sigma2' (the zero-lag sigma^2)."""
    part: str
    slots: Tuple[Slot, ...]


SIGMA2 = Symbol('sigma2', ())
DELTA = model.DISPLACEMENT


def symbol_order(s):
    return 2 if s.part == 'sigma2' else sum(slot.field.order for slot in s.slots)


def product_order(symbols):
    return sum(symbol_order(s) for s in symbols)


# ---------------------------------------------------------------------------------------------------------------
# kernel facts (from ptgen.model)

def self_contraction(op):
    """The kernel of ``op`` with two of its momenta p, -p, simplified (0 when such a contraction vanishes)."""
    kernel = model.OPERATORS[op].kernel
    if model.OPERATORS[op].order < 2:
        raise ValueError(f'{op} has one leg')
    placements = {2: [(P_VEC, -P_VEC)], 3: [(K_VEC, P_VEC, -P_VEC), (P_VEC, K_VEC, -P_VEC), (P_VEC, -P_VEC, K_VEC)]}
    return sp.simplify(sum(sp.Abs(sp.simplify(kernel(legs))) for legs in placements[model.OPERATORS[op].order]))


def delta_G2_zero_lag():
    """<K_G2(p, p')> over the directions of p at fixed p' (delta_a contracted with one leg of G2_a), -2/3."""
    kernel = sp.simplify(ept_model.K_G2(P_VEC, K_VEC))                # x^2 - 1, x = phat.khat
    assert not kernel.has(k) and not kernel.has(p), kernel
    return sp.integrate(sp.expand(kernel), (x, -1, 1)) / 2


# ---------------------------------------------------------------------------------------------------------------
# canonical form under the swap of the two points

def _swap(slot):
    f = slot.field
    return slot if f.kind == DELTA else Slot(Field(f.kind, 3 - f.point, f.order), slot.vec)


def canonical(slots):
    """The smaller of the sorted slots and the sorted slots with the points 1 <-> 2 exchanged.

    Exchanging q_1 and q_2 maps Delta -> -Delta and q -> -q.  A cumulant with r Delta's is a rank-r isotropic tensor
    in qhat (parity (-1)^r), so the exchanged cumulant is (-1)^r (-1)^r = 1 times the original, equal."""
    a = tuple(sorted(slots))
    b = tuple(sorted(_swap(s) for s in slots))
    return min(a, b)


def canonical_fields(fields):
    return tuple(s.field for s in canonical([Slot(f, '') for f in fields]))


# ---------------------------------------------------------------------------------------------------------------
# Wick diagrams, which decide whether a cumulant vanishes, is kept or is a self-pair

def _matchings(items):
    if not items:
        yield ()
        return
    a = items[0]
    for i in range(1, len(items)):
        for rest in _matchings(items[1:i] + items[i + 1:]):
            yield ((a, items[i]),) + rest


def _self_contraction_rule(field):
    if field.kind == DELTA:
        return model.DISPLACEMENT_SELF_CONTRACTION
    return model.OPERATORS[field.kind].self_contraction


class Diagram(NamedTuple):
    legs: tuple        # leg -> slot index
    pairs: tuple       # the matching, pairs of legs
    forbidden: tuple   # pairs of legs joining two distinct operator fields at one point


def diagrams(slots):
    """``(kept, n_vanishing)``: the connected Wick diagrams of the linear legs of ``slots`` that do not vanish, and the
    number of connected ones that vanish by a self-contraction (none for a first cumulant, means vanish)."""
    legs = tuple(i for i, s in enumerate(slots) for _ in range(s.field.order))
    out, n_vanishing = [], 0
    if len(slots) < 2 or len(legs) % 2:
        return out, n_vanishing
    for pairs in _matchings(list(range(len(legs)))):
        parent = list(range(len(slots)))

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i

        vanishes, forbidden = False, []
        for a, b in pairs:
            i, j = legs[a], legs[b]
            if i == j:
                vanishes |= _self_contraction_rule(slots[i].field) == 'vanishes'
                continue
            parent[root(i)] = root(j)
            fi, fj = slots[i].field, slots[j].field
            if fi.kind != DELTA and fj.kind != DELTA and fi.point == fj.point:
                forbidden.append((a, b))
        if len({root(i) for i in range(len(slots))}) != 1:
            continue
        if vanishes:
            n_vanishing += 1
            continue
        out.append(Diagram(legs, pairs, tuple(forbidden)))
    return out, n_vanishing


def classify(slots):
    """'zero' (no connected diagram), 'vanishing' (every connected diagram has a vanishing self-contraction), 'kept'
    (every diagram free of self-pairs) or 'selfpair' (every diagram has one).  A mixture of the last two raises."""
    ds, n_vanishing = diagrams(slots)
    if not ds:
        return ('vanishing' if n_vanishing else 'zero'), ds
    n_forbidden = sum(1 for d in ds if d.forbidden)
    if n_forbidden == 0:
        return 'kept', ds
    if n_forbidden == len(ds):
        return 'selfpair', ds
    raise NotImplementedError(f'cumulant with both kinds of diagrams: {slots}')


def reduce_selfpair(slots, ds, zero_lag_delta_G2):
    """The 'explicit' mode, a self-pair cumulant as [(factor, symbols)] with the self-pair contracted at zero lag.

    delta_a delta_a gives sigma^2, delta_a with a leg of G2_a gives zero_lag_delta_G2 sigma^2 and G2_a becomes a
    delta_a (its other leg).  Every diagram reduces to a diagram of the reduced cumulant, asserted to be its only one."""
    acc = defaultdict(lambda: sp.Integer(0))
    for d in ds:
        assert len(d.forbidden) == 1, (slots, d)
        a, b = d.forbidden[0]
        i, j = d.legs[a], d.legs[b]
        kinds = sorted((slots[i].field.kind, slots[j].field.kind))
        if kinds == ['delta', 'delta']:
            assert len(slots) == 2, slots          # with other fields the diagram would be disconnected
            acc[()] += 1
        elif kinds == ['G2', 'delta']:
            g, dl = (i, j) if slots[i].field.kind == 'G2' else (j, i)
            point = slots[g].field.point
            reduced = [s for t, s in enumerate(slots) if t not in (g, dl)] + [Slot(Field('delta', point, 1), '')]
            acc[canonical(reduced)] += zero_lag_delta_G2
        else:
            raise NotImplementedError(f'zero-lag contraction {kinds} in {slots}')
    out = []
    for reduced, factor in acc.items():
        if reduced == ():
            out.append((factor, (SIGMA2,)))
            continue
        status, rds = classify(reduced)
        assert status == 'kept' and len(rds) == 1, (reduced, status, len(rds))
        out.append((factor, tuple(sorted((SIGMA2, Symbol('', reduced))))))
    return out


# ---------------------------------------------------------------------------------------------------------------
# log Z

class ETerm(NamedTuple):
    coef: sp.Expr
    slot: Slot
    source: tuple      # ((operator, point),) for J_{O,a} O_a, () for a displacement term


def elementary_terms(source):
    """The elementary terms of E for ``source`` (a key of lpt_model.SOURCES), the displacements first."""
    terms = [ETerm(sp.sympify(c), Slot(Field(DELTA, 0, n), vec), ()) for n, c, vec in model.SOURCES[source]]
    for op, spec in model.OPERATORS.items():
        for a in (1, 2):
            terms.append(ETerm(sp.Integer(1), Slot(Field(op, a, spec.order), ''), ((op, a),)))
    return terms


def d_monomials(renormalisation='drop'):
    """The derivative monomials of D_a (as dicts), the sigma^2 terms only in 'explicit' mode."""
    return [dict(t.derivatives) for t in model.DERIVATIVE_OPERATOR
            if renormalisation == 'explicit' or not t.sigma2_power]


def reachable(jmono, renormalisation='drop'):
    """Whether the J monomial can be part of a term that D_1 D_2 at J = 0 keeps (at each point its exponents are
    bounded by those of one derivative monomial of D_a)."""
    mons = d_monomials(renormalisation)
    for a in (1, 2):
        part = {op: n for (op, pt), n in jmono if pt == a}
        if not any(all(m.get(op, 0) >= n for op, n in part.items()) for m in mons):
            return False
    return True


def is_A_lin(slots):
    return len(slots) == 2 and all(s.field == Field(DELTA, 0, 1) for s in slots)


def log_Z(source, renormalisation='drop', zero_lag_delta_G2=None):
    """``(L, exponent, selfpairs, vanishing)``.  L is the expanded part [(coef, J monomial, symbols)], exponent the
    exponentiated A^< part [(coef, symbol)], selfpairs and vanishing the self-pair and vanishing cumulants met
    [(slots, J monomial)].  Terms whose J monomial D_1 D_2 cannot reach are skipped (``reachable``)."""
    terms = elementary_terms(source)
    L, exponent, selfpairs, vanishing = [], [], [], []
    for size in range(1, model.MAX_ORDER + 1):
        for combo in itertools.combinations_with_replacement(range(len(terms)), size):
            order = sum(terms[i].slot.field.order for i in combo)
            if order % 2 or order > model.MAX_ORDER:
                continue
            counts = Counter(combo)
            jmono = Counter()
            for i, m in counts.items():
                for src in terms[i].source:
                    jmono[src] += m
            jmono = tuple(sorted(jmono.items()))
            if not reachable(jmono, renormalisation):
                continue
            slots = canonical([terms[i].slot for i in combo])
            status, ds = classify(slots)
            if status == 'zero':
                continue
            if status == 'vanishing':
                vanishing.append((slots, jmono))
                continue
            coef = sp.Integer(1)
            for i, m in counts.items():
                coef *= terms[i].coef**m / math.factorial(m)
            if status == 'selfpair':
                selfpairs.append((slots, jmono))
                if renormalisation == 'drop':
                    continue
                for factor, syms in reduce_selfpair(slots, ds, zero_lag_delta_G2):
                    L.append((coef * factor, jmono, syms))
                continue
            if is_A_lin(slots):
                assert not jmono
                exponent.append((coef, Symbol('<', slots)))
                L.append((coef, jmono, (Symbol('>', slots),)))
            else:
                L.append((coef, jmono, (Symbol('', slots),)))
    return L, exponent, selfpairs, vanishing


# ---------------------------------------------------------------------------------------------------------------
# Z' = exp(L) and D_1 D_2

def _merge(j1, j2):
    c = Counter(dict(j1))
    c.update(dict(j2))
    return tuple(sorted(c.items()))


def _multiply(P, Q):
    out = defaultdict(lambda: sp.Integer(0))
    for (j1, s1), c1 in P.items():
        o1 = product_order(s1)
        for (j2, s2), c2 in Q.items():
            if o1 + product_order(s2) > model.MAX_ORDER:
                continue
            out[(_merge(j1, j2), tuple(sorted(s1 + s2)))] += c1 * c2
    return out


def exponentiate(L):
    """exp(L) = sum_n L^n / n!, truncated at total order MAX_ORDER, as {(J monomial, symbols): coef}."""
    Lp = defaultdict(lambda: sp.Integer(0))
    for coef, jmono, syms in L:
        Lp[(jmono, syms)] += coef
    Z = defaultdict(lambda: sp.Integer(0))
    power = {((), ()): sp.Integer(1)}
    n = 0
    while power:
        for key, c in power.items():
            Z[key] += c / math.factorial(n)
        power = _multiply(power, Lp)
        n += 1
    return Z


def apply_D(Z, renormalisation='drop'):
    """D_1 D_2 Z at J = 0, ``({(bias names sorted, symbols): coef}, [keys whose contributions cancel])``, with the same
    bias at both points.  A term J^{e_1} J^{e_2} (e_a the exponents at point a) survives only the D terms with exactly
    these derivatives, with the factor prod e!."""
    D = defaultdict(list)
    for t in model.DERIVATIVE_OPERATOR:
        if renormalisation == 'drop' and t.sigma2_power:
            continue
        D[tuple(sorted(t.derivatives))].append(t)
    out = defaultdict(lambda: sp.Integer(0))
    for (jmono, syms), coef in Z.items():
        e1 = tuple(sorted((op, n) for (op, a), n in jmono if a == 1))
        e2 = tuple(sorted((op, n) for (op, a), n in jmono if a == 2))
        if e1 not in D or e2 not in D:
            continue
        fact = math.prod(math.factorial(n) for _, n in jmono)
        for t1 in D[e1]:
            for t2 in D[e2]:
                s = tuple(sorted(syms + (SIGMA2,) * (t1.sigma2_power + t2.sigma2_power)))
                if product_order(s) > model.MAX_ORDER:
                    continue
                bias = tuple(sorted(b for b in (t1.bias, t2.bias) if b != '1'))
                out[(bias, s)] += coef * t1.factor * t2.factor * fact
    expanded = {key: sp.expand(c) for key, c in out.items()}
    kept = {key: c for key, c in expanded.items() if c != 0}
    return kept, sorted(key for key, c in expanded.items() if c == 0)


class Result(NamedTuple):
    source: str
    renormalisation: str
    exponent: list          # [(coef, Symbol '<')], the log of the exponentiated factor is sum coef * symbol
    terms: dict             # {(bias names, symbols): coef}
    selfpairs: list         # [(slots, J monomial)] self-pair cumulants met in log Z (reachable J monomials)
    vanishing: list         # [(slots, J monomial)] cumulants whose diagrams all vanish by a self-contraction
    n_logZ: int             # the number of terms of the expanded part of log Z
    cancelled: list         # the (bias, symbols) whose contributions cancelled in D_1 D_2 Z


def derive(source='rsd', renormalisation='drop'):
    """The term list of D_1 D_2 Z|_{J=0} for ``source``, 'explicit' keeps the sigma^2 terms (lpt_model)."""
    zero_lag = delta_G2_zero_lag() if renormalisation == 'explicit' else None   # per leg of G2
    L, exponent, selfpairs, vanishing = log_Z(source, renormalisation, zero_lag)
    terms, cancelled = apply_D(exponentiate(L), renormalisation)
    return Result(source, renormalisation, exponent, terms, selfpairs, vanishing, len(L), cancelled)


def kept_terms(result):
    """The terms of ``result`` without the dropped bias coefficients (lpt_model.DROPPED_BIAS)."""
    return {key: c for key, c in result.terms.items() if not any(b in model.DROPPED_BIAS for b in key[0])}
