"""The recipes of the LPT correlator rows (generic and preferred routes), the stage-B units and the rules.

``derive()`` runs the reduction (lpt_reduce) over lpt_model.ROW_DICTIONARY and returns a :class:`Derivation` with
* ``generic``, {row: [Term]} as the reduction gives it (stage C transforms of the derived sources, stage A outputs,
  constants from Delta(q = 0) = 0),
* ``preferred``, {row: [Term]} after the declared preferences (lpt_model.PREFERENCES) and the common-subexpression rule
  (a term equal to c times an elementary row is that row),
* ``products`` and ``finite_range``, the q-space product forms of the separable rows and the closed forms of the
  Gauss-law rows.
``units_of`` gives the stage-B units of the named sources, ``attach`` the DC rules of the k-independent MQ0 terms of
the components (lpt_components) and ``kspace_terms`` the recipes of the k-space rows (lpt_model.KSPACE_ROWS).

Term(kind, l, n, qpow, coef, ref), with the kinds 'xi' (coef q^qpow xi_l^n of the spectrum ref), 'H' (q^qpow H_l^n[S]
with S = sum_s ref[s] s over the named sources, times coef), 'M' (coef M_n[S], S as for 'H'), 'prod' (coef q^qpow times
the xi monomial ref), 'row' (coef times the row ref) and a rule name of lpt_model.RULES for a k-space row (coef times
the rule applied to ref).
"""
import time
from collections import defaultdict
from typing import NamedTuple

import sympy as sp

from . import lpt_identities as idn
from . import lpt_model as model
from . import lpt_reduce as rr

ROW_ORDER = list(model.ROW_ORDER)
TERM_KINDS = ('xi', 'H', 'M', 'prod', 'row')


class Term(NamedTuple):
    kind: str
    l: int
    n: int
    qpow: int
    coef: sp.Expr
    ref: object


class Derivation(NamedTuple):
    generic: dict
    preferred: dict
    raw: dict            # {row: recipe dict of lpt_reduce, or ('diff', a, b)}
    products: dict       # {row: direct separable product or None}
    finite_range: dict   # {row: finite-range xi form or None}
    info: dict
    timings: dict


def _named(src):
    """The named source dict of a source, {'P' or 'P_lt': c} for a tree source, else its decomposition."""
    if any(key[0] == 'P' for key in src):
        assert all(key[0] == 'P' for key in src)
        return {key[1]: v for key, v in src.items()}
    dec = idn.decompose(src)
    assert dec is not None, ('source outside the span of the named sources', src)
    return dec


def xi_terms(expr):
    """[(coef, qpow, monomial string)] of a xi form (one term per monomial)."""
    out = []
    for term in sp.Add.make_args(sp.expand(expr)):
        powers = term.as_powers_dict()
        qp = int(powers.get(rr.QS, 0))
        mono = [s for s in sorted(term.free_symbols, key=lambda s: s.name) if s.name.startswith('xi_')]
        c = sp.cancel(term / (rr.QS**qp * sp.Mul(*[s**powers[s] for s in mono])))
        assert not c.free_symbols, term
        out.append((c, qp, '*'.join(s.name if powers[s] == 1 else f'{s.name}^{powers[s]}' for s in mono)))
    return out


def recipe_terms(rec):
    """The generic terms of a recipe of lpt_reduce (or of ('diff', a, b))."""
    if isinstance(rec, tuple) and rec[0] == 'diff':
        return [Term('row', 0, 0, 0, sp.Integer(1), rec[1]), Term('row', 0, 0, 0, sp.Integer(-1), rec[2])]
    terms = []
    for (l, n), src in sorted(rec['H'].items()):
        named = _named(src)
        if any(s in model.LINEAR_SPECTRA for s in named):
            terms.extend(Term('xi', l, n, 0, c, spec) for spec, c in named.items())
        else:
            terms.append(Term('H', l, n, 0, sp.Integer(1), named))
    for n, src in sorted(rec['M'].items()):
        terms.append(Term('M', 0, n, 0, sp.Integer(1), _named(src)))
    return terms


def proportional(a, b):
    """c with a = c b for two named-source dicts, else None."""
    if set(a) != set(b):
        return None
    ratios = {sp.nsimplify(a[s] / b[s]) for s in a}
    return ratios.pop() if len(ratios) == 1 else None


def _elementary(terms):
    """(l, n, named source) of a row that is one H term and nothing else, else None."""
    if len(terms) == 1 and terms[0].kind == 'H' and terms[0].qpow == 0:
        t = terms[0]
        return t.l, t.n, {s: t.coef * c for s, c in t.ref.items()}
    return None


def preferred_route(generic, products, finite_range):
    """lpt_model.PREFERENCES and the common-subexpression rule applied to the generic terms."""
    prefs = model.PREFERENCES
    pref = {}
    for row in ROW_ORDER:                                            # 1. the product route
        if row in prefs['product_route']['rows']:
            expr = finite_range.get(row)
            if expr is None:
                expr = products.get(row)
            assert expr is not None, ('no product form', row)
            pref[row] = [Term('prod', 0, 0, qp, c, mono) for c, qp, mono in xi_terms(expr)]
        else:
            pref[row] = list(generic[row])
    elementary = {row: _elementary(generic[row]) for row in ROW_ORDER}
    for row in ROW_ORDER:                                            # 2. common subexpressions
        new = []
        for t in pref[row]:
            if t.kind == 'H':
                hit = None
                for other in ROW_ORDER:
                    el = elementary.get(other)
                    if other == row or el is None or (el[0], el[1]) != (t.l, t.n):
                        continue
                    if elementary.get(row) is not None and ROW_ORDER.index(other) > ROW_ORDER.index(row):
                        continue                       # an elementary row refers only to earlier elementary rows
                    c = proportional({s: t.coef * v for s, v in t.ref.items()}, el[2])
                    if c is not None:
                        hit = (other, c)
                        break
                if hit is not None:
                    new.append(Term('row', 0, 0, 0, hit[1], hit[0]))
                    continue
            new.append(t)
        pref[row] = new
    m22 = [t for t in generic['X22'] if t.kind == 'M'][0].ref                # 3. zero-lag constants from M22
    for row in prefs['zero_lag_from_M22']['rows']:
        pref[row] = [Term('M', 0, t.n, 0, idn.full_integral_ratio(t.ref, m22) * t.coef, dict(m22))
                     if t.kind == 'M' else t for t in pref[row]]
    for row, (kind, l0, n0) in prefs['raise_n_by_recurrence']['rows'].items():   # 4. raise n by the recurrence
        new = []
        for t in pref[row]:
            if t.kind == kind and (t.l, t.n) == (l0, n0):
                c = sp.Rational(1, 2 * t.l + 1) * t.coef
                new.append(t._replace(l=t.l - 1, n=t.n + 1, qpow=t.qpow + 1, coef=c))
                new.append(t._replace(l=t.l + 1, n=t.n + 1, qpow=t.qpow + 1, coef=c))
            else:
                new.append(t)
        pref[row] = new
    return pref


def derive(transverse=True):
    """The :class:`Derivation` of the rows of lpt_model.ROW_DICTIONARY."""
    t0 = time.time()
    raw, products, info = {}, {}, {}
    for entry in model.ROW_DICTIONARY:
        rows, prods, inf = rr.derive_entry(entry, transverse=transverse)
        info[entry.name] = inf
        if entry.split:
            rows.update(rr.split_rows(entry, rows))
        raw.update(rows)
        products.update(prods)
    t1 = time.time()
    generic = {row: recipe_terms(raw[row]) for row in ROW_ORDER}
    finite_range = {}
    for row in ROW_ORDER:
        rec = raw[row]
        if isinstance(rec, dict) and len(rec['H']) == 1 and not rec['M']:
            (l, n), src = next(iter(rec['H'].items()))
            finite_range[row] = idn.finite_range_xi_form(l, n, src)
    t2 = time.time()
    preferred = preferred_route(generic, products, finite_range)
    t3 = time.time()
    return Derivation(generic, preferred, raw, {r: products.get(r) for r in ROW_ORDER}, finite_range, info,
                      dict(engine_s=t1 - t0, closed_forms_s=t2 - t1, preferences_s=t3 - t2))


# ---------------------------------------------------------------------------------------------------------------
# stage A requirements and stage-B units

def xi_factors(mono):
    """[(l, n, power)] of a product string such as 'xi_1_m1^2*xi_3_m1'."""
    out = []
    for factor in mono.split('*'):
        name, _, power = factor.partition('^')
        out.append(rr.xi_index(sp.Symbol(name)) + (int(power) if power else 1,))
    return out


def used_sources(route):
    """The named sources (not the linear spectra) of the H and M terms of a route, in name order."""
    out = set()
    for terms in route.values():
        for t in terms:
            if t.kind in ('H', 'M'):
                out |= {s for s in t.ref if s not in model.LINEAR_SPECTRA}
    return sorted(out)


def units_of(sources):
    """The stage-B units of the named sources and their composition.

    Returns ``(units, comps)``.  ``units`` maps a unit name to (kind, n, ((l, c), ..)), a 22 unit being the q-space
    combination sum c (xi_l^n)^2 of a source at one n (lpt_model.UNIT_NORMALISATION, first coefficient 1) and a 13
    unit the single block (l, n).  ``comps`` maps a source to [(unit, coefficient, k power)]."""
    named = idn.named_sources()
    units, comps = {}, {}
    for s in sources:
        by_n = defaultdict(dict)
        for (kind, l, n), c in named[s][1].items():
            by_n[(kind, n)][l] = c
        comps[s] = []
        for (kind, n), combo in sorted(by_n.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
            if kind == '22':
                norm = combo[min(combo)]
                key = ('22', n, tuple(sorted((l, sp.nsimplify(c / norm)) for l, c in combo.items())))
                name = next((u for u, spec in units.items() if spec == key), None)
                if name is None:
                    name = f'U22_{sum(1 for u in units if u.startswith("U22"))}'
                    units[name] = key
                comps[s].append((name, norm, -2 * n))
            else:
                for l, c in sorted(combo.items()):
                    name = f'U13_{l}_{n}'.replace('-', 'm')
                    units[name] = ('13', n, ((l, sp.Integer(1)),))
                    comps[s].append((name, c, 2 - n))
    return units, comps


def unit_order(units):
    """The units in table order, the 22 units by n descending, then the 13 units by n descending and l."""
    return sorted(units, key=lambda u: (units[u][0] != '22', -units[u][1], units[u][2]))


def stage_a(route, units):
    """{(spectrum, l, n)} of the stage-A outputs that a route and its units need."""
    need = set()
    for terms in route.values():
        for t in terms:
            if t.kind == 'xi':
                need.add((t.ref, t.l, t.n))
            elif t.kind == 'prod' or (t.kind not in TERM_KINDS and t.ref.startswith('xi_')):
                need |= {('P', l, n) for l, n, _ in xi_factors(t.ref)}
    for kind, n, combo in units.values():
        need |= {('P', l, n) for l, _ in combo}
    return need


# ---------------------------------------------------------------------------------------------------------------
# the DC rules of the final integrand and the k-space rows

KFACTORS = sp.symbols('k mu f Kfac')


def row_xi_expression(route, row):
    """The q-space value of a row whose terms are all xi terms of P or products, as an expression in the xi symbols
    and q (None otherwise)."""
    out = sp.Integer(0)
    for t in route[row]:
        if t.kind == 'xi' and t.ref == 'P':
            out += t.coef * rr.QS**t.qpow * rr.xi(t.l, t.n)
        elif t.kind == 'prod':
            out += t.coef * rr.QS**t.qpow * sp.Mul(*[rr.xi(l, n)**e for l, n, e in xi_factors(t.ref)])
        else:
            return None
    return sp.expand(out)


class Attachment(NamedTuple):
    component: str
    term: sp.Expr        # the k-independent MQ0 term of the component (in the row symbols)
    rule: str
    function: object     # the q-space function the rule acts on (None for the final-integral rules)
    factor: sp.Expr      # term = factor x function
    detail: str


def mq0_terms(comps):
    """{component: its k-independent MQ0 coefficient} (the terms whose transform does not vanish at k -> 0)."""
    out = {}
    for c in comps:
        term = sp.Integer(0)
        for ((a, b), rows), v in c.table.items():
            if (a, b) == (0, 0) and not (v.free_symbols & set(KFACTORS)):
                term += v * sp.Mul(*[sp.Symbol(r) for r in rows])
        if term != 0:
            out[c.name] = sp.expand(term)
    return out


def attach(der, comps):
    """The rules of every k-independent MQ0 term of the components, [Attachment] in component order."""
    tree_rows = {row for _, rows in model.LINEAR_KINDS for row in rows}
    out = []
    for comp, term in mq0_terms(comps).items():
        if term.is_number:
            out.append(Attachment(comp, term, 'q_max_subtraction', None, term, 'a constant, its transform is delta_D(k)'))
            continue
        syms = {s.name for s in term.free_symbols}
        if sp.Poly(term, *term.free_symbols).total_degree() == 1 and syms <= tree_rows:
            out.append(Attachment(comp, term, 'linear_part', None, term, 'linear theory, added back as D x Kaiser x '
                                                                         'P_lin(k)'))
            continue
        assert len(syms) == 1, (comp, term)
        (row,) = syms
        value = row_xi_expression(der.preferred, row)
        if value is not None:
            F = sp.expand(term.subs(sp.Symbol(row), value))
            G = idn.total_derivative(sp.expand(rr.QS**2 * idn.phi_form(F)))
            if G is not None and idn.small_q_order(G) > 0:
                out.append(Attachment(comp, term, 'outside_core', row, sp.cancel(term / sp.Symbol(row)),
                                      f"q^2 F = G', G = {G} (phi = xi_2^-2), G(0) = 0, so int 4 pi q^2 F dq = 0"))
            else:
                for rule in ('m0_residual', 'm0_constant'):
                    out.append(Attachment(comp, term, rule, F, sp.Integer(1),
                                          'q^2 F has no total derivative in the ansatz, a non-zero full-q integral'))
            continue
        hs = [t for t in der.preferred[row] if t.kind == 'H']
        assert len(hs) == 1 and len(der.preferred[row]) == 1 and hs[0].n >= 0 and hs[0].l == 0, (comp, row)
        h = hs[0]
        src = {s: h.coef * c for s, c in h.ref.items()}
        hit = None
        for other in ROW_ORDER:
            g = der.generic[other]
            if len(g) == 1 and g[0].kind == 'H' and (g[0].l, g[0].n) == (1, h.n - 1):
                c = proportional(src, {s: g[0].coef * v for s, v in g[0].ref.items()})
                if c is not None:
                    hit = (other, c)
                    break
        assert hit is not None, row
        out.append(Attachment(comp, term, 'inner_ball', (row, hit[0], hit[1]), sp.cancel(term / sp.Symbol(row)),
                              f'{row} = H_0^{h.n}[S], S = {hit[1]} x the source of {hit[0]} = H_1^{h.n - 1}, so '
                              f"q^2 {row} = {hit[1]} (q^2 {hit[0]})'"))
    return out


def kspace_terms(attachments):
    """``(recipes, table)`` of the k-space rows of lpt_model.KSPACE_ROWS from the attachments.

    ``recipes`` maps a k-space row to its one rule term, ``table`` lists (row, component, factor, interpolated,
    k0_const) in row order (the final integral adds factor x the row to the component's weight)."""
    recipes, table = {}, []
    for ks in model.KSPACE_ROWS:
        hits = [a for a in attachments if a.rule == ks.rule and _function_label(a) == ks.function]
        assert len(hits) == 1, (ks, hits)
        a = hits[0]
        if a.rule == 'outside_core':
            term = Term(a.rule, None, None, None, sp.Integer(1), a.function)
        elif a.rule in ('m0_residual', 'm0_constant'):
            (c, qp, mono), = xi_terms(a.function)
            factors = xi_factors(mono)
            assert qp == 0 and factors == [(0, 0, 2)], ('m0 rules are written for (xi_0^0)^2', a.function)
            # m0_residual transforms F on the padded q grid, m0_constant is its k -> 0 limit by Parseval,
            # 4 pi int q^2 (xi_0^0)^2 dq = M_0[P^2] = int dk k^2 P^2 / (2 pi^2)
            term = (Term(a.rule, None, None, None, c, mono) if a.rule == 'm0_residual'
                    else Term(a.rule, None, 0, None, c, 'P^2'))
        else:
            row, other, c = a.function
            term = Term(a.rule, None, None, None, c, other)
        recipes[ks.name] = [term]
        table.append((ks.name, a.component, a.factor, ks.interpolated, ks.k0_const))
    return recipes, table


def _function_label(a):
    """The label of an attachment's function in the notation of lpt_model.KSPACE_ROWS."""
    if a.rule == 'outside_core':
        return a.function
    if a.rule == 'inner_ball':
        return a.function[0]
    if a.rule in ('m0_residual', 'm0_constant'):
        rows = sorted(s.name for s in a.term.free_symbols)
        return f'{rows[0]}^2/2' if a.term == sp.Symbol(rows[0])**2 / 2 else str(a.term)
    return None


def rule_uses(der):
    """{rule: rows} of the recipe-level rules that apply to some rows only (grid_moment, bias_n0, raise_n), on the
    preferred route (the others, end_weights, zero_pad and low_ringing_phase, apply to every transform)."""
    uses = {'grid_moment': [], 'bias_n0': [], 'raise_n': []}

    def kinds_of(row):
        out = set()
        for t in der.preferred[row]:
            out |= kinds_of(t.ref) if t.kind == 'row' else {t.kind}
        return out
    for row in ROW_ORDER:
        if 'M' in kinds_of(row):
            uses['grid_moment'].append(row)
        if any(t.kind == 'H' and t.n >= 0 for t in der.preferred[row]):
            uses['bias_n0'].append(row)
        if row in model.PREFERENCES['raise_n_by_recurrence']['rows']:
            uses['raise_n'].append(row)
    return uses


def tables(der, comps):
    """The content of rows_sources.dat, rows_recipes.dat and rows_rules.dat (emit.lpt_rows_text).

    ``recipes`` is the preferred route of the q-space rows followed by the rule terms of the k-space rows (the order
    of PTTerms.rows), ``units`` and ``sources`` the stage-B units of the named sources the recipes use."""
    attachments = attach(der, comps)
    kspace_recipes, kspace = kspace_terms(attachments)
    recipes = {row: der.preferred[row] for row in ROW_ORDER}
    recipes.update(kspace_recipes)
    units, sources = units_of(used_sources(der.preferred))
    return dict(recipes=recipes, units=units, unit_order=unit_order(units), sources=sources,
                rules=model.RULES, uses=rule_uses(der), attachments=attachments, kspace=kspace,
                stage_a=sorted(stage_a(recipes, units)))
