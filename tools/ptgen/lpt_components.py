"""The components of the LPT final integral, their templates and linear kinds, and the component-basis table (derived).

* ``pieces(terms)``: every term of the cumulant expansion (lpt_cumulant), contracted in the frame
  (lpt_tensors.contract), split into (bias monomial, families, row product, moment (a, b), per-point factor), with the
  moment phase i^{(a+b) mod 2} applied (lpt_model.MOMENT_PHASE, asserted real).
* ``components(pieces)``: the pieces grouped by the families of their cumulants (lpt_model.COMPONENT_NAMES, the
  constant joining lpt_model.CONSTANT_JOINS), then the EFT components.  A component's templates are its bias monomials
  (asserted to carry identical tables, so that the template matrix is 0 or 1) and its linear kind the kind of its terms
  first order in one tree row (lpt_model.LINEAR_KINDS).
* ``templates(comps)``: the templates in their order with their bias exponents (lpt_model.TEMPLATE_NAMING).
* ``component_basis(comps)``: the rows R_t and the entries (component, moment, t) with the coefficients of their
  per-point factors s_e, in lpt_model.ENTRY_ORDER and lpt_model.SCALAR_FORM (basis_rows.dat, component_entries.dat).
"""
from collections import defaultdict
from fractions import Fraction
from typing import NamedTuple

import sympy as sp

from . import lpt_model as model
from . import lpt_tensors as te
from .lpt_cumulant import DELTA, product_order

ROWS = [te.ROW_SYMBOLS[n] for n in model.ROW_ORDER]
ROW_INDEX = {n: i for i, n in enumerate(model.ROW_ORDER)}
FAMILY_ORDER = list(dict.fromkeys(e.family for e in model.ROW_DICTIONARY))
SCALAR_VARIABLES = (te.k, te.mu, te.f, te.Kfac)


def canonical_families(fams):
    return tuple(sorted(fams, key=FAMILY_ORDER.index))


def row_product(names):
    return tuple(sorted(names, key=ROW_INDEX.get))


class Piece(NamedTuple):
    bias: tuple          # bias names, sorted
    families: tuple      # the families of the cumulant factors (canonical), () for the constant term
    rows: tuple          # the row product (row names in ROW_ORDER order), () for the constant row 1
    moment: tuple        # (a, b) of (Khat.qhat)^a (n.qhat)^b
    scalar: sp.Expr      # the per-point factor, a polynomial in k, mu, f, Kfac (phase applied, real)
    loop_matter: bool    # a single displacement-only cumulant of total order MAX_ORDER (SCALAR_FORM 'elf')


def pieces(terms):
    """The pieces of ``terms`` ({(bias, symbols): coef}, the kept terms of lpt_cumulant.derive)."""
    out = []
    for (bias, syms), coef in terms.items():
        fams = canonical_families(te.ENTRIES[te.entry_of(s)].family for s in syms)
        loop_matter = (len(syms) == 1 and all(sl.field.kind == DELTA for sl in syms[0].slots)
                       and product_order(syms) == model.MAX_ORDER)
        value = sp.expand(coef * sp.Mul(*[te.contract(s) for s in syms]))
        for monom, c in sp.Poly(value, te.MQ, te.NQ, *ROWS).terms():
            a, b = monom[:2]
            rows = row_product(name for name, e in zip(model.ROW_ORDER, monom[2:]) for _ in range(e))
            c = sp.expand(c * sp.I**((a + b) % 2))
            assert not c.has(sp.I), (bias, syms, monom, c)
            out.append(Piece(bias, fams, rows, (a, b), c, loop_matter))
    return out


# ---------------------------------------------------------------------------------------------------------------
# components, templates, linear kinds

def template_name(bias):
    if not bias:
        return 'ONE'
    return '_'.join(b.upper() for b in sorted(bias, key=model.KEPT_BIAS.index))


def template_exponents(bias):
    """The exponents of KEPT_BIAS of a bias monomial (the row of lpt._LPT_BIAS_DEGREES)."""
    return tuple(sum(1 for b in bias if b == name) for name in model.KEPT_BIAS)


def template_key(bias):
    """The template order, by the exponents of (bGamma3, bG2, b2, b1) ascending."""
    return tuple(reversed(template_exponents(bias)))


class Component(NamedTuple):
    name: str
    kind: str            # 'cumulant' or 'eft'
    origin: tuple        # the families (cumulant) or the copied component (eft)
    templates: tuple     # template names (bias monomials), in template order
    biases: tuple        # the bias monomials of the templates (eft: ())
    table: dict          # {(moment, row product): factor}
    linear: object       # the linear kind name or None
    loop_matter: bool


def components(ps):
    """The derived components in lpt_model.COMPONENT_NAMES order, then the EFT components."""
    names = {canonical_families(fams): name for fams, name in model.COMPONENT_NAMES}
    by_name = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: sp.Integer(0))))
    flags = defaultdict(set)
    missing = set()
    for pc in ps:
        fams = pc.families or canonical_families(model.CONSTANT_JOINS)
        if fams not in names:
            missing.add(fams)
            continue
        by_name[names[fams]][pc.bias][(pc.moment, pc.rows)] += pc.scalar
        flags[names[fams]].add(pc.loop_matter if pc.families else None)
    if missing:
        raise KeyError(f'family products without a component name: {sorted(missing)}')
    tree = {row: kind for kind, rows in model.LINEAR_KINDS for row in rows}
    out = []
    for fams, name in model.COMPONENT_NAMES:
        per_bias = {b: {key: sp.expand(v) for key, v in tab.items() if sp.expand(v) != 0}
                    for b, tab in by_name[name].items()}
        if not per_bias:
            raise KeyError(f'declared component {name} has no terms')
        biases = sorted(per_bias, key=template_key)
        table = per_bias[biases[0]]
        for b in biases[1:]:
            other = per_bias[b]
            assert set(other) == set(table) and all(sp.expand(other[y] - table[y]) == 0 for y in table), \
                (name, biases[0], b)
        kinds = {tree[rows[0]] for (_, rows) in table if len(rows) == 1 and rows[0] in tree}
        assert len(kinds) <= 1, (name, kinds)
        flag = flags[name] - {None}
        assert len(flag) <= 1, (name, flag)
        out.append(Component(name, 'cumulant', canonical_families(fams), tuple(template_name(b) for b in biases),
                             tuple(biases), table, kinds.pop() if kinds else None, bool(flag and flag.pop())))
    for name, copied, template in model.EFT_COMPONENTS:
        src = next(c for c in out if c.name == copied)
        out.append(Component(name, 'eft', (copied,), (template,), (), dict(src.table), src.linear, src.loop_matter))
    return out


def templates(comps):
    """``[(name, exponents of KEPT_BIAS or None for an EFT template)]`` in template order (TEMPLATE_NAMING)."""
    biases = sorted({b for c in comps for b in c.biases}, key=template_key)
    eft = list(dict.fromkeys(t for c in comps if c.kind == 'eft' for t in c.templates))
    return [(template_name(b), template_exponents(b)) for b in biases] + [(t, None) for t in eft]


def moment_expressions(comp):
    """{moment name: sum over the row products of factor x rows} of a component."""
    out = defaultdict(lambda: sp.Integer(0))
    for ((a, b), rows), v in comp.table.items():
        out[model.moment_name(a, b)] += v * sp.Mul(*[te.ROW_SYMBOLS[r] for r in rows])
    return {m: sp.expand(e) for m, e in out.items()}


def angular_moments(comps):
    """The moments (a, b) the components use, by total degree and then by b."""
    return sorted({m for c in comps for (m, _) in c.table}, key=lambda m: (sum(m), m[1]))


# ---------------------------------------------------------------------------------------------------------------
# the component-basis table

def factor_monomials(scalar, loop_matter, form=None):
    """``[(exponents of FACTOR_VARIABLES, coefficient)]`` of a per-point factor, written per lpt_model.SCALAR_FORM."""
    form = form or model.SCALAR_FORM
    merged = defaultdict(lambda: sp.Integer(0))
    for (pk, pmu, pf, pK), c in sp.Poly(scalar, *SCALAR_VARIABLES).terms():
        if pk == pK and not (form == 'elf' and loop_matter):
            e = (0, pmu, pf, 0, pK % 2, pK // 2)
        else:
            e = (pk, pmu, pf, pK, 0, 0)
        merged[e] += sp.Rational(c)
    return sorted((e, c) for e, c in merged.items() if c != 0)


def basis_rows(comps):
    """The distinct row products R_t, the constant, the single rows in ROW_ORDER and then the products."""
    prods = {rows for c in comps for (_, rows) in c.table}
    return sorted(prods, key=lambda r: (len(r), tuple(ROW_INDEX[y] for y in r)))


def component_basis(comps, form=None):
    """``{'rows': [row names], 'entries': [(component, moment, t)], 'monomials': [exponents], 'coef': [[Fraction]]}``,
    the entries in lpt_model.ENTRY_ORDER and ``coef[m][e]`` the coefficient of ``monomials[m]`` in s_e."""
    R = basis_rows(comps)
    t_of = {r: t for t, r in enumerate(R)}

    def row_key(rows):
        e = [0] * len(model.ROW_ORDER)
        for y in rows:
            e[ROW_INDEX[y]] += 1
        return tuple(-v for v in e)

    entries, monos = [], []
    for c in comps:
        for (a, b), rows in sorted(c.table, key=lambda key: (sum(key[0]), key[0][1], row_key(key[1]))):
            entries.append((c.name, model.moment_name(a, b), t_of[rows]))
            monos.append(factor_monomials(c.table[((a, b), rows)], c.loop_matter, form))
    monomials = sorted({e for ms in monos for e, _ in ms})
    coef = [[Fraction(0)] * len(entries) for _ in monomials]
    for i, ms in enumerate(monos):
        for e, c in ms:
            coef[monomials.index(e)][i] = Fraction(int(c.p), int(c.q))
    return dict(rows=[tuple(r) for r in R], entries=entries, monomials=monomials, coef=coef)
