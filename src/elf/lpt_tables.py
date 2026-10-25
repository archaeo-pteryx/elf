"""The LPT tables ``tables/lpt/*.dat`` (generated offline by tools/pt_autogen), loaded and checked, with their evaluators.

The tables and what this module makes of them follow.  The module needs numpy and the shared reader ``table_text``
only, so that the generator can load a copy of the two next to tables under test, and its evaluators use jax.numpy
when given JAX arrays.

* ``rows_recipes.dat``  the rows of PTTerms.rows in order, :class:`Row`, and the recipe of each (``RECIPES``, evaluated
  by lpt_stages).  ``TREE_ROWS`` is the leading block of rows linear in P_lin (their terms are transforms of the
  linear spectra or references to such rows).
* ``rows_sources.dat``  the stage-B units (``UNITS``) and the k-space sources (``SOURCES``) of the recipes.
* ``rows_rules.dat``    the discretisation rules (``RULES``) and the k-space rows of the final integral
  (``_KSPACE_ROWS``, (component, row, factor, interpolated, k0_const)).
* ``components.dat``    :class:`Template`, :class:`LinearKind` and :class:`Component`, ``_COMPONENTS``, the template
  matrix ``_TEMPLATE_MATRIX`` (templates x components), ``_LINEAR_MATRIX`` (components x linear kinds) and the bias
  exponents of the templates ``_LPT_BIAS_DEGREES``.
* ``basis_rows.dat``, ``component_entries.dat``  the component table c_j^c(q) = sum over the entries e = (c, j, t) of
  s_e R_t(q) (``CT_ROWS``, ``CT_ENTRY_NAMES``, ``CT_MONO_EXP``, ``CT_MONO_COEF``, ``COMPONENT_ENTRIES``), evaluated by
  ``row_basis`` (R_t) and ``entry_scalars`` (s_e).
* ``moments.dat``       the moments in the order of utils_lpt.get_lpt_moments (``MOMENTS``) and their angular
  polynomials (``MOMENT_POLYNOMIALS``).
* ``psi_terms.dat``, ``scalar_layer.dat``  the psi family of the moments, evaluated by ``psi_family``.

The tables are generated and byte-checked by ``python -m pt_autogen check``, so the import does not re-verify their
structure.  A repeated key raises ValueError naming the file and the lines where a duplicate would be summed silently
(psi_terms.dat, scalar_layer.dat, component_entries.dat, rows_sources.dat, rows_recipes.dat), an undefined name is left
to the KeyError or IndexError of its lookup.  The rows of basis_rows.dat are placed by their ids, ``row_basis``
multiplies all the factors of a basis row.

``psi_family(A, C, B, ir, coeffs, need) -> {(name, deriv): {(j, basis name, shift): value}}``: ``coeffs`` maps the names
of COEFFS to (N_K,) vectors, K and K^x = scalar_dmonomials(c, s, A_mu, B_mu, x) @ SCALAR_MATRIX; ``need`` is a set of
(name, deriv).  'psi': sum_t Kv[t] m_t (the terms without K for 'K' only; terms with K^x[t] identically 0 skipped);
'dA', 'dB', 'dC' (name 'K'): the total derivatives at fixed B (resp. A, C), with d ir/dA = -A ir^3 (C likewise):
    d(A^a C^c B^b ir^i)/dA = a A^(a-1) C^c B^b ir^i - i A^(a+1) C^c B^b ir^(i+2),   d(..)/dB = b A^a C^c B^(b-1) ir^i.
Exact zeros are omitted, a constant value is a Python float; only + - * (JAX, numpy float64 / longdouble / object).
"""
import os
from enum import IntEnum
from typing import NamedTuple

import numpy as np

import operator
from functools import reduce

from .table_text import fraction as _fraction, records

_DIR = os.path.join(os.path.dirname(__file__), 'tables', 'lpt')


def _records(name, key=None):
    """The records of tables/lpt/<name> in file order (``table_text.records``)."""
    return records(os.path.join(_DIR, name), key)


def _enum(name, names, doc):
    """An IntEnum ``name`` with the members ``names`` (values 0, 1, ..) and the docstring ``doc``."""
    enum = IntEnum(name, [(n, i) for i, n in enumerate(names)], module=__name__)
    enum.__doc__ = doc
    return enum


# ---------------------------------------------------------------------------------------------------------------
# moments.dat

_MOMENT_LINES = _records('moments.dat')
MOMENTS = tuple(dict.fromkeys(m for m, *_ in _MOMENT_LINES))
"""The moments in the order of utils_lpt.get_lpt_moments, MQa NQb = (Khat.qhat)^a (n.qhat)^b."""
MOMENT_POLYNOMIALS = {m: tuple((int(c), *map(int, e)) for mm, c, *e in _MOMENT_LINES if mm == m) for m in MOMENTS}
"""{moment: ((coef, pAm, pBm, a, b, by), ..)}, the terms coef A_mu^pAm B_mu^pBm mu_K^a mu_x^b mu_y^by (moments.dat)."""

# ---------------------------------------------------------------------------------------------------------------
# psi_terms.dat and scalar_layer.dat

_TERMS = [(m, b, int(s), int(t), tuple(map(int, e)), float(v)) for m, b, s, t, *e, v in _records(
    'psi_terms.dat', lambda r: (r[0], r[1], int(r[2]), int(r[3]), tuple(map(int, r[4:-1]))))]   # key (j, b, s, t, e)
BASIS = tuple(dict.fromkeys(b for _, b, *_ in _TERMS))
_KEYS = list(dict.fromkeys((MOMENTS.index(m), b, s) for m, b, s, *_ in _TERMS))   # the psi entries (j, basis, shift)
PAIRS_OF = {j: tuple((b, s) for jj, b, s in _KEYS if jj == j) for j in range(len(MOMENTS))}
PAIRS = tuple(sorted({p for ps in PAIRS_OF.values() for p in ps}, key=lambda p: (p[1], BASIS.index(p[0]))))
_SCALAR = [(int(t), tuple(map(int, e)), float(v)) for t, *e, v in _records(
    'scalar_layer.dat', lambda r: (int(r[0]), tuple(map(int, r[1:-1]))))]                   # key (t, exponents)
# (a, b, e, g) of c^a s^b A_mu^e B_mu^g in the generator's order (degree, then decreasing exponents)
SCALAR_MONOMIALS = tuple(sorted({e for _, e, _ in _SCALAR}, key=lambda e: (sum(e), tuple(-x for x in e))))
N_K = 1 + max(t for t, _, _ in _SCALAR)
_S = {(e, t): v for t, e, v in _SCALAR}
SCALAR_MATRIX = np.array([[_S.get((e, t), 0.0) for t in range(N_K)] for e in SCALAR_MONOMIALS])   # K = mono @ this
COEFFS = {'K': None, 'Kc': 'c', 'Ks': 's', 'KAm': 'Am', 'KBm': 'Bm'}   # name -> the variable x of K^x
_X = {'c': 0, 's': 1, 'Am': 2, 'Bm': 3}                               # positions in (a, b, e, g)

# ---------------------------------------------------------------------------------------------------------------
# rows_recipes.dat, rows_sources.dat and rows_rules.dat


class Term(NamedTuple):
    """A term of a recipe (rows_recipes.dat): ``kind``, the ints l, n, qpow (None where the kind has none), and
    ``refs``, ((coef, ref), ..) with one line per source of an H or M term and one line otherwise (coef a Fraction)."""
    kind: str
    l: object
    n: object
    qpow: object
    refs: tuple


def _int_or_none(text):
    return None if text == '-' else int(text)


def _recipes(records):
    """{row: (Term, ..)} of the rows_recipes.dat records, the lines of one term (row, index) joined."""
    out = {}
    for row in dict.fromkeys(r[0] for r in records):
        terms = {}
        for _, i, kind, l, n, qpow, coef, ref in (r for r in records if r[0] == row):
            terms.setdefault(int(i), []).append((kind, l, n, qpow, coef, ref))
        # an H or M term combines the sources of its lines, the other kinds have one line
        out[row] = tuple(Term(lines[0][0], _int_or_none(lines[0][1]), _int_or_none(lines[0][2]),
                              _int_or_none(lines[0][3]), tuple((_fraction(c), ref) for *_, c, ref in lines))
                         for _, lines in sorted(terms.items()))
    return out


_RECIPE_LINES = _records('rows_recipes.dat', lambda r: (r[0], int(r[1]), r[7]))   # key (row, term, ref)
ROW_NAMES = tuple(dict.fromkeys(r[0] for r in _RECIPE_LINES))
Row = _enum('Row', ROW_NAMES, """Rows of ``PTTerms.rows`` (tables/lpt/rows_recipes.dat, in its order).

    The rows of ``TREE_ROWS`` and the one-loop rows live on the core q grid, the k-space rows (``K_ROWS``, made by the
    rules of rows_rules.dat) on the core k grid.  lpt.PTTerms lists their definitions.""")
RECIPES = _recipes(_RECIPE_LINES)
"""{row name: (Term, ..)}, the recipes of rows_recipes.dat."""


def _units_and_sources(records):
    """``(UNITS, SOURCES)`` of the rows_sources.dat records."""
    units, sources = {}, {}
    for record in records:
        if record[0] == 'unit':
            _, name, kind, n, l, c = record
            lines = units.get(name, (None, None, ()))[2]       # the lines of one unit share its kind and order
            units[name] = (kind, int(n), lines + ((int(l), _fraction(c)),))
        else:
            _, name, unit, kpow, c = record
            sources[name] = sources.get(name, ()) + ((unit, int(kpow), _fraction(c)),)
    return units, sources


UNITS, SOURCES = _units_and_sources(_records('rows_sources.dat',
                                             lambda r: (r[0], r[1], r[4] if r[0] == 'unit' else r[2])))
"""UNITS = {unit: (kind, n, ((l, coef), ..))}, the stage-B units of rows_sources.dat in table order, and SOURCES =
{source: ((unit, kpow, coef), ..)}, the k-space sources (coef a Fraction)."""


def _rule_records():
    """``(RULES, k-space records)`` of rows_rules.dat (a rule line is 'rule name | applies to | action')."""
    rules, kspace = {}, []
    with open(os.path.join(_DIR, 'rows_rules.dat')) as fh:
        for line in fh:
            if line.startswith('rule '):
                name, applies, action = (part.strip() for part in line[len('rule '):].split('|'))
                rules[name] = (applies, action)
            elif line.startswith('kspace '):
                kspace.append(line.split()[1:])
    return rules, kspace


RULES, _KSPACE_RECORDS = _rule_records()
"""RULES = {rule: (applies to, runtime action)} of rows_rules.dat."""

RECIPE_KINDS = ('xi', 'H', 'M', 'prod', 'row')
"""The kinds of the q-space terms, the k-space rows use the rules of RULES."""
SPECTRA = ('P', 'P_lt')
"""The linear spectra of the xi and M terms, P_lin and P_lin exp(-(k/k_IR)^2)."""


K_ROWS = tuple(row for row in ROW_NAMES if all(t.kind in RULES for t in RECIPES[row]))
"""The k-space rows (made by a rule of rows_rules.dat), after the q-space rows."""


def _linear(row, seen=()):
    """Whether the row is linear in P_lin (its terms are xi or M terms of the spectra or references to such rows)."""
    return all((t.kind in ('xi', 'M') and all(ref in SPECTRA for _, ref in t.refs))
               or (t.kind == 'row' and all(ref not in seen and _linear(ref, seen + (row,)) for _, ref in t.refs))
               for t in RECIPES[row])


# the rows linear in P_lin are the leading block of rows_recipes.dat (the generator writes them first)
_N_TREE = next(i for i, row in enumerate(ROW_NAMES) if not _linear(row))
TREE_ROWS = slice(0, _N_TREE)
"""The rows linear in P_lin (tree level), the leading block of PTTerms.rows."""

# ---------------------------------------------------------------------------------------------------------------
# components.dat

# records 'template name b1 b2 bG2 bGamma3' (or 'template name eft'), 'linear name row [row]' and
# 'component name kind templates linear origin'
_COMPONENT_LINES = _records('components.dat')
_TEMPLATE_RECORDS = [r[1:] for r in _COMPONENT_LINES if r[0] == 'template']
_LINEAR_RECORDS = [r[1:] for r in _COMPONENT_LINES if r[0] == 'linear']
_COMPONENT_RECORDS = [r[1:] for r in _COMPONENT_LINES if r[0] == 'component']
Template = _enum('Template', [r[0] for r in _TEMPLATE_RECORDS], """Templates, the rows of ``_TEMPLATE_MATRIX``
    (components.dat).

    The bias monomials of ``_LPT_BIAS_DEGREES`` (ONE, the monomial 1, is the matter part), then the EFT templates (CTR,
    the Zel'dovich counterterm base).""")
LinearKind = _enum('LinearKind', [r[0] for r in _LINEAR_RECORDS], """Kind of a component's linear part, the columns of
    ``_LINEAR_MATRIX`` (components.dat, ``LPT._linear_rows``).""")
Component = _enum('Component', [r[0] for r in _COMPONENT_RECORDS], """The components of the final integrand, the
    columns of ``_TEMPLATE_MATRIX`` (components.dat, ``_COMPONENTS``).""")
# (component, templates, linear kind or None) of every component, in Component order
_COMPONENTS = tuple((Component[r[0]], tuple(Template[t] for t in r[2].split(',')),
                     None if r[3] == '-' else LinearKind[r[3]]) for r in _COMPONENT_RECORDS)
N_COMPONENTS = len(_COMPONENTS)
# M[t, c] = 1 if component c belongs to template t, shape (n_templates, n_components)
_TEMPLATE_MATRIX = np.array([[1.0 if t in templates else 0.0 for _, templates, _ in _COMPONENTS] for t in Template])
# N[c, kind] = 1 if component c has a linear part of that kind, shape (n_components, n_kinds)
_LINEAR_MATRIX = np.array([[1.0 if linear == kind else 0.0 for kind in LinearKind] for _, _, linear in _COMPONENTS])
# the exponents of (b1, b2, bG2, bGamma3) of the bias templates (row t is template t), the EFT templates excluded
_LPT_BIAS_DEGREES = np.array([[int(e) for e in r[1:]] for r in _TEMPLATE_RECORDS if r[1] != 'eft'], dtype=np.int32)

# k-space rows of pt_terms added after the final transform (constants are read at k_min).  Columns: 'component' is the
# component whose weight W[:, component] multiplies the row, 'row' the row of pt_terms.rows, 'factor' a fixed
# multiplier.  'interpolated' rows are Hermite-interpolated in ln k at the true k, the others are constants.
# 'k0_const' rows are skipped when subtract_k0_const=True.  Used by LPT._final_integral and lpt_rule.small_terms.
_KSPACE_ROWS = tuple((Component[c], Row[row], float(_fraction(factor)), interp == 'yes', k0 == 'yes')
                     for row, c, factor, interp, k0 in _KSPACE_RECORDS)

# ---------------------------------------------------------------------------------------------------------------
# basis_rows.dat and component_entries.dat


def _rows_by_id(records):
    """The row names of the basis_rows.dat records placed by their ids t (0 .. N_CT - 1, in any order)."""
    names = [None] * len(records)
    for t, *rows in records:
        names[int(t)] = tuple(rows)
    return tuple(names)


CT_ROW_NAMES = _rows_by_id(_records('basis_rows.dat'))     # R_t: () the row 1, else Row names
N_CT = len(CT_ROW_NAMES)
CT_ROWS = tuple(tuple(int(Row[n]) for n in names) for names in CT_ROW_NAMES)    # basis rows R_t (Row indices)
_CE = [(c, m, int(t), float(v), tuple(map(int, e))) for c, m, t, v, *e in _records(
    'component_entries.dat', lambda r: (r[0], r[1], int(r[2]), tuple(map(int, r[4:]))))]   # key (c, m, t, e)
CT_ENTRY_NAMES = tuple(dict.fromkeys((c, m, t) for c, m, t, _, _ in _CE))         # (component, moment, row t)
CT_MONO_EXP = tuple(sorted({e for *_, e in _CE}))                       # exponents of (k, mu, f, Kfac, K, Ksq)
_S = {(e, (c, m, t)): v for c, m, t, v, e in _CE}
CT_MONO_COEF = np.array([[_S.get((e, n), 0.0) for n in CT_ENTRY_NAMES] for e in CT_MONO_EXP])      # s_e = mono @ this
COMPONENT_ENTRIES = tuple(tuple((MOMENTS.index(m), e, t) for e, (cc, m, t) in enumerate(CT_ENTRY_NAMES) if cc == c.name)
                          for c in Component)
"""Per component (Component order) its entries (moment index, entry index e, basis row t) in table order."""


# ---------------------------------------------------------------------------------------------------------------
# evaluators

def _dmono(e, deriv):
    """``[(factor, exponents)]`` of d m_e / d deriv by the rule of the docstring (``m_e`` itself for 'psi')."""
    if deriv == 'psi':
        return [(1, e)]
    v = {'dA': 0, 'dC': 1, 'dB': 2}[deriv]                # the ir^i factor (dA, dC): - i A^(a+1) .. ir^(i+2)
    out = [(e[v], e[:v] + (e[v] - 1,) + e[v + 1:])] if e[v] else []
    return out + ([(-e[3], e[:v] + (e[v] + 1,) + e[v + 1:3] + (e[3] + 2,))] if deriv != 'dB' and e[3] else [])


def _plan(name, deriv):
    """``[(key, [(coefficient, t, exponents)], constant)]`` of psi[name]_deriv: the terms in table order, the terms
    without K and monomial (pure constants) summed into one float added last."""
    x = COEFFS[name]
    live = [x is None or any(SCALAR_MONOMIALS[n][_X[x]] for n in np.flatnonzero(SCALAR_MATRIX[:, t]))
            for t in range(N_K)]
    terms = {}
    for m, b, s, t, e, v in _TERMS:
        if x is None or (t >= 0 and live[t]):
            for f, e2 in _dmono(e, deriv):
                terms.setdefault((MOMENTS.index(m), b, s), []).append((v * f, t, e2))
    plan = [(key, [(c, t, e) for c, t, e in terms.get(key, []) if t >= 0 or any(e)],
             float(sum(c for c, t, e in terms.get(key, []) if t < 0 and not any(e)))) for key in _KEYS]
    return [(key, pieces, const) for key, pieces, const in plan if pieces or const != 0.0]


_PLANS = {nd: _plan(*nd) for nd in [(n, 'psi') for n in COEFFS] + [('K', d) for d in ('dA', 'dB', 'dC')]}


def _power(pows, n):
    """``x^n`` of the list ``[1.0, x, x*x, ...]``, extended by repeated multiplication (each power once)."""
    while len(pows) <= n:
        pows.append(pows[-1] * pows[1])
    return pows[n]


def _monomial(e, pows, cache):
    """A^a C^c B^b ir^i (``e`` = (a, c, b, i), not all 0) as the left-to-right product of the powers; every distinct
    monomial (and leading partial product) is built once per call."""
    if e not in cache:
        v = max(n for n in range(4) if e[n])
        head, last = e[:v] + (0,) * (4 - v), _power(pows[v], e[v])
        cache[e] = _monomial(head, pows, cache) * last if any(head) else last
    return cache[e]


def psi_family(A, C, B, ir, coeffs, need):
    """``{(name, deriv): {(j, basis, shift): value}}`` for the (name, deriv) of ``need`` (see the module docstring)."""
    pows, cache, out = ([1.0, A], [1.0, C], [1.0, B], [1.0, ir]), {}, {}
    for name, deriv in sorted(frozenset(need)):
        Kv, res = coeffs[name], {}
        for key, pieces, const in _PLANS[(name, deriv)]:
            acc = None
            for c, t, e in pieces:
                a = abs(c)
                if t >= 0:                                    # (a Kv[t]) m, the factor folded in first
                    val = Kv[t] if a == 1.0 else a * Kv[t]
                    val = val * _monomial(e, pows, cache) if any(e) else val
                else:
                    val = _monomial(e, pows, cache) if a == 1.0 else a * _monomial(e, pows, cache)
                acc = (-val if c < 0 else val) if acc is None else (acc - val if c < 0 else acc + val)
            res[key] = const if acc is None else (acc + const if const != 0.0 else acc)
        out[(name, deriv)] = res
    return out


def _xp(*xs):
    """jax.numpy if an argument is a JAX array or tracer, otherwise numpy."""
    import jax.numpy as jnp
    return jnp if any(type(x).__module__.split('.')[0] in ('jax', 'jaxlib') for x in xs) else np


def _scalar_product(pows, e, one, factor=1.0):
    """``factor`` times the powers of ``e`` = (c, s, A_mu, B_mu) exponents, left to right (``one``: empty product)."""
    acc = None if factor == 1.0 else factor
    for v, n in enumerate(e):
        if n:
            acc = _power(pows[v], n) if acc is None else acc * _power(pows[v], n)
    return one if acc is None else (acc * one if isinstance(acc, float) else acc)


def scalar_monomials(c, s, A_mu, B_mu):
    """The (N_MONO,) stack of the monomials c^a s^b A_mu^e B_mu^g of SCALAR_MONOMIALS."""
    xp = _xp(c, s, A_mu, B_mu)
    pows, one = ([1.0, c], [1.0, s], [1.0, A_mu], [1.0, B_mu]), xp.ones_like(c)
    return xp.stack([_scalar_product(pows, e, one) for e in SCALAR_MONOMIALS])


def scalar_dmonomials(c, s, A_mu, B_mu, x):
    """d scalar_monomials / dx for x in ('c', 's', 'Am', 'Bm') (the four are independent), (N_MONO,); exact zeros
    where the exponent of x is 0."""
    xp = _xp(c, s, A_mu, B_mu)
    pows, v, zero, one = ([1.0, c], [1.0, s], [1.0, A_mu], [1.0, B_mu]), _X[x], xp.zeros_like(c), xp.ones_like(c)
    return xp.stack([zero if e[v] == 0 else _scalar_product(pows, e[:v] + (e[v] - 1,) + e[v + 1:], one, float(e[v]))
                     for e in SCALAR_MONOMIALS])


def row_basis(rows):
    """The basis rows R_t(q) of CT_ROWS (1, a row, or a product of rows), ``(N_CT, nq)``."""
    xp = _xp(rows)
    return xp.stack([reduce(operator.mul, (rows[i] for i in t)) if t else xp.ones_like(rows[0]) for t in CT_ROWS])


def where_monomials(xs, E):
    """``prod_v xs[v]^E[m, v]`` for the static exponent table E (n_mono, n_var), xs (n_var,).  The powers are products
    of ``where`` selections, so that the derivatives stay finite where a variable is 0."""
    xp = _xp(xs)
    E = np.asarray(E)
    out = None
    for v in range(E.shape[1]):
        for k in range(1, int(E[:, v].max()) + 1):
            t = xp.where(E[:, v] >= k, xs[v], 1.0)
            out = t if out is None else out * t
    return out


def entry_scalars(k, mu, f, Kfac):
    """The per-point factors s_e of the entries of component_entries.dat (CT_ENTRY_NAMES order), ``(n_entries,)``,
    s_e = sum over its lines of coef k^pk mu^pmu f^pf Kfac^pKfac K^pK Ksq^pKsq with K = k Kfac and Ksq = K^2."""
    xp = _xp(k, mu, f, Kfac)
    K = k * Kfac
    mono = where_monomials(xp.stack([k, mu, f, Kfac, K, K**2]), CT_MONO_EXP)
    return mono @ xp.asarray(CT_MONO_COEF)
