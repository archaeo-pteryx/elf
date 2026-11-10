"""The LPT tables ``tables/lpt/*.dat`` (generated offline by tools/pt_autogen), loaded, with their evaluators.

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
* ``basis_rows.dat``, ``component_entries.dat``  the component table (the prefix CT of the names, not a counterterm)
  c_j^c(q) = sum over the entries e = (c, j, t) of s_e R_t(q) (``CT_ROWS``, ``CT_ENTRY_NAMES``, ``CT_MONO_EXP``,
  ``CT_MONO_COEF``, ``COMPONENT_ENTRIES``), evaluated by ``row_basis`` (R_t) and ``entry_scalars`` (s_e).
* ``moments.dat``       the moments of the final integral (``MOMENTS``) and their angular polynomials
  (``MOMENT_POLYNOMIALS``), whose angular factors ``angular.AngularPlan`` builds from ``angular.dat`` (read by
  ``angular``).

The tables are generated and byte-checked by ``python -m pt_autogen check``, so the import does not re-verify their
structure.  A repeated key raises ValueError naming the file and the lines where a duplicate would be summed silently
(component_entries.dat, rows_sources.dat, rows_recipes.dat), an undefined name is left to the KeyError or IndexError of
its lookup.  The rows of basis_rows.dat are placed by their ids, ``row_basis`` multiplies all the factors of a basis
row.
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
"""The moments in table order (the moment index j).  A name gives the powers of Khat.qhat (MQ) and of n.qhat (NQ), so
MQ2NQ1 is (Khat.qhat)^2 (n.qhat), and MQ0 is 1."""
MOMENT_POLYNOMIALS = {m: tuple((int(c), *map(int, e)) for mm, c, *e in _MOMENT_LINES if mm == m) for m in MOMENTS}
"""{moment: ((coef, pAm, pBm, a, b, by), ..)}, the terms coef A_mu^pAm B_mu^pBm mu_K^a mu_x^b mu_y^by (moments.dat)."""

# ---------------------------------------------------------------------------------------------------------------
# rows_recipes.dat, rows_sources.dat and rows_rules.dat


class Term(NamedTuple):
    """A term of a recipe (rows_recipes.dat).  ``kind`` is 'xi' (a stage-A output xi_l^n of the spectrum ref), 'H' (a
    stage-C transform H_l^n of a combination of sources), 'M' (a zero-lag constant M_n of a combination of sources or
    of a spectrum), 'prod' (a product of stage-A outputs), 'row' (another row) or the name of a rule of rows_rules.dat
    (a k-space row).  l, n and qpow are ints (None where the kind has none), and ``refs`` is ((coef, ref), ..) with one
    line per source of an H or M term and one line otherwise (coef a Fraction)."""
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

    The rows of ``TREE_ROWS`` and the one-loop rows live on the core q grid, the k-space rows (made by the rules of
    rows_rules.dat) on the core k grid.  lpt.PTTerms lists their definitions.""")
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

SPECTRA = ('P', 'P_lt')
"""The linear spectra of the xi and M terms, P_lin and P_lin exp(-(k/k_IR)^2)."""


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
# 'k0_const' rows are skipped when subtract_k0_const=True.  Used by LPT._kspace_tables and LPT._kspace_terms.
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

def _xp(*xs):
    """jax.numpy if an argument is a JAX array or tracer, otherwise numpy."""
    import jax.numpy as jnp
    return jnp if any(type(x).__module__.split('.')[0] in ('jax', 'jaxlib') for x in xs) else np


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
