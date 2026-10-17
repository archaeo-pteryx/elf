"""Numeric tables from the exact generator output, and their files in ``src/elf/tables``.

Files (read by ``src/elf/pt_coeff.py`` and ``src/elf/pt_matrix.py``):

* ``ept_hankel_22.npz``, ``ept_hankel_13.npz``: ``names`` (block names ``<prefix>_l_n_m``,
  sorted as strings), ``lnm`` (int64 ``(nblocks, 3)``), ``table`` (float64
  ``(nblocks, nterms, 7)``: the powers of ``model.MONOMIAL_COLUMNS`` and the coefficient, rows
  in ``model.TERM_ORDER`` order, zero-padded).
* ``ept_matrix.npz``: for each of ``22`` and ``13``: ``names_<b>``, ``const_<b>`` (float64,
  the rational constant of each block), ``factors_<b>`` (float64 integer coefficients
  ``[i, j]`` of ``nu1^i nu2^j``, or ``[i]`` of ``nu1^i`` for 13, zero-padded),
  ``factor_block_<b>`` (int64 block index of each factor) and ``factor_power_<b>`` (int64:
  > 0 numerator, < 0 denominator); and ``uv`` (float64 ``(nterms, 7)`` rows of ``Z1 Z3_UV``
  as in the Hankel tables, sorted by the powers of (mu, f, b1)).
* ``lpt/*.dat``: the plain-text tables of the LPT chain, the row recipes (``rows_sources.dat``,
  ``rows_recipes.dat``, ``rows_rules.dat``, :mod:`pt_autogen.lpt_rows`), the components
  (``components.dat``, ``basis_rows.dat``, ``component_entries.dat``, :mod:`pt_autogen.lpt_components`)
  and the moments (``moments.dat``, ``psi_terms.dat``, ``scalar_layer.dat``,
  :mod:`pt_autogen.lpt_angular`, :mod:`pt_autogen.lpt_moments`).  One record per line, whitespace-separated
  columns, names instead of indices, exact coefficients, and '#' header lines that state the
  defining identity, the order, the counts and the columns.  The headers name the generator
  version and carry no date, so regeneration is byte-exact.
* ``manifest.json``: the generator version, the list of files, the results of the LPT verifications
  (``comparisons``) and ``info``, the derived facts worth keeping next to the tables (the dropped
  P13 zero-lag terms and the UV limits of Z3).  It records no date, source hash or timing (git holds the
  generator sources and their history, and the timings are only printed), so it changes only when the
  generator version or a result does.
"""
import json
import os
from fractions import Fraction
from string import Template

import numpy as np
import sympy as sp

from . import GENERATOR_VERSION, kernels, model
from .vectors import mu

nu1, nu2 = sp.symbols('nu1 nu2')

HANKEL_FILES = {'22': 'ept_hankel_22.npz', '13': 'ept_hankel_13.npz'}
MATRIX_FILE = 'ept_matrix.npz'
MANIFEST_FILE = 'manifest.json'
LPT_DIR = 'lpt'
LPT_FILES = tuple(LPT_DIR + '/' + name for name in (
    'rows_sources.dat', 'rows_recipes.dat', 'rows_rules.dat', 'components.dat', 'basis_rows.dat',
    'component_entries.dat', 'moments.dat', 'psi_terms.dat', 'scalar_layer.dat'))
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))
TABLE_DIR = os.path.join(REPO_DIR, 'src', 'elf', 'tables')


def column_symbols():
    """The sympy symbols of ``model.MONOMIAL_COLUMNS``."""
    syms = {'mu': mu, model.GROWTH_RATE: kernels.f}
    syms.update({name: kernels.BIAS[name] for name in model.BIAS_NAMES})
    return tuple(syms[c] for c in model.MONOMIAL_COLUMNS)


def to_fraction(c):
    """An exact rational sympy number as a Fraction."""
    c = sp.sympify(c)
    if not c.is_Rational:
        raise ValueError(f'not a rational number: {c}')
    return Fraction(int(c.p), int(c.q))


def block_rows(poly):
    """``[(powers of model.MONOMIAL_COLUMNS, Fraction)]`` of a block, in ``model.TERM_ORDER`` order."""
    poly = sp.Poly(sp.expand(poly), *column_symbols())
    rows = [(tuple(int(e) for e in expo), to_fraction(c)) for expo, c in poly.terms()]
    bias = [model.MONOMIAL_COLUMNS.index(name) for name in model.BIAS_NAMES]
    for expo, _ in rows:
        if sum(expo[i] for i in bias) > model.MAX_BIAS_DEGREE:
            raise ValueError(f'bias degree > {model.MAX_BIAS_DEGREE}: {expo}')
    order = [model.MONOMIAL_COLUMNS.index(name) for name in model.TERM_ORDER]
    rows.sort(key=lambda row: tuple(row[0][i] for i in order), reverse=True)
    return rows


def hankel_table(blocks, prefix):
    """``(names, lnm, table)`` of the exact blocks ``{(l, n, m): polynomial}``."""
    keyed = {'%s_%d_%d_%d' % ((prefix,) + tuple(key)): key for key in blocks}
    names = sorted(keyed)
    rows = [block_rows(blocks[keyed[name]]) for name in names]
    nterms = max(len(r) for r in rows)
    ncol = len(model.MONOMIAL_COLUMNS)
    table = np.zeros((len(names), nterms, ncol + 1), dtype=np.float64)
    for i, block in enumerate(rows):
        for j, (expo, c) in enumerate(block):
            table[i, j, :ncol] = expo
            table[i, j, ncol] = float(c)
    lnm = np.array([keyed[name] for name in names], dtype=np.int64)
    return names, lnm, table


def factored(R, gens):
    """``(const, [(coefficients, power)])`` of the rational function ``R``.

    ``R = const prod_i P_i^{power_i}`` with primitive integer polynomials ``P_i`` (sympy
    ``factor_list``), positive powers for the numerator, negative for the denominator.
    ``coefficients`` is ``{exponent tuple: int}``.  Factors are sorted (numerator first, then
    by total degree and coefficients) so the output does not depend on sympy's order.
    """
    num, den = sp.fraction(sp.cancel(sp.together(R)))
    cn, fn = sp.factor_list(num, *gens)
    cd, fd = sp.factor_list(den, *gens)
    const = to_fraction(cn) / to_fraction(cd)
    factors = []
    for sign, flist in ((1, fn), (-1, fd)):
        for poly, mult in flist:
            terms = {}
            for expo, c in sp.Poly(poly, *gens).terms():
                if not c.is_Integer:
                    raise ValueError(f'non-integer factor coefficient {c}')
                terms[tuple(int(e) for e in expo)] = int(c)
            factors.append((terms, sign * int(mult)))
    factors.sort(key=lambda fp: (0 if fp[1] > 0 else 1, max(sum(e) for e in fp[0]),
                                 sorted(fp[0].items()), fp[1]))
    rebuilt = sp.Rational(const.numerator, const.denominator)
    for terms, power in factors:
        poly = sum(c * sp.Mul(*[g**e for g, e in zip(gens, expo)]) for expo, c in terms.items())
        rebuilt = rebuilt * poly**power
    if sp.cancel(rebuilt - R) != 0:
        raise AssertionError('factorisation does not rebuild R')
    return const, factors


def matrix_arrays(m22, m13, uv_rows):
    """The arrays of ``ept_matrix.npz`` (module docstring) from the exact rational functions
    ``{name: R}`` of the two block types and the rows of the UV term."""
    arrays = {}
    for prefix, blocks, gens in (('22', m22, (nu1, nu2)), ('13', m13, (nu1,))):
        names = sorted(blocks)
        consts, factors, factor_block, factor_power = [], [], [], []
        for i, name in enumerate(names):
            const, flist = factored(blocks[name], gens)
            consts.append(float(const))
            for terms, power in flist:
                factors.append(terms)
                factor_block.append(i)
                factor_power.append(power)
        shape = tuple(1 + max(expo[d] for terms in factors for expo in terms) for d in range(len(gens)))
        table = np.zeros((len(factors),) + shape, dtype=np.float64)
        for i, terms in enumerate(factors):
            for expo, c in terms.items():
                if abs(c) >= 2**53:
                    raise ValueError('factor coefficient not exact in float64')
                table[(i,) + expo] = c
        arrays['names_' + prefix] = np.array(names)
        arrays['const_' + prefix] = np.array(consts, dtype=np.float64)
        arrays['factors_' + prefix] = table
        arrays['factor_block_' + prefix] = np.array(factor_block, dtype=np.int64)
        arrays['factor_power_' + prefix] = np.array(factor_power, dtype=np.int64)
    ncol = len(model.MONOMIAL_COLUMNS)
    uv = np.zeros((len(uv_rows), ncol + 1), dtype=np.float64)
    for i, (powers, c) in enumerate(uv_rows):
        uv[i, :ncol] = powers
        uv[i, ncol] = float(c)
    arrays['uv'] = uv
    return arrays


def write_npz(path, arrays):
    np.savez(path, **arrays)


def read_npz(path):
    """All arrays of an npz file (no pickles)."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def arrays_equal(a, b):
    """Exact equality of two dicts of arrays (keys, dtypes, shapes and values)."""
    if set(a) != set(b):
        return False
    return all(a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
               and np.array_equal(a[key], b[key]) for key in a)


def write_manifest(out_dir, comparisons, info):
    """Write ``manifest.json``: the generator version, the files, the ``comparisons`` and ``info``, the derived facts
    worth keeping next to the tables.  No date, source hash or timing is recorded (git has the sources and their
    history, and the timings are only printed)."""
    manifest = dict(generator=GENERATOR_VERSION,
                    files=sorted(list(HANKEL_FILES.values()) + [MATRIX_FILE]) + list(LPT_FILES),
                    comparisons=comparisons,
                    info=info)
    with open(os.path.join(out_dir, MANIFEST_FILE), 'w') as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
        fh.write('\n')
    return manifest


# ---------------------------------------------------------------------------
# plain-text tables (LPT)

GENERATED_BY = f'# Generated by tools/pt_autogen (python -m pt_autogen regenerate), {GENERATOR_VERSION}.'

RUNTIME_READER = 'lpt_tables.py'
"""The runtime module of src/elf that reads the psi and component-basis tables."""

LPT_HEADERS = {
    'psi_terms.dat': """\
# psi_terms.dat, the psi family of the LPT moments (read by $reader).
$generated_by
# Identity: every moment M_j(l) of utils_lpt.get_lpt_moments (j = MQ0 .. MQ2NQ1 of moments.dat, without exp(-B s^2))
#   is exactly
#     M_j(l) = sum over the pairs (b, s) of moment j with s <= l of basis_b[l - s] psi_{j,b,s},
#     psi_{j,b,s} = sum over the lines (moment j, basis b, shift s) of coef K_t A^a C^c B^b ir^i,
#   with basis_b the Horner basis of utils_lpt.lpt_horner_poly (S0 from V[0], T1 .. T4 from V'[0] .. V'[3]),
#   A = k q c, C = k q s, B = -(1/2) K^2 Y_lin_lt, ir = 1/sqrt(A^2 + C^2) ((nq,) arrays) and K_t the per-mu
#   coefficients of scalar_layer.dat.  A line with t = -1 has no K factor.  Such a constant term belongs to the
#   primal K only (the vectors K^x = dK/dx of the rule drop it), and the terms whose K^x_t vanishes identically
#   are dropped too.  The moments are derived from the generating function (tools/pt_autogen/lpt_angular.py).
# Order: by moment, within a moment by shift and then basis, and the terms of an entry in descending lex order
#   of the exponents of (B, A, C, ir).  The runtime takes the entries, the pairs of each moment and the distinct
#   monomials A^a C^c B^b ir^i in order of first appearance and evaluates the sums in file order.
# Counts: $terms terms ($with_K with a K factor, $without_K without) in $entries entries (j, b, s), $pairs pairs,
#   $moments moments, $monomials monomials A^a C^c B^b ir^i other than 1.
# Coefficients: exact decimals (all are dyadic rationals).
# columns: moment basis shift t a c b i coef""",
    'scalar_layer.dat': """\
# scalar_layer.dat, the per-mu coefficients K_t of psi_terms.dat (read by $reader).
$generated_by
# Identity: K_t = sum over the lines t of coef c^a s^b A_mu^e B_mu^g, with c, s the cosine and sine of the angle
#   between K = k + f (k.n) n and k, A_mu = (1 + f) mu / Kfac and B_mu = sqrt(1 - mu^2) / Kfac (lpt._rsd_geometry).
#   The rule also uses K^x = dK/dx for x = c, s, A_mu, B_mu, the other three held fixed (c and s independent).
# Order: increasing t, within t the monomials by increasing degree a + b + e + g and then decreasing (a, b, e, g)
#   (the runtime orders the distinct monomials by the same rule).
# Counts: $scalar_entries nonzero entries, $K coefficients (t = 0 .. $last_t), $scalar_monomials distinct monomials.
# Coefficients: exact decimals.
# columns: t a b e g coef""",
    'basis_rows.dat': """\
# basis_rows.dat, the component-basis rows R_t(q) of the final integrand (read by $reader).
$generated_by
# Identity: the moment coefficient c_j^c(q) of component c (components.dat) is the sum over the entries (c, j, t) of
#   component_entries.dat of s_e R_t(q), with R_t the product of the rows of PTTerms.rows named on its line
#   (rows_recipes.dat).  A line with t alone is the constant row 1.  The table is derived from the cumulant
#   expansion of the generating function (tools/pt_autogen/lpt_cumulant.py, lpt_components.py).
# Order: the constant row, the single rows in the order of PTTerms.rows, then the products.
# Counts: $rows rows.
# columns: t [row [row]]""",
    'component_entries.dat': """\
# component_entries.dat, the per-point factors s_e of the component-basis entries (read by $reader).
$generated_by
# Identity: s_e of the entry e = (component, moment, t) is the sum over its lines of
#   coef k^pk mu^pmu f^pf Kfac^pKfac K^pK Ksq^pKsq (K = k Kfac, Ksq = K^2), with the components of components.dat,
#   the moments of moments.dat and t a row of basis_rows.dat.  Then c_j^c(q) = sum over e = (c, j, t) of s_e R_t(q),
#   and the integrand of the final integral is sum_c W_c sum_j c_j^c M_j.  A term c (Khat.qhat)^a (n.qhat)^b of the
#   cumulant expansion enters with c i^((a + b) mod 2), the phase of the moments.
# Order: the components in the order of components.dat, within a component the moments by degree a + b and then
#   by the power b of n.qhat, within a moment the rows in descending lex order of their exponents, within an entry
#   the monomials in increasing order of the exponent tuple.  The runtime sums the entries in this order.
# Counts: $entries entries, $lines lines (one nonzero monomial each), $monomials distinct monomials.
# Coefficients: exact decimals.
# columns: component moment t coef pk pmu pf pKfac pK pKsq""",
    'components.dat': """\
# components.dat, the components of the LPT final integrand, their templates and linear kinds (read by $reader).
$generated_by
# Identity: the integrand of the final integral is sum over the components c of W_c sum_j c_j^c(q) M_j
#   (component_entries.dat), with the component weights W = (template weights) @ T and T[t, c] = 1 when the
#   template t is listed on the line of c.  A cumulant component is the set of terms of the cumulant expansion whose
#   cumulants have the families listed (origin, '*'-joined), the constant 1 joining the family A.  An EFT component
#   copies the entries of its origin.  A linear kind collects the terms first order in one of its tree rows, which
#   the final integral replaces by linear theory (lpt.LPT._linear_rows).
# Records:
#   template   name b1 b2 bG2 bGamma3      the exponents of a bias monomial ('eft' for an EFT template)
#   linear     name row [row]              a linear kind and its tree rows
#   component  name kind templates linear origin
#              (kind 'cumulant' or 'eft', the templates ','-joined, linear '-' when the component has no linear part)
# Order: the templates in the order of the template weights, the linear kinds in column order, the components in
#   the column order of T.
# Counts: $templates templates ($eft_templates EFT), $kinds linear kinds, $components components ($eft_components EFT).
# columns: record name ...""",
    'moments.dat': """\
# moments.dat, the angular polynomial of each moment of the LPT final integral (read by $reader).
$generated_by
# Identity: with mu_K = Khat.qhat, mu_x = e_x.qhat (e_x = (khat - c Khat)/s), mu_y the third component and the
#   polynomial P_j = sum over the lines of moment j of coef A_mu^pAm B_mu^pBm mu_K^a mu_x^b mu_y^by (degree n),
#     (-i)^(n mod 2) int dOmega/(4 pi) e^(i k.q) e^(B mu_K^2) P_j = exp(B c^2) sum_l w_l(kq) M_j(l),
#   w_l(x) = (-2/x)^l j_l(x), B = -(1/2) K^2 Y_lin_lt, A_mu = (1 + f) mu / Kfac, B_mu = sqrt(1 - mu^2) / Kfac and
#   n.qhat = A_mu mu_K - B_mu mu_x.  M_j(l) are the moments of utils_lpt.get_lpt_moments, in its order,
#   derived by tools/pt_autogen/lpt_angular.py and factored in psi_terms.dat.
# Order: the moments in the order of get_lpt_moments, within a moment the terms in declaration order.
# Counts: $moments moments, $lines lines.
# Coefficients: integers.
# columns: moment coef pAm pBm a b by""",
    'rows_sources.dat': """\
# rows_sources.dat, the stage-B units and k-space sources of the LPT rows (read by lpt_tables.py, for lpt_stages.py).
$generated_by
# Identity (unit lines): a 22 unit is U(k) = 4 pi int dq q^2 j_0(kq) [sum over its lines of coef (xi_l^n(q))^2], the
#   q -> k transform of products of the stage-A outputs of P on the common q grid, and a 13 unit is
#   U(k) = P(k) int dq q j_l(kq) xi_l^n(q), from the own grid of xi_l^n (one line, coef 1).
# Identity (source lines): a source is S(k) = sum over its lines of coef k^kpow U(k).  The sources are the paper's
#   Q_n = int_p F P(p) P(|k-p|) and R_n = P(k) int_p F P(p) (tools/pt_autogen/lpt_model.py SOURCE_NAMES), every derived
#   row source being a unique combination of them.  xi_l^n = H_l^n[P], H_l^n as in rows_recipes.dat.
# Order: the 22 units by n descending, then the 13 units by n descending and l, the sources by name and their lines
#   in unit order.
# Counts: $units units ($units22 22, $units13 13), $sources sources, $source_lines source lines.
# Coefficients: exact rationals.
# columns: unit name kind n l coef  |  source name unit kpow coef""",
    'rows_recipes.dat': """\
# rows_recipes.dat, the recipe of every row of PTTerms.rows (read by lpt_tables.py, for lpt_stages.py).
$generated_by
# Identity: a row is the sum of its terms, the lines with the same row and term index forming one term, with
#     xi    coef q^qpow xi_l^n[ref], the stage-A transform of the linear spectrum ref (P or P_lt = P e^(-(k/k_IR)^2))
#     H     q^qpow H_l^n[sum over the term's lines of coef ref], a stage-C transform of the sources (rows_sources.dat)
#     M     M_n[sum over the term's lines of coef ref], the zero-lag constant (a grid moment of the source or spectrum)
#     prod  coef q^qpow ref, a product of stage-A outputs of P on the core q grid (xi_l_n, 'm' a minus sign)
#     row   coef ref, another row
#   and, for the rows on the k grid, a rule of rows_rules.dat applied to ref with the factor coef.
#   H_l^n[S](q) = int dk k^(n+2) S(k) j_l(kq) / (2 pi^2) and M_n[S] = H_0^n[S](q = 0).  The recipes are the derived
#   ones with the declared preferences of representation (tools/pt_autogen/lpt_model.py PREFERENCES).
# Order: the rows in the order of PTTerms.rows, the terms in derivation order.
# Counts: $rows rows ($qrows on the q grid, $krows on the k grid), $terms terms, $lines lines.
# Coefficients: exact rationals.
# columns: row term kind l n qpow coef ref  ('-' where a column does not apply)""",
    'rows_rules.dat': """\
# rows_rules.dat, the discretisation rules of the LPT rows and where they apply (read by lpt_tables.py).
$generated_by
# Records:
#   rule    name | applies to | runtime action   (the rules of tools/pt_autogen/lpt_model.py RULES)
#   uses    rule row ...                         (the rows whose recipes use a rule that applies to some rows only)
#   attach  component term rule                  (every k-independent MQ0 term of the final integrand, derived)
#   kspace  row component factor interpolated k0_const
#           (the final integral adds factor x row to the weight of the component, interpolating the row in ln k at
#           the true k when interpolated is yes and reading it at k_min otherwise, and skips the k0_const rows when
#           subtract_k0_const is set)
# Order: the rules in declaration order, the attachments in component order, the k-space rows in row order.
# Counts: $rules rules, $attachments attachments, $kspace k-space rows.
# columns: record ...""",
}
"""The '#' headers of the LPT text tables, with $-placeholders for the generator line, the reader and the counts."""


def lpt_header(name, **counts):
    """The header lines of the LPT table ``name`` with the generator line and ``counts`` filled in."""
    return Template(LPT_HEADERS[name]).substitute(generated_by=GENERATED_BY, reader=RUNTIME_READER,
                                                  **counts).splitlines()


def decimal(x):
    """The exact decimal string of the dyadic rational ``x`` (a Fraction, int or float), e.g. '-0.0625'."""
    fr = Fraction(x)
    k = fr.denominator.bit_length() - 1
    if fr.denominator != 1 << k:
        raise ValueError(f'not a dyadic rational: {x}')
    digits = str(abs(fr.numerator) * 5**k).rjust(k + 1, '0')               # |x| = digits / 10^k exactly
    out = ('-' if fr < 0 else '') + (digits[:-k] + '.' + digits[-k:] if k else digits)
    assert Fraction(out) == fr and float(out) == float(fr), (x, out)
    return out


def rational(x):
    """The exact string 'a/b' (or 'a') of a rational number ``x`` (sympy, Fraction or int)."""
    fr = to_fraction(x) if not isinstance(x, Fraction) else x
    return str(fr.numerator) if fr.denominator == 1 else f'{fr.numerator}/{fr.denominator}'


def write_text_table(path, header_lines, rows):
    """Write a plain-text table, the '#' header lines and then one record per line ('\\n' line ends)."""
    with open(path, 'w', newline='\n') as fh:
        fh.write('\n'.join(list(header_lines) + list(rows)) + '\n')


def read_text_table(path):
    """The records of a plain-text table, each split at whitespace (the '#' lines are its header)."""
    with open(path) as fh:
        return [line.split() for line in fh if line.strip() and not line.startswith('#')]


def lpt_psi_text(tables):
    """``{file: (header lines, rows)}`` of psi_terms.dat and scalar_layer.dat (``lpt_moments.make_tables``)."""
    n = tables['counts']
    psi = [f'{m:7s} {b:2s} {sh:2d} {t:3d}  {e[0]} {e[1]} {e[2]} {e[3]}  {decimal(v)}'
           for m, b, sh, t, e, v in tables['terms']]
    scalar = [f'{t:3d}  {e[0]} {e[1]} {e[2]} {e[3]}  {decimal(v)}' for t, e, v in tables['scalar']]
    return {LPT_DIR + '/psi_terms.dat': (lpt_header('psi_terms.dat', **n), psi),
            LPT_DIR + '/scalar_layer.dat': (lpt_header('scalar_layer.dat', last_t=n['K'] - 1, **n), scalar)}


def lpt_basis_text(basis):
    """``{file: (header lines, rows)}`` of basis_rows.dat and component_entries.dat (``lpt_components.component_basis``)."""
    rows = [(f'{t:2d}  ' + ' '.join(names)).rstrip() for t, names in enumerate(basis['rows'])]
    coef, mono = basis['coef'], basis['monomials']
    entries = [f'{c:6s} {j:7s} {t:2d}  {decimal(coef[m][e]):6s}  ' + ' '.join(str(x) for x in mono[m])
               for e, (c, j, t) in enumerate(basis['entries']) for m in range(len(mono)) if coef[m][e] != 0]
    return {LPT_DIR + '/basis_rows.dat': (lpt_header('basis_rows.dat', rows=len(rows)), rows),
            LPT_DIR + '/component_entries.dat': (lpt_header('component_entries.dat', entries=len(basis['entries']),
                                                            lines=len(entries), monomials=len(mono)), entries)}


def lpt_components_text(templates, kinds, comps):
    """``{file: (header lines, rows)}`` of components.dat from ``lpt_components.templates`` (name, exponents or None),
    the linear kinds ((kind, rows)) and ``lpt_components.components``."""
    lines = [f'template   {name:12s} ' + ('eft' if e is None else ' '.join(str(v) for v in e)) for name, e in templates]
    lines += [f'linear     {name:12s} ' + ' '.join(rows) for name, rows in kinds]
    lines += [f'component  {c.name:6s} {c.kind:8s} {",".join(c.templates):11s} {c.linear or "-":3s} '
              + '*'.join(c.origin) for c in comps]
    counts = dict(templates=len(templates), eft_templates=sum(1 for _, e in templates if e is None),
                  kinds=len(kinds), components=len(comps), eft_components=sum(1 for c in comps if c.kind == 'eft'))
    return {LPT_DIR + '/components.dat': (lpt_header('components.dat', **counts), lines)}


def lpt_moments_text(declaration, A_mu, B_mu):
    """``{file: (header lines, rows)}`` of moments.dat from the declaration {moment: ((coef, a, b, by), ..)}, the
    coefficients monomials c A_mu^pAm B_mu^pBm with an integer c."""
    lines = []
    for name, terms in declaration.items():
        for coef, a, b, by in terms:
            (pAm, pBm), c = sp.Poly(sp.sympify(coef), A_mu, B_mu).terms()[0]
            assert sp.Poly(sp.sympify(coef), A_mu, B_mu).length() == 1 and sp.sympify(c).is_Integer, coef
            lines.append(f'{name:7s} {int(c):3d}  {pAm} {pBm}  {a} {b} {by}')
    return {LPT_DIR + '/moments.dat': (lpt_header('moments.dat', moments=len(declaration), lines=len(lines)), lines)}


TERM_COLUMNS = {'xi': 'lnq', 'H': 'lnq', 'M': 'n', 'prod': 'q', 'row': ''}
"""The columns (l, n, qpow) each recipe kind uses (a rule term uses n when it has one)."""


def term_columns(t):
    """The printed (l, n, qpow) of a recipe term, '-' where its kind has no such column."""
    used = TERM_COLUMNS.get(t.kind, 'n' if t.n is not None else '')
    return tuple(str(v) if c in used else '-' for v, c in ((t.l, 'l'), (t.n, 'n'), (t.qpow, 'q')))


def lpt_rows_text(rows):
    """``{file: (header lines, rows)}`` of rows_sources.dat, rows_recipes.dat and rows_rules.dat from
    ``lpt_rows.tables``."""
    units, sources = rows['units'], rows['sources']
    ulines = [f'unit   {u:9s} {units[u][0]}  {units[u][1]:3d} {l:2d}  {rational(c)}'
              for u in rows['unit_order'] for l, c in units[u][2]]
    slines = [f'source {s:9s} {u:9s} {kp:2d}  {rational(c)}' for s in sorted(sources)
              for u, c, kp in sorted(sources[s], key=lambda t: rows['unit_order'].index(t[0]))]
    out = {LPT_DIR + '/rows_sources.dat': (lpt_header(
        'rows_sources.dat', units=len(units), units22=sum(1 for u in units.values() if u[0] == '22'),
        units13=sum(1 for u in units.values() if u[0] == '13'), sources=len(sources), source_lines=len(slines)),
        ulines + slines)}
    rlines, nterms = [], 0
    for row, terms in rows['recipes'].items():
        for i, t in enumerate(terms):
            nterms += 1
            parts = (t.ref.items() if t.kind in ('H', 'M') else [(t.ref, 1)])
            l, n, qpow = term_columns(t)
            for ref, c in parts:
                rlines.append(f'{row:13s} {i:2d}  {t.kind:12s} {l:>2s} {n:>3s} {qpow:>3s}  {rational(t.coef * c):12s} {ref}')
    krows = sum(1 for terms in rows['recipes'].values() if terms[0].kind not in ('xi', 'H', 'M', 'prod', 'row'))
    out[LPT_DIR + '/rows_recipes.dat'] = (lpt_header(
        'rows_recipes.dat', rows=len(rows['recipes']), qrows=len(rows['recipes']) - krows, krows=krows,
        terms=nterms, lines=len(rlines)), rlines)
    lines = [f'rule    {r.name:17s} | {r.applies_to} | {r.action}' for r in rows['rules']]
    lines += [f'uses    {rule:17s} ' + ' '.join(rs) for rule, rs in rows['uses'].items()]
    lines += [f'attach  {a.component:6s} {str(a.term).replace("**", "^"):12s} {a.rule}' for a in rows['attachments']]
    lines += [f'kspace  {row:13s} {comp:6s} {rational(f):3s} {"yes" if interp else "no":3s} {"yes" if k0 else "no"}'
              for row, comp, f, interp, k0 in rows['kspace']]
    out[LPT_DIR + '/rows_rules.dat'] = (lpt_header(
        'rows_rules.dat', rules=len(rows['rules']), attachments=len(rows['attachments']), kspace=len(rows['kspace'])),
        lines)
    return out
