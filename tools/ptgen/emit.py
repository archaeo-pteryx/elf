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
* ``manifest.json``: generator version, sha256 of ``tools/ptgen/*.py``, date, timings and
  the comparison results.
"""
import datetime
import hashlib
import json
import os
from fractions import Fraction

import numpy as np
import sympy as sp

from . import GENERATOR_VERSION, kernels, model
from .vectors import mu

nu1, nu2 = sp.symbols('nu1 nu2')

HANKEL_FILES = {'22': 'ept_hankel_22.npz', '13': 'ept_hankel_13.npz'}
MATRIX_FILE = 'ept_matrix.npz'
MANIFEST_FILE = 'manifest.json'
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


def sources_sha256():
    """``{file: sha256}`` of ``tools/ptgen/*.py``."""
    out = {}
    for name in sorted(os.listdir(PACKAGE_DIR)):
        if name.endswith('.py'):
            with open(os.path.join(PACKAGE_DIR, name), 'rb') as fh:
                out[name] = hashlib.sha256(fh.read()).hexdigest()
    return out


def write_manifest(out_dir, comparisons, timings, info):
    """Write ``manifest.json``; ``info`` holds derived facts worth keeping next to the tables."""
    manifest = dict(generator=GENERATOR_VERSION,
                    date=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
                    sources_sha256=sources_sha256(),
                    files=sorted(list(HANKEL_FILES.values()) + [MATRIX_FILE]),
                    timings_s=timings,
                    comparisons=comparisons,
                    info=info)
    with open(os.path.join(out_dir, MANIFEST_FILE), 'w') as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
        fh.write('\n')
    return manifest
