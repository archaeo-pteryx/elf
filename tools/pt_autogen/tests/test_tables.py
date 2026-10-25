"""The generated EPT tables (spec section 6), the plain-text tables ``src/elf/tables/ept/*.dat``.

* regeneration reproduces the committed tables byte for byte (headers included, as they carry no
  date), the headers are at most 120 characters wide and end with the columns line;
* the runtime (``elf.pt_coeff``, ``elf.pt_matrix``) decodes the committed tables into the exact
  generator output: the Hankel rows and the matrix constants, factors and UV rows are the floats
  of the exact rationals;
* its matrices agree with the exact rational functions.
"""
import os
import sys

import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from pt_autogen import emit, ept_matrix
from pt_autogen.__main__ import generate, tables

sys.path.insert(0, os.path.join(emit.REPO_DIR, 'src'))


@pytest.fixture(scope='module')
def generated():
    return generate(('ept',))[0]


def test_regenerated_tables_equal_committed(generated, tmp_path):
    texts = tables(generated)
    assert sorted(texts) == sorted(emit.EPT_FILES)
    for fname, (header, rows) in texts.items():
        out = tmp_path / os.path.basename(fname)
        emit.write_text_table(out, header, rows)
        with open(os.path.join(emit.TABLE_DIR, fname), 'rb') as fh:
            assert fh.read() == out.read_bytes(), fname
        assert all(line.startswith('#') and len(line) <= 120 for line in header), fname
        assert header[-1].startswith('# columns:') and emit.GENERATED_BY in header, fname


def test_runtime_reads_hankel_tables(generated):
    from elf import pt_coeff
    for prefix in ('22', '13'):
        names, lnm, table = pt_coeff.read_tables(prefix)
        blocks = {'%s_%d_%d_%d' % ((prefix,) + tuple(key)): key for key in generated['b' + prefix]}
        assert names == sorted(blocks)
        assert lnm.dtype == np.int64 and lnm.tolist() == [list(blocks[name]) for name in names]
        assert table.dtype == np.float64 and table.shape[2] == 7
        for i, name in enumerate(names):
            rows = emit.block_rows(generated['b' + prefix][blocks[name]])
            expected = np.zeros((table.shape[1], 7))
            expected[:len(rows)] = [list(map(float, expo)) + [float(c)] for expo, c in rows]
            assert np.array_equal(table[i], expected), name


def test_runtime_reads_matrix_tables(generated):
    from elf import pt_matrix
    nu1, nu2 = sp.symbols('nu1 nu2')
    decoded = pt_matrix._tables()
    for prefix, gens in (('22', (nu1, nu2)), ('13', (nu1,))):
        names, blocks, factors = decoded[prefix]
        assert list(names) == sorted(generated['m' + prefix])
        for name, (const, flist) in zip(names, blocks):
            exact_const, exact_factors = emit.factored(generated['m' + prefix][name], gens)
            assert const == float(exact_const) and [p for _, p in flist] == [p for _, p in exact_factors], name
            for (j, _), (terms, _) in zip(flist, exact_factors):
                expected = np.zeros(tuple(1 + max(e[d] for e in terms) for d in range(len(gens))))
                for e, c in terms.items():
                    expected[e] = c
                assert factors[j].shape == expected.shape and np.array_equal(factors[j], expected), name
    uv = [list(map(float, powers)) + [float(c)] for powers, c in ept_matrix.uv_rows(generated['z3_uv'])]
    assert np.array_equal(pt_matrix.uv_table(), np.array(uv))


def test_runtime_matrices_match_exact_rationals(generated):
    """``pt_matrix.evaluate`` (numpy, from the committed tables) against the exact
    rational functions evaluated with mpmath at a few complex points."""
    from elf import pt_matrix
    gen = generated
    nu1, nu2 = sp.symbols('nu1 nu2')
    mp.mp.dps = 30
    points = np.array([0.15 + 0.7j, 0.15 - 3.1j, 0.5 + 9.3j, -0.4 + 0.2j])
    a, b = np.meshgrid(points, points[::-1])
    worst = 0.0
    for prefix in ('22', '13'):
        names = pt_matrix.block_names(prefix)
        assert names == sorted(gen['m' + prefix])
        if prefix == '22':
            values = pt_matrix.evaluate('22', names, a, b)
            outer = pt_matrix.get_I(a, b)
        else:
            values = pt_matrix.evaluate('13', names, points)
            outer = np.tan(np.pi * points) / np.pi
        for name, value in zip(names, values):
            R = sp.lambdify((nu1, nu2) if prefix == '22' else (nu1,), gen['m' + prefix][name], 'mpmath')
            args = (a, b) if prefix == '22' else (points,)
            exact = np.vectorize(lambda *z: complex(R(*[mp.mpc(v.real, v.imag) for v in z])))(*args)
            worst = max(worst, np.max(np.abs(value / outer - exact) / np.abs(exact)))
    assert worst < 1e-13, worst
