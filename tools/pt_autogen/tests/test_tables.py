"""The generated tables (spec section 6).

* regeneration reproduces the committed ``src/elf/tables/*.npz`` exactly (the UV rows of
  ``ept_matrix.npz`` included);
* the runtime (``elf.pt_coeff``, ``elf.pt_matrix``) decodes the committed tables: its matrices
  agree with the exact rational functions.
"""
import os
import sys

import mpmath as mp
import numpy as np
import pytest
import sympy as sp

from pt_autogen import emit
from pt_autogen.__main__ import generate, tables

sys.path.insert(0, os.path.join(emit.REPO_DIR, 'src'))


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    gen, _ = generate()
    out = tmp_path_factory.mktemp('tables')
    for fname, arrays in tables(gen).items():
        emit.write_npz(os.path.join(out, fname), arrays)
    return gen, str(out)


def committed(fname):
    return emit.read_npz(os.path.join(emit.TABLE_DIR, fname))


def test_regenerated_tables_equal_committed(generated):
    _, out = generated
    for fname in list(emit.HANKEL_FILES.values()) + [emit.MATRIX_FILE]:
        assert emit.arrays_equal(committed(fname), emit.read_npz(os.path.join(out, fname))), fname


def test_runtime_reads_hankel_tables():
    from elf import pt_coeff
    for prefix, fname in emit.HANKEL_FILES.items():
        names, lnm, table = pt_coeff.read_tables(prefix)
        ref = committed(fname)
        assert names == [str(s) for s in ref['names']]
        assert np.array_equal(lnm, ref['lnm']) and np.array_equal(table, ref['table'])


def test_runtime_matrices_match_exact_rationals(generated):
    """``pt_matrix.evaluate`` (numpy, from the committed float tables) against the exact
    rational functions evaluated with mpmath at a few complex points."""
    from elf import pt_matrix
    gen, _ = generated
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
