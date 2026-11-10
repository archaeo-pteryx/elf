"""The Legendre table angular.dat (src/elf/tables/lpt) against its derivation and the checks of pt_autogen.lpt_legendre.

* The committed table equals the derivation exactly for L <= 12 (recomputed here), and the whole file equals a
  regeneration into a temporary directory byte for byte.
* The band of the committed table, the addition theorem for the declared (a, b), the direct angular quadrature of the
  committed C_L at two points and the plane-wave series at one point.
"""
import os

import pytest

from pt_autogen import emit, lpt_model
from pt_autogen import lpt_legendre as lg

DECLARATION = lpt_model.DENSITY_MOMENTS
L_EXACT = 12


@pytest.fixture(scope='module')
def committed():
    """The committed angular.dat as {(a, b, r): {(L, i): Fraction}}."""
    return lg.read_table(emit.TABLE_DIR)


def test_committed_table_equals_derivation(committed):
    derived = lg.table(DECLARATION, L_EXACT)
    assert list(committed) == list(derived) == lg.types_of(DECLARATION)
    for t, entries in derived.items():
        assert {key: v for key, v in committed[t].items() if key[0] <= L_EXACT} == entries, t


def test_committed_file_equals_regeneration(tmp_path):
    texts = emit.lpt_angular_text(lg.table(DECLARATION), lg.types_of(DECLARATION), lg.L_TABLE)
    (fname, (header, rows)), = texts.items()
    assert fname == lg.FILE
    out = tmp_path / os.path.basename(fname)
    emit.write_text_table(out, header, rows)
    with open(os.path.join(emit.TABLE_DIR, fname), 'rb') as fh:
        assert fh.read() == out.read_bytes()
    assert all(line.startswith('#') and len(line) <= 120 for line in header)
    assert header[-1] == '# columns: a b r L i coef' and emit.GENERATED_BY in header


def test_band(committed):
    assert lg.check_band(committed) == sum(len(entries) for entries in committed.values())


@pytest.mark.parametrize('a, b', sorted({(a, b) for a, b, _ in lg.types_of(DECLARATION)}))
def test_addition_theorem(a, b):
    assert lg.check_addition_theorem(a, b) == lg.QUADRATURE_LMAX + 1


def test_against_quadrature(committed):
    assert lg.check_against_quadrature(DECLARATION, lg.QUADRATURE_POINTS[:2], committed) < 1e-12


def test_series(committed):
    assert lg.check_series(DECLARATION, lg.SERIES_POINTS[-1:], committed) < 1e-10
