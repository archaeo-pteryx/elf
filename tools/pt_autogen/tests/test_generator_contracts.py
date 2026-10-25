"""Contracts of the generator that hold for any declaration, not only for the committed density tables.

* Exact coefficients.  The common-subexpression ratio (``lpt_rows.proportional``) and the stage-B unit normalisation
  (``lpt_rows.units_of``) keep exact rationals exact.  There is no numeric identification (sympy's default nsimplify
  turns Rational(-1648, 18865) into a product of radicals 3e-19 off, which ``emit.rational`` rejects), and a ratio c is
  accepted only when a - c b vanishes exactly for every source.  No generator module calls nsimplify without
  rational=True.
* Write after verify.  ``python -m pt_autogen regenerate`` writes and verifies the tables in a staging directory and
  moves them into the output directory only when every verification passed.  A verification that fails or raises
  leaves a copy of the committed tables bitwise unchanged and keeps the staging directory with its products.  The
  expensive derivation is replaced by small stand-ins, the command line, writers and control flow are the real ones.
"""
import ast
import glob
import os
import shutil

import pytest
import sympy as sp

from pt_autogen import __main__ as generator
from pt_autogen import emit, lpt_rows

R = sp.Rational


# ---------------------------------------------------------------------------- exact coefficients
def test_proportional_is_exact():
    c = lpt_rows.proportional({'a': R(-1648, 18865)}, {'a': 1})
    assert c == R(-1648, 18865) and c.is_Rational
    assert emit.rational(c) == '-1648/18865'
    assert lpt_rows.proportional({'Q1': R(-1648, 18865), 'Q2': R(3296, 18865)},
                                 {'Q1': R(1), 'Q2': R(-2)}) == R(-1648, 18865)


def test_proportional_rejects_what_is_not_exactly_proportional():
    assert lpt_rows.proportional({'Q1': R(1, 3), 'Q2': R(2, 3)}, {'Q1': R(1), 'Q2': R(2) + R(1, 10**30)}) is None
    assert lpt_rows.proportional({'Q1': R(1)}, {'Q2': R(1)}) is None
    assert lpt_rows.proportional({'Q1': R(1)}, {'Q1': R(0)}) is None


def test_units_of_is_exact(monkeypatch):
    """A 22 source whose l = 2 block is -1648/18865 times its l = 0 block (the unit's first coefficient 1)."""
    named = {'S': (0, {('22', 0, -2): R(3), ('22', 2, -2): R(-4944, 18865)})}
    monkeypatch.setattr(lpt_rows.idn, 'named_sources', lambda: named)
    units, comps = lpt_rows.units_of(['S'])
    assert units == {'U22_0': ('22', -2, ((0, R(1)), (2, R(-1648, 18865))))}
    assert [emit.rational(c) for _, c in units['U22_0'][2]] == ['1', '-1648/18865']
    assert comps == {'S': [('U22_0', R(3), 4)]}


def test_no_nsimplify_without_rational():
    """Every nsimplify call of tools/pt_autogen (its tests included) passes rational=True."""
    offenders = []
    for path in sorted(glob.glob(os.path.join(emit.PACKAGE_DIR, '*.py')) +
                       glob.glob(os.path.join(emit.PACKAGE_DIR, 'tests', '*.py'))):
        with open(path) as fh:
            tree = ast.parse(fh.read(), path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, 'attr', getattr(node.func, 'id', None)) == 'nsimplify':
                if not any(kw.arg == 'rational' and isinstance(kw.value, ast.Constant) and kw.value.value is True
                           for kw in node.keywords):
                    offenders.append(f'{os.path.relpath(path, emit.PACKAGE_DIR)}:{node.lineno}')
    assert not offenders, offenders


# ---------------------------------------------------------------------------- write after verify
def snapshot(directory):
    """{relative path: bytes} of every file under ``directory``."""
    return {str(p.relative_to(directory)): p.read_bytes() for p in sorted(directory.rglob('*')) if p.is_file()}


@pytest.fixture
def stand_ins(monkeypatch):
    """The derivation replaced by stand-ins writing one EPT and one LPT table over committed file names."""
    gen = dict(zero_lag={}, z3_uv=0, z3_uv_bias=0)
    monkeypatch.setattr(generator, 'generate', lambda parts=('ept', 'lpt'): (gen, {}))
    monkeypatch.setattr(generator, 'tables', lambda g: {'ept/hankel_22.dat': (['# probe ept'], ['new ept'])})
    monkeypatch.setattr(generator, 'text_tables', lambda g: {'lpt/psi_terms.dat': (['# probe'], ['new'])})
    return monkeypatch


@pytest.mark.parametrize('mode', ['false', 'raises'])
def test_failed_verification_leaves_the_tables_untouched(tmp_path, stand_ins, capsys, mode):
    out = tmp_path / 'tables'
    shutil.copytree(emit.TABLE_DIR, out)
    committed, before = snapshot(type(tmp_path)(emit.TABLE_DIR)), snapshot(out)
    if mode == 'false':
        stand_ins.setattr(generator, 'compare_lpt', lambda gen, d, t=None: {'lpt': {'probe': False}})
        assert generator.main(['regenerate', '--out', str(out)]) == 1
    else:
        def compare(gen, d, t=None):
            raise AssertionError('controlled verification failure')
        stand_ins.setattr(generator, 'compare_lpt', compare)
        with pytest.raises(AssertionError, match='controlled verification failure'):
            generator.main(['regenerate', '--out', str(out)])
    assert snapshot(out) == before                                    # bitwise unchanged, no file added
    assert snapshot(type(tmp_path)(emit.TABLE_DIR)) == committed
    staged = list(tmp_path.glob('tables.staging-*'))                  # the products are kept, and named
    assert len(staged) == 1 and (staged[0] / 'lpt' / 'psi_terms.dat').read_text() == '# probe\nnew\n'
    assert str(staged[0]) in capsys.readouterr().out


def test_passed_verification_moves_the_set_in(tmp_path, stand_ins):
    out = tmp_path / 'tables'
    shutil.copytree(emit.TABLE_DIR, out)
    before = snapshot(out)
    stand_ins.setattr(generator, 'compare_lpt', lambda gen, d, t=None: {'lpt': {'probe': True}})
    assert generator.main(['regenerate', '--out', str(out)]) == 0
    after = snapshot(out)
    assert (out / 'lpt' / 'psi_terms.dat').read_text() == '# probe\nnew\n'
    assert (out / 'ept' / 'hankel_22.dat').read_text() == '# probe ept\nnew ept\n'
    assert '"probe": true' in after['manifest.json'].decode()
    assert set(after) == set(before) and not list(tmp_path.glob('tables.staging-*'))
