"""The LPT text tables (``src/elf/tables/lpt``) against the generator, and the runtime loader of the component table.

* Regeneration reproduces the committed tables byte for byte (headers included, as they carry no date), the headers
  are at most 120 characters wide and the coefficients exact.
* The runtime loader (the tables are byte-checked by ``check``, the loader does not re-verify their structure) places
  the rows of basis_rows.dat by their ids (a copy with its lines permuted loads the same values), multiplies all the
  factors of a basis row (a copy with a third factor works), and rejects a copy of component_entries.dat with a
  repeated key (a coefficient split over two lines, the same sum).
"""
import importlib
import os
import shutil
import sys
import types
from fractions import Fraction

import numpy as np
import pytest

from pt_autogen import emit
from pt_autogen.__main__ import text_tables

RUNTIME_TABLE_MODULE = 'lpt_tables.py'
"""The runtime module of src/elf that reads the component table (numpy only), loaded next to tables under test."""
RUNTIME_TABLE_READER = 'table_text.py'
"""The shared table reader that RUNTIME_TABLE_MODULE imports (a relative import: the two are loaded as one package)."""


# ---------------------------------------------------------------------------- the committed tables
def test_committed_tables_equal_regenerated(lpt_derived, tmp_path):
    texts = text_tables(lpt_derived)
    assert sorted(texts) == sorted(emit.LPT_FILES)
    for fname, (header, rows) in texts.items():
        out = tmp_path / os.path.basename(fname)
        emit.write_text_table(out, header, rows)
        with open(os.path.join(emit.TABLE_DIR, fname), 'rb') as fh:
            assert fh.read() == out.read_bytes(), fname
        assert all(line.startswith('#') and len(line) <= 120 for line in header), fname
        assert header[-1].startswith('# columns:') and emit.GENERATED_BY in header, fname
    for record in emit.read_text_table(os.path.join(emit.TABLE_DIR, emit.LPT_DIR, 'component_entries.dat')):
        assert emit.decimal(Fraction(record[3])) == record[3], record
    for name, column in (('rows_recipes.dat', 6), ('rows_sources.dat', -1)):
        for record in emit.read_text_table(os.path.join(emit.TABLE_DIR, emit.LPT_DIR, name)):
            assert emit.rational(Fraction(record[column])) == record[column], (name, record)


# ---------------------------------------------------------------------------- the runtime loader
def load_table_module(directory, name):
    """``<directory>/<RUNTIME_TABLE_MODULE>`` imported as ``<name>.lpt_tables`` of a stand-in package over ``directory``
    (its relative import of table_text resolves to ``<directory>/<RUNTIME_TABLE_READER>``, and nothing of the package
    elf runs).  A module loaded earlier under ``name`` is replaced."""
    for key in [key for key in sys.modules if key == name or key.startswith(name + '.')]:
        del sys.modules[key]
    package = types.ModuleType(name, 'the runtime table modules loaded without the package elf')
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    return importlib.import_module(name + '.' + RUNTIME_TABLE_MODULE[:-3])


def table_module_from_copy(directory, name=None, edit=None, header_edit=None):
    """The runtime table module loaded from a copy in ``directory`` that reads a copy of the committed tables, the data
    lines of the table ``name`` replaced by ``edit(data lines)`` (its header kept, or replaced by
    ``header_edit(header lines)``)."""
    tables = directory / 'tables' / emit.LPT_DIR
    shutil.copytree(os.path.join(emit.TABLE_DIR, emit.LPT_DIR), tables)
    for module in (RUNTIME_TABLE_MODULE, RUNTIME_TABLE_READER):
        shutil.copy(os.path.join(emit.REPO_DIR, 'src', 'elf', module), directory)
    if name is not None:
        lines = (tables / name).read_text().splitlines(keepends=True)
        header = [line for line in lines if line.startswith('#')]
        header = header_edit(header) if header_edit else header
        (tables / name).write_text(''.join(header + edit([line for line in lines if not line.startswith('#')])))
    return load_table_module(directory, 'pt_autogen_test_lpt_tables_copy')


def loaded_values(module):
    """Every module-level value of a loaded table module but its directory (tables, plans, constants, arrays as lists)."""
    return {name: value.tolist() if isinstance(value, np.ndarray) else value for name, value in vars(module).items()
            if not name.startswith('__') and name != '_DIR' and not callable(value)
            and not isinstance(value, types.ModuleType)}


@pytest.mark.parametrize('permute', [lambda lines: lines[1::-1] + lines[2:], lambda lines: lines[::-1]],
                         ids=['first two swapped', 'reversed'])
def test_loader_places_basis_rows_by_id(tmp_path, permute):
    reference = table_module_from_copy(tmp_path / 'reference')
    permuted = table_module_from_copy(tmp_path / 'permuted', 'basis_rows.dat', permute)
    assert len(set(reference.CT_ROW_NAMES)) == reference.N_CT        # distinct rows, so a misplaced row would show
    assert loaded_values(permuted) == loaded_values(reference)


def test_loader_rejects_a_repeated_key(tmp_path):
    """The first data line of component_entries.dat split into two lines with half its coefficient each (the same key
    and the same sum)."""
    def split(lines):
        fields = lines[0].split()
        fields[3] = emit.decimal(Fraction(fields[3]) / 2)
        return [' '.join(fields) + '\n'] * 2 + lines[1:]
    with pytest.raises(ValueError, match=r'tables/lpt/component_entries\.dat line \d+ repeats the key .* of line \d+'):
        table_module_from_copy(tmp_path, 'component_entries.dat', split)


def test_loader_multiplies_every_factor_of_a_basis_row(tmp_path):
    """The first product of two rows (t = 25) given a third factor X_LIN: row_basis multiplies all three."""
    def third(lines):
        i = next(n for n, line in enumerate(lines) if line.split()[0] == '25')
        return lines[:i] + [lines[i].rstrip('\n') + ' X_LIN\n'] + lines[i + 1:]
    module = table_module_from_copy(tmp_path, 'basis_rows.dat', third)
    rows = np.arange(1.0, 1.0 + 40 * 3).reshape(40, 3)[:len(module.ROW_NAMES)]
    got = module.row_basis(rows)[25]
    first, second = module.CT_ROWS[25][:2]
    assert len(module.CT_ROWS[25]) == 3
    np.testing.assert_array_equal(got, rows[first] * rows[second] * rows[module.CT_ROWS[25][2]])
