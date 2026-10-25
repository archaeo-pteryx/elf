"""The LPT text tables (``src/elf/tables/lpt``) against the generator, and the derived moments against the hand code.

* Regeneration reproduces the committed tables byte for byte (headers included, as they carry no date), the headers
  are at most 120 characters wide and the coefficients exact.
* Every psi entry of psi_terms.dat (with scalar_layer.dat) equals the sympy psi exactly, for the 351 expressions of the
  eight sums of the runtime's psi_family, built from the rows and also evaluated by the runtime evaluator on sympy
  symbols.  The moments rebuilt from the committed tables equal the derived moments (exact, l <= 10).
* The transcription.  ``lpt_G`` and ``assemble`` below are literal symbolic transcriptions of
  utils_lpt.get_lpt_moments and _assemble_lpt_moments_from_derivatives (the hand derivation, moved here from
  pt_autogen.lpt_moments).  The moments derived from the generating function (pt_autogen.lpt_angular) equal them exactly for
  l <= 10, their eight derivative kinds and G00 too.  Without the representation rule LEGACY_PARITY_A the kinds
  differ in representation only (equal as functions), the flat engine equals the transcription as functions, the
  struct and flat entry algebras agree (N = 4), and the Euler basis and the angular coefficients are the runtime's.
* The runtime loader (the tables are byte-checked by ``check``, the loader does not re-verify their structure) places
  the rows of basis_rows.dat by their ids (a copy with its lines permuted loads the same values), multiplies all the
  factors of a basis row (a copy with a third factor works), and rejects a copy of psi_terms.dat, scalar_layer.dat or
  component_entries.dat with a repeated key (a coefficient split over two lines, the same sum).
"""
import importlib
import os
import re
import shutil
import types
from fractions import Fraction

import jax

jax.config.update('jax_enable_x64', True)

import numpy as np
import pytest
import sympy as sp

from pt_autogen import emit, lpt_angular, lpt_model, lpt_moments
from pt_autogen.__main__ import text_tables
from pt_autogen.lpt_angular import A, B, C, Aw, Cw, F, ir, rho, s, c, x
from pt_autogen.lpt_model import A_mu as Am, B_mu as Bm

MOMENTS = lpt_moments.MOMENTS
R = sp.Rational
LMAX = 10


def runtime_table_module():
    """The runtime table module (src/elf/lpt_tables.py with its reader table_text.py) loaded from src/elf without the
    package elf (numpy only, reads the committed tables)."""
    return lpt_moments.load_table_module(os.path.join(emit.REPO_DIR, 'src', 'elf'), 'pt_autogen_test_lpt_tables')


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
    for name, column in (('psi_terms.dat', -1), ('scalar_layer.dat', -1), ('component_entries.dat', 3)):
        for record in emit.read_text_table(os.path.join(emit.TABLE_DIR, emit.LPT_DIR, name)):
            assert emit.decimal(Fraction(record[column])) == record[column], (name, record)
    for name, column in (('rows_recipes.dat', 6), ('rows_sources.dat', -1)):
        for record in emit.read_text_table(os.path.join(emit.TABLE_DIR, emit.LPT_DIR, name)):
            assert emit.rational(Fraction(record[column])) == record[column], (name, record)


def test_psi_entries_equal_sympy(lpt_derived):
    tables = lpt_moments.read_tables(emit.TABLE_DIR)
    assert lpt_moments.verify_psi(lpt_derived['lpt_psi'], tables, runtime_table_module().psi_family) == 351


def test_tables_equal_derived_moments():
    tables = lpt_moments.read_tables(emit.TABLE_DIR)
    assert lpt_moments.verify_moments(tables) == 11 * len(MOMENTS)


# ---------------------------------------------------------------------------- the transcription
# The transcription of the hand-written moments (moved from pt_autogen.lpt_moments, 2026-10-17).

def lpt_G(S, T, A_, B_, C_, c_, s_, ir_):
    """The nine G arrays of ``get_lpt_moments`` at one l as (core, prefactor) pairs, transcribed literally.

    S = (S0, ..., S4) and T = (T0, ..., T3) at this l.  The G value is core * prefactor.
    """
    c2 = c_**2
    s2 = s_**2
    Bs2 = B_ * s2
    S0, S1, S2, S3, S4 = S
    T0, T1, T2, T3 = T
    inv_rho = ir_
    inv_rho2 = inv_rho * inv_rho
    inv_rho3 = inv_rho2 * inv_rho
    Afac = 2 * A_ * inv_rho2
    Cfac = 2 * c2 * s_ * inv_rho
    A2f = 2 * inv_rho2
    C2f = 2 * c2 * inv_rho2
    ACf = -4 * c_ * s_ * inv_rho2
    A3f = -4 * c_ * inv_rho3
    A2Cf = 4 * s_ * inv_rho3
    A4f = 4 * inv_rho2 * inv_rho2

    G = {}
    G['G00'] = (S0, sp.Integer(1))
    G['dGdA'] = ((-S1 + Bs2*S0), Afac)
    G['dGdC'] = ((T1 - Bs2*T0), Cfac)
    G['d2A'] = ((-S1
                 + 2*c2*(S1 + S2)
                 + Bs2*S0
                 - 4*c2*Bs2*(S0 + S1)
                 + 2*c2*(Bs2**2)*S0), A2f)
    G['d2C'] = ((-3*T1
                 + 2*c2*(T1 + T2)
                 + Bs2*(3*T0 - 4*c2*(T0 + T1))
                 + 2*c2*(Bs2**2)*T0), C2f)
    G['dAdC'] = ((-T1
                  + c2*(T1 + T2)
                  + Bs2*(T0 - 2*c2*(T0 + T1))
                  + c2*(Bs2**2)*T0), ACf)

    P1 = (-3)*(S1 + S2) + 2*c2*(S3 + 3*S2 + 2*S1)
    P2 = Bs2*(6*(S0 + S1) - 6*c2*(S2 + 3*S1 + 2*S0))
    P3 = 3*((-1 + 4*c2)*S0 + 2*c2*S1) * (Bs2**2)
    P4 = -2*c2*(Bs2**3)*S0
    G['d3A'] = ((P1 + P2 + P3 + P4), A3f)

    Q1 = T1
    Q2 = c2*(-5*(T1 + T2) + 2*c2*(T3 + 3*T2 + 2*T1))
    Q3 = (-Bs2*T0
          + 2*c2*Bs2*(5*(T0 + T1) - 6*c2*(T0 + T1) - 3*c2*(T1 + T2)))
    Q4 = c2*(Bs2**2)*(-5*T0 + 12*c2*T0 + 6*c2*T1)
    Q5 = -2*(c2**2)*(Bs2**3)*T0
    G['d3A2C'] = ((Q1 + Q2 + Q3 + Q4 + Q5), A2Cf)

    a0 = 3 - 24*c2 + 24*(c2**2)
    a1 = -12*c2 + 20*(c2**2)
    a2 = 4*(c2**2)
    term1 = a0*S1 + (a0 + a1)*S2 + (a1 + a2)*S3 + a2*S4

    b0 = 3 + (-36*c2 + 48*(c2**2))
    b1 = (-18*c2 + 40*(c2**2))
    b2 = (8*(c2**2))
    c0, c1, c2c, c3 = b0, (b0 + b1), (b1 + b2), b2
    term2 = -2*Bs2*(c0*S0 + c1*S1 + c2c*S2 + c3*S3)

    d0 = 1 - 24*c2 + 48*(c2**2)
    d1 = -12*c2 + 40*(c2**2)
    d2 = 8*(c2**2)
    term3 = 3*(Bs2**2)*(d0*S0 + d1*S1 + d2*S2)

    e0 = 12*c2 - 48*(c2**2)
    e1 = -16*(c2**2)
    term4 = (Bs2**3)*(e0*S0 + e1*S1)
    term5 = 4*(c2**2)*(Bs2**4)*S0
    G['d4A'] = ((term1 + term2 + term3 + term4 + term5), A4f)
    return G


def assemble(Gl, A_, C_, Am_, Bm_, L):
    """``_assemble_lpt_moments_from_derivatives`` on lists over l (Gl[l][name] the G values), returning M[j][l]."""
    zero = sp.Integer(0)

    def sh(name, l, k):
        return Gl[l - k][name] if l - k >= 0 else zero

    M = [[None] * L for _ in MOMENTS]
    for l in range(L):
        dGdAs, dGdCs = Gl[l]['dGdA'], Gl[l]['dGdC']
        d2GdA2s, d2GdC2s, d2GdAdCs = Gl[l]['d2A'], Gl[l]['d2C'], Gl[l]['dAdC']
        d3GdA3s, d3GdA2dCs, d4GdA4s = Gl[l]['d3A'], Gl[l]['d3A2C'], Gl[l]['d4A']
        G00s = Gl[l]['G00']
        G00s_lm1, G00s_lm2, G00s_lm3, G00s_lm4 = (sh('G00', l, k) for k in (1, 2, 3, 4))
        dGdAs_lm1, dGdAs_lm2, dGdAs_lm3 = (sh('dGdA', l, k) for k in (1, 2, 3))
        dGdCs_lm1, dGdCs_lm2 = (sh('dGdC', l, k) for k in (1, 2))
        d2GdAdCs_lm1 = sh('dAdC', l, 1)
        d2GdA2s_lm1, d2GdA2s_lm2 = (sh('d2A', l, k) for k in (1, 2))
        d3GdA3s_lm1 = sh('d3A', l, 1)
        Acol, Ccol = A_, C_

        G10 = dGdAs + R(1, 2) * Acol * G00s_lm1
        G01 = dGdCs + R(1, 2) * Ccol * G00s_lm1
        G20 = d2GdA2s + Acol * dGdAs_lm1 + R(1, 2) * G00s_lm1 + R(1, 4) * (Acol**2) * G00s_lm2
        G02 = d2GdC2s + Ccol * dGdCs_lm1 + R(1, 2) * G00s_lm1 + R(1, 4) * (Ccol**2) * G00s_lm2
        G11 = (d2GdAdCs + R(1, 2) * Ccol * dGdAs_lm1 + R(1, 2) * Acol * dGdCs_lm1
               + R(1, 4) * (Acol * Ccol) * G00s_lm2)
        G30 = (d3GdA3s + R(3, 2) * Acol * d2GdA2s_lm1 + R(3, 2) * dGdAs_lm1
               + R(3, 4) * (Acol**2) * dGdAs_lm2 + R(3, 4) * Acol * G00s_lm2
               + (Acol**3) / 8 * G00s_lm3)
        G21 = (d3GdA2dCs + R(1, 2) * Ccol * d2GdA2s_lm1 + Acol * d2GdAdCs_lm1 + R(1, 2) * dGdCs_lm1
               + R(1, 2) * Acol * Ccol * dGdAs_lm2 + R(1, 4) * (Acol**2) * dGdCs_lm2
               + R(1, 4) * Ccol * G00s_lm2 + (Acol**2) * Ccol / 8 * G00s_lm3)
        G40 = (d4GdA4s + 2 * Acol * d3GdA3s_lm1 + 3 * d2GdA2s_lm1
               + R(3, 2) * (Acol**2) * d2GdA2s_lm2 + 3 * Acol * dGdAs_lm2 + R(3, 4) * G00s_lm2
               + R(1, 2) * (Acol**3) * dGdAs_lm3 + R(3, 4) * (Acol**2) * G00s_lm3
               + (Acol**4) / 16 * G00s_lm4)

        mq0 = G00s
        mq1 = -G10
        mq2 = -G20
        mq3 = G30
        mq4 = G40
        nq1 = -Am_ * G10 + Bm_ * G01
        nq2 = -Am_**2 * G20 + 2 * Am_ * Bm_ * G11 - Bm_**2 * G02
        mq1_nq1 = -Am_ * G20 + Bm_ * G11
        mq2_nq1 = Am_ * G30 - Bm_ * G21
        for j, v in enumerate((mq0, mq1, mq2, mq3, mq4, nq1, nq2, mq1_nq1, mq2_nq1)):
            M[j][l] = v
    return M


def literal_moments(X, L):
    """M[j][l] of the literal transcription on the basis symbols X (S_d = s^2 T_d, T_0 = S_0 / s^2)."""
    Gl = []
    for l in range(L):
        S_l = [X['S0'][l]] + [s**2 * X[f'T{d}'][l] for d in range(1, 5)]
        T_l = [X['S0'][l] / s**2] + [X[f'T{d}'][l] for d in range(1, 4)]
        G = lpt_G(S_l, T_l, A, B, C, c, s, ir)
        Gl.append({k: core * pref for k, (core, pref) in G.items()})
    return assemble(Gl, A, C, Am, Bm, L)


def transcription_kinds(M):
    """{kind: [G(l)]} read off assemble's moments (G10 = -MQ1, G20 = -MQ2, G30 = MQ3, G40 = MQ4, G01 = dNQ1/dB_mu,
    G11 = (1/2) d^2NQ2/dA_mu dB_mu, G02 = -(1/2) d^2NQ2/dB_mu^2, G21 = -dMQ2NQ1/dB_mu), the remaining A_mu, B_mu
    structure of the four line-of-sight moments asserted."""
    j = {nm: n for n, nm in enumerate(MOMENTS)}
    L = len(M[0])
    G = {'G00': [M[j['MQ0']][l] for l in range(L)],
         'G10': [-M[j['MQ1']][l] for l in range(L)], 'G20': [-M[j['MQ2']][l] for l in range(L)],
         'G30': [M[j['MQ3']][l] for l in range(L)], 'G40': [M[j['MQ4']][l] for l in range(L)],
         'G01': [sp.diff(M[j['NQ1']][l], Bm) for l in range(L)],
         'G11': [sp.diff(M[j['NQ2']][l], Am, Bm) / 2 for l in range(L)],
         'G02': [-sp.diff(M[j['NQ2']][l], Bm, 2) / 2 for l in range(L)],
         'G21': [-sp.diff(M[j['MQ2NQ1']][l], Bm) for l in range(L)]}
    for l in range(L):
        g = {k: v[l] for k, v in G.items()}
        assert sp.expand(M[j['NQ1']][l] - (-Am * g['G10'] + Bm * g['G01'])) == 0
        assert sp.expand(M[j['NQ2']][l] - (-Am**2 * g['G20'] + 2 * Am * Bm * g['G11'] - Bm**2 * g['G02'])) == 0
        assert sp.expand(M[j['MQ1NQ1']][l] - (-Am * g['G20'] + Bm * g['G11'])) == 0
        assert sp.expand(M[j['MQ2NQ1']][l] - (Am * g['G30'] - Bm * g['G21'])) == 0
    return G


KINDS = {'G00': (0, 0), 'G10': (1, 0), 'G01': (0, 1), 'G20': (2, 0), 'G02': (0, 2), 'G11': (1, 1),
         'G30': (3, 0), 'G21': (2, 1), 'G40': (4, 0)}


def polar_normal(e):
    """The unique normal form of a function of (A, C) written in A, C, ir, c, s, A = c rho, C = s rho, ir = 1/rho and
    then every c^k -> c^(k mod 2) (1 - s^2)^(k // 2)."""
    e = sp.expand(e.xreplace({A: c * rho, C: s * rho, ir: 1 / rho}))
    out = 0
    for t in sp.Add.make_args(e):
        k = t.as_powers_dict().get(c, 0)
        out += t / c**k * c**(k % 2) * (1 - s**2)**(k // 2)
    return sp.expand(out)


def flat_on_basis(eng, entries, l, X):
    """The flat entries (generic F at shift k) on the basis symbols of l - k, in the Euler basis with x = s^2."""
    tot = 0
    for k, e in entries.items():
        if l - k < 0:
            continue
        ee, E = eng.euler(e)
        rep = {E[0]: X['S0'][l - k]}
        rep.update({E[d]: x * X[f'T{d}'][l - k] for d in range(1, len(E))})
        tot += ee.xreplace(rep)
    return sp.expand(sp.expand(tot).xreplace({x: s**2}))


@pytest.fixture(scope='module')
def transcription():
    X = lpt_angular.basis_symbols(LMAX + 1)
    M = literal_moments(X, LMAX + 1)
    return X, M, transcription_kinds(M)


@pytest.fixture(scope='module')
def engine():
    return lpt_angular.Engine(lpt_angular.max_degree(lpt_model.DENSITY_MOMENTS))


def test_derived_moments_equal_transcription(transcription, engine):
    """The nine moments and the eight derivative kinds with G00, l <= 10, exactly (99 + 99 comparisons)."""
    X, M, Gt = transcription
    derived = lpt_angular.moments(X, LMAX + 1, eng=engine)
    for j in range(len(MOMENTS)):
        for l in range(LMAX + 1):
            assert sp.expand(derived[j][l] - M[j][l]) == 0, (MOMENTS[j], l)
    for name, (a, b) in KINDS.items():
        ker = engine.shift_kernels(a, b, 0)
        for l in range(LMAX + 1):
            assert sp.expand(engine.instantiate(ker, l, X) - Gt[name][l]) == 0, (name, l)


def test_representation_rule_changes_representation_only(transcription):
    """Without LEGACY_PARITY_A the kinds with dG/dA differ in representation but are equal as functions."""
    X, _, Gt = transcription
    eng = lpt_angular.Engine(4)
    differ = []
    for name, (a, b) in KINDS.items():
        ker = eng.shift_kernels(a, b, 0, legacy=False)
        values = [eng.instantiate(ker, l, X) for l in range(LMAX + 1)]
        if any(sp.expand(v - Gt[name][l]) != 0 for l, v in enumerate(values)):
            differ.append(name)
            assert all(polar_normal(v - Gt[name][l]) == 0 for l, v in enumerate(values)), name
    assert differ == ['G10', 'G20', 'G11', 'G30', 'G21', 'G40'], differ


def test_flat_engine_equals_transcription_as_functions(transcription, engine):
    X, M, Gt = transcription
    for name, (a, b) in KINDS.items():
        entries = engine.G(a, b, 0, 'flat')
        for l in range(LMAX + 1):
            assert polar_normal(flat_on_basis(engine, entries, l, X) - Gt[name][l]) == 0, (name, l)
    for j, name in enumerate(MOMENTS):
        entries = lpt_angular.moment_flat(engine, lpt_model.DENSITY_MOMENTS[name])
        for l in range(LMAX + 1):
            assert polar_normal(flat_on_basis(engine, entries, l, X) - M[j][l]) == 0, (name, l)


def test_struct_equals_flat_and_derivatives_commute():
    N = 4
    eng = lpt_angular.Engine(N)
    kinds = [(a, b, n) for tot in range(N + 1) for n in range(tot // 2 + 1) for a in range(tot - 2 * n + 1)
             for b in [tot - 2 * n - a]]
    for a, b, n in kinds:
        flat, struct = eng.G(a, b, n, 'flat'), eng.G(a, b, n, 'struct')
        assert set(flat) == set(struct), (a, b, n)
        for k in flat:
            sub = {r: eng.raw(eng._Rkey[r]) for r in struct[k].free_symbols if r in eng._Rkey}
            sub.update({Aw: A, Cw: C})
            assert sp.expand(struct[k].xreplace(sub) - flat[k]) == 0, (a, b, n, k)
    for a, b in ((1, 1), (2, 1), (1, 2), (2, 2)):
        g1, g2 = {0: F(x)}, {0: F(x)}
        for v in 'C' * b + 'A' * a:
            g1 = eng.D(g1, v, 'flat')
        for v in 'A' * a + 'C' * b:
            g2 = eng.D(g2, v, 'flat')
        assert set(g1) == set(g2) and all(sp.expand(g1[k] - g2[k]) == 0 for k in g1), (a, b)


def test_euler_basis_and_angular_coefficients():
    lpt_angular.check_euler_basis(6, 4)
    _, utils_lpt = lpt_moments.import_utils_lpt()
    lc = importlib.import_module(utils_lpt.__name__.rsplit('.', 1)[0] + '.lpt_coefficients')
    for l in range(LMAX + 1):
        exact, ref = lpt_angular.g00_fraction(l), lc.generate_g00_coefficient(l)
        assert all(float(exact[k][i]) == ref[k, i] for k in range(l + 1) for i in range(l + 1)), l


# ---------------------------------------------------------------------------- the runtime loader
def table_module_from_copy(directory, name=None, edit=None, header_edit=None):
    """The runtime table module loaded from a copy in ``directory`` that reads a copy of the committed tables, the data
    lines of the table ``name`` replaced by ``edit(data lines)`` (its header kept, or replaced by
    ``header_edit(header lines)``)."""
    tables = directory / 'tables' / emit.LPT_DIR
    shutil.copytree(os.path.join(emit.TABLE_DIR, emit.LPT_DIR), tables)
    for module in (lpt_moments.RUNTIME_TABLE_MODULE, lpt_moments.RUNTIME_TABLE_READER):
        shutil.copy(os.path.join(emit.REPO_DIR, 'src', 'elf', module), directory)
    if name is not None:
        lines = (tables / name).read_text().splitlines(keepends=True)
        header = [line for line in lines if line.startswith('#')]
        header = header_edit(header) if header_edit else header
        (tables / name).write_text(''.join(header + edit([line for line in lines if not line.startswith('#')])))
    return lpt_moments.load_table_module(directory, 'pt_autogen_test_lpt_tables_copy')


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


@pytest.mark.parametrize('name, column', [('psi_terms.dat', -1), ('scalar_layer.dat', -1),
                                          ('component_entries.dat', 3)])
def test_loader_rejects_a_repeated_key(tmp_path, name, column):
    """The first data line split into two lines with half its coefficient each (the same key and the same sum)."""
    def split(lines):
        fields = lines[0].split()
        fields[column] = emit.decimal(Fraction(fields[column]) / 2)
        return [' '.join(fields) + '\n'] * 2 + lines[1:]
    with pytest.raises(ValueError, match=rf'tables/lpt/{re.escape(name)} line \d+ repeats the key .* of line \d+'):
        table_module_from_copy(tmp_path, name, split)


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
