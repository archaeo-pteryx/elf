"""Command line of the table generator, run from the repository root:

    PYTHONPATH=tools python -m pt_autogen regenerate [--out DIR]   (default DIR: src/elf/tables)
    PYTHONPATH=tools python -m pt_autogen check

``regenerate`` writes the npz tables, the LPT text tables (``lpt/*.dat``) and ``manifest.json`` (the generator
version, the file list, the LPT verification results and the derived facts ``info``, with no date, source hash
or timing).
``check`` regenerates into a temporary directory and compares with the committed tables (exact
array equality for the npz files, byte equality for the text tables).  Both run the LPT
verifications of ``compare_lpt`` (the tables against the derivation and the runtime, and the
derivation against the references typed once in tools/pt_autogen/tests, ``pt_autogen.lpt_verify``) and print the
generation times to stdout.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np

from . import (emit, ept_hankel, ept_matrix, lpt_components, lpt_cumulant, lpt_model, lpt_moments,
               lpt_rows, lpt_verify)


def generate(parts=('ept', 'lpt')):
    """``(outputs, timings)`` of the parts 'ept' (the exact blocks, rational functions and UV term) and 'lpt' (the
    cumulant expansion ``lpt_cumulants``, its components ``lpt_components``, the row recipes ``lpt_rows`` and the psi
    family of the derived moments ``lpt_psi``)."""
    out, timings = {}, {}

    def components():
        return lpt_components.components(lpt_components.pieces(lpt_cumulant.kept_terms(out['lpt_cumulants'])))
    steps = {'ept': (('hankel_22', lambda: {'b22': ept_hankel.p22_blocks()}),
                     ('hankel_13', lambda: dict(zip(('b13', 'zero_lag'), ept_hankel.p13_blocks()))),
                     ('matrix_22', lambda: {'m22': ept_matrix.p22_matrix()}),
                     ('matrix_13', lambda: {'m13': ept_matrix.p13_matrix()}),
                     ('uv', lambda: dict(zip(('z3_uv', 'z3_uv_bias'), ept_matrix.uv_limit())))),
             'lpt': (('lpt_cumulants', lambda: {'lpt_cumulants': lpt_cumulant.derive()}),
                     ('lpt_components', lambda: {'lpt_components': components()}),
                     ('lpt_rows', lambda: {'lpt_rows': lpt_rows.derive()}),
                     ('lpt_psi', lambda: {'lpt_psi': lpt_moments.derive()}))}
    steps = tuple(step for part in parts for step in steps[part])
    for name, step in steps:
        t0 = time.perf_counter()
        out.update(step())
        timings[name] = round(time.perf_counter() - t0, 1)
    return out, timings


def tables(gen):
    """``{file name: {array name: array}}`` of the generated tables."""
    out = {}
    for prefix in ('22', '13'):
        names, lnm, table = emit.hankel_table(gen['b' + prefix], prefix)
        out[emit.HANKEL_FILES[prefix]] = dict(names=np.array(names), lnm=lnm, table=table)
    out[emit.MATRIX_FILE] = emit.matrix_arrays(gen['m22'], gen['m13'],
                                               ept_matrix.uv_rows(gen['z3_uv']))
    return out


def text_tables(gen):
    """``{file name relative to the table directory: (header lines, rows)}`` of the LPT text tables."""
    comps = gen['lpt_components']
    out = {}
    out.update(emit.lpt_rows_text(lpt_rows.tables(gen['lpt_rows'], comps)))
    out.update(emit.lpt_components_text(lpt_components.templates(comps), lpt_model.LINEAR_KINDS, comps))
    out.update(emit.lpt_basis_text(lpt_components.component_basis(comps)))
    out.update(emit.lpt_moments_text(lpt_model.DENSITY_MOMENTS, lpt_model.A_mu, lpt_model.B_mu))
    out.update(emit.lpt_psi_text(lpt_moments.make_tables(gen['lpt_psi'])))
    return out


def compare_lpt(gen, out_dir, timings=None):
    """The LPT verifications.  The psi family of the tables written to ``out_dir`` (and its dA, dB, dC) against the
    sympy derivation, the moments rebuilt from them against the derived moments (exact, l <= LMAX_CHECK), the moment
    identity of the runtime code (JAX float64), and the groups of ``pt_autogen.lpt_verify`` (the derivation against the
    references of tools/pt_autogen/tests)."""
    tables = lpt_moments.read_tables(out_dir)
    res = {}
    checks = (('psi_entries_equal_sympy', lambda: lpt_moments.verify_psi(gen['lpt_psi'], tables)),
              ('tables_equal_derived_moments', lambda: lpt_moments.verify_moments(tables)))
    for key, fn in checks:
        try:
            res[key], res[key + '_count'] = True, fn()
        except AssertionError as err:
            res[key], res[key + '_error'] = False, str(err)
    worst = lpt_moments.moment_identity(out_dir)
    res['moment_identity_max_rel'] = {str(lmax): float('%.2e' % w) for lmax, w in worst.items()}
    res['moment_identity_below_1e-12'] = all(w < 1e-12 for w in worst.values())
    groups = (('kernels', lpt_verify.kernels),
              ('term_list', lambda: lpt_verify.term_list(gen['lpt_cumulants'])),
              ('components', lambda: lpt_verify.components(gen['lpt_components'])),
              ('rows', lambda: lpt_verify.rows(gen['lpt_rows'], gen['lpt_components'])),
              ('moments_declaration', lpt_verify.moments_declaration))
    for name, fn in groups:
        t0 = time.perf_counter()
        res[name] = fn()
        if timings is not None:
            timings['lpt_verify_' + name] = round(time.perf_counter() - t0, 1)
    return {'lpt': res}


def regenerate(out_dir):
    """Generate the tables into ``out_dir``; returns ``(arrays, texts, comparisons, timings)``.  The timings are for
    printing only, ``manifest.json`` does not record them."""
    gen, timings = generate()
    arrays = tables(gen)
    texts = text_tables(gen)
    os.makedirs(os.path.join(out_dir, emit.LPT_DIR), exist_ok=True)
    for fname, arr in arrays.items():
        emit.write_npz(os.path.join(out_dir, fname), arr)
    for fname, (header, rows) in texts.items():
        emit.write_text_table(os.path.join(out_dir, fname), header, rows)
    t0 = time.perf_counter()
    comparisons = compare_lpt(gen, out_dir, timings)
    timings['lpt_verification'] = round(time.perf_counter() - t0, 1)
    info = dict(p13_zero_lag_dropped={str(key): str(v) for key, v in gen['zero_lag'].items()},
                z3_uv_matter=str(gen['z3_uv']),
                uv_limit_bias_group_unused=str(gen['z3_uv_bias']))
    emit.write_manifest(out_dir, comparisons, info)
    return arrays, texts, comparisons, timings


def _all_true(value):
    if isinstance(value, dict):
        return all(_all_true(v) for v in value.values())
    if isinstance(value, bool):
        return value
    return True                          # counts


def check():
    """Regenerate into a temporary directory and compare with the committed tables; returns the exit status.
    The committed ``manifest.json`` has to exist, it records nothing to compare (no hash)."""
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        arrays, texts, comparisons, timings = regenerate(tmp)
        print('generation times [s]:', json.dumps(timings))
        for fname in list(arrays) + list(texts):
            committed = os.path.join(emit.TABLE_DIR, fname)
            if not os.path.exists(committed):
                print(f'FAIL {fname}: no committed table')
                ok = False
                continue
            if fname in arrays:
                same = emit.arrays_equal(emit.read_npz(committed), emit.read_npz(os.path.join(tmp, fname)))
            else:
                with open(committed, 'rb') as a, open(os.path.join(tmp, fname), 'rb') as b:
                    same = a.read() == b.read()
            print(('ok  ' if same else 'FAIL') + f' {fname}: regenerated == committed')
            ok &= same
    for key, value in comparisons.items():
        good = _all_true(value)
        print(('ok  ' if good else 'FAIL') + f' {key}: {json.dumps(value)}')
        ok &= good
    if not os.path.exists(os.path.join(emit.TABLE_DIR, emit.MANIFEST_FILE)):
        print('FAIL manifest.json missing')
        ok = False
    print('check ' + ('passed' if ok else 'FAILED'))
    return 0 if ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pt_autogen', description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    reg = sub.add_parser('regenerate', help='write the tables')
    reg.add_argument('--out', default=emit.TABLE_DIR, help='output directory')
    sub.add_parser('check', help='regenerate into a temporary directory and compare')
    args = parser.parse_args(argv)
    if args.command == 'regenerate':
        t0 = time.perf_counter()
        _, _, comparisons, timings = regenerate(args.out)
        print('wrote', args.out)
        print('generation times [s]:', json.dumps(timings), ' total %.1f' % (time.perf_counter() - t0))
        print('comparisons:', json.dumps(comparisons))
        return 0 if _all_true(comparisons) else 1
    return check()


if __name__ == '__main__':
    sys.exit(main())
