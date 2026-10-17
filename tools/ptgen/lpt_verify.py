"""The LPT generator's own verifications against the references typed once in tools/ptgen/tests.

Each function runs a group of checks and returns ``{check: result}``, a result being True, a count, or False with the
check's ``<name>_error`` message next to it.  ``python -m ptgen regenerate`` records them in manifest.json, ``check``
reports them, and the tests (tools/ptgen/tests/test_lpt_*.py) assert them.

* ``kernels``: L^(2) and the longitudinal L^(3) against their closed forms, the recursion against Matsubara (2015)
  Eqs. B8-B10, the LPT -> Euler map against ptgen's F_2, F_3, and the transverse part of L^(3) dropping out.
* ``term_list``: the cumulant expansion against the paper's integrand, the renormalised and dropped bias, the
  self-pair and vanishing cumulants, the 'explicit' renormalisation mode and the kernel facts behind the declarations.
* ``components``: the derived components against the hand code (every moment coefficient symbolically, the templates,
  the linear kinds, the template and linear matrices and the bias degrees), the moments and rows they use.
* ``rows``: the derived row sources against the appendix, the identities the recipes rest on (full-k integrals, j_l,
  the finite-range kernels, the ladder), the product forms and the preferred route against the hand code, the stage-A
  list and the rules of the k-space rows.
* ``moments_declaration``: the declared angular polynomials against MQ^a NQ^b with the declared line of sight.
"""
import random

import numpy as np
import sympy as sp

from . import lpt_components as co
from . import lpt_cumulant as cu
from . import lpt_identities as idn
from . import lpt_kernels as kn
from . import lpt_model as model
from . import lpt_reduce as rr
from . import lpt_rows as rc
from . import lpt_tensors as te
from .kernels import F as F_euler
from .vectors import K_VEC, P_VEC, R_VEC, dot, norm2
from .vectors import k as kv, p as pv, r as rv, x as xv, y as yv, z as zv

R = sp.Rational


def _run(checks):
    """{name: result} of the checks ``{name: function}``.  A failed assertion gives False and '<name>_error'."""
    out = {}
    for name, fn in checks.items():
        try:
            value = fn()
            out[name] = True if value is None else value
        except AssertionError as err:
            out[name], out[name + '_error'] = False, str(err)[:300]
    return out


def all_passed(results):
    """Whether every result of a group is truthy (counts and True)."""
    return all(v is not False for key, v in results.items() if not key.endswith('_error'))


def _points(n, seed):
    """n random exact rational points of the magnitudes k, p, r and the cosines x, y, z."""
    rng = random.Random(seed)
    for _ in range(n):
        yield {kv: R(rng.randint(1, 60), rng.randint(1, 20)), pv: R(rng.randint(1, 60), rng.randint(1, 20)),
               rv: R(rng.randint(1, 60), rng.randint(1, 20)), xv: R(rng.randint(-19, 19), 20),
               yv: R(rng.randint(-19, 19), 20), zv: R(rng.randint(-19, 19), 20)}


# ---------------------------------------------------------------------------------------------------------------
# kernels

def kernels():
    from .tests import lpt_references as ref

    def L2_closed_form():
        vec, coef = ref.L2_closed(P_VEC, R_VEC)
        assert (kn.L(P_VEC, R_VEC) - kn.VE.of(vec, coef)).simplify().is_zero()

    def L3_longitudinal():
        S3 = kn.L(K_VEC, P_VEC, R_VEC).dot(K_VEC + P_VEC + R_VEC)
        closed = (ref.L3a(K_VEC, P_VEC, R_VEC) + ref.L3a(P_VEC, R_VEC, K_VEC) + ref.L3a(R_VEC, K_VEC, P_VEC)) / 3
        for pt in _points(12, 1):
            assert (S3 - closed).subs(pt) == 0, pt
        return 12

    def recursion_appendix_B():
        S2 = kn.L_unsym((P_VEC, R_VEC))[1]
        assert sp.cancel(S2 - R(3, 7) * ref.U_angle(P_VEC, R_VEC)) == 0
        L3, S3 = kn.L_unsym((K_VEC, P_VEC, R_VEC))
        k3 = K_VEC + P_VEC + R_VEC
        # the transverse part, k x T3 / k^2 with T3 = (k1 x k23)(k1.k23)/(k1^2 k23^2) S2(k2, k3)
        kxT = (kn.VE.of(K_VEC, dot(k3, P_VEC + R_VEC)) - kn.VE.of(P_VEC + R_VEC, dot(k3, K_VEC))).scale(
            dot(K_VEC, P_VEC + R_VEC) / (norm2(K_VEC) * norm2(P_VEC + R_VEC)) * R(3, 7) * ref.U_angle(P_VEC, R_VEC)
            / norm2(k3))
        trans = L3 - kn.VE.of(k3, S3 / norm2(k3))
        for pt in _points(6, 2):
            assert (S3 - ref.S3_appendix_B(K_VEC, P_VEC, R_VEC)).subs(pt) == 0, pt
            for v in set(trans.terms) | set(kxT.terms):
                assert (trans.terms.get(v, 0) - kxT.terms.get(v, 0)).subs(pt) == 0, (pt, v)
        return 6

    def euler_F2_F3():
        # delta(k) = int d^3q e^{-ik.q}(e^{-ik.Psi} - 1) with the LPT kernels gives ptgen's EdS F_2 and F_3
        k2 = P_VEC + R_VEC
        F2 = R(1, 2) * (kn.L(P_VEC, R_VEC).dot(k2) + kn.L(P_VEC).dot(k2) * kn.L(R_VEC).dot(k2))
        assert sp.cancel(sp.together(F2 - F_euler(P_VEC, R_VEC))) == 0
        k3 = K_VEC + P_VEC + R_VEC
        l1 = {v: kn.L(v).dot(k3) for v in (K_VEC, P_VEC, R_VEC)}
        F3 = (R(1, 6) * kn.L(K_VEC, P_VEC, R_VEC).dot(k3) + R(1, 6) * l1[K_VEC] * l1[P_VEC] * l1[R_VEC]
              + R(1, 6) * (l1[K_VEC] * kn.L(P_VEC, R_VEC).dot(k3) + l1[P_VEC] * kn.L(K_VEC, R_VEC).dot(k3)
                           + l1[R_VEC] * kn.L(K_VEC, P_VEC).dot(k3)))
        F3_ref = F_euler(K_VEC, P_VEC, R_VEC)
        for pt in _points(8, 3):
            assert (F3 - F3_ref).subs(pt) == 0, pt
        return 8

    def transverse_drops_out():
        assert not (kn.L(K_VEC, P_VEC, -P_VEC) - kn.L(K_VEC, P_VEC, -P_VEC, transverse=False)).simplify().is_zero()
        entries = {e.name: e for e in model.ROW_DICTIONARY}
        for name in ('A13', 'U3'):
            a = rr.derive_entry(entries[name])[0]
            b = rr.derive_entry(entries[name], transverse=False)[0]
            assert all(rr.recipes_equal(a[row], b[row]) for row in a), name

    return _run(dict(L2_closed_form=L2_closed_form, L3_longitudinal_points=L3_longitudinal,
                     recursion_appendix_B_points=recursion_appendix_B, euler_F2_F3_points=euler_F2_F3,
                     transverse_part_drops_out=transverse_drops_out))


# ---------------------------------------------------------------------------------------------------------------
# the term list

def paper_symbol(part, fields):
    return cu.Symbol(part, cu.canonical([cu.Slot(cu.Field(*f[:3]), f[3]) for f in fields]))


def paper_terms(terms, objects):
    """The paper's terms {(bias, symbols): coefficient} with its objects expanded into order-resolved cumulants."""
    out = {}
    for coef, bias, objs in terms:
        combos = [(sp.sympify(coef), ())]
        for obj in objs:
            combos = [(c, s + (paper_symbol(part, fields),)) for c, s in combos for part, fields in objects[obj]]
        for c, syms in combos:
            key = (tuple(sorted(bias)), tuple(sorted(syms)))
            out[key] = sp.expand(out.get(key, 0) + c)
    return {key: v for key, v in out.items() if v != 0}


def term_list(result=None):
    from .tests import lpt_references as ref
    res = result or cu.derive()
    kept = cu.kept_terms(res)

    def equals_paper():
        expected = paper_terms(ref.PAPER_TREE + ref.PAPER_1LOOP, ref.PAPER_OBJECTS)
        assert set(kept) == set(expected), (sorted(set(kept) - set(expected)), sorted(set(expected) - set(kept)))
        for key in expected:
            assert sp.expand(kept[key] - expected[key]) == 0, (key, kept[key], expected[key])
        exponent = {s: c for c, s in res.exponent}
        assert exponent == {paper_symbol(part, fields): c for c, name in ref.PAPER_EXPONENT
                            for part, fields in ref.PAPER_OBJECTS[name]}
        assert len(kept) == ref.N_KEPT_TERMS
        return len(kept)

    def renormalised_absent():
        assert not any(b in model.RENORMALISED_BIAS for key in res.terms for b in key[0])
        dropped = {key: v for key, v in res.terms.items() if any(b in model.DROPPED_BIAS for b in key[0])}
        assert len(dropped) == ref.N_DROPPED_BNABLA2_TERMS
        for bias, syms in dropped:
            nabla = [s for s in syms if any(sl.field.kind == 'nabla2delta' for sl in s.slots)]
            assert nabla and all(te.entry_of(s) is None for s in nabla), (bias, syms)
        assert all(te.entry_of(s) is not None for key in kept for s in key[1])
        return len(dropped)

    def selfpairs_and_vanishing():
        kinds = {tuple(sorted((sl.field.kind, sl.field.point) for sl in s)) for s, _ in res.selfpairs}
        assert kinds == ref.SELFPAIR_KINDS, kinds
        assert res.vanishing and all(any(sl.field.kind == 'G3' for sl in s) for s, _ in res.vanishing)

    def explicit_mode():
        explicit = cu.derive(res.source, 'explicit')
        assert explicit.terms == res.terms
        assert not any(cu.SIGMA2 in key[1] for key in explicit.terms)
        cancelled = {tuple(sorted(set(b))) for b, _ in explicit.cancelled}
        assert {('b3',), ('bdG2',), ('b2',)} <= cancelled, cancelled
        assert len(explicit.cancelled) == ref.N_EXPLICIT_CANCELLED
        return len(explicit.cancelled)

    def kernel_facts():
        for op, spec in model.OPERATORS.items():
            if spec.order >= 2:
                vanishes = cu.self_contraction(op) == 0
                assert vanishes == (spec.self_contraction == 'vanishes'), (op, spec.self_contraction)
        assert cu.delta_G2_zero_lag() == R(-2, 3)

    return _run(dict(term_list_equals_paper=equals_paper, renormalised_bias_absent=renormalised_absent,
                     selfpair_and_vanishing_cumulants=selfpairs_and_vanishing, explicit_renormalisation=explicit_mode,
                     kernel_facts=kernel_facts))


# ---------------------------------------------------------------------------------------------------------------
# the components

def components(comps):
    from .tests import lpt_hand_code as hand

    def decompositions():
        assert [str(c) for c in te.decomposition('W112')[1]] == ['V1', 'V1', 'V3', 'T']
        assert [str(c) for c in te.decomposition('A10')[1]] == ['X10/2', 'Y10/2']
        assert [str(c) for c in te.decomposition('A13')[1]] == ['X13', 'Y13']
        assert [str(c) for c in te.decomposition('A_lin', '<')[1]] == ['X_LIN_LT', 'Y_LIN_LT']

    def frame():
        red = {te.Kfac**2: 1 + te.f * (2 + te.f) * te.mu**2}
        for n in model.DISPLACEMENT_ORDERS:
            for m in model.DISPLACEMENT_ORDERS:
                direct = te.k**2 * (1 + (n + m) * te.f * te.mu**2 + n * m * te.f**2 * te.mu**2)
                assert sp.expand(te.vdot(f'k{n}', f'k{m}').subs(red) - direct) == 0, (n, m)
        assert sp.expand(te.vdot('k', 'k').subs(red) - te.k**2) == 0
        assert sp.expand(te.vdot('k', 'n') - te.k * te.mu) == 0

    def moments_and_rows():
        moments = [model.moment_name(*m) for m in co.angular_moments(comps)]
        assert set(moments) == set(model.DENSITY_MOMENTS), moments
        rows = {r for c in comps for (_, rs) in c.table for r in rs}
        assert rows == set(model.ROW_ORDER) - {'X_LIN', 'Y_LIN', 'X_LIN_LT', 'Y_LIN_LT'}, rows
        return len(rows)

    def equal_hand_code():
        traced = hand.trace(te.ROW_SYMBOLS, te.k, te.mu, te.f, te.Kfac)
        assert [c.name for c in comps] == list(traced)
        n = 0
        for c in comps:
            mine = co.moment_expressions(c)
            assert set(mine) == set(traced[c.name]), c.name
            for moment, expr in mine.items():
                assert sp.expand(expr - traced[c.name][moment]) == 0, (c.name, moment)
                n += 1
        return n

    def templates_and_linear_kinds():
        meta = {c.name: (tuple(t.name for t in ts), lk.name if lk is not None else None)
                for c, ts, lk in hand._COMPONENTS}
        assert {c.name: (c.templates, c.linear) for c in comps} == meta
        templates = co.templates(comps)
        assert [t for t, _ in templates] == [t.name for t in hand.Template]
        assert np.array_equal(np.array([e for _, e in templates if e is not None]), hand._LPT_BIAS_DEGREES)
        M = np.array([[1.0 if t in c.templates else 0.0 for c in comps] for t, _ in templates])
        N = np.array([[1.0 if c.linear == kd else 0.0 for kd, _ in model.LINEAR_KINDS] for c in comps])
        assert np.array_equal(M, hand._TEMPLATE_MATRIX) and np.array_equal(N, hand._LINEAR_MATRIX)
        assert [kd for kd, _ in model.LINEAR_KINDS] == [kd.name for kd in hand.LinearKind]

    def row_layout():
        assert list(model.ROW_ORDER) + [ks.name for ks in model.KSPACE_ROWS] == [r.name for r in hand.Row]
        assert list(model.DENSITY_MOMENTS) == list(hand.MOMENT_ORDER)

    return _run(dict(decompositions=decompositions, frame=frame, moments_and_rows=moments_and_rows,
                     components_equal_hand_code=equal_hand_code, templates_and_linear_kinds=templates_and_linear_kinds,
                     row_layout=row_layout))


# ---------------------------------------------------------------------------------------------------------------
# the rows

def paper_recipe(row):
    """The appendix's row as {('H', l, n) or ('M', n): block source}."""
    from .tests import lpt_references as ref
    out = {}
    for item in ref.PAPER_ROWS[row]:
        key, srcs = (('M', item[1]), item[2]) if item[0] == 'M' else (('H', item[0], item[1]), item[2])
        out[key] = rr.add_sources(*[(c, appendix_sources()[nm][1]) for nm, c in srcs.items()])
    return out


def appendix_sources():
    """{name: (degree, normalised source)} of the appendix's typed integrands (tests/lpt_references)."""
    from .tests import lpt_references as ref
    out = {name: idn.normalised_22(F) for name, F in ref.INTEGRANDS_22.items()}
    out.update({name: idn.normalised_13(F) for name, F in ref.INTEGRANDS_13.items()})
    return out


def compare_with_paper(der):
    """{row: (H parts identical, constant 'none' / 'identical' / 'equal by full-k identity' / 'DIFFERENT')}."""
    from .tests import lpt_references as ref
    res = {}
    for row in ref.PAPER_ROWS:
        rec, paper = der.raw[row], paper_recipe(row)
        keys = {('H',) + key for key in rec['H']} | {key for key in paper if key[0] == 'H'}
        H_ok = all(rr.sources_equal(rec['H'].get(key[1:], {}), paper.get(key, {})) for key in keys)
        const = 'none'
        ma, mb = rec['M'].get(-2, {}), paper.get(('M', -2), {})
        if ma or mb:
            if rr.sources_equal(ma, mb):
                const = 'identical'
            else:
                const = ('equal by full-k identity' if idn.full_integral_zero(rr.add_sources((1, ma), (-1, mb)))[0]
                         else 'DIFFERENT')
        res[row] = (H_ok, const)
    return res


def rows(der, comps):
    from .tests import lpt_hand_code as hand
    from .tests import lpt_references as ref
    p = der.preferred
    T = rc.Term
    tables = rc.tables(der, comps)

    def named_sources_are_the_appendix():
        A = appendix_sources()
        for name in model.SOURCE_NAMES:
            assert rr.sources_equal(idn.named_sources()[name][1], A[name][1]), name
        for name, expected in ref.DEPENDENT_SOURCES.items():
            got = idn.decompose(A[name][1])
            assert got == {s: sp.nsimplify(c) for s, c in expected.items()}, (name, got)

    def rows_equal_the_paper():
        res = compare_with_paper(der)
        assert all(h for h, _ in res.values()), res
        by_identity = {row for row, (_, c) in res.items() if c not in ('none', 'identical')}
        assert by_identity == ref.CONSTANTS_BY_FULL_K_IDENTITY, res
        assert all(c != 'DIFFERENT' for _, c in res.values()), res
        return len(res)

    def full_k_identities():
        A = appendix_sources()
        for a, b in (('R1', 'Q1'), ('Q2', 'Q1'), ('Q5', 'Q1')):
            assert idn.full_integral_zero(rr.add_sources((1, A[a][1]), (-1, A[b][1])))[0], (a, b)
        assert idn.full_integral_zero(A['R2'][1])[0]
        assert not idn.full_integral_zero(rr.add_sources((1, A['Q8'][1]), (-1, A['Q1'][1])))[0]   # negative control

    def bessel_and_finite_range():
        b = idn.bessel_identities()
        assert all(b.values()), [key for key, v in b.items() if not v]
        worst = idn.finite_range_check()
        assert worst < 1e-28, worst
        return len(b)

    def ladder():
        for ln, expr in ref.PAPER_LADDER.items():
            assert sp.expand(idn.ladder(*ln) - expr) == 0, ln

    def product_forms():
        for row in ref.SEPARABLE_ROWS:
            assert idn.same_expression(der.products[row], ref.HAND_PRODUCTS[row]), row
        for row in ref.GAUSS_LAW_ROWS:
            assert idn.same_expression(der.finite_range[row], ref.HAND_PRODUCTS[row]), row
        assert idn.same_function(der.products['V12'], der.finite_range['V12'])
        assert not idn.same_expression(der.products['V12'], der.finite_range['V12'])

    def preferred_route():
        assert p['UB3'] == [T('row', 0, 0, 0, R(-24, 5), 'U3')]
        assert [(t.coef, t.ref) for t in p['V1'] if t.kind == 'row'] == [(R(-1, 5), 'T')]
        assert [(t.coef, t.ref) for t in p['X22'] if t.kind == 'row'] == [(R(-1, 3), 'Y22')]
        m = {row: [t for t in p[row] if t.kind == 'M'][0] for row in ('X22', 'X13', 'X10')}
        assert all(set(t.ref) == {'Q1'} for t in m.values())
        ratio = {row: t.coef * t.ref['Q1'] / (m['X22'].coef * m['X22'].ref['Q1']) for row, t in m.items()}
        assert ratio['X13'] == R(70, 27) and ratio['X10'] * R(2, 3) == R(28, 9), ratio
        assert [(t.l, t.n, t.qpow, t.coef) for t in p['Y_LIN']] == [(1, -1, 1, R(2, 5)), (3, -1, 1, R(2, 5))]
        assert [(t.l, t.n, t.qpow) for t in p['T']] == [(2, -2, 1), (4, -2, 1)]
        assert [t.ref for t in p['THETA']] == [{'R1': R(-8, 7)}] and p['THETA'][0].n == 0

    def stage_a_and_units():
        ln = sorted({(l, n) for _, l, n in tables['stage_a']}, key=lambda ln: (ln[0], -ln[1]))
        assert [list(v) for v in ln] == hand.LN_LIST, ln
        units = tables['units']
        units13 = [(units[u][2][0][0], units[u][1]) for u in tables['unit_order'] if units[u][0] == '13']
        assert units13 == hand.R_LN, units13

    def attachments_and_kspace_rows():
        assert {(a.component, a.rule) for a in tables['attachments']} == ref.HAND_ATTACHMENTS
        expected = [(hand.Row(row).name, hand.Component(c).name, f, interp, k0)
                    for c, row, f, interp, k0 in hand._KSPACE_ROWS]
        got = [(row, c, float(f), interp, k0) for row, c, f, interp, k0 in tables['kspace']]
        assert got == expected, (got, expected)

    return _run(dict(named_sources_are_the_appendix=named_sources_are_the_appendix,
                     rows_equal_the_paper=rows_equal_the_paper, full_k_identities=full_k_identities,
                     bessel_and_finite_range=bessel_and_finite_range, ladder_equals_the_paper=ladder,
                     product_forms_equal_hand_code=product_forms, preferred_route_is_hand_code=preferred_route,
                     stage_a_and_units_equal_hand_code=stage_a_and_units,
                     attachments_and_kspace_rows=attachments_and_kspace_rows))


# ---------------------------------------------------------------------------------------------------------------
# the declared angular polynomials

def moments_declaration():
    """The declared density moments equal mu_K^a (A_mu mu_K - B_mu mu_x)^b (lpt_model.LINE_OF_SIGHT)."""
    muK, mux, muy = sp.symbols('mu_K mu_x mu_y')

    def check():
        for name, terms in model.DENSITY_MOMENTS.items():
            a = int(name[2]) if name.startswith('MQ') else 0
            b = int(name.split('NQ')[1]) if 'NQ' in name else 0
            expected = sp.expand(muK**a * (model.A_mu * muK - model.B_mu * mux)**b)
            declared = sp.expand(sum(c * muK**i * mux**j * muy**m for c, i, j, m in terms))
            assert sp.expand(declared - expected) == 0, name
        return len(model.DENSITY_MOMENTS)

    return _run(dict(density_moments_are_MQa_NQb=check))
