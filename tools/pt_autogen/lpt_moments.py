"""The factored psi family of the LPT moments (sympy), written to psi_terms.dat and scalar_layer.dat.

The two files (``src/elf/tables/lpt``, written by ``emit.lpt_psi_text``) are read at runtime by ``elf.lpt_tables`` for the
tangent rule ``elf.lpt_rule``.  They encode the identity that every moment M_j(l) of ``elf.utils_lpt.get_lpt_moments``
(j over MOMENTS, without the factor exp(-B s^2)) is exactly

    M_j(l) = sum over the pairs (b, s) of moment j with s <= l of basis_b[l - s] psi_{j,b,s},

with basis_b the Horner basis S0, T1 .. T4 of ``utils_lpt.lpt_horner_poly`` (T_d = S_d / s^2) and psi_{j,b,s}
independent of l.  Each psi_{j,b,s} is a polynomial in the (nq,) arrays A = k q c, C = k q s, B = -(1/2) K^2 Y_lin_lt,
ir = 1/sqrt(A^2 + C^2) and in the per-mu scalars c, s, A_mu, B_mu, written psi_{j,b,s} = sum_t K_t A^a C^c B^b ir^i +
constant terms with the per-mu coefficients K_t = sum coef c^a s^b A_mu^e B_mu^g.

Provenance.  The moments are derived from the generating function by :mod:`pt_autogen.lpt_angular` for the angular
polynomials of lpt_model.DENSITY_MOMENTS.  This module factorizes them.  The exact angular coefficients inside the
Horner basis are not part of the tables (the runtime computes them with the exact Fraction recursion of
``elf.lpt_coefficients``), and the derivatives of psi that the rule needs follow from the exponents by the rule of
``elf.lpt_tables`` (verified here against sympy).

The generation has three steps.
  1. ``collect_psi`` expands every M_j(l), l <= lmax.  Each term must contain exactly one basis symbol to the first
     power, and the coefficient of basis_b[l - s] is psi_{j,b,s}, asserted to be a polynomial (no negative power of s)
     that does not depend on l.  ``derive`` collects up to LMAX_CHECK and up to l = 4 and asserts that they agree.
  2. ``register_terms`` and ``scalar_matrix`` split psi by array monomial A^a C^c B^b ir^i.  The distinct scalar
     coefficients are the K_t (a negated duplicate reuses K_t with a minus sign), expanded in monomials.
  3. ``make_tables`` gives the rows of the two files (exact Fractions, every coefficient dyadic so that its decimal is
     exact).
Three checks.  ``verify_psi`` compares the tables' psi family and its dA, dB, dC (by the exponent rule of
``elf.lpt_tables``) with the sympy psi, exactly.  ``verify_moments`` rebuilds the moments from the written tables and
compares them with the derived moments, exactly (this checks the factorization and the tables).  ``moment_identity``
compares the tables with the runtime ``get_lpt_moments`` itself (JAX float64), which checks the hand-written moments
of the runtime against the derivation to rounding.  The transcription of the hand-written moments
(tools/pt_autogen/tests/test_lpt_tables.py) is checked against the derivation exactly.

A new observable whose final integral has this structure declares its moments as angular polynomials (lpt_model) and
supplies its component table.  Everything here then follows, the pairs, the psi entries, the K_t and the scalar layer,
the monomials and their derivatives, and at runtime the residual sets of the rule.  A moment of a higher power of
Khat.qhat extends the Horner basis to T5, T6, ...
"""
import importlib
import importlib.util
import math
import os
import shutil
import sys
import tempfile
import types
from fractions import Fraction

import sympy as sp

from . import lpt_angular, lpt_model
from .lpt_angular import A, ARRAY_SYMBOLS, B, C, SCALAR_SYMBOLS, Am, Bm, basis_symbols, c, ir, s

MOMENTS = tuple(lpt_model.DENSITY_MOMENTS)
"""The moments in the order of the tuple of ``get_lpt_moments``."""

BASIS = lpt_model.HORNER_BASIS
"""The Horner basis, S0 from V[0] and T_d from V'[d - 1] (``utils_lpt.lpt_horner_poly``)."""

LMAX_CHECK = 10
"""psi is collected from the moments up to this l (and checked against the collection up to l = 4)."""

RUNTIME_TABLE_MODULE = 'lpt_tables.py'
"""The runtime module of src/elf that reads the psi tables (numpy only), loaded next to tables under test."""

COEFFICIENT_VARIABLE = {'K': None, 'Kc': c, 'Ks': s, 'KAm': Am, 'KBm': Bm}
"""The coefficient vectors of ``elf.lpt_tables.psi_family``, K and K^x = dK/dx at fixed values of the other three."""

SUMS = (('K', 'psi'), ('K', 'dA'), ('K', 'dB'), ('K', 'dC'), ('Kc', 'psi'), ('Ks', 'psi'), ('KAm', 'psi'),
        ('KBm', 'psi'))
"""The (name, deriv) sums that ``psi_family`` evaluates."""


def derived_moments(X, L):
    """M[j][l] of lpt_model.DENSITY_MOMENTS on the basis symbols X, l < L (lpt_angular)."""
    return lpt_angular.moments(X, L)


# ------------------------------------------------------------------ 1. the psi family
def collect_psi(lmax):
    """``{(j, basis, shift): psi}`` collected from the moments up to l = lmax (module docstring, step 2)."""
    L = lmax + 1
    X = basis_symbols(L)
    where = {X[b][l]: (b, l) for b in BASIS for l in range(L)}
    M = derived_moments(X, L)
    psi, seen_l = {}, {}
    for j in range(len(MOMENTS)):
        for l in range(L):
            coef = {}
            for term in sp.Add.make_args(sp.expand(M[j][l])):
                xs = [x for x in term.free_symbols if x in where]
                # linearity, exactly one basis symbol to the first power
                if len(xs) != 1 or sp.degree(term, xs[0]) != 1:
                    raise AssertionError(f'M_{MOMENTS[j]}(l={l}) not linear in the basis: term {term}')
                coef[xs[0]] = coef.get(xs[0], sp.Integer(0)) + term / xs[0]
            for x, cf in coef.items():
                b, lp = where[x]
                shift = l - lp
                cf = sp.cancel(cf)
                num, den = sp.fraction(cf)
                if den.free_symbols:
                    raise AssertionError(f'psi({MOMENTS[j]}, {b}, {shift}) at l={l} is not a polynomial: {den}')
                cf = sp.expand(cf)
                if cf == 0:
                    continue
                for t in sp.Add.make_args(cf):
                    if t.as_powers_dict().get(s, 0) < 0:
                        raise AssertionError(f'psi({MOMENTS[j]}, {b}, {shift}): negative power of s in {t}')
                key = (j, b, shift)
                if key in psi:
                    if sp.expand(psi[key] - cf) != 0:
                        raise AssertionError(f'psi{key} depends on l: l={seen_l[key][0]} vs l={l}')
                    seen_l[key].append(l)
                else:
                    psi[key] = cf
                    seen_l[key] = [l]
    # every psi occurs for every l >= shift (the shift convention X[l - s] = 0 only for l < s)
    for key, ls in seen_l.items():
        assert ls == list(range(key[2], L)), (key, ls)
    return psi


def derive():
    """psi of ``collect_psi(LMAX_CHECK)``, asserted equal to ``collect_psi(4)`` (every shift occurs by l = 4)."""
    psi = collect_psi(LMAX_CHECK)
    psi4 = collect_psi(4)
    assert set(psi4) == set(psi) and all(sp.expand(psi4[k] - psi[k]) == 0 for k in psi), 'psi depends on lmax'
    return psi


def sort_key(key):
    """The order of the psi entries, by moment, then shift, then basis."""
    j, b, shift = key
    return (j, shift, BASIS.index(b))


def split_terms(expr):
    """Expanded polynomial -> [(array monomial, scalar coefficient)] in descending lex order of (B, A, C, ir)."""
    P = sp.Poly(expr, B, A, C, ir)
    out = []
    for (nb, na, nc, nk), coef in P.terms():
        mono = B**nb * A**na * C**nc * ir**nk
        k = sp.expand(coef.as_expr() if hasattr(coef, 'as_expr') else coef)
        assert not (k.free_symbols & set(ARRAY_SYMBOLS)), k
        out.append((mono, k))
    return out


def assert_dyadic(e):
    """Every rational number of ``e`` has a power of two as denominator (so its float and decimal are exact)."""
    for r in e.atoms(sp.Rational):
        q = int(r.q)
        assert q & (q - 1) == 0, f'non-dyadic rational {r} in {e}'


# ------------------------------------------------------------------ 2. the per-mu coefficients K_t
def register_terms(psi, keys):
    """The per-entry plan [(coefficient, monomial index)], the scalar coefficients K_t and the array monomials.

    A coefficient is ('num', number) or ('sym', t, sign) with K_t = sign x the term's scalar coefficient.  The
    monomial index is None for the monomial 1.  Both lists are in order of first appearance.
    """
    split = {key: split_terms(psi[key]) for key in keys}
    scal_list, scal_idx = [], {}

    def reg(k):
        if k.is_Number:
            return ('num', k)
        if k in scal_idx:
            return ('sym', scal_idx[k], 1)
        nk = sp.expand(-k)
        if nk in scal_idx:
            return ('sym', scal_idx[nk], -1)
        scal_idx[k] = len(scal_list)
        scal_list.append(k)
        return ('sym', scal_idx[k], 1)

    mono_list, mono_idx = [], {}

    def regm(m):
        if m == 1:
            return None
        if m not in mono_idx:
            mono_idx[m] = len(mono_list)
            mono_list.append(m)
        return mono_idx[m]

    plan = {key: [(reg(k), regm(m)) for m, k in split[key]] for key in keys}
    return plan, scal_list, mono_list


def scalar_matrix(scal_list):
    """(scalar monomials, matrix [m][t] as Fractions), every K_t expanded in c, s, A_mu, B_mu.

    The monomials are sorted by increasing degree, then by decreasing exponents (the order ``elf.lpt_tables`` derives).
    """
    polys = []
    for k in scal_list:
        assert not (k.free_symbols - set(SCALAR_SYMBOLS)), k
        P = sp.Poly(sp.expand(k), *SCALAR_SYMBOLS)
        terms = {}
        for exps, coef in P.terms():
            r = sp.Rational(coef)
            assert r != 0
            q = int(r.q)
            assert q & (q - 1) == 0, f'non-dyadic coefficient {r} in K = {k}'
            terms[tuple(int(e) for e in exps)] = Fraction(int(r.p), q)
        polys.append(terms)
    monos = sorted({e for t in polys for e in t}, key=lambda e: (sum(e), tuple(-x for x in e)))
    mat = [[t.get(e, Fraction(0)) for t in polys] for e in monos]
    for row in mat:
        for v in row:
            assert Fraction(float(v)) == v, f'{v} is not exactly representable'
    return monos, mat


# ------------------------------------------------------------------ 3. the table rows
def make_tables(psi):
    """The rows of psi_terms.dat and scalar_layer.dat, ``{'terms': [...], 'scalar': [...], 'counts': {...}}``.

    ``terms`` holds (moment, basis, shift, t, (a, c, b, i), coefficient) for the term coefficient K_t A^a C^c B^b ir^i
    (t = -1 without K factor), the entries in ``sort_key`` order and the terms of an entry in ``split_terms`` order.
    ``scalar`` holds (t, (a, b, e, g), coefficient) for the nonzero entries of the scalar matrix, by t and then in
    monomial order.
    """
    keys = sorted(psi, key=sort_key)
    for key in keys:
        assert_dyadic(psi[key])
    plan, scal_list, mono_list = register_terms(psi, keys)
    monos, mat = scalar_matrix(scal_list)
    exponents = []
    for m in mono_list:
        (e, cf), = sp.Poly(m, A, C, B, ir).terms()
        assert cf == 1, m
        exponents.append(tuple(int(x) for x in e))
    terms = []
    for key in keys:
        j, b, shift = key
        for cf, mi in plan[key]:
            if cf[0] == 'num':
                r = sp.Rational(cf[1])
                t, v = -1, Fraction(int(r.p), int(r.q))
            else:
                t, v = cf[1], Fraction(cf[2])
            assert v.denominator & (v.denominator - 1) == 0, (key, v)
            terms.append((MOMENTS[j], b, shift, t, (0, 0, 0, 0) if mi is None else exponents[mi], v))
    scalar = [(t, e, mat[n][t]) for t in range(len(scal_list)) for n, e in enumerate(monos) if mat[n][t] != 0]
    counts = dict(terms=len(terms), with_K=sum(1 for r in terms if r[3] >= 0),
                  without_K=sum(1 for r in terms if r[3] < 0), entries=len(keys),
                  pairs=len({(b, shift) for _, b, shift in keys}), moments=len(MOMENTS), monomials=len(mono_list),
                  K=len(scal_list), scalar_entries=len(scalar), scalar_monomials=len(monos))
    return dict(terms=terms, scalar=scalar, counts=counts)


def read_tables(table_dir):
    """The rows of ``make_tables`` read back from ``<table_dir>/lpt/psi_terms.dat`` and ``scalar_layer.dat``."""
    from . import emit
    lpt_dir = os.path.join(table_dir, emit.LPT_DIR)
    terms = [(m, b, int(sh), int(t), tuple(int(x) for x in e), Fraction(v))
             for m, b, sh, t, *e, v in emit.read_text_table(os.path.join(lpt_dir, 'psi_terms.dat'))]
    scalar = [(int(t), tuple(int(x) for x in e), Fraction(v))
              for t, *e, v in emit.read_text_table(os.path.join(lpt_dir, 'scalar_layer.dat'))]
    return dict(terms=terms, scalar=scalar)


# ------------------------------------------------------------------ checks
def _rational(v):
    return sp.Rational(v.numerator, v.denominator)


def _exact(x):
    """x (a sympy expression or a number) with every float replaced by its exact rational."""
    e = sp.sympify(x)
    return e.xreplace({f: sp.Rational(f) for f in e.atoms(sp.Float)})


def _K_polynomials(tables):
    """K_t as polynomials in c, s, A_mu, B_mu from the scalar rows."""
    K = [sp.Integer(0)] * (1 + max(t for t, _, _ in tables['scalar']))
    for t, (a, b, e, g), v in tables['scalar']:
        K[t] += _rational(v) * c**a * s**b * Am**e * Bm**g
    return K


def total_derivative(e, d):
    """d e / dx for d = 'dA', 'dB', 'dC' (with d ir/dA = -A ir^3 and d ir/dC = -C ir^3), and e itself for 'psi'."""
    if d == 'psi':
        return e
    if d == 'dA':
        return sp.expand(sp.diff(e, A) + sp.diff(e, ir) * (-A * ir**3))
    if d == 'dB':
        return sp.expand(sp.diff(e, B))
    if d == 'dC':
        return sp.expand(sp.diff(e, C) + sp.diff(e, ir) * (-C * ir**3))
    raise ValueError(d)


def table_expressions(tables):
    """``({(name, deriv): {(j, basis, shift): expression}}, K symbols)``, the psi family built from the rows by the
    rule of ``elf.lpt_tables`` with K_t a symbol.  The terms without K enter name 'K' only.  For K^x the terms whose
    column t holds no monomial with a power of x vanish identically and are dropped.  dA, dB, dC go term by term,
        d(A^a C^c B^b ir^i)/dA = a A^(a-1) C^c B^b ir^i - i A^(a+1) C^c B^b ir^(i+2),
        d(A^a C^c B^b ir^i)/dB = b A^a C^c B^(b-1) ir^i.
    """
    Ks = sp.symbols(f'K0:{1 + max(t for t, _, _ in tables["scalar"])}')
    depends = {x: {t for t, e, _ in tables['scalar'] if e[n]} for n, x in enumerate(SCALAR_SYMBOLS)}
    out = {}
    for name, deriv in SUMS:
        x = COEFFICIENT_VARIABLE[name]
        res = {}
        for moment, b, shift, t, (a, cc, bb, i), v in tables['terms']:
            if x is not None and (t < 0 or t not in depends[x]):
                continue
            mono = A**a * C**cc * B**bb * ir**i
            if deriv == 'psi':
                dm = mono
            elif deriv == 'dA':
                dm = (a * mono / A if a else 0) - (i * mono * A * ir**2 if i else 0)
            elif deriv == 'dC':
                dm = (cc * mono / C if cc else 0) - (i * mono * C * ir**2 if i else 0)
            else:
                dm = bb * mono / B if bb else 0
            dm = sp.expand(dm)
            if dm == 0:
                continue
            key = (MOMENTS.index(moment), b, shift)
            res[key] = res.get(key, 0) + _rational(v) * (Ks[t] if t >= 0 else 1) * dm
        out[(name, deriv)] = res
    return out, Ks


def verify_psi(psi, tables, psi_family=None):
    """Assert that the psi family of the tables is the sympy psi, and return the number of expressions compared.

    For each sum of SUMS and each psi entry the table expression with K_t -> its polynomial (K^x -> dK_t/dx) equals
    psi, its total derivative or d psi/dx exactly, and an entry the tables omit is identically zero.  ``psi_family``
    (optional, ``elf.lpt_tables.psi_family``) is evaluated on sympy symbols and must give the table expressions.
    """
    tab, Ks = table_expressions(tables)
    K = _K_polynomials(tables)
    runtime = None
    if psi_family is not None:
        runtime = psi_family(A, C, B, ir, {name: list(Ks) for name in COEFFICIENT_VARIABLE}, frozenset(SUMS))
    n = 0
    for name, deriv in SUMS:
        x = COEFFICIENT_VARIABLE[name]
        sub = dict(zip(Ks, K if x is None else [sp.diff(k, x) for k in K]))
        if runtime is not None:
            assert set(runtime[(name, deriv)]) == set(tab[(name, deriv)]), ('runtime entries', name, deriv)
        for key in psi:
            truth = total_derivative(psi[key], deriv) if x is None else sp.diff(psi[key], x)
            if key not in tab[(name, deriv)]:
                assert sp.expand(truth) == 0, ('omitted but nonzero', name, deriv, key)
                continue
            expr = tab[(name, deriv)][key]
            assert sp.expand(expr.xreplace(sub) - truth) == 0, ('table vs sympy', name, deriv, key)
            if runtime is not None:
                assert sp.expand(_exact(runtime[(name, deriv)][key]) - expr) == 0, ('runtime', name, deriv, key)
            n += 1
    return n


def verify_moments(tables, lmax=LMAX_CHECK):
    """Assert that the moments rebuilt from the table rows equal the derived moments for l <= lmax (exact).

    M_j(l) = sum over the rows of moment j with shift <= l of basis_b[l - shift] coef K_t A^a C^c B^b ir^i, K_t its
    polynomial.  Returns the number of moments M_j(l) compared.
    """
    L = lmax + 1
    X = basis_symbols(L)
    derived = derived_moments(X, L)
    K = _K_polynomials(tables)
    for j, name in enumerate(MOMENTS):
        rows = [r for r in tables['terms'] if r[0] == name]
        for l in range(L):
            rebuilt = sum((X[b][l - shift] * _rational(v) * (K[t] if t >= 0 else 1) * A**a * C**cc * B**bb * ir**i
                           for _, b, shift, t, (a, cc, bb, i), v in rows if shift <= l), sp.Integer(0))
            assert sp.expand(rebuilt - derived[j][l]) == 0, ('tables vs derived moments', name, l)
    return L * len(MOMENTS)


def import_utils_lpt():
    """``(jax, utils_lpt)``: jax with float64 enabled and ``src/elf/utils_lpt.py`` loaded without the package ``elf``.

    The module is imported as a submodule of a stand-in package over ``src/elf``, so its relative import of
    lpt_coefficients works and elf/__init__.py (which reads the committed tables) is not executed."""
    import jax
    jax.config.update('jax_enable_x64', True)
    from . import emit
    name = 'pt_autogen_elf_runtime'
    if name not in sys.modules:
        package = types.ModuleType(name, 'the src/elf modules pt_autogen loads without the package elf')
        package.__path__ = [os.path.join(emit.REPO_DIR, 'src', 'elf')]
        sys.modules[name] = package
    return jax, importlib.import_module(name + '.utils_lpt')


def runtime_table_module(table_dir, name='pt_autogen_lpt_table_module'):
    """``src/elf/<RUNTIME_TABLE_MODULE>`` loaded from a copy next to a copy of ``<table_dir>/lpt`` (numpy only)."""
    from . import emit
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(os.path.join(table_dir, emit.LPT_DIR), os.path.join(tmp, 'tables', emit.LPT_DIR))
        shutil.copy(os.path.join(emit.REPO_DIR, 'src', 'elf', RUNTIME_TABLE_MODULE), tmp)
        spec = importlib.util.spec_from_file_location(name, os.path.join(tmp, RUNTIME_TABLE_MODULE))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def physical_points():
    """24 points (A, B, C, c, s, A_mu, B_mu) of the LPT geometry (lpt._rsd_geometry) with f, mu, k, q random and
    B = -U(0, 50)."""
    import numpy as np
    rng = np.random.default_rng(7)
    pts = []
    for _ in range(24):
        f, mu = rng.uniform(0.3, 1.0), rng.uniform(0.01, 0.99)
        sin = math.sqrt(1 - mu * mu)
        Kfac = math.sqrt(1 + f * (2 + f) * mu**2)
        cc, ss = (1 + f * mu**2) / Kfac, f * mu * sin / Kfac
        k, q = rng.uniform(0.005, 0.5), math.exp(rng.uniform(math.log(0.5), math.log(1000.0)))
        pts.append((k * q * cc, -rng.uniform(0, 50), k * q * ss, cc, ss, (1 + f) * mu / Kfac, sin / Kfac))
    return pts


def moment_identity(table_dir, lmaxes=(5, 8, 10)):
    """``{lmax: worst}`` of get_lpt_moments against the tables of ``table_dir`` (JAX float64, the runtime code).

    At the 24 ``physical_points``, per moment j max_l |M - M_tables| / max_l |M|, with M of
    ``elf.utils_lpt.get_lpt_moments`` and M_tables = sum over the pairs of basis_b[l - s] psi_{j,b,s}, the Horner basis
    of ``lpt_horner_poly`` and psi of the runtime table module reading the tables of ``table_dir``.
    """
    import numpy as np
    jax, ul = import_utils_lpt()
    jnp = jax.numpy
    lp = runtime_table_module(table_dir)
    m_min = (0,) + (1,) * (len(lp.BASIS) - 1)
    worst = {}
    for lmax in lmaxes:
        L, G00 = lmax + 1, ul.make_G00_coeffs(lmax)

        def both(Av, Bv, Cv, cc, ss, am, bm, G00=G00, L=L, lmax=lmax):
            s2 = ss * ss
            V = ul.compute_V_mu(s2, G00, lmax)
            hand = jnp.stack([m[:, 0] for m in ul.get_lpt_moments(Av, Bv, Cv, cc * cc, s2, am, bm, V, c=cc, s=ss)])
            basis = ul.lpt_horner_poly(Bv, jnp.concatenate([V[:1], ul.compute_V_mu_prime(s2, G00, lmax)]), 0, m_min)
            K = lp.scalar_monomials(cc, ss, am, bm) @ jnp.asarray(lp.SCALAR_MATRIX)
            psi = lp.psi_family(Av, Cv, Bv, jax.lax.rsqrt(Av * Av + Cv * Cv), {'K': K}, {('K', 'psi')})[('K', 'psi')]
            rows = [[jnp.broadcast_to(sum((basis[lp.BASIS.index(b)][l - sh] * psi[(j, b, sh)]
                                          for b, sh in lp.PAIRS_OF[j] if sh <= l), 0.0), Av.shape)[0]
                     for l in range(L)] for j in range(len(lp.MOMENTS))]
            return hand, jnp.array(rows)
        fn = jax.jit(both)
        w = 0.0
        for pt in physical_points():
            args = [jnp.asarray([v]) for v in pt[:3]] + [jnp.asarray(v) for v in pt[3:]]
            hand, tab = (np.asarray(v) for v in fn(*args))
            w = max(w, max(float(np.max(np.abs(tab[j] - hand[j])) / np.max(np.abs(hand[j]))) for j in range(len(hand))))
        worst[lmax] = w
    return worst
