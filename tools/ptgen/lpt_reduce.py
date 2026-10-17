"""Reduction of the Wick integrands (lpt_wick) to row recipes, with the EPT generator's reduction (ptgen.reduction).

For one basis contraction of a cumulant the integrand is int_k e^{ik.q} sum_j mu^j F_j with mu = khat.qhat, and
* np = qhat.p is averaged over the azimuth of p about khat (ptgen.vectors.los_average with n read as qhat),
* each F_j is reduced, 22 to ``reduce22`` blocks k^m B22_{l',n'}, 13 to ``reduce13`` + ``canonical13`` blocks
  k^m B13_{l',n'} (whose zero-lag pieces must vanish), tree to c k^d,
* the blocks are normalised to degree-0 sources (k^{-2n'} B22, k^{2-n'} B13, P) and the homogeneity degree d of the
  integrand becomes the n of the transform,
* the angular integral int dOmega/(4 pi) e^{i k q mu} mu^j = sum_l c_{jl} i^l j_l(kq) (c_{jl} the Legendre coefficients
  of mu^j) gives sum_l c_{jl} i^l H_l^d[S_j],
* the Gram matrix of the isotropic basis turns the contractions into the coefficients of the basis, the rows,
* the constant of a row with Delta's follows from Delta(q = 0) = 0, C = - sum_d M_d[S_{l=0,d}] (H_l^d[S](0) =
  delta_{l0} M_d[S]).

A recipe is ``{'H': {(l, d): source}, 'M': {d: source}}``, a source ``{block: coefficient}`` with the blocks ('P',
spectrum), ('22', l', n') and ('13', l', n') (lpt_model.TRANSFORM, CONSTANT).  The separable 22 parts also get their
q-space product form (``product_form``), a polynomial in the symbols ``xi(l, n)`` = xi_l^n(q).
"""
from collections import defaultdict
from functools import lru_cache

import sympy as sp

from . import lpt_model as model
from . import lpt_tensors as te
from . import lpt_wick as wick
from . import reduction as RD
from .vectors import k, los_average, mu, np_, nr_, p, r, x, z

QS = sp.Symbol('q', positive=True)           # the separation |q| (ptgen.vectors.q is |k - p|)


@lru_cache(maxsize=None)
def xi(l, n):
    """The symbol of xi_l^n(q), named xi_l_n with 'm' for a minus sign (xi_1_m1 = xi_1^-1)."""
    return sp.Symbol(f'xi_{l}_{n}'.replace('-', 'm'))


def xi_index(sym):
    l, n = sym.name[3:].split('_')
    return int(l), int(n.replace('m', '-'))


@lru_cache(maxsize=None)
def legendre_of_power(j):
    """{l: c_{jl}} with mu^j = sum_l c_{jl} L_l(mu)."""
    return RD.legendre_expand(mu**j, mu)


def _clean(d):
    return {key: sp.nsimplify(sp.expand(v)) for key, v in d.items() if sp.expand(v) != 0}


def add_sources(*pairs):
    out = defaultdict(lambda: sp.Integer(0))
    for c, s in pairs:
        for key, v in s.items():
            out[key] += c * v
    return _clean(out)


def sources_equal(a, b):
    keys = set(a) | set(b)
    return all(sp.expand(a.get(key, 0) - b.get(key, 0)) == 0 for key in keys)


# ---------------------------------------------------------------------------------------------------------------
# kind by kind

def _mu_coefficients(F):
    """{j: F_j} of F = sum_j mu^j F_j (F polynomial in mu after the azimuthal average)."""
    num, den = sp.fraction(sp.together(F))
    assert not den.has(mu), den
    return {int(j): sp.cancel(c / den) for (j,), c in sp.Poly(sp.expand(num), mu).terms()}


def reduce_tree(F, spectrum='P'):
    """{(l, d): source} of a tree integrand F(k, mu) (times P(k))."""
    out = defaultdict(dict)
    for j, Fj in _mu_coefficients(sp.expand(F)).items():
        Fj = sp.factor(Fj)
        d = RD.homogeneity_degree(Fj, (k,))
        c = sp.cancel(Fj / k**d)
        assert not c.has(k), Fj
        for l, cl in legendre_of_power(j).items():
            src = out[(l, d)]
            src[('P', spectrum)] = src.get(('P', spectrum), 0) + c * cl * sp.I**l
    return {key: _clean(s) for key, s in out.items() if _clean(s)}


def reduce_22(F):
    """{(l, d): source} of a 22 integrand F(k, p, x, mu, np) (times P(p) P(|k-p|))."""
    out = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    for j, Fj in _mu_coefficients(los_average(F)).items():
        blocks = RD.reduce22(Fj)
        if not blocks:
            continue
        degs = {m + 2 * n for (l, n, m) in blocks}
        assert len(degs) == 1, ('22 blocks of several degrees', degs)
        d = degs.pop()
        for l, cl in legendre_of_power(j).items():
            for (lb, nb, mb), c in blocks.items():
                out[(l, d)][('22', lb, nb)] += c * cl * sp.I**l
    return {key: _clean(s) for key, s in out.items() if _clean(s)}


def reduce_13(F):
    """``({(l, d): source}, zero_lag)`` of a 13 integrand F(k, p, x, mu, np) (times P(k) P(p))."""
    out = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    zero_lag = {}
    for j, Fj in _mu_coefficients(los_average(F)).items():
        # the even part under p -> -p (x -> -x) carries the whole loop integral (P(p) and the measure are even)
        Fj = sp.cancel((Fj + Fj.subs(x, -x)) / 2)
        if Fj == 0:
            continue
        blocks, zl, _ = RD.reduce13(Fj)
        blocks = RD.canonical13(blocks)
        if zl:
            zero_lag[j] = zl
        if not blocks:
            continue
        degs = {m + n - 2 for (l, n, m) in blocks}
        assert len(degs) == 1, ('13 blocks of several degrees', degs, blocks)
        d = degs.pop()
        for l, cl in legendre_of_power(j).items():
            for (lb, nb, mb), c in blocks.items():
                out[(l, d)][('13', lb, nb)] += c * cl * sp.I**l
    return {key: _clean(s) for key, s in out.items() if _clean(s)}, zero_lag


# ---------------------------------------------------------------------------------------------------------------
# separable 22 terms, the q-space product

def _azimuthal_z_power(j, m1, m2):
    """<z^j> over the relative azimuth, z = m1 m2 + s1 s2 cos(phi), s_i^2 = 1 - m_i^2."""
    out = sp.Integer(0)
    for i in range(0, j + 1, 2):
        out += (sp.binomial(j, i) * (m1 * m2)**(j - i) * ((1 - m1**2) * (1 - m2**2))**(i // 2)
                * sp.binomial(i, i // 2) / 2**i)
    return sp.expand(out)


def is_separable(G):
    """Whether the 22 integrand G(p, r, z, np, nr) is a polynomial in z, np, nr over monomials in p, r."""
    num, den = sp.fraction(sp.cancel(sp.together(G)))
    return not (den.has(z) or den.has(np_) or den.has(nr_) or den.has(x))


def product_form(G):
    """The q-space product of a separable 22 integrand, int_p int_r e^{i(p+r).q} G P(p) P(r), as a polynomial in the
    xi(l, n).  With qhat the polar axis np = p m1 and nr = r m2, the average over the relative azimuth replaces z^j,
    and a term m1^a m2^b p^e1 r^e2 gives [sum_l c_{al} i^l xi(l, e1)] [sum_l c_{bl} i^l xi(l, e2)]."""
    num, den = sp.fraction(sp.cancel(sp.together(G)))
    assert is_separable(G), den
    m1, m2 = sp.symbols('m1 m2')
    num = sp.expand(num.subs({np_: p * m1, nr_: r * m2}))
    acc = sp.Integer(0)
    for (j,), c in sp.Poly(num, z).terms():
        acc += c * _azimuthal_z_power(j, m1, m2)
    acc = sp.expand(acc / den)
    out = sp.Integer(0)
    for term in sp.Add.make_args(acc):
        powers = term.as_powers_dict()
        a, b = int(powers.get(m1, 0)), int(powers.get(m2, 0))
        e1, e2 = int(powers.get(p, 0)), int(powers.get(r, 0))
        c = sp.cancel(term / (m1**a * m2**b * p**e1 * r**e2))
        assert not c.free_symbols, term
        f1 = sum(cl * sp.I**l * xi(l, e1) for l, cl in RD.legendre_expand(m1**a, m1).items())
        f2 = sum(cl * sp.I**l * xi(l, e2) for l, cl in RD.legendre_expand(m2**b, m2).items())
        out += c * f1 * f2
    return sp.expand(out)


# ---------------------------------------------------------------------------------------------------------------
# entries -> rows

def combine_recipes(weights, recipes):
    """sum_t w_t recipe_t for recipes {'H': {(l, d): source}, 'M': {d: source}}."""
    out = {'H': defaultdict(dict), 'M': defaultdict(dict)}
    for w, rec in zip(weights, recipes):
        for part in ('H', 'M'):
            for key, src in rec.get(part, {}).items():
                out[part][key] = add_sources((1, out[part][key]), (w, src))
    return {part: {key: s for key, s in out[part].items() if s} for part in out}


def recipes_equal(a, b):
    for part in ('H', 'M'):
        for key in set(a.get(part, {})) | set(b.get(part, {})):
            if not sources_equal(a.get(part, {}).get(key, {}), b.get(part, {}).get(key, {})):
                return False
    return True


def derive_entry(entry, transverse=True):
    """``(rows, products, info)``: {row name: recipe}, {row name: product form of the 22 part or None} and the details
    (zero-lag pieces, matching counts, separability)."""
    ints = wick.entry_integrands(entry, transverse=transverse)
    nb = len(ints.basis)
    per_t, zero_lag = [], {}
    for t in range(nb):
        H = defaultdict(dict)
        for kind in ('tree', '13', '22'):
            F = ints.integrands.get((t, kind))
            if F is None or F == 0:
                continue
            if kind == 'tree':
                part = reduce_tree(F)
            elif kind == '22':
                part = reduce_22(F)
            else:
                part, zl = reduce_13(F)
                if zl:
                    zero_lag[t] = zl
            for key, src in part.items():
                H[key] = add_sources((1, H[key]), (1, src))
        per_t.append({'H': {key: s for key, s in H.items() if s}, 'M': {}})
    sep = {t: (is_separable(G) if G != 0 else None) for t, G in ints.integrands_pr.items()}
    prods = {t: (product_form(G) if sep[t] else None) for t, G in ints.integrands_pr.items()}
    Ginv = te.gram(ints.rank).inv()
    rmap, _ = te.row_map(entry)
    rows, products = {}, {}
    for s in range(nb):
        ws = [(Ginv[s, t], t) for t in range(nb) if Ginv[s, t] != 0]
        rec = combine_recipes([w for w, _ in ws], [per_t[t] for _, t in ws])
        if ints.rank > 0:
            const = {}
            for (l, d), src in rec['H'].items():
                if l == 0:
                    const[d] = add_sources((1, const.get(d, {})), (-1, src))
            rec['M'] = {d: src for d, src in const.items() if src}
        if ws and all(sep.get(t) for _, t in ws) and all(t in prods for _, t in ws):
            prod = sp.expand(sum(w * prods[t] for w, t in ws))
        else:
            prod = None
        name = entry.rows[rmap[s]]
        if name in rows:
            assert recipes_equal(rows[name], rec), ('orbit coefficients differ', entry.name, name)
            continue
        rows[name] = rec
        products[name] = prod
    return rows, products, dict(n_matchings=ints.n_matchings, zero_lag=zero_lag, separable=sep)


def split_rows(entry, rows):
    """The IR split rows of A^lin.  The '<' rows read the recipe with the spectrum P_lt, the '>' rows are the difference
    (recorded as ('diff', full row, '<' row))."""
    out = {}
    spec = model.SPLIT_SPECTRUM['<']
    for full, lt, gt in zip(entry.rows, entry.split['<'], entry.split['>']):
        rec = rows[full]

        def respec(src):
            return {(('P', spec) if key == ('P', 'P') else key): v for key, v in src.items()}
        out[lt] = {part: {key: respec(src) for key, src in rec[part].items()} for part in ('H', 'M')}
        out[gt] = ('diff', full, lt)
    return out
