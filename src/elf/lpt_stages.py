"""The FFTlog stages of the LPT rows, driven by the recipes of tables/lpt/rows_*.dat (``lpt_tables``).

``pt_rows(model, pk_lin)`` returns the rows of PTTerms (lpt_tables.Row, ``(n_rows, nfft)``) of the linear spectrum
``pk_lin`` on the core k grid in one pass over the static plan built here from the tables.

  A  ``model._get_xi_ln`` transforms the spectra P and P_lt = P exp(-(k/k_IR)^2) for the (l, n) of ``STAGE_A`` (the
     model's ln_list) and gives xi_l^n on the common padded q grid and on the own grid of each order.
  B  the units of rows_sources.dat.  The 22 units, sum c (xi_l^n)^2 of P on the common grid, are stacked and
     transformed q -> k at once (j_0, end weights of [k_min, 2 k_max]).  The 13 units, xi_l^n of P on their own grids,
     are transformed at once with the kernel of each order and multiplied by P (end weights of [k_min, k_max]).  A
     source is the polynomial S(k) = sum_p k^p (sum over its lines of kpow p of coef U).
  C  the H terms H_l^n[S] of the recipes (``STAGE_C``).  Equal sources up to a factor share one transform.  Every
     term's source is scaled to a first coefficient 1, and the terms with the same scaled source and n become one
     transform with the tuple of their orders.  Its input is the scaled source times the coefficient of its first
     term (``TRANSFORM_FACTORS``), so that this term enters its row with coefficient 1.  One call of
     ``model._transform_sources`` (one rfft per input, one irfft per order) takes the transforms with the same bias
     group (n < 0 or n >= 0, the rule bias_n0), the same number of orders and the same need of the 13 units.  The
     last key keeps an input made of 22 units only out of the calls that wait for the 13 units.  On the present
     tables this gives the four calls of the hand-written stages, 10 rffts and 14 irffts.
  M  the zero-lag constants M_n[S], the grid moments of the end-weighted sources (or of the padded, end-weighted
     spectra), shared in the same way.
Then ``_Context.row`` makes the rows from the products of stage-A outputs of P on the core q grid, the powers of q,
the references to other rows (resolved on demand, each row once) and the rules of the k-space rows (``RULE_HOOKS``,
one per rule of rows_rules.dat that a recipe uses).

A q-space row is the sum of its groups (``ROW_GROUPS``).  The terms with the same power p of q and the same absolute
coefficient form one group, c q^p (v_1 +- v_2 +- ..) with c the coefficient of its first term and c q^p one constant
array.  The values of the terms are taken in recipe order before the groups are summed, a referenced row being
evaluated there.

Why the factors sit in the inputs and the terms in groups.  XLA fuses the arithmetic of the rows, and the resampling
of the stage-C outputs of one order, into the per-point loops of the final integral (the forward, the tangent rule and
its transpose), where it is repeated at every (k, mu).  A multiplication in k space before stage C is made once per
call, one in q space once per point.  The calls follow the stage-B outputs they need for the same reason, the rows
and with them the final integral wait for no transform they do not use.  The hand-written stages had both
properties, and the timing of the full model is sensitive to them.
"""
from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp

from . import fftlog
from .lpt_tables import RECIPES, ROW_NAMES, RULES, SOURCES, SPECTRA, UNITS

# ---------------------------------------------------------------------------------------------------------------
# the static plan


def xi_factors(product):
    """[(l, n, power)] of a product of stage-A outputs written 'xi_1_m1^2*xi_3_m1' ('m' a minus sign)."""
    out = []
    for factor in product.split('*'):
        name, _, power = factor.partition('^')
        _, l, n = name.split('_')
        out.append((int(l), int(n.replace('m', '-')), int(power) if power else 1))
    return out


def _stage_a():
    """The (l, n) of the stage-A transforms the recipes and the units need, by l and then by n descending."""
    need = set()
    for terms in RECIPES.values():
        for t in terms:
            if t.kind == 'xi':
                need.add((t.l, t.n))
            elif t.kind == 'prod' or (t.kind in RULES and t.refs[0][1].startswith('xi_')):
                need |= {(l, n) for l, n, _ in xi_factors(t.refs[0][1])}
    for _, n, lines in UNITS.values():
        need |= {(l, n) for l, _ in lines}
    return tuple(sorted(need, key=lambda ln: (ln[0], -ln[1])))


STAGE_A = _stage_a()
"""The (l, n) of stage A, the model's ln_list."""
UNITS_22 = tuple(u for u, (kind, _, _) in UNITS.items() if kind == '22')
UNITS_13 = tuple(u for u, (kind, _, _) in UNITS.items() if kind == '13')
UNITS_13_LN = tuple((UNITS[u][2][0][0], UNITS[u][1]) for u in UNITS_13)
"""The (l, n) of the 13 units in table order (the rows of ln_list the model transforms back to k)."""
assert all(len(UNITS[u][2]) == 1 and UNITS[u][2][0][1] == 1 for u in UNITS_13), 'a 13 unit is one block'
SOURCE_POWERS = {name: tuple((p, tuple((u, c) for u, kp, c in lines if kp == p))
                             for p in sorted({kp for _, kp, _ in lines})) for name, lines in SOURCES.items()}
"""{source: ((kpow, ((unit, coef), ..)), ..)}, S = sum_p k^p sum coef U."""


def scaled(refs):
    """``(factor, source)`` of a combination ((coef, name), ..), the source ((name, coef / factor), ..) having its first
    coefficient 1 (the key under which equal transforms and constants are shared)."""
    factor = refs[0][0]
    return factor, tuple((name, c / factor) for c, name in refs)


class Transform(NamedTuple):
    source: tuple        # a scaled source, ((name, Fraction), ..)
    n: int
    orders: tuple        # the orders l of the H terms of this source and n, ascending
    factor: object       # the input is factor x source, the coefficient of the first H term of (source, n)


def _needs_13(source):
    """Whether a scaled source is made of 13 units (else of 22 units only)."""
    return any(UNITS[u][0] == '13' for name, _ in source for u, _, _ in SOURCES[name])


def _stage_c():
    """``{(bias group, number of orders, needs the 13 units): (Transform, ..)}`` of the H terms, in order of first
    appearance (the calls of ``model._transform_sources``, module docstring C)."""
    orders, factors = {}, {}
    for terms in RECIPES.values():
        for t in terms:
            if t.kind == 'H':
                factor, source = scaled(t.refs)
                orders.setdefault((source, t.n), set()).add(t.l)
                factors.setdefault((source, t.n), factor)
    batches = defaultdict(list)
    for (source, n), ls in orders.items():
        batches[(int(n >= 0), len(ls), _needs_13(source))].append(
            Transform(source, n, tuple(sorted(ls)), factors[(source, n)]))
    return {key: tuple(v) for key, v in batches.items()}


STAGE_C = _stage_c()
TRANSFORM_FACTORS = {(t.source, t.n): t.factor for transforms in STAGE_C.values() for t in transforms}
"""{(scaled source, n): the factor of the transform's input}, the coefficient of its first H term."""
CONSTANTS = tuple(dict.fromkeys((scaled(t.refs)[1], t.n) for terms in RECIPES.values() for t in terms
                                if t.kind == 'M'))
"""The distinct (scaled source or spectrum, n) of the M terms."""


def _coefficient(t):
    """The coefficient of a q-space term in front of its value (``_Context.value``).  For an H term it is the factor
    of ``scaled`` divided by the factor already in the transform's input, 1 for the first term of the transform."""
    if t.kind == 'H':
        factor, source = scaled(t.refs)
        return factor / TRANSFORM_FACTORS[(source, t.n)]
    return scaled(t.refs)[0] if t.kind == 'M' else t.refs[0][0]


def _row_groups(row):
    """``((power of q, coefficient c, ((sign, term index), ..)), ..)`` of a q-space row (module docstring).  The groups
    are the terms with the same power of q and the same |coefficient| in order of first appearance, c is the
    coefficient of the first and sign (+1 or -1) the ratio of a term's coefficient to c."""
    groups = {}
    for i, t in enumerate(RECIPES[row]):
        c, p = _coefficient(t), t.qpow or 0
        groups.setdefault((p, abs(c)), []).append((c, i))
    return tuple((p, members[0][0], tuple((1 if c == members[0][0] else -1, i) for c, i in members))
                 for (p, _), members in groups.items())


assert all(all(t.kind in RULES for t in terms) or not any(t.kind in RULES for t in terms)
           for terms in RECIPES.values()), 'rows_recipes.dat mixes rules and q-space terms in one row'
ROW_GROUPS = {row: _row_groups(row) for row, terms in RECIPES.items() if not all(t.kind in RULES for t in terms)}
"""{q-space row: its groups (``_row_groups``)}, the k-space rows (rules only) excluded."""

# ---------------------------------------------------------------------------------------------------------------
# the stages


def _times(c, x):
    """``c x`` for a Fraction c, without the exact multiplications by 1 and -1."""
    return x if c == 1 else (-x if c == -1 else float(c) * x)


def _combination(source, arrays):
    """sum over the scaled source ((name, coef), ..) of coef arrays[name]."""
    total = None
    for name, c in source:
        term = _times(c, arrays[name])
        total = term if total is None else total + term
    return total


def stage_b(model, xi_common, xi_own, pk_lin):
    """``{unit: (n_padded,) array}`` of rows_sources.dat on the padded k grid (module docstring, B)."""
    hg, st, nu = model._hankel, model._source_transform, model.fftlog_settings.nu
    units = {}
    integrands = []
    for u in UNITS_22:
        terms = [_times(c, xi_common[l, UNITS[u][1]]**2) for l, c in UNITS[u][2]]
        integrands.append(sum(terms[1:], terms[0]))
    res = fftlog.hankel(nu, st.four_pi_q3_padded * (jnp.stack(integrands) * hg.w_xi), model._q_padded, hg.k_padded,
                        st.u_m[0], hg.npad, crop=False) * st.end_weights_Q
    units.update(zip(UNITS_22, res))
    ells = st.R_source_ells
    q_l = hg.q_padded[ells]
    xis = xi_own[st.R_source_rows] * hg.w_xi
    res = fftlog.hankel(nu, q_l**2 * xis, q_l, hg.k_padded, hg.u_m[ells], hg.npad, crop=False) * (
        fftlog.pad(pk_lin, hg.grid, 'zero-pad') * st.end_weights_R)
    units.update(zip(UNITS_13, res))
    return units


def sources(model, units):
    """``{source: (n_padded,) array}``, S(k) = sum_p k^p (sum coef U) on the padded k grid."""
    k = model._hankel.k_padded
    out = {}
    for name, powers in SOURCE_POWERS.items():
        total = None
        for p, lines in powers:
            part = _combination(tuple((u, c) for u, c in lines), units)
            part = k**p * part if p else part
            total = part if total is None else total + part
        out[name] = total
    return out


def stage_c(model, source_arrays):
    """``{(scaled source, n, l): (nq,) array}``, H_l^n[S] on the core q grid (module docstring, C)."""
    crop = model._hankel.grid.crop
    out = {}
    for (_, n_orders, _), transforms in STAGE_C.items():
        arrays = [_combination(tuple((name, t.factor * c) for name, c in t.source), source_arrays) for t in transforms]
        ns = [t.n for t in transforms]
        if n_orders == 1:
            res = crop(model._transform_sources(arrays, ells=[t.orders[0] for t in transforms], ns=ns))
            out.update({(t.source, t.n, t.orders[0]): r for t, r in zip(transforms, res)})
        else:
            res = crop(model._transform_sources(arrays, ells=[t.orders for t in transforms], ns=ns))
            out.update({(t.source, t.n, l): r_l for t, r in zip(transforms, res) for l, r_l in zip(t.orders, r)})
    return out


def constants(model, source_arrays, spectra):
    """``{(scaled source, n): scalar}``, M_n[S] = grid moment of the end-weighted source (module docstring, M)."""
    hg = model._hankel
    arrays = dict(source_arrays)
    arrays.update({name: hg.pad_input(spectra[name]) for name in SPECTRA})
    return {(source, n): hg.grid.moment(_combination(source, arrays), n + 3) / (2 * jnp.pi**2)
            for source, n in CONSTANTS}


def product(xi, ref):
    """The product of stage-A outputs ``ref`` ('xi_1_m1^2*xi_3_m1') of the arrays xi[l, n]."""
    total = None
    for l, n, power in xi_factors(ref):
        factor = xi[l, n]**power if power > 1 else xi[l, n]
        total = factor if total is None else total * factor
    return total


# ---------------------------------------------------------------------------------------------------------------
# the rules of the k-space rows (rows_rules.dat)

def outside_core(model, coef, ref, ctx):
    """-4 pi int_core dq q^2 F of the row F = ref (the end-weighted core sum), constant in k."""
    q_core = fftlog.LogGrid(model._q)
    value = -4 * jnp.pi * q_core.moment(model._final_transform.end_weights_q_core * ctx.row(ref), 3)
    return jnp.broadcast_to(_times(coef, value), model._k.shape)


def m0_residual(model, coef, ref, ctx):
    """H_0[F](k) - C_q of F = coef ref (a product of stage-A outputs of P on the padded grid), C_q its k -> 0 limit,
    the end-weighted sum of the same samples."""
    hg, st = model._hankel, model._source_transform
    source = _times(coef, product(ctx.xi_padded['P'], ref))
    direct = fftlog.hankel(model.fftlog_settings.nu, st.four_pi_q3_padded * source * hg.w_xi, model._q_padded,
                           hg.k_padded, st.u_m[0], hg.npad, crop=True)
    return direct - 4 * jnp.pi * hg.q_grids[0].moment(source * hg.w_xi, 3)


def m0_constant(model, coef, ref, ctx, n=0):
    """coef M_n[P^2], the k-space grid moment of the padded, end-weighted P^2 (Parseval), constant in k."""
    hg = model._hankel
    assert ref == 'P^2', ref
    value = float(coef) * hg.grid.moment(hg.w_input * fftlog.pad(ctx.spectra['P'], hg.grid, hg.pad_mode)**2, n + 3)
    return jnp.broadcast_to(value / (2 * jnp.pi**2), model._k.shape)


def inner_ball(model, coef, ref, ctx):
    """4 pi coef q_min^2 R(q_min) of the row R = ref (the ball q < q_min of a total derivative), constant in k."""
    return jnp.broadcast_to(4 * jnp.pi * float(coef) * model._q[0]**2 * ctx.row(ref)[0], model._k.shape)


RULE_HOOKS = {'outside_core': outside_core, 'm0_residual': m0_residual, 'm0_constant': m0_constant,
              'inner_ball': inner_ball}
"""The runtime of the rules that make k-space rows (rows_rules.dat)."""
_missing = {t.kind for terms in RECIPES.values() for t in terms if t.kind in RULES} - set(RULE_HOOKS)
assert not _missing, f'rows_recipes.dat uses rules without a runtime hook: {sorted(_missing)}'

# ---------------------------------------------------------------------------------------------------------------
# the rows


class _Context:
    """The stage outputs of one call and the rows evaluated so far (``row`` resolves references on demand)."""

    def __init__(self, model, spectra, xi_padded, xi_core, H, M):
        self.model, self.spectra, self.xi_padded, self.xi_core, self.H, self.M = model, spectra, xi_padded, xi_core, H, M
        self.q = model._q
        self.values = {}

    def value(self, t):
        """The value of a q-space term without its coefficient and power of q (``_coefficient``, ``t.qpow``)."""
        if t.kind == 'xi':
            return self.xi_core[t.refs[0][1]][t.l, t.n]
        if t.kind == 'H':
            return self.H[(scaled(t.refs)[1], t.n, t.l)]
        if t.kind == 'M':
            return self.M[(scaled(t.refs)[1], t.n)]
        if t.kind == 'prod':
            return product(self.xi_core['P'], t.refs[0][1])
        return self.row(t.refs[0][1])

    def rule(self, t):
        """A term of a k-space row, its rule of rows_rules.dat applied to its reference."""
        (c, ref), = t.refs
        return RULE_HOOKS[t.kind](self.model, c, ref, self, **({} if t.n is None else {'n': t.n}))

    def row(self, name):
        if name not in self.values:
            terms = RECIPES[name]
            total = None
            if name in ROW_GROUPS:
                values = [self.value(t) for t in terms]          # in recipe order, references evaluated here
                for p, c, members in ROW_GROUPS[name]:
                    s = None
                    for sign, i in members:
                        s = values[i] if s is None else (s + values[i] if sign > 0 else s - values[i])
                    value = float(c) * self.q**p * s if p else _times(c, s)
                    total = value if total is None else total + value
            else:
                for t in terms:
                    value = self.rule(t)
                    total = value if total is None else total + value
            self.values[name] = total
        return self.values[name]


def pt_rows(model, pk_lin):
    """PTTerms.rows ``(n_rows, nfft)`` of the linear spectrum ``pk_lin`` on the core k grid (module docstring)."""
    hg = model._hankel
    spectra = {'P': pk_lin, 'P_lt': pk_lin * jnp.exp(-(model._k / model.k_IR)**2)}
    xi_padded, xi_own = model._get_xi_ln(jnp.stack([spectra[s] for s in SPECTRA], axis=0))
    padded = {s: xi_padded[i] for i, s in enumerate(SPECTRA)}
    core = {s: hg.grid.crop(xi_padded[i]) for i, s in enumerate(SPECTRA)}
    source_arrays = sources(model, stage_b(model, padded['P'], xi_own[SPECTRA.index('P')], pk_lin))
    ctx = _Context(model, spectra, padded, core, stage_c(model, source_arrays),
                   constants(model, source_arrays, spectra))
    return jnp.stack([ctx.row(name) for name in ROW_NAMES], axis=0)
