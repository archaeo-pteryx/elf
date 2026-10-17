"""Declared inputs of the LPT table generator, everything the LPT chain cannot derive, in one place.

The chain derives, from these declarations only (as :mod:`ptgen.model` is for EPT),
* the displacement kernels L^(n) (:mod:`ptgen.lpt_kernels`),
* the one-loop integrand as a list of cumulant products, its components, templates, linear kinds and angular moments
  (:mod:`ptgen.lpt_cumulant`, :mod:`ptgen.lpt_tensors`, :mod:`ptgen.lpt_components`),
* the recipes of the correlator rows of q and the rules of their discretisation (:mod:`ptgen.lpt_wick`,
  :mod:`ptgen.lpt_reduce`, :mod:`ptgen.lpt_identities`, :mod:`ptgen.lpt_rows`),
* the angular moments M_j(l) of the final integral (:mod:`ptgen.lpt_angular`) and their factored psi tables
  (:mod:`ptgen.lpt_moments`).

Every declaration below carries the reason it is declared rather than derived.  The sections are the fields and
operators, the dynamics, the bias expansion and its renormalisation, the redshift-space sources and the frame, the
truncation and the IR split, the row dictionary, the components, the angular polynomial of the density, the row recipes
(source names, preferences of representation, discretisation rules) and the conventions of the tables.
"""
from typing import Callable, NamedTuple, Optional, Tuple

import sympy as sp

from . import kernels as ept_kernels
from . import model as ept_model
from .vectors import k, norm2, p, x

R = sp.Rational
I = sp.I

# ---------------------------------------------------------------------------------------------------------------
# fields and operators

DISPLACEMENT = 'Delta'
"""Delta = Psi(q_2) - Psi(q_1), the displacement difference, and Delta^(n) its part of order n in delta_L.
Declared because the LPT power spectrum is the Fourier transform of the generating function of Delta."""

DISPLACEMENT_ORDERS = (1, 2, 3)
"""Psi = Psi^(1) + Psi^(2) + Psi^(3).  Declared because the one-loop spectrum needs the displacement to third order."""

DISPLACEMENT_SELF_CONTRACTION = 'kept'
"""A Wick contraction of two legs of one Delta^(n) is a genuine loop (the R_n-type terms of Psi^(3)).
Declared because only the kernel could say otherwise, and L^(3)(k, p, -p) does not vanish."""


class Operator(NamedTuple):
    """An elementary Lagrangian operator with its own source J_{O,a} in the generating function."""

    order: int
    """Order in the linear field delta_L (the number of linear legs)."""
    self_contraction: str
    """'none' (one leg), 'vanishes' (the kernel vanishes when two of its momenta are p and -p) or 'kept' (a genuine
    loop).  Asserted against ``kernel`` by :func:`ptgen.lpt_cumulant.self_contraction`."""
    kernel: Callable
    """The kernel K(legs) of the operator at the leg momenta (normalised like F_n, without 1/n!)."""


def _gamma3_kernel(legs):
    return ept_kernels.operator_kernel(dict(ept_model.BIAS_OPERATORS)['bGamma3'], tuple(legs))


OPERATORS = {
    'delta': Operator(1, 'none', lambda legs: sp.Integer(1)),
    'G2': Operator(2, 'vanishes', lambda legs: ept_model.K_G2(*legs)),
    'G3': Operator(3, 'vanishes', lambda legs: ept_model.K_G3(*legs)),
    'Gamma3': Operator(3, 'kept', _gamma3_kernel),
    'nabla2delta': Operator(1, 'none', lambda legs: -norm2(legs[0])),
}
"""The operators O of the paper's generating function (Eq. lpt_generating_function) with sources at the two points.
Declared because the bias expansion is a choice of the model.  The kernels are the EPT generator's (ptgen.model.K_G2,
K_G3 and the operator Gamma3 = G2[delta] - G2[theta]) read in Lagrangian space.  Products at one point (delta^2,
delta^3, delta G2) are not operators of their own, they come from repeated derivatives in D_a."""

# ---------------------------------------------------------------------------------------------------------------
# dynamics (Matsubara 2015, Eqs. 66-70, EdS fastest-growing mode)

PSI_PREFACTOR = lambda n: I / sp.factorial(n)        # noqa: E731
"""Psi^(n)(k) = (i / n!) int delta_D(k - k_1..n) L_n(k_1, .., k_n) delta(k_1) .. delta(k_n) (Matsubara 2008 Eq. 22).
Declared because it is the normalisation of the kernels of the recursion below (the paper draft omits the 1/n!)."""

LPT_SEED = "L_1(k) = k / k^2, i.e. S_1 = 1, T_1 = 0 (the Zel'dovich displacement)"
"""The first-order kernel.  Declared because it is the initial condition of the recursion."""

LPT_RECURSION = {
    'two_index': lambda n, m1, m2: 1 - sp.Rational(4 * m1 * m2, (2 * n + 3) * (n - 1)),
    'three_index': lambda n, m1, m2, m3: 1 - sp.Rational(4 * (m1 * m2 + m2 * m3 + m3 * m1), (2 * n + 3) * (n - 1)),
    'transverse': lambda n, m1, m2: sp.Integer(1),
}
"""The bracket factors of Matsubara (2015) Eqs. (67) and (68), the EdS growth of the longitudinal and transverse parts.
Declared because they are the dynamics (the Lagrangian counterpart of ptgen.model.recursion_coefficients).  The rest
of the recursion (n!/2, n!/6, the multiplicities, L_n = [k S_n + k x T_n] / k^2) is vector algebra in lpt_kernels."""

# ---------------------------------------------------------------------------------------------------------------
# the bias expansion D_a and its renormalisation


class DTerm(NamedTuple):
    """One term ``factor * (sigma^2)^sigma2_power * bias * prod_O (d/dJ_{O,a})^n_O`` of D_a."""

    bias: str
    factor: sp.Rational
    sigma2_power: int
    derivatives: Tuple[Tuple[str, int], ...]


DERIVATIVE_OPERATOR = (
    DTerm('1', R(1), 0, ()),
    DTerm('b1', R(1), 0, (('delta', 1),)),
    DTerm('b2', R(1, 2), 0, (('delta', 2),)),
    DTerm('b2', R(-1, 2), 1, ()),
    DTerm('bG2', R(1), 0, (('G2', 1),)),
    DTerm('b3', R(1, 6), 0, (('delta', 3),)),
    DTerm('b3', R(-1, 2), 1, (('delta', 1),)),
    DTerm('bG3', R(1), 0, (('G3', 1),)),
    DTerm('bdG2', R(1), 0, (('delta', 1), ('G2', 1))),
    DTerm('bdG2', R(4, 3), 1, (('delta', 1),)),
    DTerm('bGamma3', R(1), 0, (('Gamma3', 1),)),
    DTerm('bnabla2', R(1), 0, (('nabla2delta', 1),)),
)
"""D_a = 1 + b1 d_delta + (b2/2)(d_delta^2 - sigma^2) + bG2 d_G2 + (b3/6)(d_delta^3 - 3 sigma^2 d_delta) + bG3 d_G3
+ bdG2 (d_delta d_G2 + (4/3) sigma^2 d_delta) + bGamma3 d_Gamma3 + bnabla2 d_nabla2delta (the paper, after Eq.
lpt_generating_function, d_O = d/dJ_{O,a}), the same at both points (auto spectrum).  Declared because the bias
expansion with its renormalised operators is the model."""

KEPT_BIAS = ('b1', 'b2', 'bG2', 'bGamma3')
"""The bias coefficients of the result, whose monomials name the templates.  Declared because it is the model's basis
(the order is that of Params.bias)."""

RENORMALISED_BIAS = ('b3', 'bG3', 'bdG2')
"""Asserted absent from the result.  Declared because the paper states that the renormalisation removes them."""

DROPPED_BIAS = ('bnabla2',)
"""Produced by the expansion and then dropped.  Declared because the paper drops them as counterterm-like."""

RENORMALISATION = """\
The renormalised operators of F(q) are the Wick-ordered products of the elementary operators at one point, so a
joint cumulant all of whose connected Wick diagrams contract two distinct operator fields at the same Lagrangian point
(a self-pair) is dropped, and so are the sigma^2 terms of D_a ('drop' mode).  Contractions between two legs of one
field are kept, except for the operators whose kernel makes them vanish.  The 'explicit' mode keeps the sigma^2 terms
and evaluates the self-pairs with ZERO_LAG, which must give the same result."""
"""The renormalisation rule of the cumulant engine.  Declared because it is the convention of the bias expansion."""

ZERO_LAG = {
    ('delta', 'delta'): 'sigma^2, the zero-lag variance',
    ('delta', 'G2'): 'sigma^2 <K_G2> per leg of G2 (the other leg is left as a delta at the same point)',
}
"""The zero-lag contractions of two fields at one point used by the 'explicit' renormalisation mode (the value of
<K_G2> is derived from ptgen.model.K_G2).  Declared because it is the definition of the subtracted sigma^2 terms."""

# ---------------------------------------------------------------------------------------------------------------
# redshift space, the sources of the generating function and the frame

GROWTH_RATE = 'f'
"""The symbol of the growth rate.  Declared as a name."""

RSD_RULE = """Psi^{s(n)} = R^(n) Psi^(n), R^(n)_ij = delta_ij + n f n_i n_j (Eq. displacement_rsd).  With R^(n)
symmetric, i k.Delta^{s(n)} = i k_n.Delta^(n), k_n = k + n f (k.n) n."""
"""The redshift-space rule of the displacement.  Declared because it is the EdS velocity of Psi^(n) (f n at order n)."""

SOURCES = {
    'rsd': tuple((n, I, f'k{n}') for n in DISPLACEMENT_ORDERS),
}
"""The displacement terms of the exponent, (order n, coefficient, contraction vector of Delta^(n)), per observable.
Declared because the observable fixes them ('rsd' is the redshift-space density, i k_n.Delta^(n) by RSD_RULE).  The
engine takes any list, a velocity source would add terms contracted with n."""

FRAME = """The frame of the code (lpt._rsd_geometry): K = k_1 = k + f (k.n) n, |K| = k Kfac, Kfac^2 = 1 + f (2 + f) mu^2,
mu = khat.n, and the angular variables MQ = Khat.qhat and NQ = n.qhat.  Every vector is a combination of K and n."""
"""The frame of the final integral.  Declared because the angular integrals are taken with the polar axis along K."""

# ---------------------------------------------------------------------------------------------------------------
# truncation and the IR split

MAX_ORDER = 4
"""A product of cumulants is kept when the orders of all its fields add up to at most 4 (O(P_lin^2)).
Declared because the one-loop order is the model's choice."""

IR_SPLIT = """A^lin = <Delta^(1) Delta^(1)> = A^< + A^>.  The factor exp(-(1/2) K_i K_j A^<_ij) stays in the exponent
and is never expanded (its order is not counted), A^> and everything else is expanded and truncated."""
"""The IR resummation of the LPT body (the paper, after Eq. ps_1loop_lpt).  Declared because it is the model."""

LINEAR_SPECTRA = {'P': 'the linear spectrum P_lin', 'P_lt': 'P_lin exp(-(k/k_IR)^2), the long-wavelength part'}
"""The linear spectra the rows are transforms of.  Declared because the IR split cuts P_lin at k_IR (LPT k_IR)."""

SPLIT_SPECTRUM = {'<': 'P_lt'}
"""The '<' rows of the IR split read the linear spectrum P_lt, the '>' rows are A^lin - A^<.  Declared with IR_SPLIT."""

# ---------------------------------------------------------------------------------------------------------------
# the row dictionary


class Entry(NamedTuple):
    """A named cumulant of the paper and the rows of its isotropic decomposition."""

    name: str
    fields: Tuple[Tuple[str, int], ...]
    """('Delta', n) for a Delta^(n), in the order of the tensor's indices, and (operator, point) for an operator at the
    point 1 or 2."""
    rows: Tuple[str, ...]
    """One row per independent coefficient of the derived decomposition, in its order (rank 2 delta_ij, qhat_i qhat_j,
    rank 3 qhat_i delta_jk, qhat_j delta_ki, qhat_k delta_ij, qhat qhat qhat)."""
    family: str
    """The component family.  The terms whose cumulants have the same families form one component."""
    orderings: str = 'one'
    """'one' (the rows decompose this assignment of the orders to the indices) or 'all' (the sum over the distinct
    assignments, A^10_ij = <Delta_i Delta_j delta> = (12) + (21))."""
    split: Optional[dict] = None
    """The IR split of A^lin, {'<': rows, '>': rows}."""


ROW_DICTIONARY = (
    Entry('A_lin', (('Delta', 1), ('Delta', 1)), ('X_LIN', 'Y_LIN'), 'A',
          split={'<': ('X_LIN_LT', 'Y_LIN_LT'), '>': ('X_LIN_GT', 'Y_LIN_GT')}),
    Entry('A22', (('Delta', 2), ('Delta', 2)), ('X22', 'Y22'), 'A22'),
    Entry('A13', (('Delta', 1), ('Delta', 3)), ('X13', 'Y13'), 'A13'),
    Entry('W112', (('Delta', 1), ('Delta', 1), ('Delta', 2)), ('V1', 'V3', 'T'), 'W112'),
    Entry('U_lin', (('Delta', 1), ('delta', 1)), ('U_LIN',), 'U'),
    Entry('U3', (('Delta', 3), ('delta', 1)), ('U3',), 'U'),
    Entry('U11', (('Delta', 2), ('delta', 1), ('delta', 2)), ('U11',), 'U11'),
    Entry('U20', (('Delta', 2), ('delta', 1), ('delta', 1)), ('U20',), 'U20'),
    Entry('A10', (('Delta', 1), ('Delta', 2), ('delta', 1)), ('X10', 'Y10'), 'A10', orderings='all'),
    Entry('V10', (('Delta', 2), ('G2', 1)), ('V10',), 'V10'),
    Entry('V12', (('Delta', 1), ('delta', 1), ('G2', 2)), ('V12',), 'V12'),
    Entry('Upsilon', (('Delta', 1), ('Delta', 1), ('G2', 1)), ('X_UPS', 'Y_UPS'), 'Upsilon'),
    Entry('xi', (('delta', 1), ('delta', 2)), ('XI_LIN',), 'xi'),
    Entry('chi', (('delta', 1), ('delta', 1), ('G2', 2)), ('CHI',), 'chi'),
    Entry('zeta', (('G2', 1), ('G2', 2)), ('ZETA',), 'zeta'),
    Entry('U_Gamma3', (('Delta', 1), ('Gamma3', 1)), ('UB3',), 'U_Gamma3'),
    Entry('theta', (('delta', 1), ('Gamma3', 2)), ('THETA',), 'theta'),
)
"""The cumulants of the paper's appendix (sec:lpt_formulas) and the names of their rows (lpt.Row).  Declared because
the names and the grouping into families are conventions (the decompositions, V2 = V1 of W^(112) and the factor 1/2
of each A^10 ordering are derived).  A^(13) = A^(31) (each ordering is X13 delta + Y13 qhat qhat, the paper's
convention), U^10 = U^lin + U^(3) (one family 'U')."""

ROW_ORDER = ('X_LIN', 'Y_LIN', 'X_LIN_LT', 'Y_LIN_LT', 'X_LIN_GT', 'Y_LIN_GT', 'XI_LIN', 'U_LIN',
             'X22', 'Y22', 'X13', 'Y13', 'V1', 'V3', 'T', 'U3', 'U11', 'U20', 'X10', 'Y10', 'V10', 'V12',
             'X_UPS', 'Y_UPS', 'CHI', 'ZETA', 'UB3', 'THETA')
"""The q-space rows in their order (the layout of PTTerms.rows, the first 28 rows).  Declared because the layout is
the API of PTTerms."""

# ---------------------------------------------------------------------------------------------------------------
# components, templates and linear kinds (their names, the content is derived)

COMPONENT_NAMES = (
    (('A',), 'ZA'),
    (('A', 'A'), 'AA'),
    (('A22',), 'A22'),
    (('A13',), 'A13'),
    (('W112',), 'W112'),
    (('U',), 'U10'),
    (('A', 'U'), 'A_U'),
    (('A10',), 'A10'),
    (('xi',), 'XI'),
    (('A', 'xi'), 'A_XI'),
    (('U', 'U'), 'U_U'),
    (('U11',), 'U11'),
    (('U20',), 'U20'),
    (('U', 'xi'), 'XI_U'),
    (('xi', 'xi'), 'XI_XI'),
    (('Upsilon',), 'UPS'),
    (('V10',), 'V10'),
    (('V12',), 'V12'),
    (('chi',), 'CHI'),
    (('zeta',), 'ZETA'),
    (('U_Gamma3',), 'UB3'),
    (('theta',), 'THETA'),
)
"""(family product, name) of the cumulant components in their order (the columns of the template matrix).  Declared
because the names and the order are the API (lpt.Component), the content of each component is derived."""

CONSTANT_JOINS = ('A',)
"""The constant 1 joins the component of -(1/2) kk A^>, so that ZA = 1 - (1/2) kk A^> is the Zel'dovich integrand to
first order.  Declared because the grouping of the constant is a convention."""

EFT_COMPONENTS = (('CTR', 'ZA', 'CTR'),)
"""(name, copied component, template) of the components that are not cumulants.  CTR is the ZA integrand weighted by
the Zel'dovich counterterm shape (eft_terms.Counterterms.zeldovich_weight).  Declared because the counterterm is EFT."""

TEMPLATE_NAMING = """A template is a bias monomial in KEPT_BIAS, 'ONE' for 1 and otherwise the upper-case names of its
factors joined by '_' in KEPT_BIAS order (b1^2 -> B1_B1).  Templates are ordered by their exponents of (bGamma3, bG2,
b2, b1), ascending, and the EFT templates follow."""
"""Declared because the names and the order are the API (lpt.Template)."""

LINEAR_KINDS = (('ZA', ('X_LIN_GT', 'Y_LIN_GT')), ('B1', ('U_LIN',)), ('XI', ('XI_LIN',)))
"""(kind, tree rows) of the linear parts.  A component's linear kind is that of its terms first order in one tree row.
Declared because the names are the API (lpt.LinearKind)."""

# ---------------------------------------------------------------------------------------------------------------
# the angular polynomial of the density

A_mu, B_mu = sp.symbols('Am Bm', real=True)
"""The per-point scalars of the line of sight (the symbols the moments and their factorization use).  Declared as
names."""

LINE_OF_SIGHT = 'n.qhat = A_mu mu_K - B_mu mu_x, A_mu = (1 + f) mu / Kfac, B_mu = sqrt(1 - mu^2) / Kfac'
"""The line of sight in the frame of K (mu_K = Khat.qhat, mu_x along e_x = (khat - c Khat) / s).  Declared because it
fixes the per-point scalars of the moments (the sign of B_mu is that of the geometry, n.e_x = -B_mu)."""

DENSITY_MOMENTS = {
    'MQ0': ((1, 0, 0, 0),),
    'MQ1': ((1, 1, 0, 0),),
    'MQ2': ((1, 2, 0, 0),),
    'MQ3': ((1, 3, 0, 0),),
    'MQ4': ((1, 4, 0, 0),),
    'NQ1': ((A_mu, 1, 0, 0), (-B_mu, 0, 1, 0)),
    'NQ2': ((A_mu**2, 2, 0, 0), (-2 * A_mu * B_mu, 1, 1, 0), (B_mu**2, 0, 2, 0)),
    'MQ1NQ1': ((A_mu, 2, 0, 0), (-B_mu, 1, 1, 0)),
    'MQ2NQ1': ((A_mu, 3, 0, 0), (-B_mu, 2, 1, 0)),
}
"""The nine moments MQa NQb = (Khat.qhat)^a (n.qhat)^b of the density as angular polynomials, terms (coefficient, a,
b, b_y) of coefficient mu_K^a mu_x^b mu_y^b_y, in the order of utils_lpt.get_lpt_moments.  Declared because the
observable fixes which moments the final integral needs (the derived component table uses exactly these (a, b))."""

MOMENT_PHASE = """A moment of total degree n is reported real, M = (-i)^(n mod 2) <polynomial e^{i k.q} e^{B mu_K^2}>, so
a term c (Khat.qhat)^a (n.qhat)^b of the integrand enters the component table with c i^{(a+b) mod 2}."""
"""The phase convention shared by the moments (lpt_angular) and the component table (lpt_components).  Declared
because it is the convention of utils_lpt.get_lpt_moments (mq1 = -G10, mq2 = -G20, mq3 = G30, mq4 = G40)."""

LEGACY_PARITY_A = True
"""Write the parity factor c of dG/dA as A ir (2 A ir^2 (...)), as utils_lpt does, instead of c ir.  Declared because
the representation of the moments in the psi tables is a choice (the value is the same) and this one keeps the tables
the runtime was verified with byte for byte."""

# ---------------------------------------------------------------------------------------------------------------
# the row recipes, their sources, preferences of representation and discretisation rules

_kp = k * p * x
_Dm = k**2 + p**2 - 2 * k * p * x
_T = k**2 * p**2 - _kp**2

SOURCE_NAMES = {
    'Q1': ('22', (_T / (p**2 * _Dm))**2),
    'Q2': ('22', _kp * (k**2 - _kp) * _T / (p**4 * _Dm**2)),
    'Q8': ('22', _T / (p**2 * _Dm)),
    'R1': ('13', (_T / (k * p**2))**2 / _Dm),
    'R2': ('13', _kp * (k**2 - _kp) * _T / (k**2 * p**4 * _Dm)),
}
"""The names of the k-space sources, the appendix's Q_n = int_p F P(p) P(|k-p|) and R_n = P(k) int_p F P(p) by their
integrands F(k, p, x).  Declared because they are names (any basis of the span would do, these read like the paper),
the decomposition of every derived source into them is derived and unique."""

TRANSFORM = 'H_l^n[S](q) = int_0^inf dk k^{n+2} S(k) j_l(kq) / (2 pi^2),  xi_l^n = H_l^n[P]'
"""The transform of the recipes.  Declared as the convention of the runtime (lpt.LPT._transform_sources)."""

CONSTANT = 'M_n[S] = int_0^inf dk k^{n+2} S(k) / (2 pi^2) = H_0^n[S](q = 0)'
"""The zero-lag constant of the recipes.  Declared as the convention of the runtime (a grid moment)."""

UNIT_NORMALISATION = 'a stage-B unit of the 22 kind has its first coefficient 1, a 13 unit is a single block'
"""How the stage-B units (q -> k transforms) are normalised.  Declared because the normalisation of a unit is a
representation (it changes only the rounding)."""

PREFERENCES = {
    'raise_n_by_recurrence': {
        'rule': 'H_l^{n-1}[S] = q/(2l+1) (H_{l-1}^n[S] + H_{l+1}^n[S]) (an identity of j_l)',
        'rows': {'Y_LIN': ('xi', 2, -2), 'Y_LIN_LT': ('xi', 2, -2), 'T': ('H', 3, -3), 'V1': ('H', 1, -3),
                 'V3': ('H', 1, -3)},
        'reason': 'a lower n weights the low-k end of the source more strongly, where the source is cut, and the '
                  'error dominates at small q (10 to 3000 times smaller with the pair, experiments/lpt_rows_20261017)',
    },
    'product_route': {
        'rule': 'evaluate a separable 22 term as its q-space product of xi_l^n (no stage B or C)',
        'rows': ('CHI', 'ZETA', 'X_UPS', 'Y_UPS', 'V12', 'Y22', 'U20', 'V10'),
        'reason': 'no FFTlog round trip q -> k -> q, exact for the full-support (zero-padded) input',
    },
    'zero_lag_from_M22': {
        'rule': 'M13 = (70/27) M22, M10 = (28/9) M22 (full-support identities of the zero-lag constants)',
        'rows': ('X13', 'X10'),
        'reason': 'one grid moment instead of three',
    },
}
"""The representation preferences of class (c), identities that the generator proves but cannot choose between.
Declared because the choice is one of discretisation accuracy (verified numerically by the L2 prototype)."""

PRODUCT_VARIABLES = ('xi_1^-1', 'xi_3^-1')
"""A q-space product form of the xi_2^-2 family is written in (xi_1^-1, xi_3^-1), phi = xi_2^-2 = q (xi_1^-1 +
xi_3^-1) / 5 and phi' = (2 xi_1^-1 - 3 xi_3^-1) / 5.  Declared because the variables of a product are a choice."""

DERIVATIVE_RULE = 'raise'
"""d xi_l^n / dq in a product form is (l/q) xi_l^n - xi_{l+1}^{n+1} ('raise') rather than xi_{l-1}^{n+1} - ((l+1)/q)
xi_l^n ('lower').  Declared because the two are equal for the exact transforms (a choice for the discretised ones)."""

TOTAL_DERIVATIVE_ANSATZ = {'exponents': tuple(range(-4, 6)), 'monomials': 'phi^2, phi phi\', phi\'^2'}
"""The search space of a total derivative G (q^2 F = G') in the finite-range closed forms and the DC rules, G = sum c
q^e m.  Declared because an ansatz cannot be derived (it is enough for every row here)."""


class Rule(NamedTuple):
    """A discretisation rule of the runtime and the recipe kind it attaches to."""

    name: str
    applies_to: str
    action: str


RULES = (
    Rule('end_weights', 'every FFTlog input whose support ends at an interior node',
         'trapezoid end weight 1/2 (fftlog.trapezoid_weights) on the stage A input (core), the stage B integrands '
         '(cut at q_max), the 22 sources [k_min, 2 k_max], the 13 sources [k_min, k_max] and the final q integrand'),
    Rule('zero_pad', 'the stage A input', "pad_mode 'zero-pad', so P is the band-limited model and Q_n, R_n are its "
         'full integrals'),
    Rule('grid_moment', "the recipe kind M", 'M_n[S] = sum over the grid of dln k k^{n+3} S / (2 pi^2), the grid '
         'moment of the padded, end-weighted source or input'),
    Rule('low_ringing_phase', 'every transform', 'each (l, bias) uses its own low-ringing phase, stage A and C outputs '
         'are resampled to the common q grid, the 13 inputs stay on their own grids (conjugate to the k grid)'),
    Rule('bias_n0', 'the recipe kind H with n >= 0', 'kernel bias nu_n0 = 1.5 (the periodic image of the q -> 0 '
         'plateau)'),
    Rule('raise_n', 'the preference raise_n_by_recurrence', 'the n = -2 and -3 transforms are replaced by n = -1 '
         'and -2 pairs'),
    Rule('outside_core', 'a k-independent MQ0 term that is a q-space product with q^2 F = G\' (G(0) = G(inf) = 0)',
         'the k-space row -4 pi int_core dq q^2 F (the end-weighted core sum), the missing inner ball q < q_min'),
    Rule('m0_residual', 'a k-independent MQ0 term that is a q-space product with a non-zero full-q integral',
         'the k-space row H_0[F](k) - C_q, C_q the end-weighted sum of the same samples of F on the padded grid'),
    Rule('m0_constant', 'the same term as m0_residual', 'the k-space constant of F by Parseval, the k-space grid '
         'moment of the padded, end-weighted P^2, added undamped'),
    Rule('inner_ball', 'a k-independent MQ0 term given by an n >= 0 stage C transform whose source is c times that of '
         'a row R of l = 1 and n - 1', 'the k-space constant 4 pi c q_min^2 R(q_min), from (q^2 R)\' = c q^2 F'),
    Rule('q_max_subtraction', 'a k-independent constant of the MQ0 integrand', 'the final integral subtracts the value '
         'of the l = 0 integrand at q_max (the constant transforms to delta_D(k))'),
    Rule('linear_part', 'a term first order in one tree row (a linear kind of components.dat)', 'the final integral '
         'subtracts it weighted by D = exp(-K^2 (X_lt + Y_lt)(q_max) / 2) and adds back D x Kaiser x P_lin(k) '
         '(lpt.LPT._linear_rows)'),
)
"""The rules of the numerical model and their attachment criteria.  Declared because they are the discretisation (the
runtime implements each, the generator attaches them to the recipes and the components)."""


class KSpaceRow(NamedTuple):
    """A k-space row of PTTerms.rows (after the 28 q-space rows)."""

    name: str
    rule: str
    function: str
    """The q-space function F the rule acts on (a row, or a product of xi of P for m0_residual and P^2 for the
    k-space constant)."""
    interpolated: bool
    """The final integral interpolates the row in ln k at the true k (else it reads the constant at k_min)."""
    k0_const: bool
    """The row is the k -> 0 constant that subtract_k0_const removes."""


KSPACE_ROWS = (
    KSpaceRow('CHI_OUTSIDE', 'outside_core', 'CHI', True, False),
    KSpaceRow('ZETA_OUTSIDE', 'outside_core', 'ZETA', True, False),
    KSpaceRow('B2SQ_RESIDUAL', 'm0_residual', 'XI_LIN^2/2', True, False),
    KSpaceRow('B2SQ_CONST', 'm0_constant', 'XI_LIN^2/2', False, True),
    KSpaceRow('THETA_BALL', 'inner_ball', 'THETA', False, False),
)
"""The names of the k-space rows, the function each rule acts on and how the final integral reads them.  Declared
because the names and the layout are the API (rows 28 .. 32), the attachment to the components is derived.  The two
outside_core rows are constants in k but read as interpolated rows, as the runtime was verified with."""

# ---------------------------------------------------------------------------------------------------------------
# conventions of the tables

FACTOR_VARIABLES = ('k', 'mu', 'f', 'Kfac', 'K', 'Ksq')
"""The variables of the per-point factors s_e of component_entries.dat (K = k Kfac, Ksq = K^2).  Declared because
they are the arguments of the runtime's lpt_rule.fold_weights."""

SCALAR_FORM = 'elf'
"""How a per-point factor (a polynomial in k, mu, f, Kfac) is written in FACTOR_VARIABLES.  'elf' writes every
monomial k^a Kfac^a as K^(a mod 2) Ksq^(a div 2) except in the displacement-only one-loop cumulants (A22, A13,
W112), which stay in k and Kfac.  Declared because the form is a representation (the values are equal) and this one
keeps component_entries.dat byte for byte."""

ENTRY_ORDER = """component_entries.dat lists the components in COMPONENT_NAMES order (then the EFT components), within
a component the moments by total degree a + b and then by b, within a moment the basis rows in descending lex order of
their exponents over ROW_ORDER, within an entry the monomials in increasing exponent order.  basis_rows.dat lists the
constant row, the single rows in ROW_ORDER and then the products."""
"""Declared because the order is the summation order of the runtime (forward and tangent rule)."""

HORNER_BASIS = ('S0', 'T1', 'T2', 'T3', 'T4')
"""The basis of the moments M_j(l) in l, S0 = V[0] and T_d = S_d / s^2 of utils_lpt.lpt_horner_poly.  Declared
because it is the basis the runtime evaluates (pure A derivatives are written in S, the others in T)."""


def moment_name(a, b):
    """``MQa``, ``NQb`` or ``MQaNQb`` of (Khat.qhat)^a (n.qhat)^b."""
    if b == 0:
        return f'MQ{a}'
    return (f'MQ{a}' if a else '') + f'NQ{b}'
