"""The independent references of the LPT generator, each typed once (never an input of the derivation).

* The paper's one-loop integrand, Eqs. (pkmu_lpt_tree) and (pkmu_lpt_1loop) of JAX_oneloop/docs/jax_eft_power_spectrum.tex
  (lines 507-523), as (coefficient, bias monomial, [paper objects contracted with k]), the paper's objects as joint
  cumulants of order-resolved fields (appendix sec:lpt_formulas), and the exponent -(1/2) A^<(k1, k1).
* The closed forms of the displacement kernels, L^(2) = (3/7)(k/k^2)(1 - mu_12^2), Matsubara (2008) Eqs. 25-26 for the
  longitudinal L^(3), and Matsubara (2015) Eqs. B8-B10 (S_2, S_3, T_3) of the recursion.
* The appendix's sources Q1, Q2, Q5, Q8, QG2, R1, R2, F_G2 (Eqs. Q1 .. R2, FG2) and its rows in them (Sec. Cumulants
  and Eq. cumulants_H).
* The hand code's q-space product forms (lpt._get_corrs_tree, _get_corrs_matter_1loop, _get_corrs_bias of elf before
  the table-driven runtime), the paper's ladder (Eqs. xi_l0_from_xi2, xi_ladder) and the hand code's rule attachments.
"""
import sympy as sp

from pt_autogen.lpt_identities import PHI
from pt_autogen.lpt_reduce import QS, xi
from pt_autogen.vectors import dot, k, norm2, p, x

I, R = sp.I, sp.Rational

# ---------------------------------------------------------------------------------------------------------------
# the paper's one-loop integrand (in redshift space every Delta^(n) is contracted with k_n)

PAPER_EXPONENT = [(-R(1, 2), 'A<')]

PAPER_TREE = [
    (1, (), []),
    (-R(1, 2), (), ['A>']),
    (2 * I, ('b1',), ['Ulin']),
    (1, ('b1', 'b1'), ['xi']),
]

PAPER_1LOOP = [
    (-R(1, 2), (), ['A1loop']),
    (-I / 6, (), ['W']),
    (R(1, 8), (), ['A>', 'A>']),
    (2 * I, ('b1',), ['U3']),
    (2 * I * (-R(1, 2)), ('b1',), ['A>', 'Ulin']),
    (-R(1, 2), ('b1', 'b1'), ['A>', 'xi']),
    (I, ('b2',), ['U20']),
    (I, ('b1', 'b1'), ['U11']),
    (-1, ('b1',), ['A10']),
    (-1, ('b1', 'b1'), ['Ulin', 'Ulin']),
    (-1, ('b2',), ['Ulin', 'Ulin']),
    (R(1, 2), ('b2', 'b2'), ['xi', 'xi']),
    (2 * I, ('b1', 'b2'), ['xi', 'Ulin']),
    (-1, ('bG2',), ['Ups']),
    (2 * I, ('bG2',), ['V10']),
    (2 * I, ('b1', 'bG2'), ['V12']),
    (1, ('b2', 'bG2'), ['chi']),
    (1, ('bG2', 'bG2'), ['zeta']),
    (2 * I, ('bGamma3',), ['UGamma3']),
    (2, ('b1', 'bGamma3'), ['theta']),
]


def D(n):
    """Delta^(n) contracted with k_n, as (kind, point, order, vector)."""
    return ('Delta', 0, n, f'k{n}')


ORDERS = {'delta': 1, 'G2': 2, 'Gamma3': 3}


def O(kind, point):
    return (kind, point, ORDERS[kind], '')


PAPER_OBJECTS = {
    'A<': [('<', [D(1), D(1)])],
    'A>': [('>', [D(1), D(1)])],
    'A1loop': [('', [D(2), D(2)]), ('', [D(1), D(3)]), ('', [D(3), D(1)])],
    'W': [('', [D(1), D(1), D(2)]), ('', [D(1), D(2), D(1)]), ('', [D(2), D(1), D(1)])],
    'Ulin': [('', [D(1), O('delta', 1)])],
    'U3': [('', [D(3), O('delta', 1)])],
    'U20': [('', [D(2), O('delta', 1), O('delta', 1)])],
    'U11': [('', [D(2), O('delta', 1), O('delta', 2)])],
    'A10': [('', [D(1), D(2), O('delta', 1)]), ('', [D(2), D(1), O('delta', 1)])],
    'Ups': [('', [D(1), D(1), O('G2', 1)])],
    'V10': [('', [D(2), O('G2', 1)])],
    'V12': [('', [D(1), O('delta', 1), O('G2', 2)])],
    'xi': [('', [O('delta', 1), O('delta', 2)])],
    'chi': [('', [O('delta', 1), O('delta', 1), O('G2', 2)])],
    'zeta': [('', [O('G2', 1), O('G2', 2)])],
    'UGamma3': [('', [D(1), O('Gamma3', 1)])],
    'theta': [('', [O('delta', 1), O('Gamma3', 2)])],
}
"""The paper's objects as joint cumulants of order-resolved fields, A^1-loop = A^(22) + A^(13) + A^(31), W = W^(112) +
W^(121) + W^(211), A^10_ij = <Delta_i Delta_j delta> (both orderings), U^20 = <Delta delta^2>, U^11 = <Delta delta(q1)
delta(q2)>, ...  Typed independently of lpt_model.ROW_DICTIONARY."""

N_KEPT_TERMS = 25
N_DROPPED_BNABLA2_TERMS = 15
N_EXPLICIT_CANCELLED = 11
SELFPAIR_KINDS = {(('delta', 1), ('delta', 1)), (('Delta', 0), ('G2', 1), ('delta', 1)),
                  (('G2', 1), ('delta', 1), ('delta', 2)), (('G2', 1), ('delta', 1), ('nabla2delta', 2))}
"""The counts and the self-pair cumulants of the L1 prototype (experiments/lpt_cumulant_20261016)."""

# ---------------------------------------------------------------------------------------------------------------
# closed forms of the displacement kernels


def U_angle(a, b):
    """1 - (ahat.bhat)^2."""
    return 1 - dot(a, b)**2 / (norm2(a) * norm2(b))


def L2_closed(a, b):
    """L^(2)(a, b) = (3/7) (a + b) / |a + b|^2 (1 - (ahat.bhat)^2), as (vector, coefficient)."""
    return a + b, R(3, 7) * U_angle(a, b) / norm2(a + b)


def L3a(a, b, c):
    """Matsubara (2008) Eq. 25, k.L^(3a) (the longitudinal part)."""
    mu2 = dot(a, b)**2 / (norm2(a) * norm2(b))
    trip = dot(a, b) * dot(b, c) * dot(c, a) / (norm2(a) * norm2(b) * norm2(c))
    return R(5, 7) * U_angle(a, b) * U_angle(a + b, c) - R(1, 3) * (1 - 3 * mu2 + 2 * trip)


def S3_appendix_B(a, b, c):
    """Matsubara (2015) Eq. B9, the unsymmetrised S_3(k1, k2, k3)."""
    Vg = (1 - dot(a, b)**2 / (norm2(a) * norm2(b)) - dot(b, c)**2 / (norm2(b) * norm2(c))
          - dot(c, a)**2 / (norm2(c) * norm2(a)) + 2 * dot(a, b) * dot(b, c) * dot(c, a)
          / (norm2(a) * norm2(b) * norm2(c)))
    return R(5, 3) * U_angle(a, b + c) * R(3, 7) * U_angle(b, c) - R(1, 3) * Vg


# ---------------------------------------------------------------------------------------------------------------
# the appendix's sources and rows

_kp = k * p * x
_Dm = k**2 + p**2 - 2 * k * p * x
_T = k**2 * p**2 - _kp**2

INTEGRANDS_22 = {
    'Q1': (_T / (p**2 * _Dm))**2,
    'Q2': _kp * (k**2 - _kp) * _T / (p**4 * _Dm**2),
    'Q5': _kp * _T / (p**4 * _Dm),
    'Q8': _T / (p**2 * _Dm),
    'QG2': (p**2 - _kp) * _T / (p**4 * _Dm**2),
}
INTEGRANDS_13 = {
    'R1': (_T / (k * p**2))**2 / _Dm,
    'R2': _kp * (k**2 - _kp) * _T / (k**2 * p**4 * _Dm),
    # F_G2 / P(k) = 4 int_p K_G2(k - p, p) F_2(k, -p) P(p), F_2(k, -p) = 5/7 - (x/2)(k/p + p/k) + (2/7) x^2
    'FG2': 4 * ((_kp - p**2)**2 / (_Dm * p**2) - 1) * (R(5, 7) - x / 2 * (k / p + p / k) + R(2, 7) * x**2),
}

PAPER_ROWS = {
    'X22': [(0, -2, {'Q1': -R(2, 3) * R(9, 98)}), (2, -2, {'Q1': -R(2, 3) * R(9, 98)}),
            ('M', -2, {'Q1': R(2, 3) * R(9, 98)})],
    'Y22': [(2, -2, {'Q1': 2 * R(9, 98)})],
    'X13': [(0, -2, {'R1': -R(2, 3) * R(5, 21)}), (2, -2, {'R1': -R(2, 3) * R(5, 21)}),
            ('M', -2, {'R1': R(2, 3) * R(5, 21)})],
    'Y13': [(2, -2, {'R1': 2 * R(5, 21)})],
    'T': [(3, -3, {'Q1': -R(3, 7), 'Q2': -R(6, 7), 'R1': -R(6, 7), 'R2': -R(12, 7)})],
    'V1': [(1, -3, {'Q1': R(3, 35), 'Q2': R(6, 35), 'R1': -R(9, 35), 'R2': R(12, 35)}),
           (3, -3, {'Q1': R(3, 35), 'Q2': R(6, 35), 'R1': R(6, 35), 'R2': R(12, 35)})],
    'V3': [(1, -3, {'Q1': -R(12, 35), 'Q2': R(6, 35), 'R1': R(6, 35), 'R2': R(12, 35)}),
           (3, -3, {'Q1': R(3, 35), 'Q2': R(6, 35), 'R1': R(6, 35), 'R2': R(12, 35)})],
    'U3': [(1, -1, {'R1': -R(5, 21)})],
    'U11': [(1, -1, {'R1': -R(6, 7), 'R2': -R(6, 7)})],
    'U20': [(1, -1, {'Q8': -R(3, 7)})],
    'X10': [('M', -2, {'R1': R(2, 7)}), (0, -2, {'Q5': -R(2, 7), 'R2': -R(4, 7)}),
            (2, -2, {'Q5': -R(2, 7), 'R1': -R(3, 7), 'R2': -R(4, 7)})],
    'Y10': [(2, -2, {'Q5': R(6, 7), 'R1': R(9, 7), 'R2': R(12, 7)})],
    'V10': [(1, -1, {'Q1': R(3, 7)})],
    'V12': [(1, -1, {'Q5': 2})],
    'X_UPS': [(0, -2, {'Q1': R(1, 3), 'Q2': -R(1, 3), 'QG2': -1}), (2, -2, {'Q1': R(1, 3), 'Q2': R(2, 3)})],
    'Y_UPS': [(0, -2, {'Q1': -1, 'Q2': 1, 'QG2': 1}), (2, -2, {'Q1': -1, 'Q2': -2})],
    'UB3': [(1, -1, {'FG2': -R(2, 5)})],
    'THETA': [(0, 0, {'FG2': R(2, 5)})],
}
"""The appendix's rows as {row: [(l, n, {source: coefficient}) or ('M', n, {..})]}.  V1, V3 are the H_1^{-3} bracket plus
-(1/5) T, T's source being -(3/7)(Q1 + 2Q2 + 2R1 + 4R2).  X_UPS, Y_UPS are Eqs. (X_G2), (Y_G2) with QG2 entering as
k^2 QG2.  The paper's X_UPS has no constant (X_G2 -> 0 at q = 0 needs one, so it is compared without it).  UB3 and
THETA use F_G2 = 4 P(k) int_p K_G2(k - p, p) F_2(k, -p) P(p), Eq. (FG2)."""

DEPENDENT_SOURCES = {'Q5': {'Q1': R(1, 2), 'Q2': R(1, 2)}, 'QG2': {'Q1': 1, 'Q2': -1}, 'FG2': {'R1': R(-20, 7)}}
"""Q5 = (Q1 + Q2)/2, k^2 QG2 = Q1 - Q2, F_G2 = -(20/7) R1 (the L2 prototype's derived dependencies)."""

CONSTANTS_BY_FULL_K_IDENTITY = {'X10', 'X_UPS'}
"""The rows whose derived zero-lag constant equals the paper's through a full-k identity (not term by term)."""

# ---------------------------------------------------------------------------------------------------------------
# the hand code's q-space product forms and stage A

_x = xi
_Ylin_hand = 2 * QS / 5 * (_x(1, -1) + _x(3, -1))
_V10_hand = R(24, 35) * (_x(1, -1)**2 - _x(3, -1)**2) / QS
_J2 = R(2, 15) * _x(1, -1) - R(1, 5) * _x(3, -1)
_J3 = -R(1, 5) * _x(1, -1) - R(1, 5) * _x(3, -1)
HAND_PRODUCTS = {
    'Y22': R(18, 49) * _Ylin_hand**2 / QS**2,
    'U20': -R(6, 7) * _x(1, -1)**2 / QS,
    'V10': _V10_hand,
    'V12': R(14, 3) * _V10_hand - R(2, 5) * (2 * _x(3, -1) * (3 * _x(3, -1) / QS - _x(4, 0))
                                             - 2 * _x(1, -1) * (_x(1, -1) / QS - _x(2, 0))),
    'X_UPS': 4 * _J3**2,
    'Y_UPS': 12 * _J2**2 - 4 * _J3**2 - R(4, 3) * _x(1, -1)**2,
    'CHI': R(4, 3) * (_x(2, 0)**2 - _x(0, 0)**2),
    'ZETA': 2 * (R(8, 15) * _x(0, 0)**2 - R(16, 21) * _x(2, 0)**2 + R(8, 35) * _x(4, 0)**2),
}
"""The hand code's product forms (the rows whose kernel is separable or of the Gauss-law kind)."""

SEPARABLE_ROWS = ('CHI', 'ZETA', 'X_UPS', 'Y_UPS')
GAUSS_LAW_ROWS = ('Y22', 'U20', 'V10', 'V12')

PAPER_LADDER = {(1, -1): PHI[1] + 3 * PHI[0] / QS, (3, -1): -PHI[1] + 2 * PHI[0] / QS,
                (0, 0): PHI[2] + 5 * PHI[1] / QS + 3 * PHI[0] / QS**2,
                (2, 0): -PHI[2] - 2 * PHI[1] / QS + 6 * PHI[0] / QS**2,
                (4, 0): PHI[2] - 5 * PHI[1] / QS + 8 * PHI[0] / QS**2}
"""The paper's Eq. xi_l0_from_xi2 (and Eq. xi_ladder for n = -1), phi = xi_2^-2."""

HAND_ATTACHMENTS = {('ZA', 'q_max_subtraction'), ('CTR', 'q_max_subtraction'), ('XI', 'linear_part'),
                    ('XI_XI', 'm0_residual'), ('XI_XI', 'm0_constant'), ('CHI', 'outside_core'),
                    ('ZETA', 'outside_core'), ('THETA', 'inner_ball')}
"""The rules of the hand code's final integral and k-space rows (lpt._get_kspace_rows, lpt._final_integral)."""
