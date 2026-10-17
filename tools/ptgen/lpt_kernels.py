"""The LPT displacement kernels L^(n), n <= 3, from the EdS recursion of Matsubara (2015) declared in lpt_model.

Convention (Matsubara 2008 Eq. 22, Matsubara 2015 Eq. 66): Psi^(n)(k) = (i / n!) int delta_D(k - k_1..n)
L_n(k_1, .., k_n) delta(k_1) .. delta(k_n), with L_1(k) = k / k^2.  Matsubara (2015) Eqs. (67)-(69) give the
unsymmetrised kernel through S_n = k.L_n and T_n = -k x L_n (k = k_1 + .. + k_n),

  S_n = (n!/2) sum_{m1+m2=n, m1<=m2} M_{m1m2}/(m1! m2!) [two-index bracket] (k_A x k_B).(L_A x L_B)
      - (n!/6) sum_{m1+m2+m3=n, m1<=m2<=m3} M_{m1m2m3}/(m1! m2! m3!) [three-index bracket]
                                                  [k_A.(k_B x k_C)] [L_A.(L_B x L_C)],
  T_n = (n-1)! sum_{m1+m2=n, m1<m2} (m2-m1)/(m1! m2!) [transverse bracket] (k_A x k_B) (L_A . L_B),
  L_n = [k S_n + k x T_n] / k^2,

with A the first m1 momenta, B the next m2, C the rest, k_A their sums, L_A = L_{m1}(A), M the multiplicities and the
brackets of ``lpt_model.LPT_RECURSION``.  Eq. (70) symmetrises.  Everything else is vector algebra,
(a x b).(c x d) = (a.c)(b.d) - (a.d)(b.c), [a.(b x c)][d.(e x f)] = det(u_i . v_j) and k x (a x b) = a (k.b) - b (k.a),
so every kernel is a true vector, a linear combination of the momenta with scalar coefficients (:class:`VE`).
The vanishing-momentum rule (as in ptgen.kernels).  A sub-kernel of order >= 2 whose momenta sum to zero vanishes
with the cross product it sits in, so such terms are dropped.

Momenta are ptgen ``Vec`` (integer combinations of k, p, r), scalar products through ``ptgen.vectors.dot`` in the
magnitudes k, p, r and the cosines x = khat.phat, y = khat.rhat, z = phat.rhat.  ``VE.ndot`` projects on the unit
vector n of ptgen.vectors, which the Wick engine (lpt_wick) reads as the separation direction qhat.
"""
import itertools
import math
from functools import lru_cache

import sympy as sp

from . import lpt_model as model
from .vectors import Vec, dot, ndot, norm2, vsum


class VE:
    """A true vector sum_v c_v v over momenta v (``Vec``) with scalar (sympy) coefficients."""

    __slots__ = ('terms',)

    def __init__(self, terms=None):
        self.terms = {}
        for v, c in (terms or {}).items():
            if v.is_zero():
                continue
            self.terms[v] = self.terms.get(v, 0) + c

    @staticmethod
    def of(v, c=1):
        return VE({v: sp.sympify(c)})

    def __add__(self, other):
        out = dict(self.terms)
        for v, c in other.terms.items():
            out[v] = out.get(v, 0) + c
        return VE(out)

    def __sub__(self, other):
        return self + other.scale(-1)

    def scale(self, c):
        return VE({v: c * cv for v, cv in self.terms.items()})

    def dot(self, other):
        """Scalar product with a VE or a Vec."""
        if isinstance(other, Vec):
            other = VE.of(other)
        return sp.Add(*[ca * cb * dot(va, vb) for va, ca in self.terms.items() for vb, cb in other.terms.items()])

    def ndot(self):
        """Projection on the unit vector n of ptgen.vectors (k mu, np, nr for the components along k, p, r)."""
        return sp.Add(*[c * ndot(v) for v, c in self.terms.items()])

    def simplify(self):
        return VE({v: sp.cancel(sp.together(c)) for v, c in self.terms.items()})

    def is_zero(self):
        return all(sp.cancel(c) == 0 for c in self.terms.values())


def _det3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))


def _multiplicity(ms):
    """The number of distinct orderings of the orders ``ms`` (Matsubara's M_{m1m2}, M_{m1m2m3})."""
    return math.factorial(len(ms)) // math.prod(math.factorial(ms.count(m)) for m in set(ms))


def _compositions(n, parts):
    """The non-decreasing tuples of ``parts`` positive integers adding up to n."""
    return [c for c in itertools.combinations_with_replacement(range(1, n + 1), parts) if sum(c) == n]


def _vanishing(blocks):
    """Whether a block of order >= 2 of ``blocks`` (tuples of momenta) has momenta that sum to zero."""
    return any(len(b) >= 2 and vsum(b).is_zero() for b in blocks)


@lru_cache(maxsize=None)
def L_unsym(ks, transverse=True):
    """``(L_n, S_n)`` of the ordered momenta ``ks`` (a tuple of Vec), unsymmetrised (Eqs. 67-69).

    ``transverse=False`` drops k x T_n and the transverse parts of the sub-kernels."""
    n = len(ks)
    if n == 1:
        return VE.of(ks[0], 1 / norm2(ks[0])), sp.Integer(1)
    k = vsum(ks)
    if k.is_zero():
        return VE(), sp.Integer(0)
    rec = model.LPT_RECURSION
    nf = math.factorial(n)
    S = sp.Integer(0)
    for m1, m2 in _compositions(n, 2):                       # the two-index term
        A, B = ks[:m1], ks[m1:]
        if _vanishing((A, B)):
            continue
        kA, kB = vsum(A), vsum(B)
        LA, LB = L_unsym(A, transverse)[0], L_unsym(B, transverse)[0]
        coef = (sp.Rational(nf, 2) * sp.Rational(_multiplicity([m1, m2]), math.factorial(m1) * math.factorial(m2))
                * rec['two_index'](n, m1, m2))
        S += coef * (LA.dot(kA) * LB.dot(kB) - LB.dot(kA) * LA.dot(kB))
    for m1, m2, m3 in _compositions(n, 3):                   # the three-index term
        blocks = (ks[:m1], ks[m1:m1 + m2], ks[m1 + m2:])
        if _vanishing(blocks):
            continue
        kk = [vsum(b) for b in blocks]
        LL = [L_unsym(b, transverse)[0] for b in blocks]
        coef = (-sp.Rational(nf, 6) * sp.Rational(_multiplicity([m1, m2, m3]),
                                                  math.factorial(m1) * math.factorial(m2) * math.factorial(m3))
                * rec['three_index'](n, m1, m2, m3))
        S += coef * _det3([[LL[j].dot(kk[i]) for j in range(3)] for i in range(3)])
    L = VE.of(k, S)
    if transverse:
        for m1, m2 in _compositions(n, 2):
            A, B = ks[:m1], ks[m1:]
            if m1 == m2 or _vanishing((A, B)):
                continue
            kA, kB = vsum(A), vsum(B)
            LA, LB = L_unsym(A, transverse)[0], L_unsym(B, transverse)[0]
            coef = (math.factorial(n - 1) * sp.Rational(m2 - m1, math.factorial(m1) * math.factorial(m2))
                    * rec['transverse'](n, m1, m2))
            # k x (k_A x k_B) = k_A (k.k_B) - k_B (k.k_A)
            L = L + (VE.of(kA, dot(k, kB)) - VE.of(kB, dot(k, kA))).scale(coef * LA.dot(LB))
    return L.scale(1 / norm2(k)), S


@lru_cache(maxsize=None)
def L_sym(ks, transverse=True):
    """The symmetrised kernel L^(n)(ks) (Eq. 70) as a :class:`VE`, ``ks`` a tuple of Vec."""
    n = len(ks)
    if n not in model.DISPLACEMENT_ORDERS:
        raise ValueError(f'order {n} outside lpt_model.DISPLACEMENT_ORDERS')
    total = VE()
    for perm in itertools.permutations(ks):
        total = total + L_unsym(tuple(perm), transverse)[0]
    return total.scale(sp.Rational(1, math.factorial(n))).simplify()


def L(*ks, transverse=True):
    """L^(n)(k_1, .., k_n), symmetric (the momenta in a canonical order)."""
    return L_sym(tuple(sorted(ks)), transverse)


def operator_kernel(name, legs):
    """The kernel of the Lagrangian operator ``name`` (lpt_model.OPERATORS) at the leg momenta ``legs``."""
    op = model.OPERATORS[name]
    assert len(legs) == op.order, (name, legs)
    return op.kernel(tuple(legs))
