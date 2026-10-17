"""The angular moments of the LPT final integral, derived by sympy from the generating function.

The final integral needs, per (k, mu) point and per q, the angular integral over the direction of q with the polar axis
along K = k + f (k.n) n.  With rho = k q, c and s the cosine and sine of the angle between K and k, A = rho c, C = rho s,
B = -(1/2) K^2 Y_lt(q), mu_K = Khat.qhat and mu_x the component of qhat along the unit vector e_x = (khat - c Khat) / s
of the (K, k) plane (so khat.qhat = c mu_K + s mu_x), the generating function is

    G(A, C, B) = int dOmega/(4 pi) exp(i (A mu_K + C mu_x)) exp(B mu_K^2)
               = exp(B c^2) sum_l w_l(rho) F_l(s2, B),      s2 = s^2 = C^2 / (A^2 + C^2),
    w_l = (-2/rho)^l j_l(rho),      F_l = (-B)^l sum_{k, i} coeffs[l, k, i] s2^(l + i - k) (-B)^i,

with coeffs the exact rationals of elf.lpt_coefficients.generate_g00_coefficient (``g00_fraction``).  The prefactor
exp(B c^2) = exp(B) exp(-B s2), and exp(B) does not depend on A or C.  The moments of utils_lpt.get_lpt_moments omit it,
the caller's damping exp(-K^2 (X_lt + c^2 Y_lt) / 2) supplies it.  The weights obey d w_l/dA = (A/2) w_{l+1} and
d w_l/dC = (C/2) w_{l+1} (w_l = (2/rho d/drho)^l w_0).

Engine.  Every derivative of G is a sum over l of w_l times an l-independent combination of the F_{l-k} at a few
shifts k, so the generating function is a dict {k: entry} (entry = the coefficient of w_l, written with a generic F
standing for F_{l-k}).  An entry is exp(-B s2) times the stored expression.  The differentiation operators act on the
dict as the chain rule on each entry plus the weight shift,

    D_v {k: X} = {k: delta_v X} + {k + 1: (v/2) X},   v = A, C,     D_Y = C^-1 D_C,
    delta_v X = dX/dv|_explicit + dX/dir dir/dv + (dX/ds2 - B X) ds2/dv,

with ds2/dA, ds2/dC, dir/dA, dir/dC derived by sympy from s2 and ir = (A^2 + C^2)^(-1/2) (``chain_rules``) and F an
undefined function of s2.  Two entry algebras use the same dict operators.  'flat' entries are the full expressions in
A, C, s2, ir, B, d^k F/ds2^k.  'struct' entries are linear in the raw derivatives R[a, b, n] = d_A^a d_C^b (C^-1 d_C)^n
[exp(-B s2) F] / exp(-B s2) (each evaluated once by the chain rule), with polynomial coefficients in the A, C of the
weight shift.  The two agree (a test).

Angular monomials are derivatives, mu_K <-> (1/i) d/dA and mu_x <-> (1/i) d/dC.  Out of the (K, k) plane a third phase
i C_y mu_y makes G a function of C^2 + C_y^2, and sympy gives (``y_rule``) d^(2n)/dC_y^(2n) G|_{C_y=0} =
(2n - 1)!! (C^-1 d/dC)^n G, odd orders zero.  The (C^-1 d/dC)^n act first, then d_C^b and d_A^a, and
G_{a,b,2n}(l) is the coefficient of w_l in d_A^a d_C^b d_{C_y}^{2n} G |_{C_y = 0} with exp(-B s2) divided out.

Moments (lpt_model.DENSITY_MOMENTS).  A moment is an angular polynomial sum_t coef_t mu_K^a mu_x^b mu_y^by and is
reported real (lpt_model.MOMENT_PHASE), M = sum_t coef_t (-1)^ceil(n_t / 2) G_{a_t, b_t, by_t} for the total degree n.

Basis and representation.  The runtime evaluates the moments in the Horner basis S_d(l) = (s2 d/ds2)^d F_l
(utils_lpt.compute_V_mu) and T_d = S_d / s2.  An ordinary derivative is s2^k d^k F = sum_d s(k, d) S_d (signed Stirling
numbers of the first kind, the inverse of the derived relation (s2 d/ds2)^d = sum_k S(d, k) s2^k d^k).  The
factorization (lpt_moments) works on independent symbols A, C, B, ir, c, s, A_mu, B_mu, so each raw derivative R[a, b, n]
of total order t is written in one canonical form, ir^t c^(a mod 2) s^(b mod 2) sum_{m, d} B^m s^(2 e_md) Q_md(c^2) Y_d
with Y = S for pure A derivatives and T otherwise and e_md = min(m, the s^2-valuation), except that
lpt_model.LEGACY_PARITY_A writes the parity factor c of R[1, 0, 0] as A ir.  A raw derivative with b + 2n >= 3 has
negative powers of s in this basis (the factorization rejects it), none of the density moments has one.
"""
from fractions import Fraction
from math import comb, factorial

import sympy as sp
from sympy.functions.combinatorial.numbers import stirling

from . import lpt_model as model
from .lpt_model import A_mu as Am, B_mu as Bm

A, B, C, c, s, ir = sp.symbols('A B C c s ir', real=True)
x = sp.Symbol('x', positive=True)            # s2, an independent variable with its chain rule
F = sp.Function('F')                          # F_{l-k} of the entry at shift k
Aw, Cw = sp.symbols('Aw Cw', real=True)       # the A, C of the weight shift (struct entries)
rho, sig, c2 = sp.symbols('rho sigma c2', positive=True)
ARRAY_SYMBOLS = (A, B, C, ir)
SCALAR_SYMBOLS = (c, s, Am, Bm)


# ------------------------------------------------------------------ exact F_l and the Euler basis
def g00_fraction(ell):
    """The angular coefficients ``[k][i]`` of order ``ell`` as Fractions (elf.lpt_coefficients.generate_g00_coefficient
    in exact arithmetic)."""
    def rising(y, order):
        value = Fraction(1)
        for offset in range(order):
            value *= y + offset
        return value
    out = [[None] * (ell + 1) for _ in range(ell + 1)]
    for k in range(ell + 1):
        for i in range(ell + 1):
            total = sum(((-1) ** n * comb(i, n) * comb(ell + n, k) * rising(Fraction(2 * (ell + n) + 1, 2), ell - k)
                         for n in range(i + 1)), Fraction(0))
            sign = -1 if (i - k) % 2 else 1
            out[k][i] = Fraction(sign, factorial(ell - k) * factorial(i)) * total
    return out


def F_poly(l):
    """F_l(x, B) exactly, from ``g00_fraction``."""
    t = g00_fraction(l)
    return sp.expand((-B)**l * sum(sp.Rational(t[k][i].numerator, t[k][i].denominator) * x**(l + i - k) * (-B)**i
                                   for k in range(l + 1) for i in range(l + 1)))


def check_euler_basis(lmax, dmax):
    """Assert that (x d/dx)^d F_l is the V-table form (-B)^l sum_{k,i} N^d coeffs x^N (-B)^i of
    utils_lpt.compute_V_mu (N = l + i - k), for l <= lmax and d <= dmax."""
    for l in range(lmax + 1):
        t, e = g00_fraction(l), F_poly(l)
        for d in range(dmax + 1):
            ref = (-B)**l * sum(sp.Rational(t[k][i].numerator, t[k][i].denominator) * (l + i - k)**d
                                * x**(l + i - k) * (-B)**i for k in range(l + 1) for i in range(l + 1))
            assert sp.expand(e - ref) == 0, (l, d)
            e = sp.expand(x * sp.diff(e, x))


def stirling_inverse(N):
    """inv[k][d] with x^k d^k F/dx^k = sum_d inv[k][d] (x d/dx)^d F.  The forward matrix is read off sympy's expansion
    of (x d/dx)^d F(x), asserted to be S(d, k) (second kind), and inverted (asserted signed first kind)."""
    fwd = sp.zeros(N + 1, N + 1)
    e = F(x)
    for d in range(N + 1):
        ex = sp.expand(e)
        for k in range(N + 1):
            dk = sp.diff(F(x), x, k) if k else F(x)
            fwd[d, k] = ex.coeff(dk).coeff(x, k)
        e = x * sp.diff(e, x)
    for d in range(N + 1):
        for k in range(N + 1):
            assert fwd[d, k] == stirling(d, k, kind=2), (d, k)
    inv = fwd.inv()
    for k in range(N + 1):
        for d in range(N + 1):
            assert inv[k, d] == stirling(k, d, kind=1, signed=True), (k, d)
    return [[inv[k, d] for d in range(N + 1)] for k in range(N + 1)]


# ------------------------------------------------------------------ chain rules and the out-of-plane rule
def _rewrite_xir(e):
    """A derivative of s2 or ir (homogeneous in A, C) as a polynomial in A, C, x, ir.  N(A, C) / (A^2 + C^2)^p becomes
    N ir^(2p), then per monomial C^2 ir^2 -> x and A^2 ir^2 -> 1 - x (the definitions of x and ir)."""
    num, den = sp.fraction(sp.factor(sp.together(e)))
    base, power = den.as_base_exp()
    assert sp.expand(base - (A**2 + C**2)) == 0 or den == 1, den
    two_p = int(2 * power) if den != 1 else 0
    out = 0
    for (a, cc), co in sp.Poly(sp.expand(num), A, C).terms():
        i, fac = two_p, sp.Integer(1)
        while cc >= 2 and i >= 2:
            cc, i, fac = cc - 2, i - 2, fac * x
        while a >= 2 and i >= 2:
            a, i, fac = a - 2, i - 2, fac * (1 - x)
        out += co * A**a * C**cc * ir**i * fac
    return sp.expand(out)


def chain_rules():
    """{(f, v): df/dv} for f = 'x' (s2), 'ir' and v = 'A', 'C', derived and asserted to be the identities
    ds2/dA = -2 A s2 ir^2, ds2/dC = 2 C (1 - s2) ir^2, dir/dA = -A ir^3, dir/dC = -C ir^3."""
    s2e, ire = C**2 / (A**2 + C**2), (A**2 + C**2)**sp.Rational(-1, 2)
    out = {}
    for name, fe in (('x', s2e), ('ir', ire)):
        for v, V in (('A', A), ('C', C)):
            d = _rewrite_xir(sp.diff(fe, V))
            assert sp.simplify(d.subs({x: s2e, ir: ire}) - sp.diff(fe, V)) == 0, (name, v)
            out[(name, v)] = d
    expected = {('x', 'A'): -2 * A * x * ir**2, ('x', 'C'): 2 * C * (1 - x) * ir**2,
                ('ir', 'A'): -A * ir**3, ('ir', 'C'): -C * ir**3}
    assert all(sp.expand(out[key] - expected[key]) == 0 for key in expected), out
    return out


def y_rule(nmax):
    """{n: K_n} with d^(2n)/dCy^(2n) g(Cx^2 + Cy^2)|_{Cy=0} = K_n (Cx^-1 d/dCx)^n g(Cx^2), derived for an undefined g
    (the odd orders vanish, asserted)."""
    g = sp.Function('g')
    X, Y = sp.symbols('Cx Cy', positive=True)
    G = g(X**2 + Y**2)
    out = {0: sp.Integer(1)}
    for n in range(1, nmax + 1):
        lhs = sp.diff(G, Y, 2 * n).subs(Y, 0).doit()
        rhs = g(X**2)
        for _ in range(n):
            rhs = sp.diff(rhs, X) / X
        K = sp.simplify(lhs / rhs.doit())
        assert K.is_Rational and K == sp.factorial2(2 * n - 1), (n, K)
        assert sp.diff(G, Y, 2 * n - 1).subs(Y, 0).doit() == 0
        out[n] = K
    return out


# ------------------------------------------------------------------ the engine
class Engine:
    """Derivatives of the generating-function dict up to total order N (flat and struct) and their canonical forms."""

    def __init__(self, N):
        self.N = N
        self.chain = chain_rules()
        self.K = y_rule(N // 2)
        self.inv = stirling_inverse(N)
        self._R, self._Rkey, self._raw, self._canon, self._tmpl = {}, {}, {}, {}, {}
        self._G = {('flat', 0, 0, 0): {0: F(x)}, ('struct', 0, 0, 0): {0: self.Rsym((0, 0, 0))}}
        self._H = {('flat', 0): {0: F(x)}, ('struct', 0): {0: self.Rsym((0, 0, 0))}}

    # -- the two entry algebras
    def Rsym(self, key):
        if key not in self._R:
            self._R[key] = sp.Symbol('R_%d_%d_%d' % key)
            self._Rkey[self._R[key]] = key
        return self._R[key]

    def delta_flat(self, X, v):
        """The chain rule on an exp-stripped entry, v = 'A', 'C' or 'Y' (= C^-1 delta_C)."""
        if v == 'Y':
            d = sp.expand(self.delta_flat(X, 'C') / C)
            assert all(t.as_powers_dict().get(C, 0) >= 0 for t in sp.Add.make_args(d)), 'negative power of C'
            return d
        V = A if v == 'A' else C
        return sp.expand(sp.diff(X, V) + sp.diff(X, ir) * self.chain[('ir', v)]
                         + (sp.diff(X, x) - B * X) * self.chain[('x', v)])

    def delta_struct(self, X, v):
        """delta on a struct entry, R[a, b, n] -> its next raw derivative and Aw, Cw differentiated."""
        Rs = [r for r in X.free_symbols if r in self._Rkey]
        if v == 'Y':
            assert not X.has(Aw) and not X.has(Cw) and all(self._Rkey[r][:2] == (0, 0) for r in Rs), \
                'the C^-1 d/dC operators must act first'
            return sp.expand(X.xreplace({r: self.Rsym((0, 0, self._Rkey[r][2] + 1)) for r in Rs}))
        i, w = (0, Aw) if v == 'A' else (1, Cw)
        nxt = {}
        for r in Rs:
            key = list(self._Rkey[r])
            key[i] += 1
            nxt[r] = self.Rsym(tuple(key))
        return sp.expand(sp.diff(X, w) + X.xreplace(nxt))

    def D(self, Gd, v, mode):
        """D_v on the dict, the chain rule on every entry plus the weight shift (v/2) w_{l+1}."""
        delta = self.delta_flat if mode == 'flat' else self.delta_struct
        shift = {'A': (A if mode == 'flat' else Aw) / 2, 'C': (C if mode == 'flat' else Cw) / 2,
                 'Y': sp.Rational(1, 2)}[v]
        out = {}
        for k, X in Gd.items():
            out[k] = out.get(k, 0) + delta(X, v)
            out[k + 1] = out.get(k + 1, 0) + shift * X
        out = {k: sp.expand(X) for k, X in out.items()}
        return {k: X for k, X in out.items() if X != 0}

    def G(self, a, b, n, mode='struct'):
        """{k: entry} of G_{a,b,2n} = d_A^a d_C^b d_{Cy}^{2n} G|_{Cy=0} (exp(-B s2) divided out)."""
        key = (mode, a, b, n)
        if key not in self._G:
            if a > 0:
                self._G[key] = self.D(self.G(a - 1, b, n, mode), 'A', mode)
            elif b > 0:
                self._G[key] = self.D(self.G(0, b - 1, n, mode), 'C', mode)
            else:
                self._G[key] = {k: sp.expand(self.K[n] * X) for k, X in self.H(n, mode).items()}
        return self._G[key]

    def H(self, n, mode):
        """(C^-1 D_C)^n of the generating dict (without the constant K_n)."""
        if (mode, n) not in self._H:
            self._H[(mode, n)] = self.D(self.H(n - 1, mode), 'Y', mode)
        return self._H[(mode, n)]

    # -- raw derivatives (the chain rule, once each)
    def raw(self, key):
        """R[a, b, n] in the natural form, a polynomial in A, C, x, ir, B and d^k F/dx^k."""
        if key not in self._raw:
            a, b, n = key
            if a > 0:
                self._raw[key] = self.delta_flat(self.raw((a - 1, b, n)), 'A')
            elif b > 0:
                self._raw[key] = self.delta_flat(self.raw((0, b - 1, n)), 'C')
            elif n > 0:
                self._raw[key] = self.delta_flat(self.raw((0, 0, n - 1)), 'Y')
            else:
                self._raw[key] = F(x)
        return self._raw[key]

    def euler(self, e):
        """d^k F/dx^k -> x^-k sum_d s(k, d) E_d, with E_d the Euler derivatives S_d as placeholder symbols."""
        E = sp.symbols('E0:%d' % (self.N + 1))
        rep = {(sp.diff(F(x), x, k) if k else F(x)): x**(-k) * sum(self.inv[k][d] * E[d] for d in range(k + 1))
               for k in range(self.N + 1)}
        return sp.expand(e.xreplace(rep)), E

    def canonical(self, key):
        """(t, parity, basis, {(m, d): (e, Q(c2))}) of R[key] (the canonical form of the module docstring)."""
        if key in self._canon:
            return self._canon[key]
        a, b, n = key
        t = a + b + 2 * n
        e = sp.expand(self.raw(key).xreplace({A: c * rho, C: s * rho, ir: 1 / rho}) * rho**t)
        assert not e.has(rho), ('not homogeneous', key)
        e, E = self.euler(e)
        e = sp.expand(e.xreplace({x: s**2}) / (c**(a % 2) * s**(b % 2)))
        kind = 'S' if b == 0 and n == 0 else 'T'
        lift = 2 * (self.N + 2)                       # s^lift makes the Laurent polynomial in s a polynomial
        P = sp.Poly(sp.expand(e * s**lift), c, s, B, *E)
        terms = {}
        for mon, co in P.terms():
            pc, ps, m = mon[:3]
            d = [i for i, v in enumerate(mon[3:]) if v]
            assert len(d) == 1 and mon[3 + d[0]] == 1 and pc % 2 == 0 and ps % 2 == 0, (key, mon)
            j = ps // 2 - lift // 2 + (1 if kind == 'T' else 0)    # S_d = s^2 T_d in the T basis
            terms[(m, d[0])] = terms.get((m, d[0]), 0) + co * (1 - sig)**(pc // 2) * sig**j
        out = {}
        for (m, d), pe in terms.items():
            pe = sp.expand(pe)
            if pe == 0:
                continue
            val = min(sp.Poly(sp.expand(pe * sig**lift), sig).monoms()[-1][0] - lift, m)
            Q = sp.expand(sp.expand(pe * sig**(-val)).subs(sig, 1 - c2))
            assert not Q.has(sig)
            out[(m, d)] = (val, Q)
        self._canon[key] = (t, (a % 2, b % 2), kind, out)
        return self._canon[key]

    def template(self, key, legacy=None):
        """R[key] on the factorization's symbols with the basis placeholders P['S0'], P['T1'], ..."""
        legacy = model.LEGACY_PARITY_A if legacy is None else legacy
        if (key, legacy) in self._tmpl:
            return self._tmpl[(key, legacy)]
        t, (pa, pb), kind, terms = self.canonical(key)
        P = self.placeholders()
        par = (A * ir if (legacy and key == (1, 0, 0)) else c**pa) * s**pb
        out = 0
        for (m, d), (e, Q) in terms.items():
            if kind == 'S':
                Y = P['S0'] if d == 0 else s**2 * P['T%d' % d]
            else:
                Y = P['S0'] / s**2 if d == 0 else P['T%d' % d]
            out += B**m * s**(2 * e) * Q.subs(c2, c**2) * Y
        self._tmpl[(key, legacy)] = sp.expand(ir**t * par * out)
        return self._tmpl[(key, legacy)]

    def placeholders(self):
        return {name: sp.Symbol('P_' + name) for name in ['S0'] + ['T%d' % d for d in range(1, self.N + 1)]}

    def shift_kernels(self, a, b, n, legacy=None):
        """{k: entry of G_{a,b,2n} at shift k on the factorization's symbols and the placeholders} (l-independent)."""
        out = {}
        for k, X in self.G(a, b, n).items():
            sub = {r: self.template(self._Rkey[r], legacy) for r in X.free_symbols & set(self._Rkey)}
            out[k] = sp.expand(X.xreplace(sub).xreplace({Aw: A, Cw: C}))
        return out

    def instantiate(self, kernels, l, X):
        """sum_k kernel_k with the placeholders -> the basis symbols X[name][l - k] (zero for l < k)."""
        P = self.placeholders()
        tot = 0
        for k, e in kernels.items():
            if l - k >= 0:
                tot += e.xreplace({P[nm]: X[nm][l - k] for nm in P if nm in X})
        return sp.expand(tot)


# ------------------------------------------------------------------ the moments of a declaration
def phase(n):
    """(-i)^(n mod 2) (1/i)^n = (-1)^ceil(n/2), the factor of G_{a,b,by} in a reported moment of total degree n."""
    return (-1)**((n + 1) // 2)


def declared_terms(decl):
    """[(coef * phase, a, b, n_y)] of a moment's angular polynomial (odd powers of mu_y vanish, one parity)."""
    parity = {(a + b + by) % 2 for _, a, b, by in decl}
    assert len(parity) == 1, 'a reported moment needs one parity of the total degree'
    return [(sp.sympify(co) * phase(a + b + by), a, b, by // 2) for co, a, b, by in decl if by % 2 == 0]


def max_degree(declaration):
    """The highest total degree a + b + b_y of a declaration {name: terms} (the order of the engine)."""
    return max(a + b + by for terms in declaration.values() for _, a, b, by in terms)


def moment_kernels(eng, decl, legacy=None):
    """{k: shift kernel} of a declared moment on the factorization's symbols (struct engine, canonical forms)."""
    out = {}
    for co, a, b, n in declared_terms(decl):
        for k, e in eng.shift_kernels(a, b, n, legacy).items():
            out[k] = out.get(k, 0) + co * e
    return {k: sp.expand(e) for k, e in out.items() if sp.expand(e) != 0}


def moment_flat(eng, decl):
    """{k: flat entry} of a declared moment (the natural form in A, C, x, ir, B, d^k F/dx^k)."""
    out = {}
    for co, a, b, n in declared_terms(decl):
        for k, e in eng.G(a, b, n, 'flat').items():
            out[k] = out.get(k, 0) + co * e
    return {k: sp.expand(e) for k, e in out.items() if sp.expand(e) != 0}


def basis_symbols(L):
    """``{basis name: [symbol of l = 0 .. L - 1]}`` of lpt_model.HORNER_BASIS, e.g. T1_3 for T1[3]."""
    return {b: [sp.Symbol(f'{b}_{l}') for l in range(L)] for b in model.HORNER_BASIS}


def moments(X, L, declaration=None, eng=None):
    """M[j][l] of the declared moments (default lpt_model.DENSITY_MOMENTS, in its order) on the basis symbols X."""
    declaration = declaration or model.DENSITY_MOMENTS
    eng = eng or Engine(max_degree(declaration))
    out = [[None] * L for _ in declaration]
    for j, name in enumerate(declaration):
        ker = moment_kernels(eng, declaration[name])
        for l in range(L):
            out[j][l] = eng.instantiate(ker, l, X)
    return out
