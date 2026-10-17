"""The generated kernels (pt_autogen.kernels) against the prototype's hand-typed forms (spec section 3).

Every comparison is exact: the difference of the two rational functions cancels to zero.
Two-momentum kernels are compared at (k, p), which is generic (k.p, |k|, |p|, n.k, n.p are
independent symbols), and at (p, k - p); three-momentum kernels at the generic (k, p, r), at
(k, p, -p) (vanishing-momentum rule) and at the coplanar (k, p, k + p).
"""
import pytest
import sympy as sp

from pt_autogen import kernels as K
from pt_autogen.vectors import K_VEC, P_VEC, R_VEC, mu
from pt_autogen.tests import reference_kernels as R

PAIRS = [(K_VEC, P_VEC), (P_VEC, K_VEC - P_VEC)]
TRIPLES = [(K_VEC, P_VEC, R_VEC), (K_VEC, P_VEC, -P_VEC), (K_VEC, P_VEC, K_VEC + P_VEC)]


def same(a, b):
    return sp.cancel(sp.together(a - b)) == 0


@pytest.mark.parametrize('u, v', PAIRS)
def test_recursion_gives_closed_F2_G2(u, v):
    assert same(K.F(u, v), R.F2(u, v))
    assert same(K.G(u, v), R.G2(u, v))


def test_vanishing_total_momentum():
    assert K.F(P_VEC, -P_VEC) == 0 and K.G(P_VEC, -P_VEC) == 0
    assert K.FG((K_VEC, -K_VEC, P_VEC)) != (0, 0)


@pytest.mark.parametrize('qs', TRIPLES)
def test_F3_G3(qs):
    F3, G3 = K.FG(qs)
    F3_ref, G3_ref = R.FG3(*qs)
    assert same(F3, F3_ref)
    assert same(G3, G3_ref)


@pytest.mark.parametrize('u, v', PAIRS)
def test_galaxy_kernel_2(u, v):
    ref = R.b1 * R.F2(u, v) + R.b2 / 2 + R.bG2 * R.KG2(u, v)
    assert same(K.galaxy_kernel((u, v)), ref)
    # the renormalised operators are third order
    assert same(K.galaxy_kernel((u, v), renormalised=True), ref)


@pytest.mark.parametrize('qs', TRIPLES)
@pytest.mark.parametrize('renormalised', [False, True])
def test_galaxy_kernel_3(qs, renormalised):
    ref = (R.b1 * R.FG3(*qs)[0]
           + R.symmetrise3(R.bias3_unsym, *qs, include_renormalized=renormalised))
    assert same(K.galaxy_kernel(qs, renormalised=renormalised), ref)


def test_Z1():
    assert same(K.Z(K_VEC), R.b1 + R.f * mu**2)
    assert same(K.Z(K_VEC), R.Z1(K_VEC))


@pytest.mark.parametrize('u, v', PAIRS)
def test_Z2(u, v):
    assert same(K.Z(u, v), R.Z2(u, v))


@pytest.mark.parametrize('renormalised', [False, True])
def test_Z3_k_p_minus_p(renormalised):
    gen = K.Z(K_VEC, P_VEC, -P_VEC, renormalised=renormalised)
    ref = R.Z3_sym(K_VEC, P_VEC, -P_VEC, include_renormalized=renormalised)
    assert same(gen, ref)


def test_Z3_generic():
    gen = K.Z(K_VEC, P_VEC, R_VEC, renormalised=True)
    assert same(gen, R.Z3_sym(K_VEC, P_VEC, R_VEC, include_renormalized=True))


def test_comparison_detects_a_change():
    u, v = K_VEC, P_VEC
    assert not same(K.Z(u, v), R.Z2(u, v) + R.bG2 * R.KG2(u, v) / 1000)
    assert not same(K.F(u, v), R.G2(u, v))
