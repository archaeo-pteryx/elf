"""The LPT derivation against the references typed once in this directory (lpt_references.py, lpt_hand_code.py).

The checks are those of ``pt_autogen.lpt_verify``, which ``python -m pt_autogen check`` also runs and records in manifest.json.
* The kernels.  L^(2) and the longitudinal L^(3) against their closed forms, the recursion against Matsubara (2015)
  Eqs. B8-B10, the LPT -> Euler map against pt_autogen's F_2 and F_3, the transverse part of L^(3) dropping out.
* The term list.  The cumulant expansion against the paper's Eqs. (pkmu_lpt_tree) and (pkmu_lpt_1loop), b3, bG3, bdG2
  absent, the dropped b_nabla2 terms, the self-pair cumulants, the 'explicit' renormalisation and the kernel facts.
* The components.  Every moment coefficient of the 23 components against the hand code (symbolically), the templates,
  linear kinds, template and linear matrices and bias degrees, the moments and rows used, the row layout.
* The rows.  The sources against the appendix, the identities (full-k integrals, j_l, the finite-range kernels, the
  ladder), the product forms and the preferred route against the hand code, the stage-A list and the k-space rows.
* The declared angular polynomials of the density moments.
"""
from pt_autogen import lpt_verify


def assert_group(results):
    errors = {key: results.get(key + '_error') for key, value in results.items() if value is False}
    assert lpt_verify.all_passed(results), errors


def test_kernels():
    results = lpt_verify.kernels()
    assert_group(results)
    assert results['L3_longitudinal_points'] == 12 and results['euler_F2_F3_points'] == 8


def test_term_list(lpt_derived):
    results = lpt_verify.term_list(lpt_derived['lpt_cumulants'])
    assert_group(results)
    assert results['term_list_equals_paper'] == 25 and results['explicit_renormalisation'] == 11


def test_components(lpt_derived):
    results = lpt_verify.components(lpt_derived['lpt_components'])
    assert_group(results)
    assert results['components_equal_hand_code'] == 44


def test_rows(lpt_derived):
    results = lpt_verify.rows(lpt_derived['lpt_rows'], lpt_derived['lpt_components'])
    assert_group(results)
    assert results['rows_equal_the_paper'] == 18


def test_moments_declaration():
    assert_group(lpt_verify.moments_declaration())
