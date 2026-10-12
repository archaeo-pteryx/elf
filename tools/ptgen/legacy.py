"""The legacy tables (``src/elf/pt_coeff/*.txt``, ``src/elf/pt_matrix/*.txt``), main's loader of
them and main's hand-typed UV formula, for ``python -m ptgen check`` and the tests only.

The runtime no longer reads the text tables; they stay in the repository until the user
decides to remove them, and every function here returns None when they are absent.
"""
import os
import re

import sympy as sp
from sympy.parsing.mathematica import parse_mathematica

from . import kernels
from .emit import REPO_DIR
from .vectors import mu

COEFF_DIR = os.path.join(REPO_DIR, 'src', 'elf', 'pt_coeff')
MATRIX_DIR = os.path.join(REPO_DIR, 'src', 'elf', 'pt_matrix')
nu1, nu2 = sp.symbols('nu1 nu2')


def _stems(directory, prefix):
    if not os.path.isdir(directory):
        return None
    return sorted(name[:-4] for name in os.listdir(directory)
                  if name.startswith(prefix) and name.endswith('.txt'))


def read_coeff_blocks(prefix):
    """``{(l, n, m): polynomial}`` of the legacy ``pt_coeff`` files in ptgen's symbols."""
    stems = _stems(COEFF_DIR, prefix + '_')
    if stems is None:
        return None
    rep = {sp.Symbol('μ'): mu, sp.Symbol('f'): kernels.f}
    rep.update({sp.Symbol(name): kernels.BIAS[name] for name in ('b1', 'b2', 'bG2', 'bGamma3')})
    out = {}
    for stem in stems:
        with open(os.path.join(COEFF_DIR, stem + '.txt')) as fh:
            expr = parse_mathematica(fh.read())
        out[tuple(int(v) for v in stem.split('_')[1:])] = sp.expand(expr.subs(rep, simultaneous=True))
    return out


def _get_coeff_info(path):
    """main's ``pt_coeff.get_coeff_info``."""
    with open(path, 'r') as file:
        expr = file.read()
    expr = parse_mathematica(expr)
    terms = expr.as_ordered_terms()
    f, mu_, b1, b2, bG2, bGamma3 = sp.symbols('f μ b1 b2 bG2 bGamma3')
    return [[int(sp.degree(term, mu_)), int(sp.degree(term, f)), int(sp.degree(term, b1)),
             int(sp.degree(term, b2)), int(sp.degree(term, bG2)), int(sp.degree(term, bGamma3)),
             float(term.subs({f: 1, mu_: 1, b1: 1, b2: 1, bG2: 1, bGamma3: 1}))] for term in terms]


def legacy_coeff_arrays(prefix):
    """main's ``ept._read_coeff_tables(prefix)``: ``(names, lnm, table)`` from the text files."""
    import numpy as np
    names = _stems(COEFF_DIR, prefix)
    if names is None:
        return None
    lnm, coeff_info = [], []
    for name in names:
        str_list = re.split('_', name)
        lnm.append([int(str_list[1]), int(str_list[2]), int(str_list[3])])
        coeff_info.append(_get_coeff_info(os.path.join(COEFF_DIR, name + '.txt')))
    max_len = max(len(lst) for lst in coeff_info)
    padded = [lst + [[0, 0, 0, 0, 0, 0, 0.0]] * (max_len - len(lst)) for lst in coeff_info]
    return names, np.asarray(lnm), np.array(padded)


def read_matrix_blocks():
    """``{name: rational function}`` of the legacy ``pt_matrix`` files.

    22: the rational function of (nu1, nu2) as stored; 13: the stored expression times
    ``pi / tan(pi nu1)``, a rational function of nu1.
    """
    stems = _stems(MATRIX_DIR, '')
    if stems is None:
        return None
    T = sp.Symbol('T')
    out = {}
    for stem in stems:
        with open(os.path.join(MATRIX_DIR, stem + '.txt')) as fh:
            expr = parse_mathematica(fh.read())
        expr = expr.subs({sp.Symbol('nu1'): nu1, sp.Symbol('nu2'): nu2})
        if stem.startswith('13'):
            expr = sp.cancel(expr.subs(sp.tan(sp.pi * nu1), T) * sp.pi / T)
            if expr.has(T) or expr.has(sp.pi):
                raise ValueError(f'{stem}: not R(nu1) tan(pi nu1) / pi')
        out[stem] = expr
    return out


def uv_terms():
    """main's ``ept.uv_terms()``: ``[((mu_pow, f_pow, b1_pow), coefficient)]`` of the
    hand-typed ``Z1 Z3_UV``."""
    mu_, f, b1 = sp.symbols('mu f b1')
    R = sp.Rational
    z1 = b1 + f * mu_**2
    z3_uv = (-R(61, 315) * b1
             + ((-R(3, 5) + R(2, 105) * b1) * f + (-R(16, 35) - R(1, 3) * b1) * f**2) * mu_**2
             + (-R(46, 105) * f**2 - R(1, 3) * f**3) * mu_**4)
    return sorted(sp.Poly(sp.expand(z1 * z3_uv), mu_, f, b1).terms())


def z3_uv():
    """The hand-typed ``Z3_UV`` of main's ``ept.uv_terms`` docstring, in ptgen's symbols."""
    R = sp.Rational
    f, b1 = kernels.f, kernels.BIAS['b1']
    return (-R(61, 315) * b1
            + ((-R(3, 5) + R(2, 105) * b1) * f + (-R(16, 35) - R(1, 3) * b1) * f**2) * mu**2
            + (-R(46, 105) * f**2 - R(1, 3) * f**3) * mu**4)
