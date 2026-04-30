import numpy as np
from scipy.special import loggamma
from sympy.parsing.mathematica import parse_mathematica
from sympy import var, lambdify

def get_I(nu1, nu2):
    log_i = (
        -np.log(8 * np.pi**(3/2))
        + loggamma(3/2 - nu1)
        + loggamma(3/2 - nu2)
        + loggamma(nu1 + nu2 - 3/2)
        - loggamma(nu1)
        - loggamma(nu2)
        - loggamma(3 - nu1 - nu2)
    )
    return np.exp(log_i)


class PTMatrix22:

    def __init__(self, name):
        self.name = name
        with open(name,'r') as file:
            expr = file.read()
        self.expr = parse_mathematica(expr)
        nu1 = var('nu1')
        nu2 = var('nu2')
        self.func = lambdify([nu1,nu2], self.expr, modules='numpy')

    def __call__(self, nu1, nu2):
        return self.func(nu1,nu2) * get_I(nu1,nu2)


class PTMatrix13:

    def __init__(self, name):
        self.name = name
        with open(name,'r') as file:
            expr = file.read()
        self.expr = parse_mathematica(expr)
        nu1 = var('nu1')
        self.func = lambdify([nu1], self.expr, modules='numpy')

    def __call__(self, nu1):
        return self.func(nu1)
