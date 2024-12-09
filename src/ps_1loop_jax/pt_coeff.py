import sympy as sym
from sympy.parsing.mathematica import parse_mathematica

def get_coeff_info(name):
    with open(name, 'r') as file:
        expr = file.read()
    expr = parse_mathematica(expr)
    terms = expr.as_ordered_terms()

    f, mu, b1, b2, bG2, bGamma3 = sym.symbols('f μ b1 b2 bG2 bGamma3')
    
    coeff_info = [{'f': int(sym.degree(term, f)), 
                   'mu': int(sym.degree(term, mu)), 
                   'b1': int(sym.degree(term, b1)), 
                   'b2': int(sym.degree(term, b2)), 
                   'bG2': int(sym.degree(term, bG2)),
                   'bGamma3': int(sym.degree(term, bGamma3)),
                   'coeff': float(term.subs({f: 1, mu: 1, b1: 1, b2: 1, bG2: 1, bGamma3: 1}))
                   } for term in terms]
    
    return coeff_info
