from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .eft_terms import N_COUNTERTERM_COEFFICIENTS

def _as_param_array(value, dtype):
    """Convert to a JAX array, defaulting to the ambient precision.

    ``dtype=None`` means "follow ``jax_enable_x64``", which is what this package runs in.  
    Forcing float32 here would silently round every nuisance parameter to ~1e-8 relative accuracy and show up in derivatives.
    """
    return jnp.asarray(value) if dtype is None else jnp.array(value, dtype)

def _check_ctr_layout(ctr):
    """Reject a counterterm vector that is not in the shared five-slot layout."""
    if jnp.shape(ctr)[-1] != N_COUNTERTERM_COEFFICIENTS:
        raise ValueError(
            "ctr must have 5 entries (c0, c2, c4, c6, c_nlo), got shape "
            f"{jnp.shape(ctr)}"
        )

@register_pytree_node_class
@dataclass(frozen=True)
class EPTParams:
    cosmo: jnp.ndarray  # shape (2,)   [f, h]
    bias:  jnp.ndarray  # shape (4,)   [b1, b2, bG2, bGamma3]
    ctr:   jnp.ndarray  # shape (5,)   [c0, c2, c4, c6, c_nlo]  (shared layout)
    stoch: jnp.ndarray  # shape (3,)   [P_shot, a0, a2]

    def tree_flatten(self):
        return (self.cosmo, self.bias, self.ctr, self.stoch), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        cosmo, bias, ctr, stoch = children
        return cls(cosmo, bias, ctr, stoch)
    
    @property
    def f(self):     return self.cosmo[0]
    @property
    def h(self):     return self.cosmo[1]

def make_ept_params(*, f, h, bias, ctr, stoch, dtype=None):
    cosmo = _as_param_array([f, h], dtype)
    bias  = _as_param_array(bias,  dtype)
    ctr   = _as_param_array(ctr,   dtype)
    stoch = _as_param_array(stoch, dtype)
    _check_ctr_layout(ctr)
    return EPTParams(cosmo, bias, ctr, stoch)

@register_pytree_node_class
@dataclass(frozen=True)
class LPTParams:
    f:     jnp.ndarray  # shape (,)   [f]
    bias:  jnp.ndarray  # shape (4,)   [b1, b2, bG2, bGamma3]
    # Same five-slot layout as EPTParams, in the paper (Lagrangian) convention.
    # The k^4 FoG operator has no LPT tree integrand in this package yet, so ``c_nlo`` is accepted for layout compatibility but not evaluated.
    ctr:   jnp.ndarray  # shape (5,)   [cL0, cL2, cL4, cL6, c_nlo]
    stoch: jnp.ndarray  # shape (3,)   [P_shot, a0, a2]
    h:     Optional[jnp.ndarray] = None

    def tree_flatten(self):
        return (self.f, self.bias, self.ctr, self.stoch, self.h), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        f, bias, ctr, stoch, nl, h = children
        return cls(f, bias, ctr, stoch, nl, h)

def make_lpt_params(*, f, bias, ctr, stoch, h=None, dtype=None):
    f     = _as_param_array(f,     dtype)
    bias  = _as_param_array(bias,  dtype)
    ctr   = _as_param_array(ctr,   dtype)
    stoch = _as_param_array(stoch, dtype)
    h = None if h is None else _as_param_array(h, dtype)
    _check_ctr_layout(ctr)
    return LPTParams(f=f, bias=bias, ctr=ctr, stoch=stoch, h=h)
