from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

# ── indexing
F, H                 = range(2)
K_NL, NDENS          = range(2)

@register_pytree_node_class
@dataclass(frozen=True)
class EPTParams:
    scalars: jnp.ndarray  # shape (2,)   [f, h]
    bias:    jnp.ndarray  # shape (4,)   [b1, b2, bG2, bGamma3]
    ctr:     jnp.ndarray  # shape (4,)   [c0, c2, c4, cfog]
    stoch:   jnp.ndarray  # shape (3,)   [P_shot, a0, a2]
    nl:      jnp.ndarray  # shape (2,)   [k_nl, ndens]

    def tree_flatten(self):
        return (self.scalars, self.bias, self.ctr, self.stoch, self.nl), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        scalars, bias, ctr, stoch, nl = children
        return cls(scalars, bias, ctr, stoch, nl)
    
    @property
    def f(self):     return self.scalars[F]
    @property
    def h(self):     return self.scalars[H]
    @property
    def k_nl(self):  return self.nl[K_NL]
    @property
    def ndens(self): return self.nl[NDENS]

def make_ept_params(*, f, h, bias, ctr, stoch, k_nl, ndens, dtype=jnp.float32):
    scalars = jnp.array([f, h], dtype)
    bias    = jnp.asarray(bias,  dtype)
    ctr     = jnp.asarray(ctr,   dtype)
    stoch   = jnp.asarray(stoch, dtype)
    nl      = jnp.asarray([k_nl, ndens], dtype)
    assert scalars.shape == (2,) and bias.shape == (4,) and ctr.shape == (4,) and stoch.shape == (3,) and nl.shape == (2,)
    return EPTParams(scalars, bias, ctr, stoch, nl)

@register_pytree_node_class
@dataclass(frozen=True)
class LPTParams:
    scalars: jnp.ndarray  # shape (2,)   [f, h]
    bias:    jnp.ndarray  # shape (4,)   [b1, b2, bG2, bGamma3]
    ctr:     jnp.ndarray  # shape (4,)   [a0, a2, a4, a6]
    stoch:   jnp.ndarray  # shape (3,)   [P_shot, a0, a2]
    nl:      jnp.ndarray  # shape (2,)   [k_nl, ndens]

    def tree_flatten(self):
        return (self.scalars, self.bias, self.ctr, self.stoch, self.nl), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        scalars, bias, ctr, stoch, nl = children
        return cls(scalars, bias, ctr, stoch, nl)
    
    @property
    def f(self):     return self.scalars[F]
    @property
    def k_nl(self):  return self.nl[K_NL]
    @property
    def ndens(self): return self.nl[NDENS]

def make_lpt_params(*, f, h, bias, ctr, stoch, k_nl, ndens, dtype=jnp.float32):
    scalars = jnp.array([f, h], dtype)
    bias    = jnp.asarray(bias,  dtype)
    ctr     = jnp.asarray(ctr,   dtype)
    stoch   = jnp.asarray(stoch, dtype)
    nl      = jnp.asarray([k_nl, ndens], dtype)
    assert scalars.shape == (1,) and bias.shape == (4,) and ctr.shape == (4,) and stoch.shape == (3,) and nl.shape == (2,)
    return LPTParams(scalars, bias, ctr, stoch, nl)
