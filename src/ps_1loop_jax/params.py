from dataclasses import dataclass
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass(frozen=True)
class EPTParams:
    cosmo: jnp.ndarray  # shape (2,)   [f, h]
    bias:  jnp.ndarray  # shape (4,)   [b1, b2, bG2, bGamma3]
    ctr:   jnp.ndarray  # shape (5,)   [c0, c2, c4, c6, cnlo]
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

def make_ept_params(*, f, h, bias, ctr, stoch, dtype=jnp.float32):
    cosmo = jnp.array([f, h], dtype)
    bias  = jnp.array(bias,  dtype)
    ctr   = jnp.array(ctr,   dtype)
    stoch = jnp.array(stoch, dtype)
    return EPTParams(cosmo, bias, ctr, stoch)

@register_pytree_node_class
@dataclass(frozen=True)
class LPTParams:
    f:     jnp.ndarray  # shape (,)   [f]
    bias:  jnp.ndarray  # shape (4,)   [b1, b2, bG2, bGamma3]
    ctr:   jnp.ndarray  # shape (5,)   [c0, c2, c4, c6, cnlo]
    stoch: jnp.ndarray  # shape (3,)   [P_shot, a0, a2]

    def tree_flatten(self):
        return (self.f, self.bias, self.ctr, self.stoch), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        f, bias, ctr, stoch = children
        return cls(f, bias, ctr, stoch)

def make_lpt_params(*, f, bias, ctr, stoch, dtype=jnp.float32):
    f     = jnp.array(f, dtype)
    bias  = jnp.array(bias,  dtype)
    ctr   = jnp.array(ctr,   dtype)
    stoch = jnp.array(stoch, dtype)
    return LPTParams(f, bias, ctr, stoch)
