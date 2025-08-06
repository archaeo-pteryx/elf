from dataclasses import dataclass
from typing import Any
from jax import numpy as jnp
from jax import tree_util
from jax.tree_util import register_pytree_node_class

def to_jax_array(x: Any):
    return x if isinstance(x, jnp.ndarray) else jnp.asarray(x)

@register_pytree_node_class
@dataclass(frozen=True)
class BiasParams:
    b1: Any
    b2: Any
    bG2: Any
    bGamma3: Any

    def tree_flatten(self):
        children = tuple(to_jax_array(getattr(self, field)) for field in self.__dataclass_fields__)
        aux_data = tuple(self.__dataclass_fields__)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**{field: val for field, val in zip(aux_data, children)})

@register_pytree_node_class
@dataclass(frozen=True)
class CtrParams:
    c0: Any
    c2: Any
    c4: Any
    cfog: Any

    def tree_flatten(self):
        children = tuple(to_jax_array(getattr(self, field)) for field in self.__dataclass_fields__)
        aux_data = tuple(self.__dataclass_fields__)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**{field: val for field, val in zip(aux_data, children)})

@register_pytree_node_class
@dataclass(frozen=True)
class StochParams:
    P_shot: Any
    a0: Any
    a2: Any

    def tree_flatten(self):
        children = tuple(to_jax_array(getattr(self, field)) for field in self.__dataclass_fields__)
        aux_data = tuple(self.__dataclass_fields__)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**{field: val for field, val in zip(aux_data, children)})

@register_pytree_node_class
@dataclass(frozen=True)
class Params:
    f: Any
    h: Any
    bias: BiasParams
    ctr: CtrParams
    stoch: StochParams
    k_nl: Any
    ndens: Any

    def tree_flatten(self):
        children = (
            to_jax_array(self.f),
            to_jax_array(self.h),
            self.bias,
            self.ctr,
            self.stoch,
            to_jax_array(self.k_nl),
            to_jax_array(self.ndens)
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
