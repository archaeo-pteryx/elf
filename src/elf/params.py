from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from .eft_terms import N_COUNTERTERM_COEFFICIENTS

def _as_param_array(value, dtype):
    """JAX array; ``dtype=None`` follows ``jax_enable_x64`` (do not force float32)."""
    return jnp.asarray(value) if dtype is None else jnp.array(value, dtype)

def _check_shape(name, value, expected, layout_doc):
    """Reject a parameter whose static shape is not ``expected`` (safe under ``jit``)."""
    shape = jnp.shape(value)
    if shape != expected:
        raise ValueError(
            f"{name} must have shape {expected} {layout_doc}, got {shape}"
        )

@register_pytree_node_class
@dataclass(frozen=True)
class Params:
    cosmo: jnp.ndarray  # shape (2,)   [f, h]
    bias:  jnp.ndarray  # shape (4,)   [b1, b2, bG2, bGamma3]
    ctr:   jnp.ndarray  # shape (7,)   [c0, c2, c4, c6, c44, c46, c48]
    stoch: jnp.ndarray  # shape (3,)   [P_shot, a0, a2]
    # second tracer (None: auto); ctr and stoch are then the cross spectrum's own
    bias2: Optional[jnp.ndarray] = None  # shape (4,)   [b1, b2, bG2, bGamma3]

    def tree_flatten(self):
        return (self.cosmo, self.bias, self.ctr, self.stoch, self.bias2), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        cosmo, bias, ctr, stoch, bias2 = children
        return cls(cosmo, bias, ctr, stoch, bias2)
    
    @property
    def f(self):     return self.cosmo[0]
    @property
    def h(self):     return self.cosmo[1]
    @property
    def bias_b(self):
        """Bias of the second tracer: ``bias2`` for a cross spectrum, ``bias`` for auto."""
        return self.bias if self.bias2 is None else self.bias2

def make_params(*, f, h, bias, ctr, stoch, bias2=None, dtype=None):
    """Build a :class:`Params` pytree with shape checks (batch with ``jax.vmap``, not extra axes)."""
    cosmo = _as_param_array([f, h], dtype)
    bias  = _as_param_array(bias,  dtype)
    ctr   = _as_param_array(ctr,   dtype)
    stoch = _as_param_array(stoch, dtype)
    bias2 = None if bias2 is None else _as_param_array(bias2, dtype)
    _check_shape('cosmo', cosmo, (2,), '[f, h]')
    _check_shape('bias',  bias,  (4,), '[b1, b2, bG2, bGamma3]')
    _check_shape('ctr',   ctr,   (N_COUNTERTERM_COEFFICIENTS,), '[c0, c2, c4, c6, c44, c46, c48]')
    _check_shape('stoch', stoch, (3,), '[P_shot, a0, a2]')
    if bias2 is not None:
        _check_shape('bias2', bias2, (4,), '[b1, b2, bG2, bGamma3]')
    return Params(cosmo, bias, ctr, stoch, bias2)
