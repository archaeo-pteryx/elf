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

def _check_shape(name, value, expected, layout_doc):
    """Reject a parameter whose static shape is not exactly ``expected``.

    Only ``jnp.shape`` is inspected, never the values, so this is safe under ``jit``:
    ``jnp.shape`` returns a Python tuple for tracers as well as for concrete arrays.
    """
    shape = jnp.shape(value)
    if shape != expected:
        raise ValueError(
            f"{name} must have shape {expected} {layout_doc}, got {shape}"
        )

def _check_ctr_layout(ctr):
    """Reject a counterterm vector that is not in the shared five-slot layout."""
    if jnp.shape(ctr)[-1] != N_COUNTERTERM_COEFFICIENTS:
        raise ValueError(
            "ctr must have 5 entries (c0, c2, c4, c6, c_nlo), got shape "
            f"{jnp.shape(ctr)}"
        )

@register_pytree_node_class
@dataclass(frozen=True)
class Params:
    cosmo: jnp.ndarray  # shape (2,)   [f, h]
    bias:  jnp.ndarray  # shape (4,)   [b1, b2, bG2, bGamma3]
    ctr:   jnp.ndarray  # shape (5,)   [c0, c2, c4, c6, c_nlo]  (shared layout)
    stoch: jnp.ndarray  # shape (3,)   [P_shot, a0, a2]
    # Second tracer's bias for a cross spectrum; ``None`` selects the auto
    # spectrum.  ``ctr`` and ``stoch`` are then the cross spectrum's own
    # coefficients (same functional form as the auto case, no symmetrisation).
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
    """Build a :class:`Params` pytree, validating every entry's static shape.

    Batches of parameters are handled with jax.vmap over Params, not by extra leading axes.
    """
    cosmo = _as_param_array([f, h], dtype)
    bias  = _as_param_array(bias,  dtype)
    ctr   = _as_param_array(ctr,   dtype)
    stoch = _as_param_array(stoch, dtype)
    bias2 = None if bias2 is None else _as_param_array(bias2, dtype)
    _check_shape('cosmo', cosmo, (2,), '[f, h]')
    _check_shape('bias',  bias,  (4,), '[b1, b2, bG2, bGamma3]')
    _check_shape('ctr',   ctr,   (N_COUNTERTERM_COEFFICIENTS,), '[c0, c2, c4, c6, c_nlo]')
    _check_shape('stoch', stoch, (3,), '[P_shot, a0, a2]')
    if bias2 is not None:
        _check_shape('bias2', bias2, (4,), '[b1, b2, bG2, bGamma3]')
    _check_ctr_layout(ctr)
    return Params(cosmo, bias, ctr, stoch, bias2)
