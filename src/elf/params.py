import dataclasses
from dataclasses import dataclass, field
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

def _check_bias_form(bias, bias_a, bias_b):
    """Reject anything but ``bias`` alone (auto) or ``bias_a`` and ``bias_b`` together (cross)."""
    given = [name for name, value in (('bias', bias), ('bias_a', bias_a), ('bias_b', bias_b))
             if value is not None]
    if given not in (['bias'], ['bias_a', 'bias_b']):
        raise ValueError(
            "give either bias (auto spectrum) or both bias_a and bias_b (cross spectrum), "
            f"got {', '.join(given) if given else 'none of them'}"
        )

@register_pytree_node_class
@dataclass(frozen=True)
class Params:
    """Cosmology, bias, counterterm and stochastic parameters (a JAX pytree).

    An auto spectrum sets ``bias`` (``bias_a`` and ``bias_b`` None), a cross spectrum sets
    ``bias_a`` and ``bias_b`` (``bias`` None).  The code reads only ``bias_a`` and ``bias_b``,
    both set to ``bias`` for an auto spectrum at the entry that reads the biases.
    """
    cosmo: jnp.ndarray  # shape (2,)   [f, h]
    # shape (4,)   [b1, b2, bG2, bGamma3], or [b1, b2, b_s2, bGamma3] for LPT(bias_basis='bs2')
    # (None for a cross spectrum)
    bias:  Optional[jnp.ndarray]
    ctr:   jnp.ndarray  # shape (7,)   [c0, c2, c4, c6, c44, c46, c48]
    stoch: jnp.ndarray  # shape (3,)   [e00, e20, e22]
    # the two tracers of a cross spectrum, same layout as bias (None for an auto spectrum).
    # For a cross spectrum ctr and stoch are the pair's own.
    bias_a: Optional[jnp.ndarray] = field(default=None, kw_only=True)
    bias_b: Optional[jnp.ndarray] = field(default=None, kw_only=True)

    def tree_flatten(self):
        # leaves (cosmo, bias, ctr, stoch) for auto, (cosmo, bias_a, bias_b, ctr, stoch) for cross
        return (self.cosmo, self.bias, self.bias_a, self.bias_b, self.ctr, self.stoch), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        cosmo, bias, bias_a, bias_b, ctr, stoch = children
        return cls(cosmo, bias, ctr, stoch, bias_a=bias_a, bias_b=bias_b)
    
    @property
    def f(self):     return self.cosmo[0]
    @property
    def h(self):     return self.cosmo[1]


def _with_tracer_biases(params):
    """``params`` with ``bias_a = bias_b = bias`` (and ``bias`` None) for an auto spectrum.

    A cross spectrum is returned unchanged.  Called at the entry that reads the biases.
    Inside a traced function ``bias_a`` and ``bias_b`` are then the same tracer as ``bias``,
    so the auto gradient with respect to ``bias`` is complete.
    """
    _check_bias_form(params.bias, params.bias_a, params.bias_b)
    if params.bias is not None:
        return dataclasses.replace(params, bias=None, bias_a=params.bias, bias_b=params.bias)
    return params


def make_params(*, f, h, bias=None, ctr, stoch, bias_a=None, bias_b=None, dtype=None):
    """Build a :class:`Params` pytree with shape checks (batch with ``jax.vmap``, not extra axes).

    Give ``bias`` for an auto spectrum, or ``bias_a`` and ``bias_b`` (the two tracers) for a
    cross spectrum.
    """
    _check_bias_form(bias, bias_a, bias_b)
    cosmo = _as_param_array([f, h], dtype)
    bias, bias_a, bias_b = (None if b is None else _as_param_array(b, dtype)
                            for b in (bias, bias_a, bias_b))
    ctr   = _as_param_array(ctr,   dtype)
    stoch = _as_param_array(stoch, dtype)
    _check_shape('cosmo', cosmo, (2,), '[f, h]')
    for name, b in (('bias', bias), ('bias_a', bias_a), ('bias_b', bias_b)):
        if b is not None:
            _check_shape(name, b, (4,), '[b1, b2, bG2, bGamma3]')
    _check_shape('ctr',   ctr,   (N_COUNTERTERM_COEFFICIENTS,), '[c0, c2, c4, c6, c44, c46, c48]')
    _check_shape('stoch', stoch, (3,), '[e00, e20, e22]')
    return Params(cosmo, bias, ctr, stoch, bias_a=bias_a, bias_b=bias_b)
