"""Shared EFT counterterm and stochasticity building blocks.

The numerical backends own the construction of the base spectrum: 
an EPT backend supplies a k-space array, while the LPT backend can supply a fused
Zel'dovich integrand.
This module only owns the parameter-dependent shapes.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp


CountertermBase = Literal["linear", "linear_ir_resum", "zeldovich"]
_VALID_COUNTERTERM_BASES = frozenset(("linear", "linear_ir_resum", "zeldovich"))


#: Number of entries in the shared ``ctr`` coefficient vector.
N_COUNTERTERM_COEFFICIENTS = 5


@dataclass(frozen=True)
class Counterterms:
    """Immutable counterterm policy and shared coefficient contractions.

    Both backends use the same five-slot coefficient vector

        ``ctr = (c0, c2, c4, c6, c_nlo)``

    in the paper convention: the first four multiply the leading ``k^2`` shape and the fifth is the ``k^4`` FoG coefficient.  
    The slot layout is identical for EPT and LPT so that a coefficient vector never changes meaning when it is handed to the other backend; 
    Both backends implement both operators; each supplies its own tree spectrum to ``Counterterms.nlo`` (see ``Counterterms.leading`` and ``Counterterms.nlo``).  
    Mapping Eulerian coefficients to their Lagrangian counterparts is deliberately the caller's responsibility.
    """

    base: CountertermBase

    def __post_init__(self):
        if self.base not in _VALID_COUNTERTERM_BASES:
            raise ValueError(
                "counterterm_base must be 'linear', 'linear_ir_resum', or "
                f"'zeldovich', got {self.base!r}"
            )

    @property
    def needs_kspace_base(self):
        """True when the backend must build ``P_base(k, mu)`` as a k-space array.

        ``'zeldovich'`` is instead supplied by the LPT backend as a q-space integrand fused into the final transform, so it never takes this route.
        """
        return self.base != "zeldovich"

    @staticmethod
    def split_coefficients(ctr):
        """Split ``ctr`` into ``((c0, c2, c4, c6), c_nlo)``.

        The length is validated eagerly.  ``ctr.shape`` is static even under ``jax.jit``, 
        so a wrong layout fails at trace time rather than silently reinterpreting a slot.
        """
        ctr = jnp.asarray(ctr)
        if ctr.shape[-1] != N_COUNTERTERM_COEFFICIENTS:
            raise ValueError(
                "ctr must have 5 entries (c0, c2, c4, c6, c_nlo), got shape "
                f"{ctr.shape}"
            )
        return ctr[..., :4], ctr[..., 4]

    @staticmethod
    def leading_shape(k, mu, f, leading_a, leading_b=None):
        """Return the leading counterterm factor multiplying the base P(k, mu).

        For a cross spectrum the two tracers contribute
        ``<d1_a d_ctr_b> + <d_ctr_a d1_b>``, i.e. the coefficients add.  Each
        counterterm term is linear in its coefficient, so this is the same
        degree-1 symmetrisation ``(x_a + x_b) / 2`` that
        :func:`~.utils.cross_bias_factor` applies to the bias monomials, and it
        reduces exactly to the auto case when ``leading_b is None``.
        """
        if leading_b is None:
            leading_b = leading_a
        c0, c2, c4, c6 = 0.5 * (jnp.asarray(leading_a) + jnp.asarray(leading_b))
        mu2 = mu**2
        coefficient = c0 + c2 * f * mu2 + c4 * f**2 * mu2**2 + c6 * f**3 * mu2**3
        return -2.0 * k**2 * coefficient

    @classmethod
    def leading(cls, k, mu, f, ctr_a, ctr_b, base_pk):
        """Leading ``k^2`` counterterm from the full five-slot ``ctr`` vectors.

        Pass the same vector twice for an auto spectrum.
        """
        leading_a, _ = cls.split_coefficients(ctr_a)
        leading_b, _ = cls.split_coefficients(ctr_b)
        return cls.leading_shape(k, mu, f, leading_a, leading_b) * base_pk

    @staticmethod
    def nlo_shape(k, mu, f, c_nlo):
        """Return the common k^4 FoG factor multiplying a backend tree spectrum."""
        return -c_nlo * k**4 * f**4 * mu**4

    @classmethod
    def nlo(cls, k, mu, f, ctr_a, ctr_b, tree_pk):
        """NLO ``k^4`` counterterm from the full five-slot ``ctr`` vectors."""
        _, c_nlo_a = cls.split_coefficients(ctr_a)
        _, c_nlo_b = cls.split_coefficients(ctr_b)
        return cls.nlo_shape(k, mu, f, 0.5 * (c_nlo_a + c_nlo_b)) * tree_pk


def stochasticity(k, mu, stochastic_coefficients):
    """Canonical stochastic contribution used by both EPT and LPT.

    ``k`` and ``mu`` must be broadcast-compatible.  The result is additive and is not multiplied by any linear or Zel'dovich power spectrum.
    """
    p_shot, a0, a2 = stochastic_coefficients
    return (p_shot + (a0 + a2 * mu**2) * k**2)
