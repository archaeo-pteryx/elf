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
N_COUNTERTERM_COEFFICIENTS = 7


@dataclass(frozen=True)
class Counterterms:
    """Immutable counterterm policy and shared coefficient contractions.

    Both backends use the same seven-slot coefficient vector

        ``ctr = (c0, c2, c4, c6, c44, c46, c48)``

    in the paper convention: the first four multiply the leading ``k^2`` shape and the last three the NLO ``k^4`` fingers-of-God shape.  
    Both shapes ride on the same base spectrum ``P_base(k, mu)``, selected by ``counterterm_base``.  
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
        """Split ``ctr`` into ``((c0, c2, c4, c6), (c44, c46, c48))``.

        The length is validated eagerly.  ``ctr.shape`` is static even under ``jax.jit``, 
        so a wrong layout fails at trace time rather than silently reinterpreting a slot.
        """
        ctr = jnp.asarray(ctr)
        if ctr.shape[-1] != N_COUNTERTERM_COEFFICIENTS:
            raise ValueError(
                "ctr must have 7 entries (c0, c2, c4, c6, c44, c46, c48), got shape "
                f"{ctr.shape}"
            )
        return ctr[..., :4], ctr[..., 4:7]

    @staticmethod
    def leading_shape(k, mu, f, leading_coefficients):
        """Return the leading counterterm factor multiplying the base P(k, mu).

        The same form is used for auto and cross spectra; for a cross spectrum
        the caller passes that spectrum's own coefficients.
        """
        c0, c2, c4, c6 = leading_coefficients
        mu2 = mu**2
        coefficient = c0 + c2 * f * mu2 + c4 * f**2 * mu2**2 + c6 * f**3 * mu2**3
        return -2.0 * k**2 * coefficient

    @classmethod
    def leading(cls, k, mu, f, ctr, base_pk):
        """Leading ``k^2`` counterterm from the full seven-slot ``ctr`` vector."""
        leading_coefficients, _ = cls.split_coefficients(ctr)
        return cls.leading_shape(k, mu, f, leading_coefficients) * base_pk

    @staticmethod
    def nlo_shape(k, mu, f, nlo_coefficients):
        """Return the NLO ``k^4`` fingers-of-God factor multiplying the base P(k, mu).

        Same structure as :meth:`leading_shape`, one order higher in ``k mu f``:

            ``-2 (c44 + c46 f mu^2 + c48 f^2 mu^4) (k mu f)^4``
        """
        c44, c46, c48 = nlo_coefficients
        mu2 = mu**2
        coefficient = c44 + c46 * f * mu2 + c48 * f**2 * mu2**2
        return -2.0 * coefficient * (k * mu * f)**4

    @classmethod
    def nlo(cls, k, mu, f, ctr, base_pk):
        """NLO ``k^4`` counterterm from the full seven-slot ``ctr`` vector.

        ``base_pk`` is the same base spectrum the leading counterterm rides on.
        """
        _, nlo_coefficients = cls.split_coefficients(ctr)
        return cls.nlo_shape(k, mu, f, nlo_coefficients) * base_pk


def stochasticity(k, mu, stochastic_coefficients):
    """Canonical stochastic contribution used by both EPT and LPT.

    ``k`` and ``mu`` must be broadcast-compatible.  The result is additive and is not multiplied by any linear or Zel'dovich power spectrum.
    """
    p_shot, a0, a2 = stochastic_coefficients
    return (p_shot + (a0 + a2 * mu**2) * k**2)
