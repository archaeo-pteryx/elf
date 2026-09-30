"""Shared EFT counterterm and stochasticity shapes.

The backends build the base spectrum (a k-space array, or the fused Zel'dovich
integrand in LPT); this module owns only the parameter-dependent shapes.
"""

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp


CountertermBase = Literal["linear", "linear_ir_resum", "zeldovich"]
_VALID_COUNTERTERM_BASES = frozenset(("linear", "linear_ir_resum", "zeldovich"))


N_COUNTERTERM_COEFFICIENTS = 7


@dataclass(frozen=True)
class Counterterms:
    """Counterterm base and the contractions of ``ctr = (c0, c2, c4, c6, c44, c46, c48)``.

    Both orders multiply the same base spectrum, with no bias or Kaiser factor;
    for a cross spectrum the coefficients belong to the pair.
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
        """True unless the base is the fused LPT 'zeldovich' integrand."""
        return self.base != "zeldovich"

    @staticmethod
    def split_coefficients(ctr):
        """Split ``ctr`` into ``((c0, c2, c4, c6), (c44, c46, c48))``; a wrong length fails at trace time."""
        ctr = jnp.asarray(ctr)
        if ctr.ndim != 1 or ctr.shape[0] != N_COUNTERTERM_COEFFICIENTS:
            raise ValueError(
                "ctr must have 7 entries (c0, c2, c4, c6, c44, c46, c48), got shape "
                f"{ctr.shape}"
            )
        return ctr[:4], ctr[4:]

    @staticmethod
    def leading_shape(k, mu, f, leading_coefficients):
        """``-2 k^2 (c0 + c2 f mu^2 + c4 f^2 mu^4 + c6 f^3 mu^6)``."""
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
        """Return ``-2 (c44 + c46 f mu^2 + c48 f^2 mu^4) (k mu f)^4``."""
        c44, c46, c48 = nlo_coefficients
        mu2 = mu**2
        coefficient = c44 + c46 * f * mu2 + c48 * f**2 * mu2**2
        return -2.0 * coefficient * (k * mu * f)**4

    @classmethod
    def nlo(cls, k, mu, f, ctr, base_pk):
        """NLO ``k^4`` counterterm from the full seven-slot ``ctr`` vector."""
        _, nlo_coefficients = cls.split_coefficients(ctr)
        return cls.nlo_shape(k, mu, f, nlo_coefficients) * base_pk


def stochasticity(k, mu, stochastic_coefficients):
    """``P_shot + (a0 + a2 mu^2) k^2`` (additive, not multiplied by any spectrum)."""
    p_shot, a0, a2 = stochastic_coefficients
    return (p_shot + (a0 + a2 * mu**2) * k**2)
