"""Shared EFT counterterm and stochasticity shapes.

:class:`Counterterms` decides where the counterterms enter.  For the 'linear' and
'linear_ir_resum' bases they are added in k space to the spectrum (the backends
supply ``P_lin``, ``P_nw``, ``P_w`` and the damping exponent at the true k).  For
'zeldovich' they weight LPT's fused Zel'dovich integrand (template 12).
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
    for a cross spectrum the coefficients belong to the pair.  The base decides
    where they enter (:meth:`kspace_base_pk`, :meth:`add_kspace`, :meth:`zeldovich_weight`).
    """

    base: CountertermBase

    def __post_init__(self):
        if self.base not in _VALID_COUNTERTERM_BASES:
            raise ValueError(
                "counterterm_base must be 'linear', 'linear_ir_resum', or "
                f"'zeldovich', got {self.base!r}"
            )

    def kspace_base_pk(self, linear_at):
        """Base spectrum of the k-space counterterms, None for 'zeldovich'.

        ``linear_at()`` returns ``(pk, pk_nw, pk_w, damp_exponent)`` at the true k
        (``base.PowerSpectrum._linear_at``) and is called only for the k-space bases.
        'linear' gives ``pk`` and 'linear_ir_resum' ``pk_nw + exp(-damp_exponent) pk_w``.
        """
        if self.base == "zeldovich":
            return None
        pk, pk_nw, pk_w, damp_exponent = linear_at()
        if self.base == "linear":
            return pk
        return pk_nw + jnp.exp(-damp_exponent) * pk_w

    def add_kspace(self, pkmu, k, mu, f, ctr, ctr_base_pk):
        """``pkmu`` plus the leading and NLO counterterms on ``ctr_base_pk`` (``pkmu`` itself if None)."""
        if ctr_base_pk is None:
            return pkmu
        return (pkmu + self.leading(k, mu, f, ctr, ctr_base_pk)
                + self.nlo(k, mu, f, ctr, ctr_base_pk))

    def zeldovich_weight(self, k, mu, f, ctr):
        """Weight of LPT's Zel'dovich template, the leading plus NLO shape for 'zeldovich' and 0 otherwise.

        The k-space bases are added by :meth:`add_kspace` instead.
        """
        if self.base != "zeldovich":
            return jnp.zeros_like(k)
        leading_coefficients, nlo_coefficients = self.split_coefficients(ctr)
        return (self.leading_shape(k, mu, f, leading_coefficients)
                + self.nlo_shape(k, mu, f, nlo_coefficients))

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
    def leading(cls, k, mu, f, ctr, ctr_base_pk):
        """Leading ``k^2`` counterterm from the full seven-slot ``ctr`` vector."""
        leading_coefficients, _ = cls.split_coefficients(ctr)
        return cls.leading_shape(k, mu, f, leading_coefficients) * ctr_base_pk

    @staticmethod
    def nlo_shape(k, mu, f, nlo_coefficients):
        """Return ``-2 (c44 + c46 f mu^2 + c48 f^2 mu^4) (k mu f)^4``."""
        c44, c46, c48 = nlo_coefficients
        mu2 = mu**2
        coefficient = c44 + c46 * f * mu2 + c48 * f**2 * mu2**2
        return -2.0 * coefficient * (k * mu * f)**4

    @classmethod
    def nlo(cls, k, mu, f, ctr, ctr_base_pk):
        """NLO ``k^4`` counterterm from the full seven-slot ``ctr`` vector."""
        _, nlo_coefficients = cls.split_coefficients(ctr)
        return cls.nlo_shape(k, mu, f, nlo_coefficients) * ctr_base_pk


def stochasticity(k, mu, stochastic_coefficients):
    """``e00 + (e20 + e22 mu^2) k^2`` (additive, not multiplied by any spectrum)."""
    e00, e20, e22 = stochastic_coefficients
    return (e00 + (e20 + e22 * mu**2) * k**2)
