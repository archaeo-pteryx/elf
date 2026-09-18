"""Cosmology-independent coefficients for the LPT redshift-space moments."""

from fractions import Fraction
from functools import lru_cache
from math import comb, factorial

import numpy as np


def _rising_half_integer(x: Fraction, order: int) -> Fraction:
    value = Fraction(1)
    for offset in range(order):
        value *= x + offset
    return value


@lru_cache(maxsize=None)
def generate_g00_coefficient(ell: int) -> np.ndarray:
    """Return the exact-rational Appendix-G coefficient matrix for one ell.

    The returned float64 values implement Eq. (70) of
    ``docs/JAX_EFT_power_spectrum.pdf``.  Exact arithmetic is used only during
    model construction; the cached result is converted to the model's JAX
    dtype afterwards.
    """
    if isinstance(ell, (bool, np.bool_)) or not isinstance(ell, (int, np.integer)):
        raise TypeError("ell must be a non-negative integer")
    ell = int(ell)
    if ell < 0:
        raise ValueError("ell must be a non-negative integer")

    result = np.empty((ell + 1, ell + 1), dtype=np.float64)
    for k in range(ell + 1):
        for i in range(ell + 1):
            total = Fraction(0)
            for n in range(i + 1):
                total += (
                    (-1) ** n
                    * comb(i, n)
                    * comb(ell + n, k)
                    * _rising_half_integer(
                        Fraction(2 * (ell + n) + 1, 2), ell - k
                    )
                )
            sign = -1 if (i - k) % 2 else 1
            result[k, i] = float(
                Fraction(sign, factorial(ell - k) * factorial(i)) * total
            )

    # Prevent accidental mutation of the process-wide cached table.
    result.setflags(write=False)
    return result


def validate_lmax(lmax: int) -> int:
    """Validate and normalize the inclusive angular-expansion order."""
    if isinstance(lmax, (bool, np.bool_)) or not isinstance(lmax, (int, np.integer)):
        raise TypeError("lmax must be a non-negative integer")
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be a non-negative integer")
    return lmax
