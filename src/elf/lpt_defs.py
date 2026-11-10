"""The hand-written definition of the LPT final integrand that is not a table, the redshift-space geometry of one mu.

``_rsd_geometry`` is shared by the forward (``lpt``) and the tangent rule (``lpt_rule``).  Everything else the backend
needs comes from the tables (``lpt_tables``).  Importing this module does no JAX computation.
"""
from typing import NamedTuple

import jax.numpy as jnp


class _RSDGeometry(NamedTuple):
    """Redshift-space geometry of one mu (:func:`_rsd_geometry`), with K = k + f (k.n) n."""
    Kfac: jnp.ndarray               # |K| / k
    c: jnp.ndarray                  # cosine of the angle between K and k
    s: jnp.ndarray                  # sine of that angle
    A_mu: jnp.ndarray               # n.Khat = (1 + f) mu / Kfac, the line of sight n along K
    B_mu: jnp.ndarray               # -n.e_x = sqrt(1 - mu^2) / Kfac, with e_x = (khat - c Khat) / s across K


def _rsd_geometry(f, mu, sin_mu):
    """The :class:`_RSDGeometry` of ``mu`` (``sin_mu`` its AD-regular sine)."""
    Kfac = jnp.sqrt(1 + f * (2 + f) * mu**2)
    c = (1 + f * mu**2) / Kfac
    s = f * mu * sin_mu / Kfac
    return _RSDGeometry(Kfac=Kfac, c=c, s=s, A_mu=(1 + f) * mu / Kfac, B_mu=sin_mu / Kfac)
