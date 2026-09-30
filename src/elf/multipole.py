import jax.numpy as jnp
import numpy as np


def prepare_mu_gauleg(ngauss, ells=(0, 2, 4)):
    """``(mu_positive, weights)``: the ``ngauss`` positive Gauss-Legendre nodes of [-1, 1] and
    ``weights[i, j] = (2 l + 1)/2 w_j L_l(mu_j)`` on all ``2 ngauss`` nodes, ``l = ells[i]``.
    """
    ells = tuple(ells)
    if not ells:
        raise ValueError("ells must contain at least one non-negative even integer")
    for ell in ells:
        if int(ell) != ell or ell < 0 or ell % 2:
            raise ValueError(f"ells must be non-negative even integers, got {ells!r}")
    ells = tuple(int(ell) for ell in ells)
    max_ell = max(ells)
    if max_ell >= 2 * ngauss:
        raise ValueError(
            f"max(ells)={max_ell} requires ngauss>={max_ell // 2 + 1}, "
            f"got ngauss={ngauss}"
        )

    mu, ws = np.polynomial.legendre.leggauss(2 * ngauss)
    mu_positive = jnp.array(mu[ngauss:])
    legendre_weights = jnp.stack([
        (2 * ell + 1) / 2 * ws * np.polynomial.legendre.Legendre([0] * ell + [1])(mu)
        for ell in ells
    ], axis=0)  # (len(ells), 2*nmu)

    return mu_positive, legendre_weights


def get_legendre_multipoles(pkmu, weights):
    pkmu = pkmu.T
    pkmu = jnp.concatenate([jnp.flip(pkmu, axis=0), pkmu], axis=0)
    pk_ells = jnp.einsum("ln,nk->lk", weights, pkmu)  # (nells, nk)
    return pk_ells


def get_k_mu_true_sin_for_ap(k, mu, alpha_perp, alpha_para):
    """AP map plus ``sin_true``.

    ``sin_true`` takes the root of the observed ``1 - mu^2``, not of ``1 - mu_true^2``,
    which gives ``inf * 0`` in AD at ``|mu| = 1``.
    """
    k = jnp.atleast_1d(k)
    mu = jnp.atleast_1d(mu)

    ratio = alpha_perp / alpha_para
    fac = jnp.sqrt(1 + mu**2 * (ratio**2 - 1))
    mu_true = mu * ratio / fac
    sin_true = jnp.sqrt(jnp.maximum(1 - mu**2, 0.0)) / fac
    k_true = jnp.outer(k, fac) / alpha_perp
    return k_true, mu_true, sin_true
