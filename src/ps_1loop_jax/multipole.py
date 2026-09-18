import jax.numpy as jnp
import numpy as np

def prepare_mu_gauleg(ngauss):

    mu, ws = np.polynomial.legendre.leggauss(2 * ngauss)
    leg0 = np.polynomial.legendre.Legendre((1))(mu)
    leg2 = np.polynomial.legendre.Legendre((0,0,1))(mu)
    leg4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(mu)

    mu_positive = jnp.array(mu[ngauss:])
    legendre_weights = jnp.stack([
        0.5 * ws * leg0,   # (2*nmu,)
        2.5 * ws * leg2,
        4.5 * ws * leg4
    ], axis=0)  # (3, 2*nmu)

    return mu_positive, legendre_weights

def get_legendre_multipoles(pkmu, weights):
    pkmu = pkmu.T
    pkmu = jnp.concatenate([jnp.flip(pkmu, axis=0), pkmu], axis=0)
    pk_ells = jnp.einsum("ln,nk->lk", weights, pkmu)  # (3, nk)
    return pk_ells

def get_k_mu_true_for_ap(k, mu, alpha_perp, alpha_para):
    k = jnp.atleast_1d(k)
    mu = jnp.atleast_1d(mu)

    fac = jnp.sqrt(1 + mu**2 * ((alpha_perp / alpha_para)**2 - 1))
    mu_true = mu * (alpha_perp / alpha_para) / fac
    k_true = jnp.outer(k, fac) / alpha_perp

    return k_true, mu_true


def get_k_mu_true_sin_for_ap(k, mu, alpha_perp, alpha_para):
    """AP map plus a derivative-regular transverse direction cosine.

    Computing ``sqrt(1 - mu_true**2)`` after the AP map creates the
    indeterminate AD product ``inf * 0`` at ``|mu|=1``.  The algebraically
    identical expression below takes the square root only of the observed
    coordinate, which is constant when differentiating the AP parameters.
    """
    k = jnp.atleast_1d(k)
    mu = jnp.atleast_1d(mu)

    ratio = alpha_perp / alpha_para
    fac = jnp.sqrt(1 + mu**2 * (ratio**2 - 1))
    mu_true = mu * ratio / fac
    sin_true = jnp.sqrt(jnp.maximum(1 - mu**2, 0.0)) / fac
    k_true = jnp.outer(k, fac) / alpha_perp
    return k_true, mu_true, sin_true
