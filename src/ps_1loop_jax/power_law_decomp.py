import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp

@jit
def get_decomp_data(nu, x, fx):
    # setup
    nfft = len(x)
    eta_m = 2 * jnp.pi / (nfft * jnp.log(x[1] / x[0])) * (jnp.arange(nfft) - nfft // 2)
    nu_m = nu + eta_m * 1j

    # c_m
    c_m = jnp.fft.rfft(fx * x**(-nu)) / nfft
    c_m = x[0]**(-eta_m * 1j) * jnp.hstack((c_m[1:][::-1].conj(), c_m[:-1]))

    # decomposed data
    f_q = c_m[:, None] * x[None, :]**nu_m[:, None]
    f_rec = jnp.sum(f_q, axis=0).real
    f_x0 = c_m * x[0]**nu_m

    return f_q, f_rec, f_x0
