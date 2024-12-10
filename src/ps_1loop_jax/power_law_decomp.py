import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
from functools import partial


class PowerLawDecomp:

    def __init__(self, nu, x):
        self.nu = nu
        self.x = x
        self.nfft = len(x)
        self.eta_m = 2 * jnp.pi / (self.nfft * jnp.log(x[1] / x[0])) * (jnp.arange(self.nfft + 1) - self.nfft // 2)
        self.nu_m = self.nu + self.eta_m * 1j
        self.x_tile = jnp.tile(self.x, (len(self.nu_m), 1))
        self.nu_m_tile = jnp.tile(self.nu_m, (len(self.x), 1)).T

    @partial(jit, static_argnames=['self'])
    def get_c_m(self, fx):
        fn_biased = fx * (self.x / self.x[0])**(-self.nu)
        c_m = jnp.fft.fft(fn_biased) / self.nfft
        c_m = self.x[0]**(-self.nu_m) * jnp.hstack((c_m[1:self.nfft//2+1][::-1].conj(), c_m[:self.nfft//2+1]))
        c_m = c_m.at[0].set(c_m[0] / 2)
        c_m = c_m.at[-1].set(c_m[-1] / 2)
        return c_m

    @partial(jit, static_argnames=['self'])
    def get_decomp_data(self, fx):
        c_m = self.get_c_m(fx)
        c_m_tile = jnp.tile(c_m, (len(self.x), 1)).T

        func_q = c_m_tile * self.x_tile**self.nu_m_tile
        func_rec = jnp.sum(func_q, axis=0).real
        func_x0 = c_m * self.x[0]**self.nu_m

        return func_q, func_rec, func_x0
    