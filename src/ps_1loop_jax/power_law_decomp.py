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
        self.eta_m = 2 * jnp.pi / (self.nfft * jnp.log(x[1] / x[0])) * (jnp.arange(self.nfft) - self.nfft // 2)
        self.nu_m = self.nu + self.eta_m * 1j

    @partial(jit, static_argnames=['self'])
    def get_c_m(self, fx):
        c_m = jnp.fft.fft(fx * self.x**(-self.nu)) / self.nfft
        c_m = self.x[0]**(- self.eta_m * 1j) * jnp.hstack((c_m[1:self.nfft//2+1][::-1].conj(), c_m[:self.nfft//2]))
        return c_m

    @partial(jit, static_argnames=['self'])
    def get_decomp_data(self, fx):
        c_m = self.get_c_m(fx)
        
        f_q = c_m[:, None] * self.x[None, :]**self.nu_m[:, None]
        f_rec = jnp.sum(f_q, axis=0).real
        f_x0 = c_m * self.x[0]**self.nu_m

        return f_q, f_rec, f_x0
