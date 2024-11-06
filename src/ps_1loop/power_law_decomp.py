import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
from functools import partial


class PowerLawDecomp:

    def __init__(self, nu, kmin, kmax, nfft):
        self.nu = nu
        self.kmin = kmin
        self.kmax = kmax
        self.nfft = nfft
        self.kn = jnp.geomspace(kmin, kmax, nfft)
        self.eta_m = 2 * jnp.pi / (nfft / (nfft - 1) * jnp.log(kmax / kmin)) * (jnp.arange(nfft + 1) - nfft // 2)
        self.nu_m = self.nu + self.eta_m * 1j
        self.kn_tile = jnp.tile(self.kn, (len(self.nu_m), 1))
        self.nu_m_tile = jnp.tile(self.nu_m, (len(self.kn), 1)).T

    @partial(jit, static_argnames=['self'])
    def get_c_m(self, data_array):
        fn_biased = data_array * (self.kn / self.kmin)**(-self.nu)
        c_m = jnp.fft.fft(fn_biased) / self.nfft
        c_m = self.kmin**(-self.nu_m) * jnp.hstack((c_m[1:self.nfft//2+1][::-1].conj(), c_m[:self.nfft//2+1]))
        c_m = c_m.at[0].set(c_m[0] / 2)
        c_m = c_m.at[-1].set(c_m[-1] / 2)
        return c_m

    @partial(jit, static_argnames=['self'])
    def get_decomposed_data(self, data_array):
        c_m = self.get_c_m(data_array)
        c_m_tile = jnp.tile(c_m, (len(self.kn), 1)).T

        func_q = c_m_tile * self.kn_tile**self.nu_m_tile
        func_rec = jnp.sum(func_q, axis=0).real
        func_k0 = c_m * self.kmin**self.nu_m

        return func_q, func_rec, func_k0
    