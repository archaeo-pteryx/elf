import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from functools import partial


class PowerLawDecomp:

    def __init__(self, nu, kmin, kmax, nmax):
        self.nu = nu
        self.kmin = kmin
        self.kmax = kmax
        self.nmax = nmax
        self.kn = jnp.geomspace(kmin, kmax, nmax)
        self.eta_m = 2 * jnp.pi / (nmax / (nmax - 1) * jnp.log(kmax / kmin)) * (jnp.arange(nmax + 1) - nmax // 2)
        self.nu_m = self.nu + self.eta_m * 1j
        self.kn_tile = jnp.tile(self.kn, (len(self.nu_m), 1))
        self.nu_m_tile = jnp.tile(self.nu_m, (len(self.kn), 1)).T

    def compute(self, func, kwarg={}):
        fn_biased = func(self.kn, **kwarg) * (self.kn / self.kmin)**(-self.nu)
        self.c_m = get_c_m(fn_biased, self.nmax, self.kmin, self.nu_m)

        # reconstruct
        self.c_m_tile = jnp.tile(self.c_m, (len(self.kn), 1)).T
        self.func_q = self.c_m_tile * self.kn_tile**self.nu_m_tile
        self.func_rec = jnp.sum(self.func_q, axis=0)

        # k -> 0 limit
        self.func_k0 = self.c_m * self.kmin**self.nu_m

    def reconstruct(self):
        return self.func_rec
    
@partial(jax.jit, static_argnums=1)
def get_c_m(fn, nmax, kmin, nu_m):
    c_m = jnp.fft.fft(fn) / nmax
    c_m_sym = kmin**(-nu_m) * jnp.hstack((c_m[1:nmax//2+1][::-1].conj(), c_m[:nmax//2+1]))
    c_m_sym = c_m_sym.at[0].set(c_m_sym[0] / 2)
    c_m_sym = c_m_sym.at[-1].set(c_m_sym[-1] / 2)
    return c_m_sym