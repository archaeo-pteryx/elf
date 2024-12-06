import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
import numpy as np
from scipy.special import gamma
from functools import partial


class Hankel:

    def __init__(self, l, nu, x):
        nfft = len(x)
        dlnx = jnp.log(x[1] / x[0])
        eta_m = 2 * jnp.pi / (nfft * dlnx) * jnp.arange(nfft//2+1)
        g_l = get_g_l(l, nu + 1j * eta_m)
        y = (l + 1.) / x[::-1]
        
        self.u_m = (x[0] * y[0])**(-1j * eta_m) * g_l
        self.l = l
        self.nu = nu
        self.x = x
        self.y = y

    @partial(jit, static_argnames=['self'])
    def __call__(self, fx):
        c_m = jnp.fft.rfft(fx * self.x**(-self.nu))
        res = jnp.fft.irfft(jnp.conj(c_m * self.u_m)) * self.y**(-self.nu) * jnp.sqrt(jnp.pi) / 4.
        return res

def get_g_l(l, z):
    return 2.**z * get_g_base(l+0.5, z-1.5)

def get_g_base(mu, x):
    g_base = np.zeros(x.size, dtype=complex)
    
    cut = 200
    
    # normal formula
    sel = (np.abs(np.imag(x)) + np.abs(mu) <= cut) & (x != mu + 1 + 0.0j)
    g_base[sel] = gamma((mu + 1 + x[sel]) / 2.) / gamma((mu + 1 - x[sel]) / 2.)
    
    # asymptotic formula
    sel = np.abs(np.imag(x)) + np.abs(mu) > cut
    asym_plus = (mu + 1 + x[sel]) / 2.
    asym_minus = (mu + 1 - x[sel]) / 2.
    g_base[sel] = np.exp((asym_plus - 0.5) * np.log(asym_plus) - (asym_minus - 0.5) * np.log(asym_minus) - x[sel] \
        + 1./12. * (1./asym_plus - 1./asym_minus) + 1./360. * (1./asym_minus**3 - 1./asym_plus**3) + 1./1260 * (1./asym_plus**5 - 1./asym_minus**5))
    
    g_base[np.where(x == mu + 1 + 0.0j)[0]] = 0. + 0.0j
    
    return g_base