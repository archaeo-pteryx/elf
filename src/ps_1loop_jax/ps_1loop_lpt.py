import os
import glob, re

import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp
import numpy as np
import quadax
import interpax

from .hankel import Hankel

from . import pt_coeff
from . import pt_matrix
from . import utils_loop
from .utils_loop import get_pk, get_pk_int, get_pk_int2
from .utils_math import legendre
from . import utils_lpt

from . import ir_resum


class PowerSpectrum1LoopLPT:

    def __init__(self, 
                 cross=False,
                 subtract_k0_limit=True,
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=256,
                 ):
        
        self.cross = cross # flag to enable the calculation of cross power spectra
        self.subtract_k0_limit = subtract_k0_limit # flag to subtract k -> 0 limit from 2-2 terms

        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self._k = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._q = 1 / self._k[::-1]
        self._mu = jnp.linspace(0., 1., 51)

        self._initialize_loop_coeff()
            
    def _initialize_loop_coeff(self):
        l_list = [0,1,2,3,4]
        
        self.ln_list = [(0,0), (0,-2), (0,2), (1,-1), (1,1), (2,0), (2,-2), (2,2), (3,-1), (3,1), (4,0)]

        # set the Hankel transforms
        self.hankel_pk2xi = {}
        self.hankel_xi2pk = {}
        nu = 1.1
        for l in l_list:
            self.hankel_pk2xi[l] = Hankel(l, nu, self._k, npad=(self._nfft//2), x_high=(self._kmax/10.), c_window_width=0.2)
            self.hankel_xi2pk[l] = Hankel(l, nu, self._q, npad=(self._nfft//2), x_high=1e4, c_window_width=0.2)
        
        # load the coefficients of G_0_0
        loaded = jnp.load(os.path.dirname(__file__)+'/lpt_rsd_coeff/G_0_0_coeffs.npz')
        self.G_0_0_coeffs = {int(k): jnp.array(loaded[k]) for k in loaded.files}

    @partial(jit, static_argnames=['self'])
    def get_xi_ln(self, pk_data):
        pk_lin = get_pk(self._k, pk_data, kmin=self._kmin, kmax=self._kmax)

        xi_ln = {}
        for (l, n) in self.ln_list:
            _, xi = self.hankel_pk2xi[l](pk_lin * self._k**(n+3))
            xi_ln[(l, n)] = xi / (2 * jnp.pi**2)
        
        return xi_ln
    
    @partial(jit, static_argnames=['self'])
    def get_corrs(self, pk_data):
        xi_ln = self.get_xi_ln(pk_data)

        corrs = {}

        corrs['X_lin'] = 2./3 * (xi_ln[(0,-2)][0] - xi_ln[(0,-2)] - xi_ln[(2,-2)])
        corrs['Y_lin'] = 2 * xi_ln[(2,-2)]
        
        corrs['U_lin'] = - xi_ln[(1,-1)]
        corrs['cor_lin'] = xi_ln[(0,0)]
        
        return corrs
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_zel(self, k, mu, f, pk_data):
        corrs = self.get_corrs(pk_data)
        
        X = corrs['X_lin']
        Y = corrs['Y_lin']
        
        Ksq = k**2 * (1 + f * (2 + f) * mu**2)
        s = f * mu * jnp.sqrt(1 - mu**2) / jnp.sqrt(1 + f * (2 + f) * mu**2)

        base = 4 * jnp.pi * self._q**3 * jnp.exp(- 0.5 * Ksq * (X + Y))
        
        pk = 0.
        for l in range(self.lmax + 1):
            coeff = self.G_0_0_coeffs[l]
            integrand = base * (-2 / (k * self._q))**(l) * utils_lpt.get_G_0_0(0.5 * Ksq * Y, s**2, coeff)
            k_fft, pk_fft = self.hankel_xi2pk[l](integrand)
            pk = pk + interpax.interp1d(jnp.log(k), jnp.log(k_fft), pk_fft)

        return pk