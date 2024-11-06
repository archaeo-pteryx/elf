import os
import glob, re

import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
from functools import partial

import jax.numpy as jnp
import quadax
import interpax

from .power_law_decomp import PowerLawDecomp
from . import pt_matrix
from . import utils_loop
from .utils_loop import get_pk, get_pk_int
from .utils_math import legendre
from . import ir_resum


class PowerSpectrum1Loop:

    def __init__(self, 
                 do_irres=True,
                 rbao=110.,
                 ks=0.2,
                 cross=False,
                 subtract_k0_limit=True,
                 kmin_fft=1e-5,
                 kmax_fft=1e3,
                 nfft=256,
                 ):
        
        # set up the FFTLog-based power-law decomposition
        self.config_fft = {
            'pk_lin nu=-0.3': {'nu':-0.3, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
            'pk_lin nu=-0.7': {'nu':-0.7, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
            'pk_lin nu=-1.6': {'nu':-1.6, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
        }
        self._set_power_law_decomp(self.config_fft)
        self._kmin = kmin_fft
        self._kmax = kmax_fft
        self._nfft = nfft
        self._kn = jnp.geomspace(kmin_fft, kmax_fft, nfft)
        self._mu = jnp.linspace(0., 1., 51)

        # store the names of 1-loop terms calculated with the FFTLog-based method
        self.name_pk_terms = ['22_dd','13_dd','I_d2','I_G2','I_d2_d2','I_G2_G2','I_d2_G2','F_G2']
        self.name_pkmu_terms = {}
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/M22_*.txt')
        self.name_pkmu_terms['22'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/M13_*.txt')
        self.name_pkmu_terms['13'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        self.name_pkmu_terms['tot'] = self.name_pkmu_terms['22'] + self.name_pkmu_terms['13']

        self.mat = {}
        self.matrix = {}
        # precompute the PT matrices
        self._set_matrix(self.name_pk_terms + self.name_pkmu_terms['tot'])

        self.do_irres = do_irres # flag to perform the IR resummation
        self.rbao = rbao
        self.ks = ks
        self.cross = cross # flag to enable the calculation of cross power spectra
        self.subtract_k0_limit = subtract_k0_limit # flag to subtract k -> 0 limit from 2-2 terms

    def _set_power_law_decomp(self, config_fft):
        # set multiple instances of PowerLawDecomp class.
        self.decomp = {}
        for name, config in config_fft.items():
            nu = config['nu']
            kmin = config['kmin']
            kmax = config['kmax']
            nfft = config['nfft']
            self.decomp[name] = PowerLawDecomp(nu, kmin, kmax, nfft)

    def _set_matrix(self, names=[]):
        for name in names:
            matfile = glob.glob(os.path.dirname(__file__)+'/pt_matrix/*/*/%s.txt' % (name))[0]
            if '22' in name or 'I' in name or '12' in name:
                self.mat[name] = pt_matrix.PTMatrix22(matfile)
            elif '13' in name or 'F' in name:
                self.mat[name] = pt_matrix.PTMatrix13(matfile)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))
        
        # precompute the PT matrices for appropriate FFT settings.
        for name in names:
            if name in utils_loop.kernel_to_decomp_dict.keys():
                name_dec = utils_loop.kernel_to_decomp_dict[name]
            else:
                species_list = list(self.name_pkmu_terms.keys())
                species_list.remove('tot')
                for species in species_list:
                    if name in self.name_pkmu_terms[species]:
                        name_dec = utils_loop.kernel_to_decomp_dict[species]
                        break

            if '22' in name or 'I' in name or '12' in name:
                nu_m1 = -0.5 * self.decomp[name_dec[0]].nu_m
                nu_m2 = -0.5 * self.decomp[name_dec[1]].nu_m
                nu_m1, nu_m2 = jnp.meshgrid(nu_m1, nu_m2)
                self.matrix[name] = self.mat[name](nu_m1, nu_m2).T
            elif '13' in name or 'F' in name:
                nu_m1 = -0.5 * self.decomp[name_dec[0]].nu_m
                self.matrix[name] = self.mat[name](nu_m1)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))
    
    @partial(jit, static_argnames=['self'])
    def get_pk_dict(self, pk_data):
        pk_lin = get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)

        name_list = ['22_dd','13_dd','I_d2','I_G2','I_d2_d2','I_G2_G2','I_d2_G2','F_G2']
        pk_dict = {}
        pk_dict['tree'] = pk_lin

        p1_q, p1_k, p1_k0 = self.decomp['pk_lin nu=-0.3'].get_decomposed_data(pk_lin)
        p2_q, p2_k, p2_k0 = self.decomp['pk_lin nu=-0.3'].get_decomposed_data(pk_lin)
        
        for name in name_list:

            if '22' in name or 'I' in name:
                pk = self._kn**3 * jnp.diag(jnp.dot(p1_q.T, jnp.dot(self.matrix[name], p2_q)).real)
                if self.subtract_k0_limit:
                    pk_k0 = self._kmin**3 * jnp.dot(p1_k0, jnp.dot(self.matrix[name], p2_k0)).real
                    pk = pk - pk_k0

            elif '13' in name or 'F' in name:
                pk = self._kn**3 * p2_k * jnp.dot(self.matrix[name], p1_q).real
                if name == '13_dd':
                    pk_lin_int = get_pk_int(pk_data)
                    pk = pk - (61. / 315.) * self._kn**2 * p2_k * pk_lin_int

            pk_dict[name] = pk

        return pk_dict

    @partial(jit, static_argnames=['self'])
    def get_source(self, pk_data, coeff):
        pk_dict = self.get_pk_dict(pk_data)

        Sk = coeff['b1']**2 * (pk_dict['tree'] + pk_dict['22_dd'] + pk_dict['13_dd']) \
            + coeff['b1'] * coeff['b2'] * pk_dict['I_d2'] \
            + 2 * coeff['b1'] * coeff['bG2'] * pk_dict['I_G2'] \
            + coeff['b2']**2 / 4 * pk_dict['I_d2_d2'] \
            + coeff['bG2']**2 * pk_dict['I_G2_G2'] \
            + coeff['b2'] * coeff['bG2'] * pk_dict['I_d2_G2'] \
            + 2 * coeff['b1'] * coeff['bG2'] * pk_dict['F_G2'] \
            + (4 / 5) * coeff['b1'] * coeff['bGamma3'] * pk_dict['F_G2']

        return Sk
    
    @partial(jit, static_argnames=['self'])
    def get_pk_1loop(self, k, pk_data, coeff):
        Sk = self.get_source(pk_data, coeff)
        pk_interp = interpax.Interpolator1D(self._kn, Sk)
        pk = pk_interp(k)
        return pk

    @partial(jit, static_argnames=['self'])
    def get_pkmu(self, k, mu, pk_data, params):
        # tree + 1-loop
        if self.do_irres:
            pkmu = self.get_pkmu_irres_LO_NLO(k, mu, pk_data, params)
        else:
            pkmu_tree = self.get_pkmu_lin(k, mu, pk_data, params)
            pkmu_1loop = self.get_pkmu_1loop(k, mu, pk_data, params)
            pkmu = pkmu_tree + pkmu_1loop
        
        # counterterm
        pkmu_ctr_k2 = self.get_pkmu_ctr_k2(k, mu, pk_data, params)
        pkmu_ctr_k4 = self.get_pkmu_ctr_k4(k, mu, pk_data, params)
        pkmu = pkmu + pkmu_ctr_k2 + pkmu_ctr_k4
        
        # NOTE: can be removed
        # stochasticity
        pkmu_stoch = self.get_pkmu_stoch(k, mu, params)
        pkmu = pkmu + pkmu_stoch
        
        return pkmu

    @partial(jit, static_argnames=['self', 'num'])
    def get_pk_ell(self, k, l, pk_data, params, num=256):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.linspace(0., 1., num)

        pkmu = self.get_pkmu(k, mu, pk_data, params)

        leg = jnp.tile((2*l+1) * legendre(l, mu), (len(k), 1))
        pk_ell = quadax.simpson(pkmu * leg, x=mu, axis=1)

        return pk_ell

    @partial(jit, static_argnames=['self'])
    def get_pkmu_ref(self, k, mu, alpha_perp, alpha_para, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        # mapping of (k, mu)
        fac = jnp.sqrt(1 + mu**2 * ((alpha_perp / alpha_para)**2 - 1))
        mu_true = mu * (alpha_perp / alpha_para) / fac
        k_true = jnp.kron(k, fac).reshape(len(k), len(mu)) / alpha_perp

        # spline interpolation
        pkmu_grid = self.get_pkmu(self._kn, self._mu, pk_data, params)
        mu_tile = jnp.tile(mu_true, (len(k), 1))
        pkmu = interpax.interp2d(jnp.ravel(k_true), jnp.ravel(mu_tile), self._kn, self._mu, pkmu_grid, extrap=True)
        pkmu = pkmu.reshape(len(k), len(mu)) / (alpha_perp**2 * alpha_para)

        return pkmu

    @partial(jit, static_argnames=['self', 'num'])
    def get_pk_ell_ref(self, k, l, alpha_perp, alpha_para, pk_data, params, num=256):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.linspace(0., 1., num)

        pkmu = self.get_pkmu_ref(k, mu, alpha_perp, alpha_para, pk_data, params)

        leg = jnp.tile((2*l+1) * legendre(l, mu), (len(k), 1))
        pk_ell = quadax.simpson(pkmu * leg, x=mu, axis=1)

        return pk_ell

    @partial(jit, static_argnames=['self'])
    def get_pkmu_lin(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        Z1_1 = bias1['b1'] + f * mu**2
        Z1_2 = bias2['b1'] + f * mu**2
        pk_lin = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pkmu = jnp.kron(pk_lin, Z1_1 * Z1_2).reshape(len(k), len(mu))

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict(self, pk_data):
        pk_lin = get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)
        p_q, p_k, p_k0 = self.decomp['pk_lin nu=-0.7'].get_decomposed_data(pk_lin)
        
        pk_dict = {}

        for name in self.name_pkmu_terms['22']:
            pk = self._kn**3 * jnp.diag(jnp.dot(p_q.T, jnp.dot(self.matrix[name], p_q)).real)
            if self.subtract_k0_limit:
                pk_k0 = self._kmin**3 * jnp.dot(p_k0, jnp.dot(self.matrix[name], p_k0)).real
                pk = pk - pk_k0
            pk_dict[name] = pk
        
        for name in self.name_pkmu_terms['13']:
            pk = self._kn**3 * p_k * jnp.dot(self.matrix[name], p_q).real
            pk_dict[name] = pk

        return pk_dict
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_data(self, pk_data, params):
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        # pk_lin = get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_dict = self.get_pkmu_dict(pk_data)

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))
        for name in self.name_pkmu_terms['tot']:
            # degrees of f, mu, and galaxy bias parameters
            nf, nmu, bias_degree_dict = utils_loop.get_degree_info(name)

            # calculate the coefficient that consists of bias parameters
            bias_factor = utils_loop.get_bias_factor(bias_degree_dict, bias1, bias2)

            pkmu_term = bias_factor * f**nf * jnp.kron(pk_dict[name], self._mu**nmu).reshape(len(self._kn), len(self._mu))
            pkmu = pkmu + pkmu_term

        pkmu = pkmu + self.get_pkmu_13_UV(self._kn, self._mu, pk_data, params)

        return pkmu
    
    @partial(jax.jit, static_argnames=['self'])
    def get_pkmu_13_UV(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        Z1_g = bias1['b1'] + f * mu**2
        Z3_g_UV = - 61./315. * bias2['b1'] - 64./21. * bias2['bG2'] - 128./105. * bias2['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * bias2['b1']) * f * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * bias2['b1']) * f**2 * mu**2
        Z3_g_UV += (- 46./105.) * f**2 * mu**4
        Z3_g_UV += (- 1./3.) * f**3 * mu**4
        Z1Z3_UV_1 = Z1_g * Z3_g_UV

        Z1_g = bias2['b1'] + f * mu**2
        Z3_g_UV = - 61./315. * bias1['b1'] - 64./21. * bias1['bG2'] - 128./105. * bias1['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * bias1['b1']) * f * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * bias1['b1']) * f**2 * mu**2
        Z3_g_UV += (- 46./105.) * f**2 * mu**4
        Z3_g_UV += (- 1./3.) * f**3 * mu**4
        Z1Z3_UV_2 = Z1_g * Z3_g_UV

        Z1Z3_UV = (Z1Z3_UV_1 + Z1Z3_UV_2) / 2

        pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_int = get_pk_int(pk_data)

        pkmu_13 = jnp.kron(k**2 * pk * pk_int, Z1Z3_UV).reshape(len(k), len(mu))

        return pkmu_13
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_1loop(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        pkmu_data = self.get_pkmu_data(pk_data, params)

        k_tile = jnp.tile(k, (len(mu), 1)).T
        mu_tile = jnp.tile(mu, (len(k), 1))
        pkmu = interpax.interp2d(jnp.ravel(k_tile), jnp.ravel(mu_tile), self._kn, self._mu, pkmu_data)
        pkmu = pkmu.reshape(len(k), len(mu))

        return pkmu
    
    @partial(jax.jit, static_argnames=['self'])
    def get_pkmu_irres_LO_NLO(self, k, mu, pk_data, params):
        h = params['h']
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        # wiggly-non-wiggly decomposition
        pk_nw_data = ir_resum.get_pk_nw_data(pk_data, h, khmin=7e-5, khmax=7., kmin_interp=self._kmin, kmax_interp=self._kmax)

        pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_nw = get_pk(k, pk_nw_data, kmin=self._kmin, kmax=self._kmax)
        pk_w = pk - pk_nw

        # BAO damping factor in redshift space
        Sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.rbao, self.ks)
        dSigma2 = ir_resum.get_dSigma2(pk_nw_data, self.rbao, self.ks)
        Sigma2_tot = (1 + mu**2 * f * (2 + f)) * Sigma2 + f**2 * mu**2 * (mu**2 - 1) * dSigma2
        damp_fac = jnp.kron(k**2, Sigma2_tot).reshape(len(k), len(mu))

        # LO term
        pk_nw_tile = jnp.tile(pk_nw, (len(mu), 1)).T
        pk_w_tile = jnp.tile(pk_w, (len(mu), 1)).T
        Z1_1 = bias1['b1'] + f * mu**2
        Z1_2 = bias2['b1'] + f * mu**2
        Z1_factor = jnp.tile(Z1_1 * Z1_2, (len(k), 1))
        pkmu_irres_tree = Z1_factor * (pk_nw_tile + jnp.exp(-damp_fac) * pk_w_tile * (1 + damp_fac))
        
        # NLO term
        pkmu_1loop = self.get_pkmu_1loop(k, mu, pk_data, params)
        pkmu_1loop_nw = self.get_pkmu_1loop(k, mu, pk_nw_data, params)
        pkmu_1loop_w = pkmu_1loop - pkmu_1loop_nw
        pkmu_irres_1loop = pkmu_1loop_nw + jnp.exp(-damp_fac) * pkmu_1loop_w

        pkmu = pkmu_irres_tree + pkmu_irres_1loop

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_ctr_k2(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        h = params['h']
        f = params['f']
        ctr1 = params['ctr']
        ctr2 = params['ctr2'] if self.cross else params['ctr']

        ctr_k2_mu = (ctr1['c0'] + ctr2['c0']) / 2
        ctr_k2_mu = ctr_k2_mu + (ctr1['c2'] + ctr2['c2']) / 2 * f * mu**2
        ctr_k2_mu = ctr_k2_mu + (ctr1['c4'] + ctr2['c4']) / 2 * f**2 * mu**4

        if self.do_irres:
            pk = self._get_pk_irres_rsd(k, mu, pk_data, h, f)
            pkmu_ctr_k2 = - 2 * jnp.kron(k**2, ctr_k2_mu).reshape(len(k), len(mu)) * pk
        else:
            pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
            pkmu_ctr_k2 = - 2 * jnp.kron(k**2 * pk, ctr_k2_mu).reshape(len(k), len(mu))

        return pkmu_ctr_k2
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_ctr_k4(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        h = params['h']
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']
        ctr1 = params['ctr']
        ctr2 = params['ctr2'] if self.cross else params['ctr']

        ctr_k4_mu = (ctr1['cfog'] + ctr2['cfog']) / 2 * f**4 * mu**4 * (bias1['b1'] + f * mu**2) * (bias2['b1'] + f * mu**2)

        if self.do_irres:
            pk = self._get_pk_irres_rsd(k, mu, pk_data, h, f)
            pkmu_ctr_k4 = - jnp.kron(k**4, ctr_k4_mu).reshape(len(k), len(mu)) * pk
        else:
            pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
            pkmu_ctr_k4 = - jnp.kron(k**4 * pk, ctr_k4_mu).reshape(len(k), len(mu))

        return pkmu_ctr_k4

    @partial(jit, static_argnames=['self'])
    def _get_pk_irres_rsd(self, k, mu, pk_data, h, f):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        pk_nw, pk_w, damp_fac = self._get_irres_components(k, mu, pk_data, h, f)

        pk_nw_tile = jnp.tile(pk_nw, (len(mu), 1)).T
        pk_w_tile = jnp.tile(pk_w, (len(mu), 1)).T

        pk = pk_nw_tile + jnp.exp(-damp_fac) * pk_w_tile
        return pk

    @partial(jit, static_argnames=['self'])
    def _get_irres_components(self, k, mu, pk_data, h, f):
        # wiggly-non-wiggly decomposition
        pk_nw_data = ir_resum.get_pk_nw_data(pk_data, h, khmin=7e-5, khmax=7., kmin_interp=self._kmin, kmax_interp=self._kmax)

        pk = get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_nw = get_pk(k, pk_nw_data, kmin=self._kmin, kmax=self._kmax)
        pk_w = pk - pk_nw

        # BAO damping factor in redshift space
        Sigma2 = ir_resum.get_Sigma2(pk_nw_data, self.rbao, self.ks)
        dSigma2 = ir_resum.get_dSigma2(pk_nw_data, self.rbao, self.ks)
        Sigma2_tot = (1 + mu**2 * f * (2 + f)) * Sigma2 + f**2 * mu**2 * (mu**2 - 1) * dSigma2

        damp_fac = jnp.kron(k**2, Sigma2_tot).reshape(len(k), len(mu))

        return pk_nw, pk_w, damp_fac

    # NOTE: can be removed
    @partial(jit, static_argnames=['self'])
    def get_pkmu_stoch(self, k, mu, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        stoch = params['stoch']
        k_nl = params['k_nl']

        ndens = params['ndens']
        pkmu = stoch['P_shot'] \
            + stoch['a0'] * jnp.kron((k / k_nl)**2, mu**0).reshape(len(k), len(mu)) \
            + stoch['a2'] * jnp.kron((k / k_nl)**2, mu**2).reshape(len(k), len(mu))
        pkmu = (1. / ndens) * pkmu

        return pkmu