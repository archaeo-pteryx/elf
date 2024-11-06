import os
import glob, re

import jax
jax.config.update('jax_enable_x64', True)
from jax import jit
import jax.numpy as jnp
from functools import partial

from .ps_1loop import PowerSpectrum1Loop
from . import utils_loop
from . import ir_resum


class PowerSpectrum1LoopLPNG(PowerSpectrum1Loop):

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
            'pk_1phi nu=-0.9': {'nu':-0.9, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
            'pk_1phi nu=-1.6': {'nu':-1.6, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
            'pk_1phi nu=-2.1': {'nu':-2.1, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
            'Mk nu=0.2': {'nu':0.2, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nfft':nfft},
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
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M22_*.txt')
        self.name_pkmu_terms['12_lpng1'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M12_*lpng2*.txt')
        self.name_pkmu_terms['12_lpng2'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        self.name_pkmu_terms['12_lpng'] = self.name_pkmu_terms['12_lpng1'] + self.name_pkmu_terms['12_lpng2']
        self.name_pkmu_terms['22_lpng'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M13_*lpng1*.txt')
        self.name_pkmu_terms['13_lpng1'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M13_*lpng3*.txt')
        self.name_pkmu_terms['13_lpng3'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M12_*lpng1*.txt')
        self.name_pkmu_terms['13_lpng'] = self.name_pkmu_terms['13_lpng1'] + self.name_pkmu_terms['13_lpng3']
        self.name_pkmu_terms['tot'] = self.name_pkmu_terms['22'] + self.name_pkmu_terms['13']
        self.name_pkmu_terms['tot'] += self.name_pkmu_terms['12_lpng'] + self.name_pkmu_terms['22_lpng'] + self.name_pkmu_terms['13_lpng']

        self.mat = {}
        self.matrix = {}
        # precompute the PT matrices
        self._set_matrix(self.name_pk_terms + self.name_pkmu_terms['tot'])
        self._compute_matrix(self.name_pk_terms + self.name_pkmu_terms['tot'])

        self.do_irres = do_irres # flag to perform the IR resummation
        self.rbao = rbao
        self.ks = ks
        self.cross = cross # flag to enable the calculation of cross power spectra
        self.subtract_k0_limit = subtract_k0_limit # flag to subtract k -> 0 limit from 2-2 terms

    @partial(jit, static_argnames=['self'])
    def get_pkmu_lin(self, k, mu, pk_data, params):
        k = jnp.atleast_1d(k).astype(float)
        mu = jnp.atleast_1d(mu).astype(float)

        f_nl = params['f_nl']
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        Z1_1 = bias1['b1'] + f * mu**2
        Z1_lpng_1 = bias1['bphi'] * f_nl / self.get_Mk(k)
        Z1_2 = bias2['b1'] + f * mu**2
        Z1_lpng_2 = bias2['bphi'] * f_nl / self.get_Mk(k)

        Z1_1_tile = jnp.tile(Z1_1, (len(k), 1)) + jnp.tile(Z1_lpng_1, (len(mu),1)).T
        Z1_2_tile = jnp.tile(Z1_2, (len(k), 1)) + jnp.tile(Z1_lpng_2, (len(mu),1)).T

        pk_lin = self.get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_lin_tile = jnp.tile(pk_lin, (len(mu),1)).T

        pkmu = Z1_1_tile * Z1_2_tile * pk_lin_tile

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict_22(self, pk_data):
        name_list = self.name_pkmu_terms['22']
        pk_dict = {}

        pk_lin = self.get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)

        for name in name_list:
            name_dec = utils_loop.kernel_to_decomp_dict['22_gg']

            p1_q, p1_k, p1_k0 = self.decomp[name_dec[0]].get_decomposed_data(pk_lin)
            p2_q, p2_k, p2_k0 = self.decomp[name_dec[1]].get_decomposed_data(pk_lin)

            pk = self._kn**3 * jnp.diag(jnp.dot(p1_q.T, jnp.dot(self.matrix[name], p2_q)).real)
            if self.subtract_k0_limit:
                pk_k0 = self._kmin**3 * jnp.dot(p1_k0, jnp.dot(self.matrix[name], p2_k0)).real
                pk = pk - pk_k0
            
            pk_dict[name] = pk

        return pk_dict
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict_13(self, pk_data):
        name_list = self.name_pkmu_terms['13']
        pk_dict = {}

        pk_lin = self.get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)
        
        for name in name_list:
            name_dec = utils_loop.kernel_to_decomp_dict['13_gg']

            p1_q, p1_k, p1_k0 = self.decomp[name_dec[0]].get_decomposed_data(pk_lin)
            p2_q, p2_k, p2_k0 = self.decomp[name_dec[1]].get_decomposed_data(pk_lin)

            pk = self._kn**3 * p2_k * jnp.dot(self.matrix[name], p1_q).real
            
            pk_dict[name] = pk

        return pk_dict
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict_12_lpng(self, pk_lin, Mk, pk_1phi):
        name_list = self.name_pkmu_terms['12_lpng']
        pk_dict = {}
        
        for name in name_list:

            if 'lpng1' in name:
                name_dec = utils_loop.kernel_to_decomp_dict['12_lpng1']
                p1_q, p1_k, p1_k0 = self.decomp[name_dec[0]].get_decomposed_data(pk_1phi)
                p2_q, p2_k, p2_k0 = self.decomp[name_dec[1]].get_decomposed_data(Mk)

            elif 'lpng2' in name:
                name_dec = utils_loop.kernel_to_decomp_dict['12_lpng2']
                p1_q, p1_k, p1_k0 = self.decomp[name_dec[0]].get_decomposed_data(pk_1phi)
                p2_q, p2_k, p2_k0 = self.decomp[name_dec[1]].get_decomposed_data(pk_1phi)

            pk = self._kn**3 * jnp.diag(jnp.dot(p1_q.T, jnp.dot(self.matrix[name], p2_q)).real)
            
            pk_dict[name] = pk

        return pk_dict
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict_22_lpng(self, pk_lin, Mk, pk_1phi):
        name_list = self.name_pkmu_terms['22_lpng']
        pk_dict = {}
        
        for name in name_list:

            name_dec = utils_loop.kernel_to_decomp_dict['22_lpng']
            p1_q, p1_k, p1_k0 = self.decomp[name_dec[0]].get_decomposed_data(pk_lin)
            p2_q, p2_k, p2_k0 = self.decomp[name_dec[1]].get_decomposed_data(pk_1phi)        

            pk = self._kn**3 * jnp.diag(jnp.dot(p1_q.T, jnp.dot(self.matrix[name], p2_q)).real)
            if self.subtract_k0_limit:
                pk_k0 = self._kmin**3 * jnp.dot(p1_k0, jnp.dot(self.matrix[name], p2_k0)).real
                pk = pk - pk_k0
            
            pk_dict[name] = pk

        return pk_dict
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_dict_13_lpng(self, pk_lin, Mk, pk_1phi):
        name_list = self.name_pkmu_terms['13_lpng']
        pk_dict = {}
        
        for name in name_list:
            
            if 'lpng1' in name:
                name_dec = utils_loop.kernel_to_decomp_dict['13_lpng1']
                p1_q, p1_k, p1_k0 = self.decomp[name_dec[0]].get_decomposed_data(pk_lin)
                p2_q, p2_k, p2_k0 = self.decomp[name_dec[1]].get_decomposed_data(pk_1phi)
            
            elif 'lpng3' in name:
                name_dec = utils_loop.kernel_to_decomp_dict['13_lpng3']
                p1_q, p1_k, p1_k0 = self.decomp[name_dec[0]].get_decomposed_data(pk_lin)
                p2_q, p2_k, p2_k0 = self.decomp[name_dec[1]].get_decomposed_data(pk_1phi)

            pk = self._kn**3 * p2_k * jnp.dot(self.matrix[name], p1_q).real
            
            pk_dict[name] = pk

        return pk_dict
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_data_gauss(self, pk_data, params):
        pkmu_22 = self.get_pkmu_data_22(pk_data, params)
        pkmu_13 = self.get_pkmu_data_13(pk_data, params)
        pkmu = pkmu_22 + pkmu_13
        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_data_lpng(self, pk_data, Mk_data, params):
        pkmu_12_lpng = self.get_pkmu_data_12_lpng(pk_data, Mk_data, params)
        pkmu_22_lpng = self.get_pkmu_data_22_lpng(pk_data, Mk_data, params)
        pkmu_13_lpng = self.get_pkmu_data_13_lpng(pk_data, Mk_data, params)
        pkmu = pkmu_12_lpng + pkmu_22_lpng + pkmu_13_lpng
        return pkmu

    @partial(jit, static_argnames=['self'])
    def get_pkmu_data_22(self, pk_data, params):
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        pk_dict = self.get_pkmu_dict_22(pk_data)

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))

        for name in self.name_pkmu_terms['22']:
            # degrees of f, mu, and galaxy bias parameters
            nf, nmu, bias_degree_dict = utils_loop.get_degree_info(name)

            # calculate the coefficient that consists of bias parameters
            bias_factor = utils_loop.get_bias_factor(bias_degree_dict, bias1, bias2)

            pkmu_term = bias_factor * f**nf * jnp.kron(pk_dict[name], self._mu**nmu).reshape(len(self._kn), len(self._mu))
            pkmu = pkmu + pkmu_term

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_data_13(self, pk_data, params):
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        pk_dict = self.get_pkmu_dict_13(pk_data)

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))

        for name in self.name_pkmu_terms['13']:
            # degrees of f, mu, and galaxy bias parameters
            nf, nmu, bias_degree_dict = utils_loop.get_degree_info(name)

            # calculate the coefficient that consists of bias parameters
            bias_factor = utils_loop.get_bias_factor(bias_degree_dict, bias1, bias2)

            pkmu_term = bias_factor * f**nf * jnp.kron(pk_dict[name], self._mu**nmu).reshape(len(self._kn), len(self._mu))
            pkmu = pkmu + pkmu_term

        pkmu_13_UV = self.get_pkmu_13_UV(self._kn, self._mu, pk_data, params)
        pkmu = pkmu + pkmu_13_UV

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_data_12_lpng(self, pk_data, Mk_data, params):
        f_nl = params['f_nl']
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        pk_lin = self.get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)
        Mk = self.get_pk(self._kn, Mk_data, kmin=self._kmin, kmax=self._kmax)
        pk_1phi = pk_lin / Mk

        pk_dict = self.get_pkmu_dict_12_lpng(pk_lin, Mk, pk_1phi)

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))

        for name in self.name_pkmu_terms['12_lpng1']:
            # degrees of f, mu, and galaxy bias parameters
            nf, nmu, bias_degree_dict = utils_loop.get_degree_info(name)

            # calculate the coefficient that consists of bias parameters
            bias_factor = utils_loop.get_bias_factor(bias_degree_dict, bias1, bias2)

            pkmu_term = bias_factor * f**nf * jnp.kron(pk_dict[name], self._mu**nmu).reshape(len(self._kn), len(self._mu))
            pkmu = pkmu + pkmu_term

        pk_factor = jnp.tile(pk_1phi, (len(self._mu), 1)).T
        pkmu_12_lpng1 = 2 * pk_factor * pkmu

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))
        for name in self.name_pkmu_terms['12_lpng2']:
            # degrees of f, mu, and galaxy bias parameters
            nf, nmu, bias_degree_dict = utils_loop.get_degree_info(name)

            # calculate the coefficient that consists of bias parameters
            bias_factor = utils_loop.get_bias_factor(bias_degree_dict, bias1, bias2)

            pkmu_term = bias_factor * f**nf * jnp.kron(pk_dict[name], self._mu**nmu).reshape(len(self._kn), len(self._mu))
            pkmu = pkmu + pkmu_term

        pk_factor = jnp.tile(Mk, (len(self._mu), 1)).T
        pkmu_12_lpng2 = 2 * pk_factor * pkmu

        pkmu = f_nl * (pkmu_12_lpng1 + pkmu_12_lpng2)

        return pkmu
    
    @partial(jit, static_argnames=['self'])
    def get_pkmu_data_22_lpng(self, pk_data, Mk_data, params):
        f_nl = params['f_nl']
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        pk_lin = self.get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)
        Mk = self.get_pk(self._kn, Mk_data, kmin=self._kmin, kmax=self._kmax)
        pk_1phi = pk_lin / Mk

        pk_dict = self.get_pkmu_dict(pk_lin, Mk, pk_1phi)

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))

        for name in self.name_pkmu_terms['22_lpng']:
            # degrees of f, mu, and galaxy bias parameters
            nf, nmu, bias_degree_dict = utils_loop.get_degree_info(name)

            # calculate the coefficient that consists of bias parameters
            bias_factor = utils_loop.get_bias_factor(bias_degree_dict, bias1, bias2)

            pkmu_term = bias_factor * f**nf * jnp.kron(pk_dict[name], self._mu**nmu).reshape(len(self._kn), len(self._mu))
            pkmu = pkmu + pkmu_term

        pkmu = f_nl * pkmu

        return pkmu

    @partial(jit, static_argnames=['self'])
    def get_pkmu_data_13_lpng(self, pk_data, Mk_data, params):
        f_nl = params['f_nl']
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        pk_lin = self.get_pk(self._kn, pk_data, kmin=self._kmin, kmax=self._kmax)
        Mk = self.get_pk(self._kn, Mk_data, kmin=self._kmin, kmax=self._kmax)
        pk_1phi = pk_lin / Mk

        pk_dict = self.get_pkmu_dict(pk_lin, Mk, pk_1phi)

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))

        for name in self.name_pkmu_terms['13_lpng1']:
            # degrees of f, mu, and galaxy bias parameters
            nf, nmu, bias_degree_dict = utils_loop.get_degree_info(name)

            # calculate the coefficient that consists of bias parameters
            bias_factor = utils_loop.get_bias_factor(bias_degree_dict, bias1, bias2)

            pkmu_term = bias_factor * f**nf * jnp.kron(pk_dict[name], self._mu**nmu).reshape(len(self._kn), len(self._mu))
            pkmu = pkmu + pkmu_term

        pkmu_UV = self.get_pkmu_13_lpng1_UV()
        pkmu_13_lpng1 = pkmu + pkmu_UV

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))

        Z1_g1 = jnp.tile(bias1['b1'] + f * self._mu**2, (len(self._kn), 1))
        Z1_g2 = jnp.tile(bias2['b1'] + f * self._mu**2, (len(self._kn), 1))
        pk_1phi = jnp.tile(self.get_pk_1phi(self._kn), (len(self._mu), 1)).T
        factor = jnp.kron(self._kn**2, (1 + f**2 * self._mu**2)).reshape(len(self._kn), len(self._mu))
        
        pkmu_13_lpng2 = - (bias1['bphi'] * Z1_g2 + bias2['bphi'] * Z1_g1) / 2 * factor * sigmav2 * pk_1phi

        pkmu = jnp.zeros((len(self._kn), len(self._mu)))

        for name in self.name_pkmu_terms['13_lpng3']:
            # degrees of f, mu, and galaxy bias parameters
            nf, nmu, bias_degree_dict = utils_loop.get_degree_info(name)

            # calculate the coefficient that consists of bias parameters
            bias_factor = utils_loop.get_bias_factor(bias_degree_dict, bias1, bias2)

            pkmu_term = bias_factor * f**nf * jnp.kron(pk_dict[name], self._mu**nmu).reshape(len(self._kn), len(self._mu))
            pkmu = pkmu + pkmu_term

        pkmu_13_lpng3 = pkmu

        pkmu = f_nl * (pkmu_13_lpng1 + pkmu_13_lpng2 + pkmu_13_lpng3)

        return pkmu

    @partial(jit, static_argnames=['self'])
    def get_pkmu_13_lpng1_UV(self, k, mu, pk_data, pk_1phi_data, params):
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        Z1_lpng = bias1['bphi']
        Z3_g_UV = - 61./315. * bias2['b1'] - 64./21. * bias2['bG2'] - 128./105. * bias2['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * bias2['b1']) * f * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * bias2['b1']) * f**2 * mu**2
        Z3_g_UV += (- 46./105.) * f**2 * mu**4
        Z3_g_UV += (- 1./3.) * f**3 * mu**4
        Z1Z3_UV_1 = Z1_lpng * Z3_g_UV

        Z1_lpng = bias2['bphi']
        Z3_g_UV = - 61./315. * bias1['b1'] - 64./21. * bias1['bG2'] - 128./105. * bias1['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * bias1['b1']) * f * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * bias1['b1']) * f**2 * mu**2
        Z3_g_UV += (- 46./105.) * f**2 * mu**4
        Z3_g_UV += (- 1./3.) * f**3 * mu**4
        Z1Z3_UV_2 = Z1_lpng * Z3_g_UV

        Z1Z3_UV = (Z1Z3_UV_1 + Z1Z3_UV_2) / 2

        pk = self.get_pk(k, pk_1phi_data, kmin=self._kmin, kmax=self._kmax)
        pk_int = self.get_pk_int(pk_data)

        pkmu_13 = jnp.kron(k**2 * pk * pk_int, Z1Z3_UV).reshape(len(k),len(mu))
        
        return pkmu_13
    
    @partial(jax.jit, static_argnames=['self'])
    def get_pkmu_irres_LO_NLO(self, k, mu, pk_data, params):
        f_nl = params['f_nl']
        h = params['h']
        f = params['f']
        bias1 = params['bias']
        bias2 = params['bias2'] if self.cross else params['bias']

        # wiggly-non-wiggly decomposition
        pk_nw_data = ir_resum.get_pk_nw_data(pk_data, h, khmin=7e-5, khmax=7., kmin_interp=self._kmin, kmax_interp=self._kmax)

        pk = self.get_pk(k, pk_data, kmin=self._kmin, kmax=self._kmax)
        pk_nw = self.get_pk(k, pk_nw_data, kmin=self._kmin, kmax=self._kmax)
        pk_w = pk - pk_nw

        # BAO damping factor in redshift space
        Sigma2_1 = (1 + mu**2 * f * (2 + f)) * ir_resum.get_Sigma2(pk_nw_data, self.rbao, self.ks)
        Sigma2_2 = f**2 * mu**2 * (mu**2 - 1) * ir_resum.get_dSigma2(pk_nw_data, self.rbao, self.ks)
        Sigma2_tot = Sigma2_1 + Sigma2_2
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