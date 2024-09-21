import os
import glob, re
import copy
import jax.numpy as jnp
import quadax
import interpax
from scipy.optimize import fsolve

from .power_law_decomp import PowerLawDecomp
from . import pt_matrix
from . import utils_loop
from .utils_loop import get_log_extrap
from .utils_math import get_legendre
from .ir_resum import IRResum


class PowerSpectrum1Loop:

    def __init__(self, kmin_fft=1e-5, kmax_fft=1e+3, nmax_fft=256, precompute=True):
        
        # set up the FFTLog-based power-law decomposition
        self.config_fft = {
            'plin nu=-0.3': {'nu':-0.3, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nmax':nmax_fft},
            'plin nu=-0.7': {'nu':-0.7, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nmax':nmax_fft},
            'plin nu=-1.6': {'nu':-1.6, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nmax':nmax_fft},
            'plin nu=-0.3 (no-wiggle)': {'nu':-0.3, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nmax':nmax_fft},
            'plin nu=-0.7 (no-wiggle)': {'nu':-0.7, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nmax':nmax_fft},
            'plin nu=-1.6 (no-wiggle)': {'nu':-1.6, 'kmin':kmin_fft, 'kmax':kmax_fft, 'nmax':nmax_fft},
        }
        self.set_power_law_decomp(self.config_fft)
        self._kmin_fft = kmin_fft
        self._kmax_fft = kmax_fft
        self._nmax_fft = nmax_fft

        # store the names of 1-loop terms calculated with the FFTLog-based method
        self.name_pkmu_gg_terms = {}
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/M22_*.txt')
        self.name_pkmu_gg_terms['22_gg'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/M13_*.txt')
        self.name_pkmu_gg_terms['13_gg'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        self.name_pkmu_gg_terms['tot'] = self.name_pkmu_gg_terms['22_gg'] + self.name_pkmu_gg_terms['13_gg']

        self.mat = {}
        self.matrix = {}
        # precompute the PT matrices
        if precompute:
            self.set_matrix(self.name_pkmu_gg_terms['tot'])
            self.compute_matrix(self.name_pkmu_gg_terms['tot'])

    def set_power_law_decomp(self, config_fft):
        # set multiple instances of PowerLawDecomp class.
        self.decomp = {}
        for name, config in config_fft.items():
            nu = config['nu']
            kmin = config['kmin']
            kmax = config['kmax']
            nmax = config['nmax']
            self.decomp[name] = PowerLawDecomp(nu, kmin, kmax, nmax)

    def set_matrix(self, names=[]):
        for name in names:
            matfile = glob.glob(os.path.dirname(__file__)+'/pt_matrix/*/*/%s.txt' % (name))[0]
            if '22' in name or 'I' in name or '12' in name:
                self.mat[name] = pt_matrix.PTMatrix22(matfile)
            elif '13' in name or 'F' in name:
                self.mat[name] = pt_matrix.PTMatrix13(matfile)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))

    def compute_matrix(self, names=[]):
        # precompute the PT matrices for appropriate FFT settings.
        for name in names:
            if name in utils_loop.kernel_to_decomp_dict.keys():
                name_dec = utils_loop.kernel_to_decomp_dict[name]
            else:
                species_list = list(self.name_pkmu_gg_terms.keys())
                species_list.remove('tot')
                for species in species_list:
                    if name in self.name_pkmu_gg_terms[species]:
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

    def set_pk_lin(self, k, pk_lin, kmin=1e-7, kmax=1e+7):
        k_extrap, pk_extrap = get_log_extrap(k, pk_lin, kmin, kmax)
        self.pk_lin_spl = interpax.Interpolator1D(jnp.log(k_extrap), jnp.log(pk_extrap))

    def set_Dgrowth(self, Dgrowth):
        self.Dgrowth = Dgrowth

    def set_fgrowth(self, fgrowth):
        self.fgrowth = fgrowth

    def set_1loop(self, hubble, ks=0.2, rbao=110., kmin=1e-6, kmax=1e+6, khigh=None):
        self.Dgrowth = 1.

        # FFTLog-based power-law decomposition
        self.decomp['plin nu=-0.3'].compute(self.get_pk_lin, kwarg={'khigh':khigh})
        self.decomp['plin nu=-0.7'].compute(self.get_pk_lin, kwarg={'khigh':khigh})
        self.decomp['plin nu=-1.6'].compute(self.get_pk_lin, kwarg={'khigh':khigh})

        # set up the IR resummation
        self.irres = IRResum(self.get_pk_lin, hubble=hubble, rbao=rbao, 
                            khmin=7e-5, khmax=7,
                            kmin_interp=1e-7, kmax_interp=1e+7, kwarg={'khigh':khigh})
        self.decomp['plin nu=-0.3 (no-wiggle)'].compute(self.irres.get_pk_nw)
        self.decomp['plin nu=-0.7 (no-wiggle)'].compute(self.irres.get_pk_nw)
        self.decomp['plin nu=-1.6 (no-wiggle)'].compute(self.irres.get_pk_nw)

        # BAO damping factors
        self.Sigma2 = self.irres.get_Sigma2(ks=ks)
        self.dSigma2 = self.irres.get_dSigma2(ks=ks)

        # compute the power spectrum integral for the UV part of P13
        self.pk_lin_int = self.get_pk_int(self.get_pk_lin, kmin=kmin, kmax=kmax, kwarg={'khigh':khigh})
        self.pk_lin_nw_int = self.get_pk_int(self.irres.get_pk_nw, kmin=kmin, kmax=kmax)
        self.sigmav2 = self.pk_lin_int / 3

        self.compute_1loop_terms()

    def set_bias_params(self, bias1={}, bias2={}):
        self.bias = bias1
        self.bias1 = bias1
        self.bias2 = bias2
        if bias2 == {}:
            self.bias2 = copy.deepcopy(bias1)

    def set_ctr_params(self, ctr1={}, ctr2={}, sigma_vel=0.):
        self.ctr = ctr1
        self.ctr1 = ctr1
        self.ctr2 = ctr2
        if ctr2 == {}:
            self.ctr2 = copy.deepcopy(ctr1)
        self.sigma_vel = sigma_vel

    def set_stoch_params(self, stoch={}, ndens=1, ndens2=None, k_nl=1):
        self.stoch = stoch
        self.ndens = ndens
        self.ndens1 = ndens
        self.ndens2 = ndens2
        if ndens2 == None:
            self.ndens2 = ndens
        self.k_nl = k_nl

    def get_pk_lin(self, k, khigh=None): # in unit of [h^{-3} Mpc^3]
        pk_lin = jnp.exp(self.pk_lin_spl(jnp.log(k))) * self.Dgrowth**2
        if khigh != None:
            pk_lin = pk_lin * jnp.exp(-(k / khigh))
        return pk_lin

    def get_pk_int(self, get_pk, kmin=1e-6, kmax=1e+6, num=1000, kwarg={}):
        q = jnp.geomspace(kmin, kmax, num)
        integrand = q * get_pk(q)
        res = quadax.simpson(integrand, x=jnp.log(q)) / (2 * jnp.pi**2)
        return res
    
    def get_k_nl(self, k0=0.5):
        def func(logk):
            k = jnp.exp(logk)
            Delta2_lin = k**3 * self.get_pk_lin(k) / (2 * jnp.pi**2)
            return jnp.log(Delta2_lin)
        
        crit = False
        while crit == False:
            root = fsolve(func, x0=jnp.log(k0), xtol=1e-6, maxfev=1000) # solve Delta2_lin(k_nl) = 1
            k0 = jnp.exp(root[0])
            crit = (jnp.abs(func(jnp.log(k0))) < 1e-6)

        k_nl = jnp.exp(root[0])
        return k_nl

    def get_pk_mm_irres(self, k, mode='LO'):
        plin = self.get_pk_lin(k)
        plin_nw = self.irres.get_pk_nw(k) * self.Dgrowth**2
        plin_w = plin - plin_nw
        if mode == 'LO':
            pk = plin_nw + jnp.exp(-k**2 * self.Dgrowth**2 * self.Sigma2) * plin_w
        return pk

    def get_pk_1loop_data(self, name, sub_k0=True, mode='full'):
        name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict[name])
        if mode != 'full':
            for i in range(len(name_dec)):
                if 'plin' in name_dec[i]: name_dec[i] += ' (%s)' % (mode)

        if '22' in name or 'I' in name:
            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_q = self.decomp[name_dec[1]].func_q
            pk_data = kn**3 * jnp.diag(jnp.dot(p1_q.T, jnp.dot(self.matrix[name], p2_q)).real)
            if sub_k0:
                kmin = self.decomp[name_dec[0]].kmin
                p1_k0 = self.decomp[name_dec[0]].func_k0
                p2_k0 = self.decomp[name_dec[1]].func_k0
                pk_data_k0 = kmin**3 * jnp.dot(p1_k0, jnp.dot(self.matrix[name], p2_k0)).real
                pk_data = pk_data - pk_data_k0

        elif '13' in name or 'F' in name:
            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_k = self.decomp[name_dec[1]].func_rec.real
            pk_data = kn**3 * p2_k * jnp.dot(self.matrix[name], p1_q).real
            if name == '13_dd':
                pk_data = pk_data - 61./315. * kn**2 * p2_k * self.pk_lin_int
            elif name == '13_dv':
                pk_data = pk_data - 25./63. * kn**2 * p2_k * self.pk_lin_int
            elif name == '13_vv':
                pk_data = pk_data - 3./5. * kn**2 * p2_k * self.pk_lin_int
        else:
            raise KeyError('PT kernel name is invalid.')

        return kn, pk_data

    def get_pk_1loop(self, k, name, sub_k0=True):
        alpha = 1.5
        kn, pk_data = self.get_pk_1loop_data(name=name, sub_k0=sub_k0)
        pk_interp = interpax.Interpolator1D(kn, kn**alpha * pk_data)
        pk = pk_interp(k) * k**(-alpha)
        pk = self.Dgrowth**4 * pk
        return pk

    def get_pk_gg_raw(self, k):
        pk_tree = self.get_pk_lin(k)

        name_list = ['22_dd','13_dd','I_d2','I_G2','I_d2_d2','I_G2_G2','I_d2_G2','F_G2']
        pks = {}
        for name in name_list:
            pks[name] = self.get_pk_1loop(k, name=name, sub_k0=True)

        # cross power spectrum
        pk_gg = (self.bias1['b1'] * self.bias2['b1']) * (pk_tree + pks['22'] + pks['13'])
        pk_gg = pk_gg + (self.bias1['b1'] * self.bias2['b2'] + self.bias1['b2'] * self.bias2['b1']) / 2. * pks['I_d2']
        pk_gg = pk_gg + (self.bias1['b1'] * self.bias2['bG2'] + self.bias1['bG2'] * self.bias2['b1']) * pks['I_G2']
        pk_gg = pk_gg + (self.bias1['b2'] * self.bias2['b2']) / 4. * pks['I_d2_d2']
        pk_gg = pk_gg + (self.bias1['bG2'] * self.bias2['bG2']) * pks['I_G2_G2']
        pk_gg = pk_gg + (self.bias1['b2'] * self.bias2['bG2'] + self.bias1['bG2'] * self.bias2['b2']) / 2. * pks['I_d2_G2']
        pk_gg = pk_gg + (self.bias1['b1'] * self.bias2['bG2'] + self.bias1['bG2'] * self.bias2['b1']) * pks['F_G2']
        pk_gg = pk_gg + (2./5.) * (self.bias1['b1'] * self.bias2['bGamma3'] + self.bias1['bGamma3'] * self.bias2['b1']) * pks['F_G2']

        return pk_gg

    def get_pk_gg(self, k, irres=True, cross=False):
        pk = self.get_pkmu_gg(k, 0, irres=irres, cross=cross)
        return pk

    def get_pkmu_gg(self, k, mu, irres=True, cross=False):
        # tree + 1-loop
        if irres:
            pkmu = self.get_pkmu_gg_irres(k, mu, mode='LO+NLO')
        else:
            pkmu_tree = self.get_pkmu_gg_lin(k, mu)
            pkmu_1loop = self.get_pkmu_gg_1loop(k, mu, name='tot', mode='full')
            pkmu = pkmu_tree + pkmu_1loop

        # counterterm
        pkmu_ctr = self.get_pkmu_ctr(k, mu, irres=irres)
        pkmu = pkmu + pkmu_ctr

        # stochasticity
        pkmu_stoch = self.get_pkmu_stoch(k, mu, cross=cross)
        pkmu = pkmu + pkmu_stoch

        # Finger-of-God damping factor
        if self.sigma_vel != 0.:
            pkmu = pkmu * self.get_damp_factor(k, mu, self.sigma_vel)

        return pkmu

    def get_pk_ell_gg(self, k, ells, irres=True, ctr_multipole=True, cross=False):
        num = 256
        mu = jnp.linspace(0., 1., num)

        pkmu = self.get_pkmu_gg(k, mu, irres=irres, cross=cross)
        
        # subtract ctr1 part from P(k, mu)
        if ctr_multipole:
            pkmu_ctr1 = self.get_pkmu_ctr1(k, mu, irres=irres)
            pkmu = pkmu - pkmu_ctr1

        # compute the Legendre multipole moments
        pkmu = jnp.tile(pkmu, (len(ells),1,1))
        legendre = jnp.array([jnp.tile((2*l+1) * get_legendre(l, mu), (len(k),1)) for l in ells])
        pk_ell = quadax.simpson(pkmu * legendre, x=mu, axis=2)

        # add ctr1 part to P_ell(k)
        if ctr_multipole:
            pk_ell_ctr1 = self.get_pk_ell_ctr1(k, ells, irres=irres)
            pk_ell = pk_ell + pk_ell_ctr1

        return pk_ell

    def get_pk_gg_ref(self, k_ref, alpha_perp, alpha_para, irres=True, cross=False):
        pk = self.get_pkmu_gg_ref(k_ref, 0., alpha_perp, alpha_para, irres=irres, cross=cross)
        return pk

    def get_pkmu_gg_ref(self, k_ref, mu_ref, alpha_perp, alpha_para, irres=True, ctr_multipole=False, cross=False):
        k_ref = jnp.atleast_1d(k_ref)
        mu_ref = jnp.atleast_1d(mu_ref)

        # mapping of (k, mu)
        fac = jnp.sqrt(1 + mu_ref**2 * ((alpha_perp / alpha_para)**2 - 1))
        mu = mu_ref * (alpha_perp / alpha_para) / fac
        k = jnp.kron(k_ref, fac).reshape(len(k_ref), len(mu_ref)) / alpha_perp

        # spline interpolation
        k_grid = jnp.geomspace(jnp.max(jnp.array([jnp.min(k), self._kmin_fft])), jnp.min(jnp.array([jnp.max(k), self._kmax_fft])), self._nmax_fft)
        mu_grid = jnp.linspace(0., 1., 51)
        pkmu_grid = self.get_pkmu_gg(k_grid, mu_grid, irres=irres, cross=cross)

        # subtract ctr1 part from P(k, mu)
        if ctr_multipole:
            pkmu_ctr1 = self.get_pkmu_ctr1(k_grid, mu_grid, irres=irres)
            pkmu_grid = pkmu_grid - pkmu_ctr1

        mu_tile = jnp.tile(mu, (len(k_ref), 1))
        pkmu = interpax.interp2d(jnp.ravel(k), jnp.ravel(mu_tile), k_grid, mu_grid, pkmu_grid)
        pkmu = pkmu.reshape(len(k_ref), len(mu_ref))
        pkmu = pkmu / (alpha_perp**2 * alpha_para)

        if len(k_ref) == 1 or len(mu_ref) == 1:
            pkmu = jnp.ravel(pkmu)
        return pkmu
    
    def get_pk_ell_gg_ref(self, k_ref, ells, alpha_perp, alpha_para, irres=True, ctr_multipole=True, cross=False):
        k_ref = jnp.atleast_1d(k_ref)
        num = 256
        mu_ref = jnp.linspace(0., 1., num)

        pkmu_ref = self.get_pkmu_gg_ref(k_ref, mu_ref, alpha_perp, alpha_para, irres=irres, ctr_multipole=ctr_multipole, cross=cross)

        pkmu_ref = jnp.tile(pkmu_ref, (len(ells),1,1))
        legendre = jnp.array([jnp.tile((2*l+1) * get_legendre(l,mu_ref), (len(k_ref),1)) for l in ells])
        pk_ell = quadax.simpson(pkmu_ref * legendre, x=mu_ref, axis=2)

        # add ctr1 part to P_ell(k)
        if ctr_multipole:
            pk_ell_ctr1 = self.get_pk_ell_ctr1_ref(k_ref, ells, alpha_perp, alpha_para, irres=irres)
            pk_ell = pk_ell + pk_ell_ctr1

        return pk_ell

    def get_pk_gg_lin(self, k):
        pk = self.get_pkmu_gg_lin(k, 0.)
        return pk

    def get_pkmu_gg_lin(self, k, mu):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        Z1_1 = self.bias1['b1'] + self.fgrowth * mu**2
        Z1_2 = self.bias2['b1'] + self.fgrowth * mu**2
        pk_lin = self.get_pk_lin(k)
        pkmu = jnp.kron(pk_lin, Z1_1 * Z1_2).reshape(len(k),len(mu))

        if len(k) == 1 or len(mu) == 1:
            pkmu = jnp.ravel(pkmu)
        return pkmu

    def get_pk_ell_gg_lin(self, k, ells):
        k = jnp.atleast_1d(k)
        num = 256
        mu = jnp.linspace(0., 1., num)
        pkmu = self.get_pkmu_gg_lin(k, mu)
        pkmu = jnp.tile(pkmu, (len(ells),1,1))
        legendre = jnp.array([jnp.tile((2*l+1) * get_legendre(l, mu), (len(k),1)) for l in ells])
        pk_ell = quadax.simpson(pkmu * legendre, x=mu, axis=2)
        return pk_ell
    
    def compute_1loop_terms(self):
        term_names = self.name_pkmu_gg_terms['tot']
        self.pk_data_dict = {}

        for mode in ['full', 'no-wiggle']:
            for term in term_names:
                kn, pk_data = self.get_pk_rsd_1loop_data(term, mode=mode)
                self.pk_data_dict[(term, mode)] = (kn, pk_data)

    def get_pkmu_13_UV(self, k, mu, mode='full'):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)
        
        # UV limit of the 1-3 term

        Z1_g = self.bias1['b1'] + self.fgrowth * mu**2
        Z3_g_UV = - 61./315. * self.bias2['b1'] - 64./21. * self.bias2['bG2'] - 128./105. * self.bias2['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * self.bias2['b1']) * self.fgrowth * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * self.bias2['b1']) * self.fgrowth**2 * mu**2
        Z3_g_UV += (- 46./105.) * self.fgrowth**2 * mu**4
        Z3_g_UV += (- 1./3.) * self.fgrowth**3 * mu**4
        Z1Z3_UV_1 = Z1_g * Z3_g_UV

        Z1_g = self.bias2['b1'] + self.fgrowth * mu**2
        Z3_g_UV = - 61./315. * self.bias1['b1'] - 64./21. * self.bias1['bG2'] - 128./105. * self.bias1['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * self.bias1['b1']) * self.fgrowth * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * self.bias1['b1']) * self.fgrowth**2 * mu**2
        Z3_g_UV += (- 46./105.) * self.fgrowth**2 * mu**4
        Z3_g_UV += (- 1./3.) * self.fgrowth**3 * mu**4
        Z1Z3_UV_2 = Z1_g * Z3_g_UV

        Z1Z3_UV = (Z1Z3_UV_1 + Z1Z3_UV_2) / 2

        if mode == 'full':
            pk_fac = self.get_pk_lin(k) * (self.pk_lin_int * self.Dgrowth**2)
        elif mode == 'no-wiggle':
            pk_fac = (self.irres.get_pk_nw(k) * self.pk_lin_nw_int * self.Dgrowth**4)
        else:
            raise ValueError('Invalid mode.')

        pkmu_13 = jnp.kron(k**2 * pk_fac, Z1Z3_UV).reshape(len(k),len(mu))
        return pkmu_13

    def get_pk_rsd_1loop_data(self, name, sub_k0=True, mode='full'):
        if '22' in name:
            name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict['22_gg'])
            if mode != 'full':
                for i in range(len(name_dec)):
                    if 'plin' in name_dec[i]: name_dec[i] += ' (%s)' % (mode)

            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_q = self.decomp[name_dec[1]].func_q
            pk_data = kn**3 * jnp.diag(jnp.dot(p1_q.T, jnp.dot(self.matrix[name], p2_q)).real)
            if sub_k0:
                kmin = self.decomp[name_dec[0]].kmin
                p1_k0 = self.decomp[name_dec[0]].func_k0
                p2_k0 = self.decomp[name_dec[1]].func_k0
                pk_data_k0 = kmin**3 * jnp.dot(p1_k0, jnp.dot(self.matrix[name], p2_k0)).real
                pk_data = pk_data - pk_data_k0

        elif '13' in name:
            name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict['13_gg'])
            if mode != 'full':
                for i in range(len(name_dec)):
                    if 'plin' in name_dec[i]: name_dec[i] += ' (%s)' % (mode)

            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_q = self.decomp[name_dec[1]].func_rec.real
            pk_data = kn**3 * p2_q * jnp.dot(self.matrix[name], p1_q).real

        else:
            raise KeyError('PT kernel name is invalid.')
        return kn, pk_data

    def get_pkmu_gg_1loop_data(self, mu, name='tot', mode='full'):
        mu = jnp.atleast_1d(mu)

        if name == 'tot':
            kn, pkmu_22 = self.get_pkmu_gg_1loop_data(mu, name='22_gg', mode=mode)
            kn, pkmu_13 = self.get_pkmu_gg_1loop_data(mu, name='13_gg', mode=mode)
            pkmu_data = pkmu_22 + pkmu_13
            return kn, pkmu_data

        term_names = self.name_pkmu_gg_terms[name]
        pkmu_tab = []
        for term in term_names:
            nf, nmu, bias_deg = utils_loop.get_deg_info(term)
            
            keys = list(bias_deg.keys())
            if len(bias_deg) == 0:
                bias_fac = 1
            elif len(bias_deg) == 1:
                key = keys[0]
                if bias_deg[key] == 1:
                    bias_fac = (self.bias1[key] + self.bias2[key]) / 2
                elif bias_deg[key] == 2:
                    bias_fac = self.bias1[key] * self.bias2[key]
            elif len(bias_deg) == 2:
                key1 = keys[0]
                key2 = keys[1]
                bias_fac = (self.bias1[key1] * self.bias2[key2] + self.bias2[key1] * self.bias1[key2]) / 2
            else:
                raise ValueError('Invalid numbers of bias parameters.')

            fac = bias_fac * self.fgrowth**nf

            kn, pk_data = self.pk_data_dict[(term, mode)]

            pkmu = fac * jnp.kron(pk_data, mu**nmu).reshape(len(kn),len(mu))
            pkmu_tab.append(pkmu)
        pkmu_tab = jnp.array(pkmu_tab)
        pkmu_data = self.Dgrowth**4 * jnp.sum(pkmu_tab, axis=0)

        if name == '13_gg':
            pkmu_UV = self.get_pkmu_13_UV(kn, mu, mode=mode)
            pkmu_data += pkmu_UV

        return kn, pkmu_data

    def get_pkmu_gg_1loop(self, k, mu, name='tot', mode='full'):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        kn, pkmu_data = self.get_pkmu_gg_1loop_data(mu, name=name, mode=mode)

        if len(mu) == 1:
            pkmu = interpax.interp1d(k, kn, jnp.ravel(pkmu_data))
        else:
            k_tile = jnp.tile(k, (len(mu), 1)).T
            mu_tile = jnp.tile(mu, (len(k), 1))
            pkmu = interpax.interp2d(jnp.ravel(k_tile), jnp.ravel(mu_tile), kn, mu, pkmu_data)
            pkmu = pkmu.reshape(len(k), len(mu))

        return pkmu

    def get_pk_ell_gg_1loop(self, k, ells, name='tot', mode='full'):
        k = jnp.atleast_1d(k)
        num = 256
        mu = jnp.linspace(0., 1., num)
        pkmu = self.get_pkmu_gg_1loop(k, mu, name=name, mode=mode)
        pkmu = jnp.tile(pkmu, (len(ells),1,1))
        legendre = jnp.array([jnp.tile((2*l+1) * get_legendre(l, mu), (len(k),1)) for l in ells])
        pk_ell = quadax.simpson(pkmu * legendre, x=mu, axis=2)
        return pk_ell

    def get_pkmu_gg_irres(self, k, mu, mode='LO+NLO'):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        # wiggly-non-wiggly decomposition
        plin = self.get_pk_lin(k)
        plin_nw = self.irres.get_pk_nw(k) * self.Dgrowth**2
        plin_w = plin - plin_nw

        # BAO damping factor in redshift space
        Sigma2_1 = (1 + mu**2 * self.fgrowth * (2 + self.fgrowth)) * self.Sigma2
        Sigma2_2 = self.fgrowth**2 * mu**2 * (mu**2 - 1) * self.dSigma2
        Sigma2_tot = Sigma2_1 + Sigma2_2

        plin_nw_tile = jnp.tile(plin_nw, (len(mu),1)).T
        plin_w_tile = jnp.tile(plin_w, (len(mu),1)).T
        damp_fac = jnp.kron(k**2, self.Dgrowth**2 * Sigma2_tot).reshape(len(k),len(mu))

        Z1_tile1 = jnp.tile(self.bias1['b1'] + self.fgrowth * mu**2, (len(k), 1))
        Z1_tile2 = jnp.tile(self.bias2['b1'] + self.fgrowth * mu**2, (len(k), 1))

        if len(k) == 1 or len(mu) == 1:
            Z1_tile1 = jnp.ravel(Z1_tile1)
            Z1_tile2 = jnp.ravel(Z1_tile2)
            damp_fac = jnp.ravel(damp_fac)
            plin_nw_tile = jnp.ravel(plin_nw_tile)
            plin_w_tile = jnp.ravel(plin_w_tile)

        # IR-resummed power spectrum
        if mode == 'LO':
            # leading-order IR resummation
            pkmu = Z1_tile1 * Z1_tile2 * (plin_nw_tile + jnp.exp(-damp_fac) * plin_w_tile)
        elif mode == 'tree':
            # leading-order IR resummation + additional term to prevent the double counting
            pkmu = Z1_tile1 * Z1_tile2 * (plin_nw_tile + (1 + damp_fac) * jnp.exp(-damp_fac) * plin_w_tile)
        elif mode == '1loop no-wiggle':
            # 1-loop term computed with the non-wiggly component of the linear power spectrum
            pkmu = self.get_pkmu_gg_1loop(k, mu, name='tot', mode='no-wiggle')
        elif mode == '1loop wiggle':
            # 1-loop term computed with the wiggly component of the linear power spectrum
            pkmu = self.get_pkmu_gg_1loop(k, mu, name='tot') - self.get_pkmu_gg_1loop(k, mu, name='tot', mode='no-wiggle')
        elif mode == '1loop':
            # next-to-leading order term of the IR-resummed power spectrum
            pkmu_1loop_nw = self.get_pkmu_gg_irres(k, mu, mode='1loop no-wiggle')
            pkmu_1loop_w = self.get_pkmu_gg_1loop(k, mu, name='tot') - pkmu_1loop_nw
            pkmu = pkmu_1loop_nw + jnp.exp(-damp_fac) * pkmu_1loop_w
        elif mode == 'LO+NLO':
            # LO+NLO IR-resummed power spectrum
            pkmu_tree = self.get_pkmu_gg_irres(k, mu, mode='tree')
            pkmu_1loop = self.get_pkmu_gg_irres(k, mu, mode='1loop')
            pkmu = pkmu_tree + pkmu_1loop

        if len(k) == 1 or len(mu) == 1:
            pkmu = jnp.ravel(pkmu)
        return pkmu
    
    def get_pk_lin_irres_rsd(self, k, mu):
        # wiggly-non-wiggly decomposition
        plin = self.get_pk_lin(k)
        plin_nw = self.irres.get_pk_nw(k) * self.Dgrowth**2
        plin_w = plin - plin_nw

        # BAO damping factor in redshift space
        Sigma2_1 = (1 + mu**2 * self.fgrowth * (2 + self.fgrowth)) * self.Sigma2
        Sigma2_2 = self.fgrowth**2 * mu**2 * (mu**2 - 1) * self.dSigma2
        Sigma2_tot = Sigma2_1 + Sigma2_2

        plin_nw_tile = jnp.tile(plin_nw, (len(mu),1)).T
        plin_w_tile = jnp.tile(plin_w, (len(mu),1)).T
        damp_fac = jnp.kron(k**2, self.Dgrowth**2 * Sigma2_tot).reshape(len(k),len(mu))

        pk = plin_nw_tile + jnp.exp(-damp_fac) * plin_w_tile
        return pk
    
    def get_pkmu_ctr(self, k, mu, irres=True):
        pkmu_ctr = self.get_pkmu_ctr1(k, mu, irres=irres) + self.get_pkmu_ctr2(k, mu, irres=irres)
        return pkmu_ctr

    def get_pkmu_ctr1(self, k, mu, irres=True):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        # cross power spectrum
        ctr1_mu = (self.ctr1['c0'] + self.ctr2['c0']) / 2 
        ctr1_mu = ctr1_mu + (self.ctr1['c2'] + self.ctr2['c2']) / 2 * self.fgrowth * mu**2
        ctr1_mu = ctr1_mu + (self.ctr1['c4'] + self.ctr2['c4']) / 2 * self.fgrowth**2 * mu**4

        if irres:
            pk = self.get_pk_lin_irres_rsd(k, mu)
            pkmu_ctr1 = - 2 * jnp.kron(k**2, ctr1_mu).reshape(len(k),len(mu)) * pk
        else:
            pk = self.get_pk_lin(k)
            pkmu_ctr1 = - 2 * jnp.kron(k**2 * pk, ctr1_mu).reshape(len(k),len(mu))

        pkmu = pkmu_ctr1
        if len(k) == 1 or len(mu) == 1:
            pkmu = jnp.ravel(pkmu)
        return pkmu
    
    def get_pkmu_ctr2(self, k, mu, irres=True):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)

        # cross power spectrum
        ctr2_mu = (self.ctr1['cfog'] + self.ctr2['cfog']) / 2 * self.fgrowth**4 * mu**4 * (self.bias1['b1'] + self.fgrowth * mu**2) * (self.bias2['b1'] + self.fgrowth * mu**2)

        if irres:
            pk = self.get_pk_lin_irres_rsd(k, mu)
            pkmu_ctr2 = - jnp.kron(k**4, ctr2_mu).reshape(len(k),len(mu)) * pk
        else:
            pk = self.get_pk_lin(k)
            pkmu_ctr2 = - jnp.kron(k**4 * pk, ctr2_mu).reshape(len(k),len(mu))

        pkmu = pkmu_ctr2
        if len(k) == 1 or len(mu) == 1:
            pkmu = jnp.ravel(pkmu)
        return pkmu
    
    def get_coeff_ctr1_multipole(self, l):
        cl = (self.ctr1['c%s' % (l)] + self.ctr2['c%s' % (l)]) / 2. if l in [0,2,4] else 0.
        return cl

    def get_pk_ell_ctr1(self, k, ells, irres=True):
        k = jnp.atleast_1d(k)
        num = 256
        mu = jnp.linspace(0., 1., num)

        coeffs = jnp.array([self.get_coeff_ctr1_multipole(l) for l in ells])
        coeffs = jnp.tile(coeffs, (len(k), 1)).T

        if irres:
            pk = self.get_pk_lin_irres_rsd(k, mu)
        else:
            pk_lin = self.get_pk_lin(k)
            pk = jnp.tile(pk_lin, (len(mu),1)).T
        
        pk = jnp.tile(pk, (len(ells),1,1))
        legendre = jnp.array([jnp.tile((2*l+1) * get_legendre(l, mu) * mu**l * self.fgrowth**(l/2), (len(k),1)) for l in ells])
        pk_ell_ctr1 = - 2 * quadax.simpson(pk * legendre, x=mu, axis=2) * jnp.tile(k**2, (len(ells), 1))
        pk_ell_ctr1 = coeffs * pk_ell_ctr1

        return pk_ell_ctr1
    
    def get_pkmu_for_ctr1_ref(self, k_ref, mu_ref, alpha_perp, alpha_para, irres=True):
        k_ref = jnp.atleast_1d(k_ref)
        mu_ref = jnp.atleast_1d(mu_ref)

        # mapping of (k, mu)
        fac = jnp.sqrt(1 + mu_ref**2 * ((alpha_perp / alpha_para)**2 - 1))
        mu = mu_ref * (alpha_perp / alpha_para) / fac
        k = jnp.kron(k_ref, fac).reshape(len(k_ref), len(mu_ref)) / alpha_perp

        # spline interpolation of mu^l k^2 P_lin(k)
        k_grid = jnp.geomspace(jnp.max(jnp.array([jnp.min(k), self._kmin_fft])), jnp.min(jnp.array([jnp.max(k), self._kmax_fft])), self._nmax_fft)
        mu_grid = jnp.linspace(0., 1., 51)
        # k2mul = jnp.kron(k_grid**2, mu_grid**l).reshape(len(k_grid), len(mu_grid))
        if irres:
            pkmu_grid = self.get_pk_lin_irres_rsd(k_grid, mu_grid)
        else:
            pkmu_grid = jnp.tile(self.get_pk_lin(k_grid), (len(mu_grid),1)).T

        mu_tile = jnp.tile(mu, (len(k_ref), 1))
        pkmu = interpax.interp2d(jnp.ravel(k), jnp.ravel(mu_tile), k_grid, mu_grid, pkmu_grid)
        pkmu = pkmu.reshape(len(k_ref), len(mu_ref))
        pkmu = pkmu / (alpha_perp**2 * alpha_para)

        if len(k_ref) == 1 or len(mu_ref) == 1:
            pkmu = jnp.ravel(pkmu)
        return pkmu
    
    def get_pk_ell_ctr1_ref(self, k_ref, ells, alpha_perp, alpha_para, irres=True):
        k_ref = jnp.atleast_1d(k_ref)
        num = 256
        mu_ref = jnp.linspace(0., 1., num)

        coeffs = jnp.array([self.get_coeff_ctr1_multipole(l) for l in ells])
        coeffs = jnp.tile(coeffs, (len(k_ref), 1)).T

        # mapping of (k, mu)
        fac = jnp.sqrt(1 + mu_ref**2 * ((alpha_perp / alpha_para)**2 - 1))
        mu = mu_ref * (alpha_perp / alpha_para) / fac
        k = jnp.kron(k_ref, fac).reshape(len(k_ref), len(mu_ref)) / alpha_perp

        pkmu_ref = self.get_pkmu_for_ctr1_ref(k_ref, mu_ref, alpha_perp, alpha_para, irres=irres)

        pkmu_ref = jnp.tile(pkmu_ref, (len(ells),1,1))
        legendre = jnp.array([jnp.tile((2*l+1) * get_legendre(l,mu_ref) * mu**l * self.fgrowth**(l/2), (len(k_ref),1)) * k**2 for l in ells])
        pk_ell_ctr1 = - 2 * quadax.simpson(pkmu_ref * legendre, x=mu, axis=2)
        pk_ell_ctr1 = coeffs * pk_ell_ctr1

        return pk_ell_ctr1
    
    def get_damp_factor(self, k, mu, sigma_vel=0.):
        kmu = jnp.kron(k, mu).reshape(len(k), len(mu))
        return jnp.exp(- (kmu * self.fgrowth * sigma_vel)**2 / 2.)

    def get_pkmu_stoch(self, k, mu, cross=False):
        k = jnp.atleast_1d(k)
        mu = jnp.atleast_1d(mu)
        
        # auto power spectrum
        if cross == False:
            pkmu = self.stoch['P_shot']
            pkmu = pkmu + self.stoch['a0'] * jnp.kron((k / self.k_nl)**2, get_legendre(0,mu)).reshape(len(k), len(mu))
            pkmu = pkmu + self.stoch['a2'] * jnp.kron((k / self.k_nl)**2, get_legendre(2,mu)).reshape(len(k), len(mu))
            pkmu = 1. / self.ndens * pkmu
        # cross power spectrum
        else:
            try:
                pkmu = self.stoch['P_shot_cross']
                pkmu = pkmu + self.stoch['a0_cross'] * jnp.kron((k / self.k_nl)**2, get_legendre(0,mu)).reshape(len(k), len(mu))
                pkmu = pkmu + self.stoch['a2_cross'] * jnp.kron((k / self.k_nl)**2, get_legendre(2,mu)).reshape(len(k), len(mu))
                pkmu = (1. / self.ndens1 + 1. / self.ndens2) / 2. * pkmu
            except KeyError:
                pkmu = jnp.zeros((len(k), len(mu)))

        if len(k) == 1 or len(mu) == 1:
            pkmu = jnp.ravel(pkmu)
        return pkmu
