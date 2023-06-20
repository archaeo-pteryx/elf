import os, sys, time
import glob, re
import copy
import numpy as np
from scipy.integrate import quad, romb
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.special import lpmv

from camb_wrapper import CambWrapper
from power_1loop import PowerSpectrum1loop
from power_law_decomp import PowerLawDecomp
import pt_matrix
import utils_loop
from ir_resum import IRResum


class PowerSpectrum1loopLPNG(PowerSpectrum1loop):

    def __init__(self, config_fft=None):
        self.params = None
        if config_fft == None:
            config_fft = {
                'plin nu=-0.3': {'nu':-0.3, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-0.7': {'nu':-0.7, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-1.6': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-0.3 (no-wiggle)': {'nu':-0.3, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-0.7 (no-wiggle)': {'nu':-0.7, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-1.6 (no-wiggle)': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'p1phi nu=-0.9': {'nu':-0.9, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'p1phi nu=-1.6': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'p1phi nu=-2.1': {'nu':-2.1, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                # 'T nu=-1.6': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'M nu=0.2': {'nu':0.2, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
            }
        self.config_fft = config_fft
        self.set_power_law_decomp(config_fft)
        self.mat = {}
        self.matrix = {}

        self.name_pkmu_gg_terms = {}
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/M22_*.txt')
        self.name_pkmu_gg_terms['22_gg'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/gauss/M13_*.txt')
        self.name_pkmu_gg_terms['13_gg'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M22_*.txt')
        self.name_pkmu_gg_terms['22_gg_lpng'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M13_*lpng1*.txt')
        self.name_pkmu_gg_terms['13_gg_lpng1'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M13_*lpng3*.txt')
        self.name_pkmu_gg_terms['13_gg_lpng3'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M12_*lpng1*.txt')
        self.name_pkmu_gg_terms['12_gg_lpng1'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M12_*lpng2*.txt')
        self.name_pkmu_gg_terms['12_gg_lpng2'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        self.name_pkmu_gg_terms['tot'] = self.name_pkmu_gg_terms['22_gg'] + self.name_pkmu_gg_terms['13_gg']
        self.name_pkmu_gg_terms['tot'] += self.name_pkmu_gg_terms['22_gg_lpng']
        self.name_pkmu_gg_terms['tot'] += self.name_pkmu_gg_terms['13_gg_lpng1'] + self.name_pkmu_gg_terms['13_gg_lpng3']
        self.name_pkmu_gg_terms['tot'] += self.name_pkmu_gg_terms['12_gg_lpng1'] + self.name_pkmu_gg_terms['12_gg_lpng2']

    def set_cosmology(self, cparam, redshift=0., omega_nu0=0.00064, Omega_K0=0., kmin=1e-7, kmax=1e+7, khigh=None):
        # run CAMB
        self.set_camb(cparam, omega_nu0=omega_nu0, Omega_K0=Omega_K0, kmin=kmin, kmax=kmax)
        self.redshift = redshift
        self.fgrowth = self.cosmo.get_fgrowth_lcdm(redshift, mode='z')

        # FFTLog-based power-law decomposition
        self.decomp['plin nu=-0.3'].compute(self.get_pk_lin, kwarg={'khigh':khigh})
        self.decomp['plin nu=-0.7'].compute(self.get_pk_lin, kwarg={'khigh':khigh})
        self.decomp['plin nu=-1.6'].compute(self.get_pk_lin, kwarg={'khigh':khigh})
        self.decomp['p1phi nu=-0.9'].compute(self.get_pk_1phi, kwarg={'khigh':khigh})
        self.decomp['p1phi nu=-1.6'].compute(self.get_pk_1phi, kwarg={'khigh':khigh})
        self.decomp['p1phi nu=-2.1'].compute(self.get_pk_1phi, kwarg={'khigh':khigh})
        self.decomp['M nu=0.2'].compute(self.get_M)

        # set up the IR resummation
        self.irres = IRResum(self.get_pk_lin, hubble=self.params['h'], rbao=110, 
                            khmin=7e-5, khmax=7, n_min=120, n_max=240,
                            kmin_interp=kmin, kmax_interp=kmax, kwarg={'khigh':khigh})
        self.decomp['plin nu=-0.3 (no-wiggle)'].compute(self.irres.get_pk_nw)
        self.decomp['plin nu=-0.7 (no-wiggle)'].compute(self.irres.get_pk_nw)
        self.decomp['plin nu=-1.6 (no-wiggle)'].compute(self.irres.get_pk_nw)

        # compute the power spectrum integrals for UV part of P13
        self.plin_int = self.get_pk_int(self.get_pk_lin, kmin=1e-7, kmax=1e+7, limit=1000, kwarg={'khigh':khigh})
        self.plin_nw_int = self.get_pk_int(self.irres.get_pk_nw, kmin=1e-7, kmax=1e+7, limit=1000)
        self.sigma2_v = self.plin_int / 3

    def set_f_nl(self, f_nl):
        self.f_nl = f_nl

    def get_M(self, k):
        h = self.params['h']
        Omega_m0 = self.params['Omega_m0']
        H_0 = self.params['H_0_in_Mpc_inv']
        Tk = self.get_matter_transfer(k*h) # normalized as T(k) -> 1 (k -> 0)
        Dgrowth = self.cosmo.get_Dgrowth_lcdm(self.redshift, mode='z') # normalized as D(a) -> a (a -> 0)
        return 2./3 * (k*h)**2 * (Tk * Dgrowth) / (Omega_m0 * H_0**2)

    def get_T(self, k):
        Delta_phi = 1 # should be modified later.
        M = self.get_M(k)
        return Delta_phi * M / k**2

    def get_pk_1phi(self, k, khigh=None):
        plin = self.get_pk_lin(k, khigh=khigh)
        M = self.get_M(k)
        return plin / M

    def get_pk_gg_lin(self, k, mu):
        k = np.atleast_1d(k)

        Z1_1 = self.bias1['b1'] + self.bias1['bphi'] * self.f_nl / self.get_M(k)
        Z1_2 = self.bias2['b1'] + self.bias2['bphi'] * self.f_nl / self.get_M(k)
        pk_lin = self.get_pk_lin(k)
        pk = Z1_1 * Z1_2 * pk_lin

        return pk

    def get_pkmu_gg_lin(self, k, mu):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        Z1_1 = self.bias1['b1'] + self.fgrowth * mu**2
        Z1_lpng_1 = self.bias1['bphi'] * self.f_nl / self.get_M(k)
        Z1_2 = self.bias2['b1'] + self.fgrowth * mu**2
        Z1_lpng_2 = self.bias2['bphi'] * self.f_nl / self.get_M(k)

        Z1_1_tile = np.tile(Z1_1, (len(k), 1)) + np.tile(Z1_lpng_1, (len(mu),1)).T
        Z1_2_tile = np.tile(Z1_2, (len(k), 1)) + np.tile(Z1_lpng_2, (len(mu),1)).T

        pk_lin = self.get_pk_lin(k)
        pk_lin_tile = np.tile(pk_lin, (len(mu),1)).T

        pkmu = Z1_1_tile * Z1_2_tile * pk_lin_tile

        if len(k) == 1 or len(mu) == 1:
            pkmu = np.ravel(pkmu)
        return pkmu

    def get_pl_gg_lin(self, l, k):
        k = np.atleast_1d(k)
        mu = np.linspace(0.,1.,2**8+1)
        dmu = mu[1]-mu[0]
        pkmu = self.get_pkmu_gg_lin(k,mu)
        legendre = np.tile(lpmv(0,l,mu), (len(k),1))
        pl = (2*l+1) * romb(pkmu * legendre, axis=1, dx=dmu)
        return pl

    def get_pkmu_13_lpng1_UV(self, k, mu, mode='full'):
        Z1_lpng = self.bias1['bphi']
        Z3_g_UV = - 61./315. * self.bias2['b1'] - 64./21. * self.bias2['bG2'] - 128./105. * self.bias2['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * self.bias2['b1']) * self.fgrowth * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * self.bias2['b1']) * self.fgrowth**2 * mu**2
        Z3_g_UV += (- 46./105.) * self.fgrowth**2 * mu**4
        Z3_g_UV += (- 1./3.) * self.fgrowth**3 * mu**4
        Z1Z3_UV_1 = Z1_lpng * Z3_g_UV

        Z1_lpng = self.bias2['bphi']
        Z3_g_UV = - 61./315. * self.bias1['b1'] - 64./21. * self.bias1['bG2'] - 128./105. * self.bias1['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * self.bias1['b1']) * self.fgrowth * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * self.bias1['b1']) * self.fgrowth**2 * mu**2
        Z3_g_UV += (- 46./105.) * self.fgrowth**2 * mu**4
        Z3_g_UV += (- 1./3.) * self.fgrowth**3 * mu**4
        Z1Z3_UV_2 = Z1_lpng * Z3_g_UV

        Z1Z3_UV = (Z1Z3_UV_1 + Z1Z3_UV_2) / 2

        if mode == 'full':
            pk_fac = self.get_pk_1phi(k) * self.plin_int
        elif mode == 'no-wiggle':
            pk_fac = self.get_pk_1phi(k) * self.plin_nw_int
        else:
            raise ValueError('Invalid mode')

        pkmu_13 = np.kron(k**2 * pk_fac, Z1Z3_UV).reshape(len(k),len(mu))
        
        return pkmu_13

    def get_pk_rsd_1loop_data(self, name, sub_k0=True, mode='full'):
        if '12' in name:
            if 'lpng1' in name:
                name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict['12_gg_lpng1'])
            else:
                name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict['12_gg_lpng2'])
            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_q = self.decomp[name_dec[1]].func_q
            pk_data = kn**3 * np.diag(np.dot(p1_q.T, np.dot(self.matrix[name], p2_q)).real)

        elif '22' in name:
            if 'lpng' in name:
                name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict['22_gg_lpng'])
            else:
                name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict['22_gg'])
            if mode != 'full':
                for i in range(len(name_dec)):
                    if 'plin' in name_dec[i]: name_dec[i] += ' (%s)' % (mode)

            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_q = self.decomp[name_dec[1]].func_q
            pk_data = kn**3 * np.diag(np.dot(p1_q.T, np.dot(self.matrix[name], p2_q)).real)
            if sub_k0:
                kmin = self.decomp[name_dec[0]].kmin
                p1_k0 = self.decomp[name_dec[0]].func_k0
                p2_k0 = self.decomp[name_dec[1]].func_k0
                pk_data_k0 = kmin**3 * np.dot(p1_k0, np.dot(self.matrix[name], p2_k0)).real
                pk_data = pk_data - pk_data_k0

        elif '13' in name:
            if 'lpng1' in name:
                name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict['13_gg_lpng1'])
            elif 'lpng3' in name:
                name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict['13_gg_lpng3'])
            else:
                name_dec = copy.deepcopy(utils_loop.kernel_to_decomp_dict['13_gg'])
            if mode != 'full':
                for i in range(len(name_dec)):
                    if 'plin' in name_dec[i]: name_dec[i] += ' (%s)' % (mode)

            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_q = self.decomp[name_dec[1]].func_rec.real
            pk_data = kn**3 * p2_q * np.dot(self.matrix[name], p1_q).real

        else:
            raise KeyError('PT kernel name is invalid.')
        return kn, pk_data

    def get_pkmu_gg_1loop_raw(self, k, mu, name='tot', mode='full'):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        if name == 'tot':
            # Gaussian terms
            pkmu_22 = self.get_pkmu_gg_1loop_raw(k, mu, name='22_gg', mode=mode)
            pkmu_13 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg', mode=mode)

            # LPNG 1-2 term (first order in f_NL)
            pkmu_12_lpng1 = self.get_pkmu_gg_1loop_raw(k, mu, name='12_gg_lpng1', mode=mode)
            pkmu_12_lpng2 = self.get_pkmu_gg_1loop_raw(k, mu, name='12_gg_lpng2', mode=mode)
            pkmu_12_lpng = pkmu_12_lpng1 + pkmu_12_lpng2

            # LPNG 2-2 term (first order in f_NL)
            pkmu_22_lpng = self.get_pkmu_gg_1loop_raw(k, mu, name='22_gg_lpng', mode=mode)

            # LPNG 1-3 term (first order in f_NL)
            pkmu_13_lpng1 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng1', mode=mode)
            pkmu_13_lpng2 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng2', mode=mode)
            pkmu_13_lpng3 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng3', mode=mode)
            pkmu_13_lpng = pkmu_13_lpng1 + pkmu_13_lpng2 + pkmu_13_lpng3

            pkmu = (pkmu_22 + pkmu_13) + self.f_nl * (pkmu_12_lpng + pkmu_22_lpng + pkmu_13_lpng)
            return pkmu

        elif name == 'gauss_tot':
            # Gaussian terms
            pkmu_22 = self.get_pkmu_gg_1loop_raw(k, mu, name='22_gg', mode=mode)
            pkmu_13 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg', mode=mode)
            pkmu = pkmu_22 + pkmu_13
            return pkmu

        elif name == 'lpng_tot':
            pkmu_12_lpng = self.get_pkmu_gg_1loop_raw(k, mu, name='12_gg_lpng', mode=mode)
            pkmu_22_lpng = self.get_pkmu_gg_1loop_raw(k, mu, name='22_gg_lpng', mode=mode)
            pkmu_13_lpng = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng', mode=mode)
            pkmu = pkmu_12_lpng + pkmu_22_lpng + pkmu_13_lpng
            return pkmu

        elif name == '12_gg_lpng':
            # LPNG 1-2 term (first order in f_NL)
            pkmu_12_lpng1 = self.get_pkmu_gg_1loop_raw(k, mu, name='12_gg_lpng1', mode=mode)
            pkmu_12_lpng2 = self.get_pkmu_gg_1loop_raw(k, mu, name='12_gg_lpng2', mode=mode)
            pkmu_12_lpng = pkmu_12_lpng1 + pkmu_12_lpng2
            return pkmu_12_lpng

        elif name == '13_gg_lpng':
            # LPNG 1-3 term (first order in f_NL)
            pkmu_13_lpng1 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng1', mode=mode)
            pkmu_13_lpng2 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng2', mode=mode)
            pkmu_13_lpng3 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng3', mode=mode)
            pkmu_13_lpng = pkmu_13_lpng1 + pkmu_13_lpng2 + pkmu_13_lpng3
            return pkmu_13_lpng

        elif name == '13_gg_lpng2':
            Z1_g1 = np.tile(self.bias1['b1'] + self.fgrowth * mu**2, (len(k),1))
            Z1_g2 = np.tile(self.bias2['b1'] + self.fgrowth * mu**2, (len(k),1))
            
            pk_1phi = np.tile(self.get_pk_1phi(k), (len(mu),1)).T
            factor = np.kron(k**2, (1 + self.fgrowth**2 * mu**2)).reshape(len(k),len(mu))
            
            pkmu = - (self.bias1['bphi'] * Z1_g2 + self.bias2['bphi'] * Z1_g1) / 2 * factor * self.sigma2_v * pk_1phi

            if len(k) == 1 or len(mu) == 1:
                pkmu = np.ravel(pkmu)
            return pkmu

        term_names = self.name_pkmu_gg_terms[name]
        pkmu_tab = []
        for term in term_names:
            nf, nmu, bias_deg = utils_loop.get_deg_info(term)

            # only for auto power spectrum
            # bias_fac = np.prod([self.bias[key]**val for key, val in bias_deg.items()])

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
                raise ValueError('Invalid number of bias')

            fac = bias_fac * self.fgrowth**nf

            kn, pk_data = self.get_pk_rsd_1loop_data(term, mode=mode)

            pkmu = fac * np.kron(pk_data, mu**nmu).reshape(len(kn),len(mu))
            pkmu_tab.append(pkmu)
        pkmu_tab = np.array(pkmu_tab)
        pkmu_data = np.sum(pkmu_tab, axis=0)

        if name == '13_gg':
            pkmu_UV = self.get_pkmu_13_UV(kn, mu, mode=mode)
            pkmu_data += pkmu_UV
        elif name == '13_gg_lpng1':
            pkmu_UV = self.get_pkmu_13_lpng1_UV(kn, mu, mode=mode)
            pkmu_data += pkmu_UV
        elif name == '12_gg_lpng1':
            pk_1phi = np.tile(self.get_pk_1phi(kn), (len(mu),1)).T
            pkmu_data = 2 * pk_1phi * pkmu_data
        elif name == '12_gg_lpng2':
            Mk = np.tile(self.get_M(kn), (len(mu),1)).T
            pkmu_data = Mk * pkmu_data

        if len(mu) == 1:
            pkmu_interp = ius(kn, np.ravel(pkmu_data))
            pkmu = pkmu_interp(k)
        else:
            pkmu_interp = rbs(kn, mu, pkmu_data)
            pkmu = pkmu_interp(k, mu)

        return pkmu

    def get_pkmu_gg_irres(self, k, mu, mode='LO+NLO', Sigma2=None, dSigma2=None, ks=0.2):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        # wiggly-non-wiggly decomposition
        plin = self.get_pk_lin(k)
        plin_nw = self.irres.get_pk_nw(k)
        plin_w = plin - plin_nw

        # BAO damping factor in redshift space
        if Sigma2 == None:
            Sigma2 = self.irres.get_Sigma2(ks=ks)
        if dSigma2 == None:
            dSigma2 = self.irres.get_dSigma2(ks=ks)
        Sigma2_1 = (1 + mu**2 * self.fgrowth * (2 + self.fgrowth)) * Sigma2
        Sigma2_2 = self.fgrowth**2 * mu**2 * (mu**2 - 1) * dSigma2
        Sigma2_tot = Sigma2_1 + Sigma2_2

        Z1_tile1 = np.tile(self.bias1['b1'] + self.fgrowth * mu**2, (len(k), 1)) + np.tile(self.bias1['bphi'] * self.f_nl / self.get_M(k), (len(mu),1)).T
        Z1_tile2 = np.tile(self.bias2['b1'] + self.fgrowth * mu**2, (len(k), 1)) + np.tile(self.bias2['bphi'] * self.f_nl / self.get_M(k), (len(mu),1)).T
        plin_nw_tile = np.tile(plin_nw, (len(mu),1)).T
        plin_w_tile = np.tile(plin_w, (len(mu),1)).T
        damp_fac = np.kron(k**2, Sigma2_tot).reshape(len(k),len(mu))

        if len(k) == 1 or len(mu) == 1:
            Z1_tile1 = np.ravel(Z1_tile1)
            Z1_tile2 = np.ravel(Z1_tile2)
            damp_fac = np.ravel(damp_fac)
            plin_nw_tile = np.ravel(plin_nw_tile)
            plin_w_tile = np.ravel(plin_w_tile)

        # IR-resummed power spectrum
        if mode == 'LO':
            # leading-order IR resummation
            pkmu = Z1_tile1 * Z1_tile2 * (plin_nw_tile + np.exp(-damp_fac) * plin_w_tile)
        elif mode == 'tree':
            # leading-order IR resummation + additional term to prevent the double counting
            pkmu = Z1_tile1 * Z1_tile2 * (plin_nw_tile + (1 + damp_fac) * np.exp(-damp_fac) * plin_w_tile)
        elif mode == '1loop no-wiggle':
            # 1-loop term computed with the non-wiggly component of the linear power spectrum
            pkmu = self.get_pkmu_gg_1loop_raw(k, mu, name='tot', mode='no-wiggle')
        elif mode == '1loop wiggle':
            # 1-loop term computed with the wiggly component of the linear power spectrum
            pkmu = self.get_pkmu_gg_1loop_raw(k, mu, name='tot') - self.get_pkmu_gg_1loop_raw(k, mu, name='tot', mode='no-wiggle')
        elif mode == '1loop':
            # next-to-leading order term of the IR-resummed power spectrum
            pkmu_1loop_nw = self.get_pkmu_gg_irres(k, mu, mode='1loop no-wiggle', Sigma2=Sigma2, dSigma2=dSigma2, ks=ks)
            pkmu_1loop_w = self.get_pkmu_gg_1loop_raw(k, mu, name='tot') - pkmu_1loop_nw
            pkmu = pkmu_1loop_nw + np.exp(-damp_fac) * pkmu_1loop_w
        elif mode == 'LO+NLO':
            # LO+NLO IR-resummed power spectrum
            pkmu_tree = self.get_pkmu_gg_irres(k, mu, mode='tree', Sigma2=Sigma2, dSigma2=dSigma2, ks=ks)
            pkmu_1loop = self.get_pkmu_gg_irres(k, mu, mode='1loop', Sigma2=Sigma2, dSigma2=dSigma2, ks=ks)
            pkmu = pkmu_tree + pkmu_1loop

        if len(k) == 1 or len(mu) == 1:
            pkmu = np.ravel(pkmu)
        return pkmu
