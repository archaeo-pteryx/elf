import os, sys, time
import glob, re
import numpy as np
from scipy.integrate import quad, romb
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.special import lpmv
import fftlog
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
                'plin nu=-1.6 (no-wiggle)': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-0.3 (LO IR-res)': {'nu':-0.3, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-1.6 (LO IR-res)': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'p1phi nu=-0.9': {'nu':-0.9, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'p1phi nu=-1.6': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'p1phi nu=-2.1': {'nu':-2.1, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'T nu=-1.6': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
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
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M13_*.txt')
        self.name_pkmu_gg_terms['13_gg_lpng3'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M12_*lpng1*.txt')
        self.name_pkmu_gg_terms['12_gg_lpng1'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/lpng/M12_*lpng2*.txt')
        self.name_pkmu_gg_terms['12_gg_lpng2'] = [re.split('/', fname)[-1][:-4] for fname in fnames]
        self.name_pkmu_gg_terms['tot'] = self.name_pkmu_gg_terms['22_gg'] + self.name_pkmu_gg_terms['13_gg']
        self.name_pkmu_gg_terms['tot'] += self.name_pkmu_gg_terms['22_gg_lpng'] + self.name_pkmu_gg_terms['13_gg_lpng3']
        self.name_pkmu_gg_terms['tot'] += self.name_pkmu_gg_terms['12_gg_lpng1'] + self.name_pkmu_gg_terms['12_gg_lpng2']

    def set_cosmology(self, cparam, z=0, omega_nu0=0.00064, Omega_K0=0., kmin=1e-7, kmax=1e+7, khigh=40.):
        # run CAMB
        self.set_camb(cparam, omega_nu0=omega_nu0, Omega_K0=Omega_K0, kmin=kmin, kmax=kmax)
        self.fgrowth = self.cosmo.get_fgrowth_lcdm(z, mode='z')

        # FFTLog-based power-law decomposition
        self.decomp['plin nu=-0.3'].compute(self.get_pk_lin, kwarg={'z':z, 'khigh':khigh})
        self.decomp['plin nu=-0.7'].compute(self.get_pk_lin, kwarg={'z':z, 'khigh':khigh})
        self.decomp['plin nu=-1.6'].compute(self.get_pk_lin, kwarg={'z':z, 'khigh':khigh})
        self.decomp['p1phi nu=-0.9'].compute(self.get_pk_1phi, kwarg={'z':z, 'khigh':khigh})
        self.decomp['p1phi nu=-1.6'].compute(self.get_pk_1phi, kwarg={'z':z, 'khigh':khigh})
        self.decomp['p1phi nu=-2.1'].compute(self.get_pk_1phi, kwarg={'z':z, 'khigh':khigh})
        self.decomp['M nu=0.2'].compute(self.get_M, kwarg={'z':z})

        # set up the IR resummation
        self.irres = IRResum(self.get_pk_lin, self.params['h'], rbao=110, kmin=7e-5, kmax=7, n_min=120, n_max=240, kwarg={'z':z})
        Sigma2_ref = self.irres.get_Sigma2(ks=0.2)
        self.decomp['plin nu=-0.3 (no-wiggle)'].compute(self.irres.get_pk_nw)
        self.decomp['plin nu=-1.6 (no-wiggle)'].compute(self.irres.get_pk_nw)
        self.decomp['plin nu=-0.3 (LO IR-res)'].compute(self.get_pk_mm_irres, kwarg={'Sigma2': Sigma2_ref})
        self.decomp['plin nu=-1.6 (LO IR-res)'].compute(self.get_pk_mm_irres, kwarg={'Sigma2': Sigma2_ref})

        # compute IR and UV parts of P13 and P22
        self.plin_int = self.get_pk_lin_int(khigh=khigh, kmin=1e-7, kmax=1e+7, limit=1000)
        self.plin_int2 = self.get_pk_lin_int2(khigh=khigh, kmin=1e-7, kmax=1e+7, limit=1000)
        self.alpha = {}
        self.alpha['UV13'] = -(122/315) * self.plin_int
        self.alpha['IR13'] = -(2/3) * self.plin_int
        self.alpha['UV22'] = (9/49) * self.plin_int2
        self.alpha['IR22'] = (2/3) * self.plin_int
        self.sigma2_v = self.plin_int / 3

    def set_f_nl(self, f_nl):
        self.f_nl = f_nl

    def get_M(self, k, z=0):
        h = self.params['h']
        Omega_m0 = self.params['Omega_m0']
        H_0 = self.params['H_0_in_Mpc_inv']
        Tk = self.get_matter_transfer(k*h) # normalized as T(k) -> 1 (k -> 0)
        Dgrowth = self.cosmo.get_Dgrowth_lcdm(z, mode='z') # normalized as D(a) -> a (a -> 0)
        return 2./3 * (k*h)**2 * (Tk * Dgrowth) / (Omega_m0 * H_0**2)

    def get_T(self, k, z=0):
        Delta_phi = 1 # should be modified later.
        M = self.get_M(k, z=z)
        return Delta_phi * M / k**2

    def get_pk_1phi(self, k, z=0, khigh=None):
        plin = self.get_pk_lin(k, z=z, khigh=khigh)
        M = self.get_M(k, z=z)
        return plin / M

    def get_pk_gg_lin(self, k, mu):
        k = np.atleast_1d(k)

        b1 = self.bias['b1']
        bphi = self.bias['bphi']
        fgrowth = self.fgrowth
        f_nl = self.f_nl

        Z1 = b1 + bphi * f_nl / self.get_M(k)
        pk_lin = self.get_pk_lin(k)
        pk = Z1**2 * pk_lin

        return pk

    def get_pkmu_gg_lin(self, k, mu):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        b1 = self.bias['b1']
        bphi = self.bias['bphi']
        fgrowth = self.fgrowth
        f_nl = self.f_nl
        pk_lin = self.get_pk_lin(k)

        Z1_g = b1 + fgrowth * mu**2
        Z1_lpng = bphi * f_nl / self.get_M(k)
        Z1_tile = np.tile(Z1_g, (len(k), 1)) + np.tile(Z1_lpng, (len(mu),1)).T
        pk_lin_tile = np.tile(pk_lin, (len(mu),1)).T
        pkmu = Z1_tile**2 * pk_lin_tile

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

    def get_pk_rsd_1loop_data(self, name, sub_k0=True, mode='full'):
        if '12' in name:
            if 'lpng1' in name:
                name_dec = utils_loop.kernel_to_decomp_dict['12_gg_lpng1']
            else:
                name_dec = utils_loop.kernel_to_decomp_dict['12_gg_lpng2']
            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_q = self.decomp[name_dec[1]].func_q
            pk_data = kn**3 * np.diag(np.dot(p1_q.T, np.dot(self.matrix[name], p2_q)).real)

        elif '22' in name:
            if 'lpng' in name:
                name_dec = utils_loop.kernel_to_decomp_dict['22_gg_lpng']
            else:
                name_dec = utils_loop.kernel_to_decomp_dict['22_gg']
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
                name_dec = utils_loop.kernel_to_decomp_dict['13_gg_lpng1']
            elif 'lpng3' in name:
                name_dec = utils_loop.kernel_to_decomp_dict['13_gg_lpng3']
            else:
                name_dec = utils_loop.kernel_to_decomp_dict['13_gg']
            if mode != 'full':
                for i in range(len(name_dec)):
                    if 'plin' in name_dec[i]: name_dec[i] += ' (%s)' % (mode)

            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_q = self.decomp[name_dec[1]].func_rec.real
            p13_int = np.dot(self.matrix[name], p1_q).real
            pk_data = kn**3 * p2_q * p13_int

        else:
            raise KeyError('PT kernel name is invalid.')
        return kn, pk_data

    def get_pkmu_gg_1loop_raw(self, k, mu, name='tot', mode='full'):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        b1 = self.bias['b1']
        bphi = self.bias['bphi']
        fgrowth = self.fgrowth
        f_nl = self.f_nl

        if name == 'tot':
            # Gaussian terms
            pkmu_22 = self.get_pkmu_gg_1loop_raw(k, mu, name='22_gg', mode=mode)
            pkmu_13 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg', mode=mode)

            # 2-2 term
            pkmu_22_lpng = self.get_pkmu_gg_1loop_raw(k, mu, name='22_gg_lpng', mode=mode)

            # 1-3 term
            pkmu_13_lpng1 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng1', mode=mode)
            pkmu_13_lpng2 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng2', mode=mode)
            pkmu_13_lpng3 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg_lpng3', mode=mode)
            pkmu_13_lpng = pkmu_13_lpng1 + pkmu_13_lpng2 + pkmu_13_lpng3

            ## 1-2 term
            pkmu_12_lpng1 = self.get_pkmu_gg_1loop_raw(k, mu, name='12_gg_lpng1', mode=mode)
            pkmu_12_lpng2 = self.get_pkmu_gg_1loop_raw(k, mu, name='12_gg_lpng2', mode=mode)
            pkmu_12_lpng = pkmu_12_lpng1 + pkmu_12_lpng2

            pkmu = pkmu_22 + pkmu_13
            pkmu += pkmu + f_nl * (pkmu_22_lpng + pkmu_13_lpng + pkmu_12_lpng)

            return pkmu

        elif name == '13_gg_lpng1':
            Z1_g = b1 + fgrowth * mu**2
            Z1_g = np.tile(Z1_g, (len(k), 1))
            Z1_lpng = bphi * f_nl / self.get_M(k)
            Z1_lpng = np.tile(Z1_lpng, (len(mu), 1)).T
            pkmu_13 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg', mode=mode)
            factor = Z1_lpng / Z1_g
            if len(k) == 1 or len(mu) == 1:
                factor = np.ravel(factor)
            pkmu = factor * pkmu_13
            return pkmu

        elif name == '13_gg_lpng2':
            Z1_g = b1 + fgrowth * mu**2
            Z1_g = np.tile(Z1_g, (len(k),1))
            pk_1phi = self.get_pk_1phi(k)
            pk_1phi = np.tile(pk_1phi, (len(mu),1)).T
            factor = np.kron(k**2, (1 + fgrowth**2 * mu**2)).reshape(len(k),len(mu))
            pkmu = - bphi * f_nl * Z1_g * factor * self.sigma2_v * pk_1phi
            if len(k) == 1 or len(mu) == 1:
                pkmu = np.ravel(pkmu)
            return pkmu

        term_names = self.name_pkmu_gg_terms[name]
        pkmu_tab = []
        for term in term_names:
            nf, nmu, bias_deg = utils_loop.get_deg_info(term)

            bias_fac = np.prod([self.bias[key]**val for key, val in bias_deg.items()])
            fac = bias_fac * self.fgrowth**nf
            if fac == 0.: continue
            
            kn, pk_data = self.get_pk_rsd_1loop_data(term, mode=mode)

            pkmu = fac * np.kron(pk_data, mu**nmu).reshape(len(kn),len(mu))
            pkmu_tab.append(pkmu)
        pkmu_tab = np.array(pkmu_tab)
        pkmu_data = np.sum(pkmu_tab, axis=0)

        Z1_g = b1 + fgrowth * mu**2
        Z1_g = np.tile(Z1_g, (len(kn), 1))

        if name == '13_gg':
            pkmu_data = Z1_g * pkmu_data
            pkmu_UV = self.get_pkmu_13_UV(kn, mu)
            pkmu_data += pkmu_UV
        elif name == '13_gg_lpng3':
            pkmu_data = Z1_g * pkmu_data
        elif name == '12_gg_lpng1':
            pk_1phi = self.get_pk_1phi(kn)
            pk_1phi = np.tile(pk_1phi, (len(mu),1)).T
            pkmu_data = 2 * f_nl * Z1_g * pk_1phi * pkmu_data
        elif name == '12_gg_lpng2':
            Mk = self.get_M(kn)
            Mk = np.tile(Mk, (len(mu),1)).T
            pkmu_data = f_nl * Z1_g * Mk * pkmu_data

        if len(mu) == 1:
            pkmu_interp = ius(kn, np.ravel(pkmu_data))
            pkmu = pkmu_interp(k)
        else:
            pkmu_interp = rbs(kn, mu, pkmu_data)
            pkmu = pkmu_interp(k, mu)

        return pkmu
