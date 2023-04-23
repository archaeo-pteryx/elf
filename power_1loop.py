import os, sys, time
import glob, re
import numpy as np
from scipy.integrate import quad, romb
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
from scipy.special import lpmv
import fftlog
from camb_wrapper import CambWrapper

from power_law_decomp import PowerLawDecomp
import pt_matrix
import utils_loop
from ir_resum import IRResum


class PowerSpectrum1loop:

    def __init__(self, config_fft=None):
        self.params = None
        if config_fft == None:
            config_fft = {
                'plin nu=-0.3': {'nu':-0.3, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-1.6': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-0.3 (no-wiggle)': {'nu':-0.3, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-1.6 (no-wiggle)': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-0.3 (LO IR-res)': {'nu':-0.3, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256},
                'plin nu=-1.6 (LO IR-res)': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':256}
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
        self.name_pkmu_gg_terms['tot'] = self.name_pkmu_gg_terms['22_gg'] + self.name_pkmu_gg_terms['13_gg']

    def set_camb(self, cparam, omega_nu0=0.00064, Omega_K0=0., kmin=1e-7, kmax=1e+7):
        self.cosmo = CambWrapper(cparam, omega_nu0=omega_nu0, Omega_K0=Omega_K0)
        self.params = self.cosmo.params
        self.cosmo.set_matter_power(z=0.)
        
        k, Tk = self.cosmo.get_matter_transfer_data()
        Tk[1] = Tk[0]
        self.Tk_lowk = Tk[0]

        dlnk_low = np.log(k[1]/k[0])
        dlnk_high = np.log(k[-1]/k[-2])
        N_extrap_low = int(np.log(k[0]/kmin) / dlnk_low) + 1
        N_extrap_high = int(np.log(kmax/k[-1]) / dlnk_high) + 1
        k_extrap = fftlog.log_extrap(k, N_extrap_low, N_extrap_high)
        Tk_extrap = fftlog.log_extrap(Tk, N_extrap_low, N_extrap_high)
        self.matter_transfer_spl = ius(np.log(k_extrap),Tk_extrap/self.Tk_lowk)

    def get_matter_transfer(self, k): # normalized as T(k) -> 1 (k -> 0)
        return self.matter_transfer_spl(np.log(k))

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
                nu_m1, nu_m2 = np.meshgrid(nu_m1, nu_m2)
                self.matrix[name] = self.mat[name](nu_m1, nu_m2).T
            elif '13' in name or 'F' in name:
                nu_m1 = -0.5 * self.decomp[name_dec[0]].nu_m
                self.matrix[name] = self.mat[name](nu_m1)
            else:
                raise KeyError('PT kernel name %s is invalid.' % (name))

    def set_cosmology(self, cparam, z=0, omega_nu0=0.00064, Omega_K0=0., kmin=1e-7, kmax=1e+7, khigh=40.):
        # run CAMB
        self.set_camb(cparam, omega_nu0=omega_nu0, Omega_K0=Omega_K0, kmin=kmin, kmax=kmax)
        self.fgrowth = self.cosmo.get_fgrowth_lcdm(z, mode='z')

        # FFTLog-based power-law decomposition
        self.decomp['plin nu=-0.3'].compute(self.get_pk_lin, kwarg={'z':z, 'khigh':khigh})
        self.decomp['plin nu=-1.6'].compute(self.get_pk_lin, kwarg={'z':z, 'khigh':khigh})

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

    def set_bias(self, bias={}):
        self.bias = bias

    def get_pk_prim(self, kh):
        # primordial power spectrum of the curvature perturbations in unit of [Mpc^3]
        As = self.params['As']
        ns = self.params['ns']
        k_pivot = self.params['k_pivot']
        pk_prim = As * (kh / k_pivot)**(ns-1)
        return pk_prim

    def get_pk_lin(self, k, z=0, khigh=None): # in unit of [h^{-3} Mpc^3]
        h = self.params['h']
        pk_prim = self.get_pk_prim(k*h)
        Tk = self.get_matter_transfer(k*h) * self.Tk_lowk

        plin_z0 = Tk**2 * pk_prim * (k*h)**4 / ((k*h)**3 / (2*np.pi**2))
        plin_z0 *= h**3
        
        Dgrowth = self.cosmo.get_Dgrowth_lcdm(np.array([0,z]), mode='z')
        Dgrowth = Dgrowth[1] / Dgrowth[0]
        plin = Dgrowth**2 * plin_z0

        if khigh != None:
            plin *= np.exp(-(k/khigh))

        return plin

    def get_pk_lin_int(self, khigh=None, kmin=1e-7, kmax=1e7, limit=1000):
        res = quad(lambda logk: self.get_pk_lin(np.exp(logk), khigh=khigh) * np.exp(logk), np.log(kmin), np.log(kmax), limit=limit)
        return res[0] / (2*np.pi**2)

    def get_pk_lin_int2(self, khigh=None, kmin=1e-7, kmax=1e7, limit=1000):
        res = quad(lambda logk: self.get_pk_lin(np.exp(logk), khigh=khigh)**2 / np.exp(logk), np.log(kmin), np.log(kmax), limit=limit)
        return res[0] / (2*np.pi**2)

    def get_pk_mm_irres(self, k, mode='LO', Sigma2=None, ks=0.2):
        plin = self.get_pk_lin(k)
        plin_nw = self.irres.get_pk_nw(k)
        plin_w = plin - plin_nw
        if Sigma2 == None:
            Sigma2 = self.irres.get_Sigma2(ks=ks)
        if mode == 'LO':
            pk = plin_nw + np.exp(-k**2 * Sigma2) * plin_w
        elif mode == 'tree':
            pk = plin_nw + np.exp(-k**2 * Sigma2) * plin_w * (1 + k**2 * Sigma2)
        return pk

    def get_pk_1loop_data(self, name, sub_k0=True, mode='full'):
        name_dec = utils_loop.kernel_to_decomp_dict[name]
        if mode != 'full':
            for i in range(len(name_dec)):
                if 'plin' in name_dec[i]: name_dec[i] += ' (%s)' % (mode)

        if '22' in name or 'I' in name:
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

        elif '13' in name or 'F' in name:
            kn = self.decomp[name_dec[0]].kn
            p1_q = self.decomp[name_dec[0]].func_q
            p2_k = self.decomp[name_dec[1]].func_rec.real
            p13_int = np.dot(self.matrix[name], p1_q).real
            if name == '13':
                p13_int += self.alpha['UV13'] / kn
            pk_data = kn**3 * p2_k * p13_int

        else:
            raise KeyError('PT kernel name is invalid.')

        return kn, pk_data

    def get_pk_1loop(self, k, name='22', sub_k0=True):
        alpha = 1.5
        kn, pk_data = self.get_pk_1loop_data(name=name, sub_k0=sub_k0)
        pk_interp = ius(kn, kn**alpha * pk_data)
        pk = pk_interp(k) * k**(-alpha)
        return pk

    def get_pk_gg_raw(self, k, sub_k0_22=True):
        pk_tree = self.get_pk_lin(k)

        name_list = ['22','13','I_d2','I_G2','I_d2_d2','I_G2_G2','I_d2_G2','F_G2']
        pks = {}
        for name in name_list:
            pks[name] = self.get_pk_1loop(k, name=name, sub_k0=sub_k0_22)

        pk_gg = self.bias['b1']**2 * (pk_tree + pks['22'] + pks['13'])
        pk_gg = pk_gg + self.bias['b1'] * self.bias['b2'] * pks['I_d2']
        pk_gg = pk_gg + 2 * self.bias['b1'] * self.bias['bG2'] * pks['I_G2']
        pk_gg = pk_gg + self.bias['b2']**2 / 4. * pks['I_d2_d2']
        pk_gg = pk_gg + self.bias['bG2']**2 * pks['I_G2_G2']
        pk_gg = pk_gg + self.bias['b2'] * self.bias['bG2'] * pks['I_d2_G2']
        pk_gg = pk_gg + 2 * self.bias['b1'] * self.bias['bG2'] * pks['F_G2']
        pk_gg = pk_gg + (4./5.) * self.bias['b1'] * self.bias['bGamma3'] * pks['F_G2']

        # pk_gg = pk_gg - self.bias['c0'] * k**2 * pk_tree

        return pk_gg

    def get_pkmu_gg(self, k, mu, irres=False, Sigma2=None, dSigma2=None, ks=0.2):
        if irres:
            pkmu = self.get_pkmu_gg_irres(k, mu, mode='full', Sigma2=Sigma2, dSigma2=dSigma2, ks=ks)
        else:
            pkmu_tree = self.get_pkmu_gg_lin(k, mu)
            pkmu_1loop = self.get_pkmu_gg_1loop_raw(k, mu, name=name, mode='full')
            pkmu = pkmu_tree + pkmu_1loop
        return pkmu

    def get_pl_gg(self, l, k, irres=False, Sigma2=None, dSigma2=None, ks=0.2):
        k = np.atleast_1d(k)
        mu = np.linspace(0.,1.,2**8+1)
        dmu = mu[1]-mu[0]
        pkmu = self.get_pkmu_gg(k, mu, irres=irres, Sigma2=Sigma2, dSigma2=dSigma2, ks=ks)
        legendre = np.tile(lpmv(0,l,mu), (len(k),1))
        pl = (2*l+1) * romb(pkmu * legendre, axis=1, dx=dmu)
        return pl

    def get_pkmu_gg_lin(self, k, mu):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        b1 = self.bias['b1']
        fgrowth = self.fgrowth
        Z1 = b1 + fgrowth * mu**2

        pk_lin = self.get_pk_lin(k)
        pkmu = np.kron(pk_lin, Z1**2).reshape(len(k),len(mu))

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

    def get_pl_gg_lin_analytic(self, l, k):
        k = np.atleast_1d(k)
        b1 = self.bias['b1']
        fgrowth = self.fgrowth
        beta = fgrowth / b1
        kaiser = [
            1 + 2/3 * beta + 1/5 * beta**2, 
            4/3 * beta + 4/7 * beta**2,
            8/35 * beta**2
            ]
        pk_lin = self.get_pk_lin(k)
        return b1**2 * kaiser[int(l/2)] * pk_lin

    def get_pkmu_13_UV(self, k, mu):
        Z1_g = self.bias['b1'] + self.fgrowth * mu**2

        Z3_g_UV = - 61./315. * self.bias['b1'] - 64./21. * self.bias['bG2'] - 128./105. * self.bias['bGamma3']
        Z3_g_UV += (- 3./5. + 2./105. * self.bias['b1']) * self.fgrowth * mu**2
        Z3_g_UV += (- 16./35. - 1./3. * self.bias['b1']) * self.fgrowth**2 * mu**2
        Z3_g_UV += (- 46./105.) * self.fgrowth**2 * mu**4
        Z3_g_UV += (- 1./3.) * self.fgrowth**3 * mu**4
        
        pk_lin = self.get_pk_lin(k)
        pkmu_13 = np.kron(k**2 * pk_lin * self.plin_int, Z1_g * Z3_g_UV).reshape(len(k),len(mu))
        
        return pkmu_13

    def get_pk_rsd_1loop_data(self, name, sub_k0=True, mode='full'):
        if '22' in name:
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

        if name == 'tot':
            pkmu_22 = self.get_pkmu_gg_1loop_raw(k, mu, name='22_gg', mode=mode)
            pkmu_13 = self.get_pkmu_gg_1loop_raw(k, mu, name='13_gg', mode=mode)
            pkmu = pkmu_22 + pkmu_13
            return pkmu

        term_names = self.name_pkmu_gg_terms[name]
        pkmu_tab = []
        for term in term_names:
            nf, nmu, bias_deg = utils_loop.get_deg_info(term)

            bias_fac = np.prod([self.bias[key]**val for key, val in bias_deg.items()])
            fac = bias_fac * self.fgrowth**nf

            kn, pk_data = self.get_pk_rsd_1loop_data(term, sub_k0=True, mode=mode)

            pkmu = fac * np.kron(pk_data, mu**nmu).reshape(len(kn),len(mu))
            pkmu_tab.append(pkmu)
        pkmu_tab = np.array(pkmu_tab)
        pkmu_data = np.sum(pkmu_tab, axis=0)

        if name == '13_gg':
            fgrowth = self.fgrowth
            b1 = self.bias['b1']
            Z1_g = b1 + fgrowth * mu**2
            Z1_g = np.tile(Z1_g, (len(kn), 1))
            pkmu_data = Z1_g * pkmu_data

            pkmu_UV = self.get_pkmu_13_UV(kn, mu)
            pkmu_data += pkmu_UV

        if len(mu) == 1:
            pkmu_interp = ius(kn, np.ravel(pkmu_data))
            pkmu = pkmu_interp(k)
        else:
            pkmu_interp = rbs(kn, mu, pkmu_data)
            pkmu = pkmu_interp(k, mu)

        return pkmu

    def get_pl_gg_1loop_raw(self, l, k, name='tot', mode='full'):
        k = np.atleast_1d(k)
        mu = np.linspace(0.,1.,2**8+1)
        dmu = mu[1]-mu[0]
        pkmu = self.get_pkmu_gg_1loop_raw(k, mu, name=name, mode=mode)
        legendre = np.tile(lpmv(0,l,mu), (len(k),1))
        pl = (2*l+1) * romb(pkmu * legendre, axis=1, dx=dmu)
        return pl

    def get_pkmu_gg_irres(self, k, mu, mode='full', Sigma2=None, dSigma2=None, ks=0.2):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        plin = self.get_pk_lin(k)
        plin_nw = self.irres.get_pk_nw(k)
        plin_w = plin - plin_nw
        if Sigma2 == None:
            Sigma2 = self.irres.get_Sigma2(ks=ks)
        if dSigma2 == None:
            dSigma2 = self.irres.get_dSigma2(ks=ks)

        b1 = self.bias['b1']
        fgrowth = self.fgrowth

        Sigma2_1 = (1+mu**2*fgrowth*(2+fgrowth)) * Sigma2
        Sigma2_2 = fgrowth**2*mu**2*(mu**2-1) * dSigma2
        Sigma2_tot = Sigma2_1 + Sigma2_2

        Z1 = b1 + fgrowth * mu**2
        Z1_tile = np.tile(Z1, (len(k), 1))
        plin_nw_tile = np.tile(plin_nw, (len(mu),1)).T
        plin_w_tile = np.tile(plin_w, (len(mu),1)).T
        damping = np.kron(k**2, Sigma2_tot).reshape(len(k),len(mu))

        if len(k) == 1 or len(mu) == 1:
            Z1_tile = np.ravel(Z1_tile)
            damping = np.ravel(damping)
            plin_nw_tile = np.ravel(plin_nw_tile)
            plin_w_tile = np.ravel(plin_w_tile)

        if mode == 'LO':
            pkmu = Z1_tile**2 * (plin_nw_tile + np.exp(-damping) * plin_w_tile)
        elif mode == 'tree':
            pkmu = Z1_tile**2 * (plin_nw_tile + np.exp(-damping) * plin_w_tile * (1 + damping))
        elif mode == '1loop no-wiggle':
            pkmu = self.get_pkmu_gg_1loop_raw(k, mu, name='tot', mode='no-wiggle')
        elif mode == '1loop wiggle':
            pkmu = self.get_pkmu_gg_1loop_raw(k, mu, name='tot') - self.get_pkmu_gg_1loop_raw(k, mu, name='tot', mode='no-wiggle')
        elif mode == '1loop':
            pkmu_1loop_nw = self.get_pkmu_gg_irres(k, mu, mode='1loop no-wiggle', Sigma2=Sigma2, dSigma2=dSigma2, ks=ks)
            pkmu_1loop_w = self.get_pkmu_gg_1loop_raw(k, mu, name='tot') - pkmu_1loop_nw
            pkmu = pkmu_1loop_nw + np.exp(-damping) * pkmu_1loop_w
        elif mode == 'full':
            pkmu_tree = self.get_pkmu_gg_irres(k, mu, mode='tree', Sigma2=Sigma2, dSigma2=dSigma2, ks=ks)
            pkmu_1loop_nw = self.get_pkmu_gg_irres(k, mu, mode='1loop no-wiggle', Sigma2=Sigma2, dSigma2=dSigma2, ks=ks)
            pkmu_1loop_w = self.get_pkmu_gg_1loop_raw(k, mu, name='tot') - pkmu_1loop_nw
            pkmu = pkmu_tree + pkmu_1loop_nw + np.exp(-damping) * pkmu_1loop_w

        if len(k) == 1 or len(mu) == 1:
            pkmu = np.ravel(pkmu)
        return pkmu

    def get_pl_gg_irres(self, l, k, mode='full', Sigma2=None, dSigma2=None, ks=0.2):
        k = np.atleast_1d(k)
        mu = np.linspace(0.,1.,2**8+1)
        dmu = mu[1]-mu[0]
        pkmu = self.get_pkmu_gg_irres(k, mu, mode=mode, Sigma2=Sigma2, dSigma2=dSigma2, ks=ks)
        legendre = np.tile(lpmv(0,l,mu), (len(k),1))
        pl = (2*l+1) * romb(pkmu * legendre, axis=1, dx=dmu)
        return pl
