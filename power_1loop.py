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

def window_tophat(k,R):
    return 3 / (k*R)**3 * (np.sin(k*R) - (k*R)*np.cos(k*R))

class PowerSpectrum1loop:

    def __init__(self, config_fft=None):
        self.params = None
        if config_fft == None:
            config_fft = {
                'plin nu=-0.3': {'nu':-0.3, 'kmin':1e-6, 'kmax':1e+4, 'nmax':512},
                'plin nu=-1.6': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':512},
                'p1phi nu=-1.6': {'nu':-1.6, 'kmin':1e-6, 'kmax':1e+4, 'nmax':512},
                'M nu=0.2': {'nu':0.2, 'kmin':1e-6, 'kmax':1e+4, 'nmax':512},
            }
        self.config_fft = config_fft
        self.set_power_law_decomp(config_fft)
        self.mat = {}
        self.matrix = {}
        fnames = glob.glob(os.path.dirname(__file__)+'/pt_matrix/redshift_space/M*_rsd_*.txt')
        self.name_pkmu_gg_terms = [re.split('/', fname)[-1][:-4] for fname in fnames]

    def set_camb(self, cparam, omega_nu0=0.00064, Omega_K0=0., kmin=1e-7, kmax=1e+5):
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
            dirname = 'redshift_space' if 'rsd' in name else 'real_space'
            if '22' in name or 'I' in name:
                matfile = os.path.dirname(__file__)+'/pt_matrix/%s/%s.txt' % (dirname, name)
                self.mat[name] = pt_matrix.PTMatrix22(matfile)
            elif '13' in name or 'F' in name:
                matfile = os.path.dirname(__file__)+'/pt_matrix/%s/%s.txt' % (dirname, name)
                self.mat[name] = pt_matrix.PTMatrix13(matfile)
            else:
                raise KeyError('PT kernel name is invalid.')

    def compute_matrix(self, names=[]):
        # precompute the PT matrices for appropriate FFT settings.
        for name in names:
            name_dec = utils_loop.kernel_to_decomp_dict[name]
            if '22' in name or 'I' in name:
                nu_m1 = -0.5 * self.decomp[name_dec[0]].nu_m
                nu_m2 = -0.5 * self.decomp[name_dec[1]].nu_m
                nu_m1, nu_m2 = np.meshgrid(nu_m1, nu_m2)
                self.matrix[name] = self.mat[name](nu_m1, nu_m2).T
            elif '13' in name or 'F' in name:
                nu_m1 = -0.5 * self.decomp[name_dec[0]].nu_m
                self.matrix[name] = self.mat[name](nu_m1)

    def set_cosmology(self, cparam, z=0, omega_nu0=0.00064, Omega_K0=0., kmin=1e-7, kmax=1e+5, khigh=40., irres=False):
        self.set_camb(cparam, omega_nu0=omega_nu0, Omega_K0=Omega_K0, kmin=kmin, kmax=kmax)
        self.fgrowth = self.cosmo.get_fgrowth_lcdm(z, mode='z')

        # FFTLog-based power-law decomposition
        if irres:
            self.irres = IRResum(self.get_pk_lin, rbao=110, kmin=7e-5, kmax=7, n_min=175, n_max=275, kwarg={'z':z})
            Sigma2_ref = self.irres.get_Sigma2(ks=0.2)
            self.decomp['plin nu=-0.3'].compute(self.get_pk_mm_irres, kwarg={'Sigma2': Sigma2_ref})
            self.decomp['plin nu=-1.6'].compute(self.get_pk_mm_irres, kwarg={'Sigma2': Sigma2_ref})
            self.decomp['p1phi nu=-1.6'].compute(self.get_pk_1phi, kwarg={'z':z, 'khigh':khigh})
            self.decomp['M nu=0.2'].compute(self.get_M, kwarg={'z':z})
        else:
            self.decomp['plin nu=-0.3'].compute(self.get_pk_lin, kwarg={'z':z, 'khigh':khigh})
            self.decomp['plin nu=-1.6'].compute(self.get_pk_lin, kwarg={'z':z, 'khigh':khigh})
            self.decomp['p1phi nu=-1.6'].compute(self.get_pk_1phi, kwarg={'z':z, 'khigh':khigh})
            self.decomp['M nu=0.2'].compute(self.get_M, kwarg={'z':z})

        # IR and UV parts of P13 and P22
        plin_int = self.get_pk_lin_int(alpha=0, khigh=khigh, kmin=0, kmax=np.inf, limit=1000)
        plin_int2 = self.get_pk_lin_int(alpha=-2, khigh=khigh, kmin=0, kmax=np.inf, limit=1000)
        self.alpha = {}
        self.alpha['UV13'] = -(122/315) * plin_int
        self.alpha['IR13'] = -(2/3) * plin_int
        self.alpha['UV22'] = (9/49) * plin_int2
        self.alpha['IR22'] = (2/3) * plin_int

    def set_bias(self, bias={}):
        self.bias = bias

    def get_pk_prim(self, kh):
        # primordial power spectrum of the curvature perturbations in unit of [Mpc^3]
        As = self.params['As']
        ns = self.params['ns']
        k_pivot = self.params['k_pivot']
        pk_prim = As * (kh/k_pivot)**(ns-1)
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

    def get_pkmu_mm_irres(self, k, mu, mode='LO', Sigma2=None, dSigma2=None, ks=0.2):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        plin = self.get_pk_lin(k)
        plin_nw = self.irres.get_pk_nw(k)
        plin_w = plin - plin_nw
        if Sigma2 == None:
            Sigma2 = self.irres.get_Sigma2(ks=ks)

        fgrowth = self.fgrowth
        Sigma2_1 = (1+mu**2*fgrowth*(2+fgrowth)) * Sigma2
        Sigma2_2 = fgrowth**2*mu**2*(mu**2-1) * dSigma2
        Sigma2_tot = Sigma2_1 + Sigma2_2

        if mode == 'LO':
            kaiser = (b1 + fgrowth * mu**2)**2
            kaiser_tile = np.tile(kaiser, (len(k), 1))
            plin_nw_tile = np.tile(plin_nw, (len(mu),1)).T
            plin_w_tile = np.tile(plin_w, (len(mu),1)).T
            damp_fac = np.exp(-np.kron(k**2, Sigma2_tot)).reshape(len(k),len(mu))
            pkmu = kaiser_tile * (plin_nw_tile + damp_fac * plin_w_tile)

        if len(k) == 1 or len(mu) == 1:
            pkmu = np.ravel(pkmu)
        return pkmu

    def get_M(self, k, z=0):
        h = self.params['h']
        Omega_m0 = self.params['Omega_m0']
        H_0 = self.params['H_0_in_Mpc_inv']
        Tk = self.get_matter_transfer(k*h) # normalized as T(k) -> 1 (k -> 0)
        Dgrowth = self.cosmo.get_Dgrowth_lcdm(z, mode='z') # normalized as D(a) -> a (a -> 0)
        return 2./3 * (k*h)**2 * (Tk * Dgrowth) / (Omega_m0 * H_0**2)

    def get_pk_1phi(self, k, z=0, khigh=None):
        plin = self.get_pk_lin(k, z=z, khigh=khigh)
        M = self.get_M(k, z=z)
        return plin / M

    def get_pk_lin_png_local(self, k, z=0, f_nl=0, b1=1, b_phi=0):
        M = self.get_M(k,z)
        factor = b1**2 + 2*b1*b_phi*f_nl/M + (b_phi*f_nl/M)**2
        plin = self.get_pk_lin(k,z)
        return factor * plin

    def get_pk_lin_int(self, alpha=0, khigh=40., kmin=0., kmax=np.inf, limit=1000):
        res = quad(lambda k: self.get_pk_lin(k, khigh=khigh) * k**alpha, kmin, kmax, limit=limit)
        return res[0] / (2*np.pi)**2

    def get_pk_1loop_data(self, name='22', sub_k0=False):
        name_dec = utils_loop.kernel_to_decomp_dict[name]

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
            p13_int = np.dot(self.matrix[name], p1_q).real
            if name == '13':
                p13_int += self.alpha['UV13'] / kn
            if name == 'F_G2_LPNG':
                pk_data = kn**3 * self.get_pk_lin(kn) / self.get_M(kn) * p13_int
            else:
                pk_data = kn**3 * self.get_pk_lin(kn) * p13_int

        else:
            raise KeyError('PT kernel name is invalid.')

        return kn, pk_data

    def get_pk_1loop(self, k, name='22', sub_k0=False):
        alpha = 1.5
        kn, pk_data = self.get_pk_1loop_data(name=name, sub_k0=sub_k0)
        pk_interp = ius(kn, kn**alpha * pk_data)
        pk = pk_interp(k) * k**(-alpha)
        # if z == 0.: Dgrowth = 1
        # else:
        #     Dgrowth = self.cosmo.get_Dgrowth_lcdm(np.array([0,z]), mode='z')
        #     Dgrowth = Dgrowth[1] / Dgrowth[0]
        # pk = Dgrowth**4 * pk
        return pk

    def get_pk_rsd_1loop_data(self, name):
        if '22' in name or 'I' in name:
            kn = self.decomp['plin nu=-1.6'].kn
            p1_q = self.decomp['plin nu=-1.6'].func_q
            p2_q = self.decomp['plin nu=-1.6'].func_q
            pk_data = kn**3 * np.diag(np.dot(p1_q.T, np.dot(self.matrix[name], p2_q)).real)
        elif '13' in name or 'F' in name:
            kn = self.decomp['plin nu=-1.6'].kn
            p1_q = self.decomp['plin nu=-1.6'].func_q
            p13_int = np.dot(self.matrix[name], p1_q).real
            if name == '13':
                p13_int += self.alpha['UV13'] / kn
            if name == 'F_G2_LPNG':
                pk_data = kn**3 * self.get_pk_lin(kn) / self.get_M(kn) * p13_int
            else:
                pk_data = kn**3 * self.get_pk_lin(kn) * p13_int
        else:
            raise KeyError('PT kernel name is invalid.')
        return kn, pk_data

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

    def get_pkmu_gg_1loop(self, k, mu):
        k = np.atleast_1d(k)
        mu = np.atleast_1d(mu)

        names = self.name_pkmu_gg_terms
        pkmu_tab = []
        for name in names:
            nf, nmu, bias_deg = utils_loop.get_deg_info(name)

            bias_fac = np.prod([self.bias[key]**val for key, val in bias_deg.items()])
            fac = bias_fac * self.fgrowth**nf
            kn, pk_data = self.get_pk_rsd_1loop_data(name)

            pkmu = fac * np.kron(pk_data, mu**nmu).reshape(len(kn),len(mu))
            pkmu_tab.append(pkmu)
        pkmu_tab = np.array(pkmu_tab)

        pkmu_data = np.sum(pkmu_tab, axis=0)
        pkmu_interp = rbs(kn, mu, pkmu_data)
        pkmu_data = pkmu_interp(k,mu)
        return pkmu
