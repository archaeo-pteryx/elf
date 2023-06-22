import numpy as np
from copy import deepcopy
from scipy.integrate import romb
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.interpolate import RectBivariateSpline as rbs
import camb
from background import Cosmo

class CambWrapper(Cosmo):

    def __init__(self, cparam, omega_nu0=0.00064, Omega_K0=0, k_pivot=0.05):
        super(CambWrapper, self).__init__(cparam, omega_nu0, Omega_K0)
        self.pars = camb.CAMBparams()
        mnu = self.params['omega_nu0'] * 94.12
        self.params['k_pivot'] = k_pivot
        self.pars.set_cosmology(H0=self.params['H_0'], ombh2=self.params['omega_b0'], omch2=self.params['omega_c0'], omk=self.params['Omega_K0'], num_massive_neutrinos=1, mnu=mnu, nnu=3.046, YHe=0.24, TCMB=2.7255, tau=0.079)
        self.pars.set_dark_energy(w=self.params['w_de'])
        self.pars.InitPower.set_params(As=self.params['As'], ns=self.params['ns'], r=0, pivot_scalar=self.params['k_pivot'])

    def set_matter_power(self, z=0, kmax=1e+3):
        self.pars.set_matter_power(redshifts=[z], kmax=kmax)
        self.pars.NonLinear = camb.model.NonLinear_none
        self.pars.DoLensing = False
        self.results = camb.get_results(self.pars)
        self.params['sigma8'] = self.results.get_sigma8()[0]

    def get_rdrag(self):
        return self.results.get_derived_params()['rdrag'] * self.params['h']

    def get_matter_transfer_data(self, name='tot'):
        trans = self.results.get_matter_transfer_data()
        k = trans.q

        # see https://camb.readthedocs.io/en/latest/transfer_variables.html
        if name == 'tot':
            tk = trans.transfer_data[camb.model.Transfer_tot-1,:,0] # total transfer
        elif name == 'cb':
            tk = trans.transfer_data[camb.model.Transfer_nonu-1,:,0] # CDM+baryon transfer

        return k, tk

    def get_matter_power_data(self, minkh=2e-5, maxkh=10., npoints=400):
        kh, z, pk = self.results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=npoints)
        return kh, pk[0]

    def get_matter_power(self, k):
        k = np.atleast_1d(k)
        kh, z, pk = self.results.get_matter_power_spectrum(minkh=2e-5, maxkh=10, npoints=400)
        pk_spl = ius(np.log(kh), np.log(pk[0]))
        return np.exp(pk_spl(np.log(k)))

    def convert_sigma8_to_As(self, sigma8):
        return self.params['As'] * (sigma8 / self.params['sigma8'])**2
        