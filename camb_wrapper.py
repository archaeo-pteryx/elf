import numpy as np
import camb
from background import Cosmo


class CambWrapper(Cosmo):

    def __init__(self, cparam, omega_nu0=0., Omega_K0=0., k_pivot=0.05):
        super(CambWrapper, self).__init__(cparam, omega_nu0, Omega_K0)
        self.pars = camb.CAMBparams()
        mnu = self.params['omega_nu0'] * 94.12
        self.params['k_pivot'] = k_pivot
        self.pars.set_cosmology(H0=self.params['H_0'], ombh2=self.params['omega_b0'], omch2=self.params['omega_c0'], omk=self.params['Omega_K0'], num_massive_neutrinos=1, mnu=mnu, nnu=3.046, YHe=0.24, TCMB=2.7255, tau=0.079)
        self.pars.set_dark_energy(w=self.params['w_0'])
        self.pars.InitPower.set_params(As=self.params['As'], ns=self.params['ns'], r=0, pivot_scalar=self.params['k_pivot'])

    def set_matter_power(self, z=0, kmax=100.):
        self.pars.set_matter_power(redshifts=[z], kmax=kmax)
        self.pars.NonLinear = camb.model.NonLinear_none
        self.pars.DoLensing = False
        self.results = camb.get_results(self.pars)
        self.params['sigma8'] = self.results.get_sigma8()[0]

    def get_matter_transfer_data(self, name='tot'):
        trans = self.results.get_matter_transfer_data()
        k = trans.q

        # see https://camb.readthedocs.io/en/latest/transfer_variables.html
        if name == 'tot':
            tk = trans.transfer_data[camb.model.Transfer_tot-1,:,0] # total transfer
        elif name == 'cb':
            tk = trans.transfer_data[camb.model.Transfer_nonu-1,:,0] # CDM+baryon transfer

        return k, tk
