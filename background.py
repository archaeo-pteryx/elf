import numpy as np
from scipy.integrate import quad, romb
from scipy.interpolate import InterpolatedUnivariateSpline as ius

Mpc_in_m = 3.085677581 * 1e+16 * 1e+6 # in unit of m

SpeedofLight = 299792458. # m/s
G = 6.67408 * 1e-11 # m^3 kg^-1 s^-2. based on CODATA value 2014
GMsolar = 1.32712442099 * 1e+20 # in unit of m^3 s^-2. based on ASTRONOMICAL CONSTANTS http://asa.usno.navy.mil/static/files/2018/Astronomical_Constants_2018.pdf
Msolar = GMsolar / G # in unit of kg
G_in_Mpc_Msolar_inv = GMsolar / (SpeedofLight**2) / Mpc_in_m # in unit of Mpc Msolar^-1


class Cosmo:

    def __init__(self, cparam, omega_nu0=0., Omega_K0=0.):
        self.params = {'omega_b0':cparam[0],'omega_c0':cparam[1],'Omega_de0':cparam[2],'ln10p10As':cparam[3],'ns':cparam[4],'w_0':cparam[5]}
        self.params['omega_nu0'] = omega_nu0
        self.params['Omega_K0'] = Omega_K0
        self.params['Omega_m0'] = 1-self.params['Omega_de0']-self.params['Omega_K0']
        self.params['omega_m0'] = self.params['omega_b0'] + self.params['omega_c0'] + self.params['omega_nu0']
        self.params['h'] = np.sqrt(self.params['omega_m0'] / self.params['Omega_m0'])
        self.params['H_0'] = self.params['h'] * 100
        self.params['H_0_in_Mpc_inv'] = self.params['H_0'] / (SpeedofLight * 1e-3) # in unit of Mpc^-1
        self.params['Omega_b0'] = self.params['omega_b0'] / self.params['h']**2
        self.params['Omega_c0'] = self.params['omega_c0'] / self.params['h']**2
        self.params['Omega_nu0'] = self.params['omega_nu0'] / self.params['h']**2
        self.params['As'] = np.exp(self.params['ln10p10As']) * 1e-10
        self.params['rho_c0'] = 3 * self.params['H_0_in_Mpc_inv']**2 / (8*np.pi*G_in_Mpc_Msolar_inv) / self.params['h']**2 # in unit of h^2 Msolar Mpc^-3
        self.params['rho_m0'] = self.params['rho_c0'] * self.params['Omega_m0']; # in unit of h^2 Msolar Mpc^-3

        a = np.hstack((np.geomspace(1e-10,0.5,500), np.linspace(0.5,1,501)[1:]))
        de_dependence = np.exp([3 * quad(lambda t: (1+self.get_w_0(t)) / t, ai, 1)[0] for ai in a])
        self._de_dependence_spl = ius(a, de_dependence)

    @staticmethod
    def get_a(x, mode):
        x = np.atleast_1d(x)
        if mode == 'a': a = x
        elif mode == 'z': a = 1/(1+x)
        else: raise ValueError('The argument must be either a scale factor or a redshift.')
        return a

    def get_w_0(self, x, mode='a'):
        return self.params['w_0']

    def get_E(self, x, mode='a'):
        a = self.get_a(x, mode)
        Omega_m0 = self.params['Omega_m0']
        Omega_de0 = self.params['Omega_de0']
        Omega_K0 = self.params['Omega_K0']
        E = np.sqrt(Omega_m0/a**3 + Omega_K0/a**2 + Omega_de0 * self._de_dependence_spl(a))
        if len(E) == 1: E = E[0]
        return E

    # in unit of km/s/Mpc
    def get_hubble(self, x, mode='a'):
        return self.params['H_0'] * self.get_E(x, mode)

    def get_hubble_in_Mpc_inv(self, x, mode='a'):
        return self.params['H_0_in_Mpc_inv'] * self.get_E(x, mode)

    def get_hubble_comoving(self, x, mode='a'):
        a = self.get_a(x, mode)
        return a * self.get_hubble_in_Mpc_inv(a)

    def get_Omega_m(self, x, mode='a'):
        a = self.get_a(x, mode)
        return self.params['Omega_m0'] / self.get_E(a)**2 / a**3

    def get_Omega_de(self, x, mode='a'):
        a = self.get_a(x, mode)
        Omega_de = self.params['Omega_de0'] / self.get_E(a)**2 * self._de_dependence_spl(a)
        if len(Omega_de) == 1: Omega_de = Omega_de[0]
        return Omega_de

    def get_Omega_K(self, x, mode='a'):
        a = self.get_a(x, mode)
        return 1 - self.get_Omega_m(a) - self.get_Omega_de(a)

    def get_Dgrowth_lcdm(self, x, mode='a'):
        if self.params['w_0'] != -1.:
            raise ValueError('This function is valid only for LambdaCDM cosmology.')
        a = self.get_a(x, mode)
        num = 9
        t = np.linspace(1e-10,1,2**num+1)
        dt = t[1]-t[0]
        integrand = np.kron(self.get_Omega_m(a),1/t) + np.kron(self.get_Omega_de(a),t**2) + np.kron(self.get_Omega_K(a),np.ones(len(t)))
        integrand = integrand.reshape((len(a),len(t)))
        res = romb(integrand**(-3./2), axis=1, dx=dt)
        # res = quad(lambda t: (self.get_Omega_m(a)/t + self.get_Omega_de(a)*t**2 + self.get_Omega_K(a))**(-3./2), 0.,1.)[0]
        Dgrowth = 5./2 * a * self.get_Omega_m(a) * res
        if len(Dgrowth) == 1: Dgrowth = Dgrowth[0]
        return Dgrowth

    def get_fgrowth_lcdm(self, x, mode='a'):
        if self.params['w_0'] != -1.:
            raise ValueError('This function is valid only for LambdaCDM cosmology.')
        a = self.get_a(x, mode)
        num = 9
        t = np.linspace(1e-10,1,2**num+1)
        dt = t[1]-t[0]
        integrand = np.kron(self.get_Omega_m(a),1/t) + np.kron(self.get_Omega_de(a),t**2) + np.kron(self.get_Omega_K(a),np.ones(len(t)))
        integrand = integrand.reshape((len(a),len(t)))
        res = romb(integrand**(-3./2), axis=1, dx=dt)
        fgrowth = -1 - self.get_Omega_m(a)/2 + self.get_Omega_de(a) + 1/res
        if len(fgrowth) == 1: fgrowth = fgrowth[0]
        return fgrowth

    # in unit of Mpc
    def get_comoving_dist(self, x, mode='a'):
        a = self.get_a(x, mode)
        z = 1./a-1
        comoving_dist = np.array([quad(lambda t: 1./self.get_hubble_in_Mpc_inv(t, mode='z'), 0., zi)[0] for zi in z])
        if len(comoving_dist) == 1: comoving_dist = comoving_dist[0]
        return comoving_dist

    # in unit of Mpc/h
    def get_comoving_dist_in_h_inv_Mpc(self, x, mode='a'):
        return self.get_comoving_dist(x, mode) * self.params['h']

    @staticmethod
    def comoving_to_radial(x, K):
        if K == 0: return x
        elif K > 0: return np.sin(np.sqrt(K)*x) / np.sqrt(K)
        elif K < 0: return np.sinh(np.sqrt(-K)*x) / np.sqrt(-K)
        else: raise ValueError('The curvature K is not specified.')

    # in comoving Mpc
    def get_D_angular(self, x, mode='a', K=0):
        return self.comoving_to_radial(self.get_comoving_dist(x, mode), K)

    # in unit of Mpc/h
    def get_D_angular_in_h_inv_Mpc(self, x, mode='a', K=0):
        return self.get_D_angular(x, mode, K) * self.params['h']
