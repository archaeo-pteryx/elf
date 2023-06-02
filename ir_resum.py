import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import quad
from scipy.special import spherical_jn
from scipy.fft import dst, idst
from utils_loop import get_log_extrap

class IRResum:

    def __init__(self, pk_lin, hubble, rbao=110, khmin=7e-5, khmax=7, n_min=120, n_max=240,
                kmin_interp=1e-7, kmax_interp=1e+4, kwarg={}):
        self.rbao = rbao
        kh = np.linspace(khmin, khmax, 2**16)
        plin = pk_lin(kh/hubble, **kwarg) * hubble**(-3)
        plin_nw = self.remove_wiggle(kh, plin, n_min, n_max)
        k_extrap, plin_nw_extrap = get_log_extrap(kh / hubble, plin_nw * hubble**3, kmin_interp, kmax_interp)
        self.pk_nw_interp = ius(np.log(k_extrap), np.log(plin_nw_extrap))

    def remove_wiggle(self, kh, plin, n_min, n_max):
        harms = dst(np.log(kh * plin))

        n = np.arange(1,len(harms)+1)
        i_odd = np.arange(0,len(harms)-1,2)
        i_even = np.arange(1,len(harms),2)

        n_odd = n[i_odd]
        n_even = n[i_even]
        harms_odd = harms[i_odd]
        harms_even = harms[i_even]

        n = n[:int(len(harms)/2)]
        n_sd = np.hstack((n[n <= n_min], n[n >= n_max]))
        harms_odd_sd = np.hstack((harms_odd[n <= n_min], harms_odd[n >= n_max]))
        harms_even_sd = np.hstack((harms_even[n <= n_min], harms_even[n >= n_max]))

        harms_odd_s = ius(n_sd, harms_odd_sd)(n)
        harms_even_s = ius(n_sd, harms_even_sd)(n)

        i_rec = np.argsort(np.hstack((n_odd, n_even)))
        harms_s = np.hstack((harms_odd_s, harms_even_s))[i_rec]
        plin_nw = np.exp(idst(harms_s)) / kh

        return plin_nw

    def get_pk_nw(self, k):
        return np.exp(self.pk_nw_interp(np.log(k)))

    def get_Sigma2(self, ks=0.2):
        res = quad(lambda q: self.get_pk_nw(q) * (1 - spherical_jn(0,self.rbao*q) + 2 * spherical_jn(2,self.rbao*q)), 1e-4, ks, limit=1000)
        return res[0] / (6*np.pi**2)

    def get_dSigma2(self, ks=0.2):
        res = quad(lambda q: self.get_pk_nw(q) * spherical_jn(2,self.rbao*q), 1e-4, ks, limit=1000)
        return res[0] / (2*np.pi**2)

    def get_sigmav2(self):
        res = quad(lambda q: self.get_pk_nw(q), 1e-4, 1e+2, limit=1000)
        return res[0] / (6*np.pi**2)

    def get_Sigma2_rsd(self, fgrowth, mu, ks=0.2):
        Sigma2_1 = (1+mu**2*fgrowth*(2+fgrowth)) * self.get_Sigma2(ks)
        Sigma2_2 = fgrowth**2*mu**2*(mu**2-1) * self.get_dSigma2(ks)
        return Sigma2_1 + Sigma2_2
