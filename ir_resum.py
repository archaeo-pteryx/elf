import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as ius
from scipy.integrate import quad
from scipy.special import spherical_jn
from scipy.fft import dst, idst

class IRResum:

    def __init__(self, pk_lin, rbao=110, kmin=7e-5, kmax=7, n_min=175, n_max=275, kwarg={}):
        self.rbao = rbao
        k = np.linspace(kmin, kmax, 2**16)
        plin = pk_lin(k, **kwarg)
        plin_nw = self.remove_wiggle(k, plin, n_min, n_max)
        self.pk_nw_interp = ius(np.log(k), np.log(plin_nw))

    def remove_wiggle(self, k, plin, n_min, n_max):
        harms = dst(np.log(k * plin))

        n = np.arange(1,len(harms)+1)
        i_odd = np.arange(0,len(harms)-1,2)
        i_even = np.arange(1,len(harms),2)

        n_odd = n[i_odd]
        n_even = n[i_even]
        harms_odd = harms[i_odd]
        harms_even = harms[i_even]

        n_odd_sd, harms_odd_sd = self.remove_data(n_odd, harms_odd, n_min, n_max)
        n_even_sd, harms_even_sd = self.remove_data(n_even, harms_even, n_min, n_max)
        harms_odd_s = ius(n_odd_sd, harms_odd_sd)(n_odd)
        harms_even_s = ius(n_even_sd, harms_even_sd)(n_even)

        i_rec = np.argsort(np.hstack((n_odd, n_even)))
        harms_s = np.hstack((harms_odd_s, harms_even_s))[i_rec]
        plin_nw = np.exp(idst(harms_s)) / k

        return plin_nw

    def remove_data(self, n, harms, n_min, n_max):
        n_s = np.hstack((n[n <= n_min], n[n >= n_max]))
        harms_s = np.hstack((harms[n <= n_min], harms[n >= n_max]))
        return n_s, harms_s

    def get_pk_nw(self, k):
        return np.exp(self.pk_nw_interp(np.log(k)))

    def get_Sigma2(self, ks=0.2):
        rbao = self.rbao
        res = quad(lambda q: self.get_pk_nw(q) * (1 - spherical_jn(0,rbao*q) + 2 * spherical_jn(2,rbao*q)), 1e-4, ks, limit=1000)
        return res[0] / (6*np.pi**2)

    def get_dSigma2(self, ks=0.2):
        rbao = self.rbao
        res = quad(lambda q: self.get_pk_nw(q) * spherical_jn(2,rbao*q), 1e-4, ks, limit=1000)
        return res[0] / (2*np.pi**2)

    def get_sigmav2(self):
        res = quad(lambda q: self.get_pk_nw(q), 1e-4, 100, limit=1000)
        return res[0] / (6*np.pi**2)

    def get_Sigma2_rsd(self, fgrowth, mu, ks=0.2):
        Sigma2_1 = (1+mu**2*fgrowth*(2+fgrowth)) * self.get_Sigma2(ks)
        Sigma2_2 = fgrowth**2*mu**2*(mu**2-1) * self.get_dSigma2(ks)
        return Sigma2_1 + Sigma2_2
