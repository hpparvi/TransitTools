import sys
import numpy as np
import pylab as pl

from TransitParameterization import PhysicalTransitParameterization as tpp
from TransitParameterization import KippingTransitParameterization as tpk

from TransitLightCurve import TransitLightCurve
from diffeval import de
from base import fold, bin

class Fitter(object): pass

class DiffEvolFitter(Fitter):
    def __init__(self, time, flux, parm, mean_std=None, ldbnd=None, binning=None, oversample=False):
        self.time  = time
        self.flux  = flux
        self.parm  = parm
        self.ldbnd = np.array(ldbnd)
        self.bnds  = np.array([parm.p_low, parm.p_high]).transpose()

        self.mean_std = 1e-4 if mean_std is None else mean_std
        self.mean_inv_var = 1./self.mean_std**2
        
        if ldbnd is not None:
            self.bnds = np.concatenate((self.bnds, self.ldbnd))

        self.tlc   = TransitLightCurve(parm, ldpar=np.zeros(self.ldbnd.shape[0]), mode='phase')

        _p, _m = fold(self.time, 4, 1., clip_range=[0.45,  0.55])
        _p *= 2*np.pi
        
        self.phase_f = _p
        self.flux_f = self.flux[_m]
        
        self.n_population = 50
        self.n_generations = 50
        self.F = 0.5
        self.C = 0.5
        
        if binning is not None:
            self.b_min = binning['min']
            self.b_max = binning['max']
            if 'nbins' in binning.keys():
                self.nbins = binning['nbins']
                self.wbins = (self.b_max - self.b_min) / float(self.nbins)
                self._pb, self._fb, self._ferr = bin(self.phase, self.flux, 
                                                     bn=self.nbins, 
                                                     lim=[self.b_min, self.b_max])
            else:
                self.wbins = binning['wbins']
                self.nbins = (self.b_max - self.b_min) / float(self.wbins)
                self._pb, self._fb, self._ferr = bin(self.phase, self.flux, 
                                                     bw=self.nbins, 
                                                     lim=[self.b_min, self.b_max])
            self.phase_f = self._pb
            self.flux_f  = self._fb

    def minfun(self, p):
        return ((self.flux_f-self.tlc(self.phase_f, p))**2 * self.mean_inv_var).sum() / (self.phase_f.size - self.bnds.shape[0])

    def minfun_oversampled(p): 
        ph_os = 2.*np.pi*np.linspace(0.45, 0.55, 200)
        lc_fit = transitlc(ph_os, p)
        pfb, ffb, ffe = bin(ph_os, lc_fit, bn=pb.size, lim=[2*np.pi*0.45, 2*np.pi*0.55])
        return ((fb-pfb)**2).sum()

    def __call__(self, npop=None, ngen=None, F=None, C=None):
        np = self.n_population  if npop is None else npop
        ng = self.n_generations if ngen is None else ngen
        C  = self.C if C is None else C
        F  = self.F if F is None else F
        
        self.X, self.p_fit = de(self.minfun, self.bnds, np, ng, F, C)
        
        return self.X, self.p_fit

def main():
    
    def t(arr):
        s = ''
        for a in arr:
            s += '%8.2f' %a
        return s
    
    np.set_string_function(t, repr=False)
    
    def print_p_difference(p1, p2):
        print '\n%8s%s'%('Original', str(p1))
        print '%8s%s'%('Fit', str(p2))
        print 10*' '+ (len(str(p2))-2)*'-'
        print '%8s%s\n'%('Error', str(1e2*(p2-p1)/p1))
    
    time = np.linspace(0., 5., 3000)
    phase, mask = fold(time, 4., 1., clip_range=[0.45,  0.55])
    phase *= 2*np.pi

    p_original = np.array([1., 4., 0.10, 5.0, 0.8, 0.25])
    p_low      = np.array([1., 4., 0.05, 4.5, 0.5])
    p_high     = np.array([1., 4., 0.15, 5.5, 0.9])
    p_bounds   = np.array([p_low, p_high]).transpose()
    
    parameterization_p = tpp(p_original[:5], p_low[:5], p_high[:5], eccentric=False)
    parameterization = parameterization_p
    parameterization = tpk(parameterization_p)
    
    transitlc = TransitLightCurve(parameterization, ldpar=p_original[5:], mode='time')
    lc_original = transitlc(time)
    lc_observed = lc_original + np.random.normal(0.0,  2e-4, lc_original.size)
    
    fit = DiffEvolFitter(time, lc_observed, parameterization, mean_std=2e-4, ldbnd=[[0.2, 0.3]])
    X, p_fit = fit(npop=60, ngen=250, C=0.8)

    lc_fit = TransitLightCurve(parameterization, ldpar=p_original[5:], mode='phase')(phase, p_fit)
        
    p_fit[:5] = parameterization_p.mapped_from_orbit_c(parameterization.mapped_to_orbit(p_fit[:5]))
    print_p_difference(p_original, p_fit)
    
    pl.plot(phase, lc_original[mask], c='0.5')
    pl.plot(phase, lc_observed[mask], '.', c='0')
    pl.plot(phase, lc_fit, c='0')
    pl.show()

if __name__ == '__main__':
    main()
