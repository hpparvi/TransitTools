import sys
import numpy as np
import pylab as pl

from TransitParameterization import PhysicalTransitParameterization as tpp
from TransitParameterization import KippingTransitParameterization as tpk

from TransitLightCurve import TransitLightCurve
from diffeval import de
from base import fold#, bin
from futils import bin

class Fitter(object):
    def __init__(self): 
        pass
        
    def __call__(self, npop=None, ngen=None, F=None, C=None):
        np = self.n_population  if npop is None else npop
        ng = self.n_generations if ngen is None else ngen
        C  = self.C if C is None else C
        F  = self.F if F is None else F
        
        self.X, self.p_fit = de(self.minfun, self.bnds, np, ng, F, C)
        
        return self.X, self.p_fit.copy()

class TTVFitter(Fitter):
    def __init__(self, time, flux, parm, mean_std=None, ldbnd=None, phase_lim=[-0.5, 0.5]):
        self._time = time
        self._flux = flux
        self.parm  = parm
        self.ldbnd = np.array(ldbnd)
        self.bnds  = np.array([parm.p_low, parm.p_high]).transpose()
        self.phase_lim = phase_lim

        self.t_center = 0.5 * (parm.p_low[0] + parm.p_high[0])
        self.p_mean   = 0.5*(parm.p_low[1]+parm.p_high[1])

        ## --- Transit sets ---
        ##
        self.nsets = 20
        self.tfrac = 0.05

        self.timesets  = []
        self.fluxsets  = []
        self.phasesets = []
        
        tmask = np.zeros(time.size, np.bool)
        halfp = 0.2 * 0.5*self.p_mean
        
        nt = int((time[-1] - time[0]) / self.p_mean)
        ft = int(self.tfrac * nt)
        trsel = np.zeros(nt)
        
        print "Creating %i transit sets with %i transits each" %(self.nsets, ft)
        for iset in range(self.nsets):
            print "Set %i" %(iset+1)
            trset = np.random.permutation(nt)[:ft]
            trsel[trset] += 1
            
            tmask[:] = False
            for tr in trset:
                tt = self.t_center+tr*self.p_mean
                tmask = np.logical_or(tmask, np.logical_and(time > tt - halfp, time < tt + halfp))
  
            self.timesets.append(time[tmask].copy())
            self.fluxsets.append(flux[tmask].copy())

        self.mean_std = 1e-4 if mean_std is None else mean_std
        self.mean_inv_var = 1./self.mean_std**2
        
        phase = fold(self._time, self.p_mean, self.t_center, 0.5) - 0.5
        pmask = np.logical_and(phase>phase_lim[0], phase<phase_lim[1])
        self.time = self._time[pmask]
        self.flux = self._flux[pmask]
        self.dt = self.time-self.time[0]
        
        if ldbnd is not None:
            self.bnds = np.concatenate((self.bnds, self.ldbnd))

        self.bnds = np.concatenate((self.bnds, [[-1e-3, 1e-3]]))

        self.tlc   = TransitLightCurve(parm, ldpar=np.zeros(self.ldbnd.shape[0]), mode='time')

        self.n_population = 50
        self.n_generations = 50
        self.F = 0.5
        self.C = 0.5
    
        self.setid = 0

    def increase_setid(self):
        self.setid = (self.setid + 1) % self.nsets

    def minfun(self, p):
        #X = ((self.fluxsets[self.setid]-self.tlc(self.timesets[self.setid]+self.dt*p[-1], p[:-1]))**2 * self.mean_inv_var).sum() / (self.timesets[self.setid].size - self.bnds.shape[0])
        X = ((self.fluxsets[self.setid]-self.tlc(self.timesets[self.setid], p[:-1]))**2).sum()
        self.increase_setid()
        return X

    def minfun_t(self, p):
        return ((self.flux-self.tlc(self.time+self.dt*p[-1], p[:-1]))**2 * self.mean_inv_var).sum() / (self.time.size - self.bnds.shape[0])

    def get_fitted_data(self, binning=False, normalize_phase=True):
        phase = fold(self._time, self.p_fit[1], self.p_fit[0], 0.5) - 0.5
        pm = np.logical_and(phase>self.phase_lim[0], phase<self.phase_lim[1])
        phase = 2.*np.pi*phase[pm]
        if normalize_phase:
            phase /= 2.*np.pi
        if binning:
            pb, fb, fe = bin(phase, self._flux[pm], 100)
            return pb, fb
        else:
            return phase, self.flux

    def get_fitted_lc(self, resolution=500, normalize_phase=True):
        phase = 2.*np.pi*np.linspace(self.phase_lim[0], self.phase_lim[1], resolution)
        tlc   = TransitLightCurve(self.parm, ldpar=np.zeros(self.ldbnd.shape[0]), mode='phase')
        flux = tlc(phase, self.p_fit)
        
        if normalize_phase:
            phase /= 2.*np.pi
        return phase, flux

class DiffEvolFitter(Fitter):
    def __init__(self, time, flux, parm, mean_std=None, ldbnd=None, binning=None, phase_lim=[-0.5, 0.5], oversample=False):
        self._time = time
        self._flux = flux
        self.parm  = parm
        self.ldbnd = np.array(ldbnd)
        self.bnds  = np.array([parm.p_low, parm.p_high]).transpose()
        self.phase_lim = phase_lim

        self.mean_std = 1e-4 if mean_std is None else mean_std
        self.mean_inv_var = 1./self.mean_std**2
        
        ## --- Limb darkening parameters ---
        ## If limb darkening parameter bounds are definend, they 
        ## are concatenated to the parameter boundary vector.
        ##
        if ldbnd is not None:
            self.bnds = np.concatenate((self.bnds, self.ldbnd))

        ## -- Transit center fitting ---
        ##
        if np.abs(parm.p_high[0]-parm.p_low[0]) < 1e-12:
            self.t_center = parm.p_low[0]
            self.fit_center = False
        else:
            self.t_center = 0.5 * (parm.p_low[0] + parm.p_high[0])
            self.fit_center = True
            
        ## --- Dynamic folding and binning --- 
        ## If we are fitting the period, the data must be refolded
        ## and rebinned for each fitting generation.
        ##
        if np.abs(parm.p_high[1]-parm.p_low[1]) < 1e-12:
            self.dynamic_binning = False
            phase = fold(self._time, parm.p_low[1], self.t_center, 0.5) - 0.5
            pm = np.logical_and(phase>self.phase_lim[0], phase<self.phase_lim[1])
            self.time_f = self._time[pm]
            self.phase_f = 2.*np.pi * phase[pm]
            self.flux_f = self._flux[pm]
        else:
            #raise NotImplementedError
            self.dynamic_binning = True
            self.time_f = self._time
            self.flux_f = self._flux
            self.nbins = binning['nbins']


        self.tlc   = TransitLightCurve(parm, ldpar=np.zeros(self.ldbnd.shape[0]), mode='phase')

        self.n_population = 50
        self.n_generations = 50
        self.F = 0.5
        self.C = 0.5
        
        if binning is not None and not self.dynamic_binning:
            #self.b_min = binning['min']
            #self.b_max = binning['max']
            if 'nbins' in binning.keys():
                self.nbins = binning['nbins']
                #self.wbins = (self.b_max - self.b_min) / float(self.nbins)
                self._pb, self._fb, self._ferr = bin(self.phase_f, self.flux_f, 
                                                     self.nbins) #, 
                                                     #lim=[self.b_min, self.b_max])
            else:
                self.wbins = binning['wbins']
                self.nbins = (self.b_max - self.b_min) / float(self.wbins)
                self._pb, self._fb, self._ferr = bin(self.phase, self.flux, 
                                                     bw=self.nbins, 
                                                     lim=[self.b_min, self.b_max])
            self.phase_f = self._pb[self._fb == self._fb]
            self.flux_f  = self._fb[self._fb == self._fb]

        if self.dynamic_binning:
            self.minfun = self.minfun_dynamic_binning
        else:
            self.minfun = self.minfun_basic


    def minfun_dynamic_binning(self, p):
        self.phase_f = fold(self._time, p[1], p[0], 0.5) - 0.5
        self._pb, self._fb, self._ferr = bin(self.phase_f, self._flux, 
                                                     self.nbins)

        self.phase_f = self._pb[self._fb == self._fb]
        self.flux_f  = self._fb[self._fb == self._fb]
        
        return ((self.flux_f-self.tlc(self.phase_f, p))**2 * self.mean_inv_var).sum() / (self.phase_f.size - self.bnds.shape[0])

    def minfun_basic(self, p):
        phase_offset = 2.*np.pi*(self.t_center-p[0])/p[1]
        return ((self.flux_f-self.tlc(self.phase_f+phase_offset, p))**2 * self.mean_inv_var).sum() / (self.phase_f.size - self.bnds.shape[0])

    def minfun_oversampled(self, p): 
        ph_os = 2.*np.pi*np.linspace(0.45, 0.55, 200)
        lc_fit = transitlc(ph_os, p)
        pfb, ffb, ffe = bin(ph_os, lc_fit, bn=pb.size, lim=[2*np.pi*0.45, 2*np.pi*0.55])
        return ((fb-pfb)**2).sum()

    def get_fitted_data(self, binning=False, normalize_phase=True):
        phase = fold(self._time, self.p_fit[1], self.p_fit[0], 0.5) - 0.5
        pm = np.logical_and(phase>self.phase_lim[0], phase<self.phase_lim[1])
        phase = 2.*np.pi*phase[pm]
        if normalize_phase:
            phase /= 2.*np.pi
        if binning:
            pb, fb, fe = bin(phase, self._flux[pm], self.nbins)
            return pb, fb
        else:
            return phase, self.flux

    def get_fitted_lc(self, resolution=500, normalize_phase=True):
        phase = 2.*np.pi*np.linspace(self.phase_lim[0], self.phase_lim[1], resolution)
        flux = self.tlc(phase, self.p_fit)
        
        if normalize_phase:
            phase /= 2.*np.pi
        return phase, flux


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
