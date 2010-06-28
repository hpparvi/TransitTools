import sys
import numpy   as np
from Gimenez import  Gimenez
from Orbit import CircularOrbit
from TransitParameterization import PhysicalTransitParameterization as Physical

class TransitLightCurve(object):
    def __init__(self, parm, model=None, orbit=None, ldpar=None, mode='time'):
        self.parm  = parm 
        self.model = model if model is not None else Gimenez(method='fortran')
        self.orbit = orbit if orbit is not None else CircularOrbit(mode=mode)
        self.ldpar = ldpar if ldpar is not None else []
        self.orbit.update(self.parm)

    def __call__(self, time, p=None):
        if p is not None: self.update(p)
        return self.model(self.orbit.projected_distance(time), self.orbit.p[2], self.ldpar)

    def update(self, p):
        self.parm.update(p[:self.parm.npar])
        self.orbit.update(self.parm)
        self.ldpar = p[self.parm.npar:]
        

class TestLightCurve(TransitLightCurve):
    def __init__(self, tc=1., P=4., p=0.1, a=10., b=0., noise=1e-3,  ldpar=None, mode='time', time_lim=[0., 6.], resolution=500):
        parm = Physical([tc, P, p, a, b])
        super(TestLightCurve, self).__init__(parm, ldpar=ldpar, mode=mode)
        self.noise = noise
        self.resolution = resolution
        self.time_lim = time_lim
        self.time = np.linspace(time_lim[0], time_lim[1], self.resolution)
        
    def __call__(self):
        flux = super(TestLightCurve, self).__call__(self.time)
        flux += np.random.normal(0., self.noise, self.resolution)
        return self.time, flux
        
if __name__ == '__main__':
    import pylab as pl
    lc_test = TestLightCurve(noise=2e-4, resolution=2000, ldpar=[0.2])
    time, flux = lc_test()
    
    pl.plot(time, flux)
    pl.show()
