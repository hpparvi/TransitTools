import sys
from numpy import linspace
from numpy.random import normal

from transitmodel.gimenez import  Gimenez
from geometry import Geometry
from transitparameterization import TransitParameterization

class TransitLightcurve(object):
    def __init__(self, parm, model=None, orbit=None, ldpar=None, mode='time', method='fortran'):
        self.parm  = parm 
        self.model = model if model is not None else Gimenez(method=method)
        self.orbit = orbit if orbit is not None else Geometry(mode=mode)
        self.ldpar = ldpar if ldpar is not None else []
        self.orbit.update(self.parm)

    def __call__(self, time, p=None):
        if p is not None: self.update(p)
        return self.model(self.orbit.projected_distance(time), self.orbit.k, self.ldpar)

    def update(self, p):
        self.parm.update(p[:self.parm.npar])
        self.orbit.update(self.parm)
        self.ldpar = p[self.parm.npar:]
        

class TestLightcurve(TransitLightcurve):
    def __init__(self, tc=1., p=4., k=0.1, a=10., b=0., noise=1e-3,  ldpar=None, mode='time', method='fortran', time_lim=[0., 2.], resolution=500):
        parm = TransitParameterization('physical', [k, tc, p, a, b])
        super(TestLightcurve, self).__init__(parm, ldpar=ldpar, mode=mode, method=method)
        self.noise = noise
        self.resolution = resolution
        self.time_lim = time_lim
        self.time = linspace(time_lim[0], time_lim[1], self.resolution)
        
    def __call__(self):
        flux = super(TestLightcurve, self).__call__(self.time)
        flux += normal(0., self.noise, self.resolution)
        return self.time, flux
        
if __name__ == '__main__':
    import pylab as pl
    lc_test = TestLightcurve(noise=2e-4, resolution=2000, ldpar=[0.2], method="python")
    time, flux = lc_test()
    
    pl.plot(time, flux)
    pl.show()
