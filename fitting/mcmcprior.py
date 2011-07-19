from __future__ import division

from math import exp, log, sqrt, pi
from numpy.random import normal, uniform

class Prior(object):
    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        self.width = b - a

    def limits(self): return self.a, self.b 

    def min(self): return self.a

    def max(self): return self.b
    

class UniformPrior(Prior):
    def __init__(self, a, b):
        super(UniformPrior, self).__init__(a,b)
        self._f = 1. / self.width

    def __call__(self, x):
        return self._f if self.a < x < self.b else 0.

    def random(self):
        return uniform(self.a, self.b)

class JeffreysPrior(Prior):
    def __init__(self, a, b):
        super(JeffreysPrior, self).__init__(a,b)
        self._f = log(b/a)

    def __call__(self, x):
        return 1. / (x*self._f) if self.a < x < self.b else 0.

    def random(self): raise NotImplementedError

class GaussianPrior(Prior):
    def __init__(self, mean, std):
        super(GaussianPrior, self).__init__(mean-10*std, mean+10*std)
        self.mean = float(mean)
        self.std = float(std)
        self._f1 = 1./ sqrt(2.*pi*std*std)
        self._f2 = 1./ (2.*std*std)

    def __call__(self, x):
        return self._f1 * exp(-(x-self.mean)**2 * self._f2) if self.a < x < self.b else 0.

    def random(self):
        return normal(self.mean, self.std)
    
class InverseSqrtPrior(Prior):
    def __init__(self, a, b):
        if a < 0 or b < 0:
            print 'Error: bad values for the InverseSqrtPrior.'
            exit()

        a = a if a >1e-8 else 1e-8
        super(InverseSqrtPrior, self).__init__(a,b)
        self._f = 1/(2*(sqrt(b)-sqrt(a)))

    def __call__(self, x):
        return 1/sqrt(x)

mcmcpriors = {'uniform':UniformPrior,
              'jeffreys':JeffreysPrior,
              'gaussian':GaussianPrior}
