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

    def __call__(self, x, pv=None):
        return self._f if self.a < x < self.b else 0.

    def random(self):
        return uniform(self.a, self.b)

class JeffreysPrior(Prior):
    def __init__(self, a, b):
        super(JeffreysPrior, self).__init__(a,b)
        self._f = log(b/a)

    def __call__(self, x, pv=None):
        return 1. / (x*self._f) if self.a < x < self.b else 0.

    def random(self): raise NotImplementedError

class GaussianPrior(Prior):
    def __init__(self, mean, std):
        super(GaussianPrior, self).__init__(mean-10*std, mean+10*std)
        self.mean = float(mean)
        self.std = float(std)
        self._f1 = 1./ sqrt(2.*pi*std*std)
        self._f2 = 1./ (2.*std*std)

    def __call__(self, x, pv=None):
        return self._f1 * exp(-(x-self.mean)**2 * self._f2) if self.a < x < self.b else 0.

    def random(self):
        return normal(self.mean, self.std)
    
class InverseSqrtPrior(Prior):
    def __init__(self, a, b):
        if a < 0 or b < 0:
            print 'Error: bad values for the InverseSqrtPrior.'
            exit()

        a = a if a > 1e-8 else 1e-8
        super(InverseSqrtPrior, self).__init__(a,b)
        self._f = 1/(2*(sqrt(b)-sqrt(a)))

    def __call__(self, x, pv=None):
        return 1/sqrt(x) if self.a < x < self.b else 0.


class B2Prior(Prior):
    def __init__(self, a, b):
        if a < 0 or b < 0:
            print 'Error: bad values for the B2SqrtPrior.'
            exit()

        super(B2Prior, self).__init__(a,b)
        self._f = b**2

    def __call__(self, x, pv=None):
        return 1/(2*x)



class LinCombPrior(Prior):
    def __init__(self, P1, P2, idx1, idx2):
        super(LinCombPrior, self).__init__(-1, 1) # This is a quick hack, fix!
        self.p1, self.i1 = P1, idx1
        self.p2, self.i2 = P2, idx2
        
    def __call__(self, x, pv):
        x1 = 0.5*(pv[self.i1]+pv[self.i2])
        x2 = 0.5*(pv[self.i1]-pv[self.i2])

        return self.p1(x1) * self.p2(x2)


mcmcpriors = {'uniform':UniformPrior,
              'jeffreys':JeffreysPrior,
              'gaussian':GaussianPrior}
