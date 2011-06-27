from math import exp, log, sqrt, pi

class Prior(object):
    def __init__(self, a, b):
        self.a = float(a)
        self.b = float(b)
        self.width = b - a


class UniformPrior(Prior):
    def __init__(self, a, b):
        super(UniformPrior, self).__init__(a,b)
        self._f = 1. / self.width

    def __call__(self, x):
        return self._f if self.a < x < self.b else 0.


class JeffreysPrior(Prior):
    def __init__(self, a, b):
        super(JeffreysPrior, self).__init__(a,b)
        self._f = log(b/a)

    def __call__(self, x):
        return 1. / (x*self._f) if self.a < x < self.b else 0.
        

class GaussianPrior(Prior):
    def __init__(self, mean, std):
        super(GaussianPrior, self).__init__(mean-10*std, mean+10*std)
        self.mean = float(mean)
        self.std = float(std)
        self._f1 = 1./ sqrt(2.*pi*std*std)
        self._f2 = 1./ (2.*std*std)

    def __call__(self, x):
        return self._f1 * exp(-(x-self.mean)**2 * self._f2) if self.a < x < self.b else 0.
        
mcmcpriors = {'uniform':UniformPrior,
              'jeffreys':JeffreysPrior,
              'gaussian':GaussianPrior}
