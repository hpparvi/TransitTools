from transitLightCurve.core import *
from transitfitter import Fitter
from fitnessfunction import FitnessFunction
from ptgibbsmcmc import PTMCMC
#from ptmcmc import PTMCMC

from gibbsmcmc import GibbsMCMC

class MultiTransitMCMC(Fitter):
    def __init__(self, lcdata, parameterization, mcmc_pars={}, **kwargs):

        lcdata    = lcdata if isinstance(lcdata, list) else [lcdata]
        nchannels = len(lcdata)
        method    = kwargs.get('mcmcmethod', 'Gibbs')

        methods = {'Gibbs':GibbsMCMC, 'PT':PTMCMC}

        self.p = parameterization
        self.fitfun = FitnessFunction(self.p, lcdata, **kwargs)
        self.fitter = methods[method](self.fitfun, self.p, **mcmc_pars)

    def __call__(self):
        return self.fitter()
