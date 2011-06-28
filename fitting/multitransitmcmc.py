from transitLightCurve.core import *
from transitfitter import Fitter
from fitparameterization import MTFitParameterization
from fitnessfunction import FitnessFunction
from ptmcmc import PTMCMC
from gibbsmcmc import GibbsMCMC

class MultiTransitMCMC(Fitter):
    def __init__(self, lcdata, pars, stellar_prm, parameter_defs, p0, mcmc_pars={}, **kwargs):

        lcdata    = lcdata if isinstance(lcdata, list) else [lcdata]
        nchannels = len(lcdata)
        method    = kwargs.get('mcmcmethod', 'Gibbs')

        methods = {'Gibbs':GibbsMCMC, 'PT':PTMCMC}

        ## Generate the fitting parameterization
        ## =====================================
        self.p = MTFitParameterization(pars, stellar_prm, nchannels, lcdata[0].n_transits,
                                       initial_parameter_values = p0,
                                       **kwargs)

        ## Setup the minimization function
        ## ================================
        self.fitfun = FitnessFunction(self.p, lcdata, **kwargs)
        self.fitter = methods[method(self.fitfun, parameter_defs, **mcmc_pars)]

    def __call__(self):
        return self.fitter()
