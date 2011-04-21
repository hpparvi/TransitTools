import sys
import numpy as np

from numpy import array, asarray, ones
from types import MethodType

from transitLightCurve.core import *
from transitLightCurve.transitlightcurve import TransitLightcurve
from transitLightCurve.utilities import bin, fold

from fitparameterization import MTFitParameterization
from fitnessfunction import FitnessFunction
from transitfitter import Fitter
from mcmc import MCMC

class MultiTransitMCMC(Fitter):
    def __init__(self, lcdata, pars, stellar_prm, parameter_defs, p0, mcmc_pars={}, **kwargs):

        lcdata    = lcdata if isinstance(lcdata, list) else [lcdata]
        nchannels = len(lcdata)

        ## Generate the fitting parameterization
        ## =====================================
        self.p = MTFitParameterization(pars, stellar_prm, nchannels, lcdata[0].n_transits,
                                       initial_parameter_values = p0,
                                       **kwargs)
        
        ## Setup the minimization function
        ## ================================
        self.fitfun = FitnessFunction(self.p, lcdata, **kwargs)
 
        self.fitter = MCMC(self.fitfun, parameter_defs, **mcmc_pars)


    def __call__(self):
        return self.fitter()
