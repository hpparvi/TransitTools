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
    def __init__(self, lcdata, bounds, stellar_prm, parameters, **kwargs):
        self.verbose = kwargs.get('verbose', False)
    
        mcmc_pars = {'n_steps':150, 'n_chains': 1, 'seed':0, 'verbose':True, 'use_mpi':True}
        mcmc_pars.update(kwargs.get('mcmc_pars',{}))

        lcdata    = lcdata if isinstance(lcdata, list) else [lcdata]
        nchannels = len(lcdata)
        method    = kwargs.get('method', 'python')

        ## Generate the fitting parameterization
        ## =====================================
        self.p = MTFitParameterization(bounds, stellar_prm, nchannels, lcdata[0].n_transits, **kwargs)
        
        ## Setup the minimization function
        ## ================================
        self.fitfun = FitnessFunction(self.p, lcdata, **kwargs)

        self.p.l_min[1] = self.p.l_min[1] + self.p.l_min[2]
        self.p.l_min[2] = self.p.l_min[1] - self.p.l_min[2]
        self.p.l_max[1] = self.p.l_max[1] + self.p.l_max[2]
        self.p.l_max[2] = self.p.l_max[1] - self.p.l_max[2]

        self.p_0     = ones(self.p.n_fitted_parameters)
        self.p_sigma = ones(self.p.n_fitted_parameters)
        self.p_free  = ones(self.p.n_fitted_parameters, np.bool)
        self.p_names = self.p.get_parameter_names()
        self.p_descr = self.p.get_parameter_descriptions()
        self.hard_limits = self.p.get_hard_limits().transpose()

        try:
            for name in self.p_names:
                pm = self.p_names == name
                self.p_0[pm] = parameters[name][0]
                self.p_sigma[pm] = parameters[name][1]
        except KeyError:
            print "Error: no sigma given for parameter %s" %name
            sys.exit()
                
        #self.p_0[1] = self.p_0[1] + self.p_0[2]
        #self.p_0[2] = self.p_0[1] - self.p_0[2]
        #self.p_sigma[1] = self.p_sigma[1] + self.p_sigma[2]
        #self.p_sigma[2] = self.p_sigma[1] + self.p_sigma[2]

        self.fitter = MCMC(self.fitfun, p0=self.p_0, p_limits=self.hard_limits, p_free=self.p_free, p_sigma=self.p_sigma, 
                           p_names=self.p_names, p_descr=self.p_descr, **mcmc_pars)


    def __call__(self):
        return self.fitter()
