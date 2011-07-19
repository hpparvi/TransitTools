"""
Markov Chain Monte Carlo
------------------------

A module to carry out parameter fitting error estimation using Markov Chain
Monte Carlo simulation. 
"""
import sys
import time
import numpy as np
import pylab as pl
import scipy as sp
import pyfits as pf

from cPickle import dump, load

from math import exp, log, sqrt
from numpy import pi, sin, arccos, asarray, concatenate
from numpy.random import normal

from transitLightCurve.core import *
from mcmcprior import mcmcpriors, UniformPrior, JeffreysPrior


class DrawSample(object): pass

class DrawGaussian(DrawSample):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, x): return normal(x, self.sigma)
                    
draw_functions = {'gaussian':DrawGaussian}


class MCMCParameter(object):
    def __init__(self, name, descr, start_value, draw_function, prior):
        self.name = name
        self.description = descr
        self.start_value = start_value
        self._draw_method = draw_function
        self._prior = prior

    def draw(self, x):
        return self._draw_method(x)

    def prior(self, x):
        return self._prior(x)


class MCMC(object):
    """
    Class encapsulating the mcmc run.
    """
    def __init__(self, chifun, parameterization, **kwargs):
        
        self.seed     = kwargs.get('seed', 0)
        self.n_chains = kwargs.get('n_chains', 1)
        self.n_steps  = kwargs.get('n_steps', 200) 
        self.thinning = kwargs.get('thinning', 1) 
        self.use_mpi  = kwargs.get('use_mpi', True) 
        self.verbose  = kwargs.get('verbose', True)
 
        self.monitor  = kwargs.get('monitor', True)
        self.mfile    = kwargs.get('monitor_file', 'mcmc_monitor')
        self.minterval= kwargs.get('monitor_interval', 150)

        self.sinterval= kwargs.get("autosave_interval", 250)
        self.sname    = kwargs.get("autosave_filename", 'mcmc_autosave.pkl')

        self.autotune_length = kwargs.get('autotune_length', 2000)
        self.autotune_strength = kwargs.get('autotune_strength', 0.5)
        self.autotune_interval = kwargs.get('autotune_interval', 25)

        self.use_curses = kwargs.get('use_curses', True)

        if with_mpi and self.use_mpi:
            self.seed += mpi_rank
            self.n_chains_g = self.n_chains
            self.n_chains_l = self.n_chains//mpi_size
            if mpi_rank == mpi_size-1: self.n_chains_l += self.n_chains % mpi_size
            logging.info('Created node %i with %i chains'%(mpi_rank, self.n_chains_l))
        else:
            self.n_chains_g = self.n_chains
            self.n_chains_l = self.n_chains

        np.random.seed(self.seed)

        if self.monitor:
            self.fig_progress = pl.figure(10, figsize=(34,20))

        self.chifun = chifun
        self.parameters = parameterization.fp_defs
        self.fitpar = parameterization
        self.n_parms = self.fitpar.n_fp

        self.n_points = concatenate(self.chifun.times).size

        self.p0 = self.fitpar.fp_vect.copy()
        self.p  = self.p0.copy()
        
        self.p_names = self.fitpar.fp_names
        self.p_descr = [p.description for p in self.parameters if p.free == True]

        self.result = MCMCResult(self.n_chains_l, self.n_steps, self.n_parms, self.p_names, self.p_descr)

        acceptionTypes = {'ChiLikelihood':self._acceptStepChiLikelihood}
        self.acceptStep = acceptionTypes['ChiLikelihood']


    def log_likelihood_Gaussian(self, chi_sqr, sigma, sigma_scale, n):
        return 0.5*(n*log(sigma_scale/(2*pi*sigma**2)) - sigma_scale*chi_sqr)

    def log_likelihood_ratio_gaussian(self, X0, X1, p0, p1, b0, b1, n):
        return 0.5*(n*log(b0/b1) + b0*X0 - b1*X1)
    
    def log_likelihood_Cauchy(self): pass
    def log_likelihood_ratio_Cauchy(self): pass

    def log_likelihood_Laplace(self): pass
    def log_likelihood_ratio_Laplace(self): pass


    def _acceptStepChiLikelihood(self, X0, Xt, prior_ratio, error_ratio):
        """Decides whether we should accept the step or not based on the likelihood ratios.

        Decides whether we should accept the step or not based on the ratio of the two given
        likelihood values, ratio of priors and ratio of error scale parameters.

        likelihood ratio = p1/p0 (b1/b0)^(n/2) exp(b0 ChiSqr0/2 - b1 ChiSqr1/2)        (1)

                         = p1/p0 exp(n/2 log(b1/b0)) exp([b0 ChiSqr0 - b1 ChiSqr1]/2)  (2)

                         = p1/p0 exp([n log(b1/b0) + b0 ChiSqr0 - b1 ChiSqr1]/2)       (3)

        where p1 and p0 are the trial and current values of the prior, respectively, b1 and b0 the
        error scale factors, n the number of data points, ChiSqr1 and ChiSqr0 the Chi^2 values. We
        must make the transformation from (1) to (3) to increase numerical stability with large
        datasets (such as CoRoT lightcurves). For example, raising any value to the 100000 power
        directly would be a bit silly thing to do.
        
        Notes:
          The likelihood ratios are supposed to be premultiplied by the error scale factor b.
        """
        if prior_ratio <= 0: # This should never ever happen! Find the bug!
            return False
        else:
            log_posterior = log(prior_ratio) + 0.5*( self.n_points*log(error_ratio) + X0 - Xt)
            if log_posterior > 0 or np.random.random() < exp(log_posterior):
                return True
            else:
                return False


    def __call__(self):
        raise NotImplementedError

    def _mcmc(self, *args):
        raise NotImplementedError

    def get_results(self, parameter=None, burn_in=None, cor_len=None, separate_chains=False):
        return self.result(parameter, burn_in, cor_len, separate_chains)

    def init_curses(self):
        raise NotImplementedError

    def plot_simulation_progress(self, i_s, chain):
        raise NotImplementedError


class MCMCCursesUI(object):
    def __init__(self, mcmc):
        self.mcmc = mcmc
        
        self.screen = curses.initscr()

        curses.use_default_colors()

        self.height, self.width = self.screen.getmaxyx()
        self.main_window = CTitleWindow('MCMC', self.width, self.height, 0, 0)

        ## Info window
        self.info_window = CTitleWindow('Information', 40, 10, 3, 2)
        self.iw = self.info_window
        self.iw.addstr(1,1,'Chains')
        self.iw.addstr(2,1,'Steps')
        self.iw.addstr(4,1,'Autotune length {}'.format(self.mcmc.autotune_length))

        self.accept_window = CTitleWindow('Accept ratio', self.width-6, 4, 3, 12)
        self.aw = self.accept_window
        self.aw.addstr(1,1,' '.join(['{0:^10s}'.format(n) for n in self.mcmc.p_names]))

        self.sigma_window = CTitleWindow('Parameter Sigma', self.width-6, 4, 3, 16)
        self.sw = self.sigma_window
        self.sw.addstr(1,1,' '.join(['{0:^10s}'.format(n) for n in self.mcmc.p_names]))

        self.fitted_p_window = CTitleWindow('Fitted parameters', self.width-6, 4, 3, 20)

    def update(self):
        self.iw.addstr(1,15,"{0:6d}/{1:6d}".format(self.mcmc.i_chain+1, self.mcmc.n_chains))
        self.iw.addstr(2,15,"{0:6d}/{1:6d}".format(self.mcmc.i_step+1, self.mcmc.n_steps))

        self.aw.addstr(2,1,' '.join(['{0:^10.1%}'.format(a) for a in self.mcmc.result.get_acceptance()]))
        self.sw.addstr(2,1,' '.join(['{0:^10.6f}'.format(s) for s in [p._draw_method.sigma for p in self.mcmc.parameters]]))


class MCMCResult(FitResult):
    def __init__(self, n_chains, n_steps, n_parms, p_names, p_descr, burn_in = 0.2, cor_len=1):
        self.n_chains = n_chains
        self.n_steps  = n_steps
        self.n_parms  = n_parms
        self.burn_in  = burn_in
        self.cor_len  = cor_len
        self.p_names  = p_names
        self.p_description = p_descr
        
        self.steps    = np.zeros([n_chains, n_steps, n_parms])
        self.chi      = np.zeros([n_chains, n_steps])
        self.accepted = np.zeros([n_chains, n_parms, 2])

    def __call__(self, parameter=None, burn_in=None, thinning=None):
        raise NotImplementedError

    def get_acceptance(self):
        raise NotImplementedError

    def save(self, filename):
        f = open(filename, 'wb')
        dump(self, f)
        f.close()

    def save_fits(self, filename):
        raise NotImplementedError

    def plot(self, i_c, burn_in, thinning, fign=100, s_max=-1, c='b', alpha=1.0):
        raise NotImplementedError

def load_MCMCResult(filename):
    f = open(filename, 'rb')
    res = load(f)
    f.close()
    return res
