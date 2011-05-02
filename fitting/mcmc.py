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
import tables as tbl
import scipy.signal

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


class DrawGaussianIndependence(DrawSample):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std

    def __call__(self, x): raise NotImplementedError                      

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
    def __init__(self, chifun, parameter_defs, **kwargs):
        
        self.seed     = kwargs.get('seed', 0)
        self.n_chains = kwargs.get('n_chains', 1)
        self.n_steps  = kwargs.get('n_steps', 200) 
        self.thinning = kwargs.get('thinning', 1) 
        self.use_mpi  = kwargs.get('use_mpi', True) 
        self.verbose  = kwargs.get('verbose', True) 

        self.autotune_length = kwargs.get('autotune_length', 2000)
        self.autotune_strength = kwargs.get('autotune_strength', 0.5)

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

        self.chifun = chifun
        self.fitpar = self.chifun.parm
        self.n_parms = self.fitpar.n_fitted_parameters + 1

        self.n_points = concatenate(self.chifun.times).size

        self.parameters = []
        self.p_names = concatenate([self.fitpar.get_parameter_names(), ['error scale']])
        self.p_descr = concatenate([self.fitpar.get_parameter_descriptions(), ['Error scale']])

        try:
            for name, descr in zip(self.p_names, self.p_descr):
                self.parameters.append(MCMCParameter(name, descr,
                                                     parameter_defs[name]['start_value'],
                                                     parameter_defs[name]['draw_function'],
                                                     parameter_defs[name]['prior']))
        except KeyError:
            print "Error: no sigma given for parameter %s" %name
            sys.exit()

            
        self.p0 = asarray([p.start_value for p in self.parameters])
        self.p  = self.p0.copy()
        
        self.result = MCMCResult(self.n_chains_l, self.n_steps, self.n_parms, self.p_names, self.p_descr)

        acceptionTypes = {'ChiLikelihood':self._acceptStepChiLikelihood}
        self.acceptStep = acceptionTypes['ChiLikelihood']


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
        if prior_ratio < 0: 
            return False
        elif np.random.random() < (prior_ratio*exp(0.5*( self.n_points*log(error_ratio) + X0 - Xt))):
            return True
        else:
            return False


    def __call__(self):
        curses.wrapper(self._mcmc)

    def _mcmc(self, *args):
        """The main Markov Chain Monte Carlo routine in all its simplicity."""
        self.init_curses()

        ## MCMC main loop
        ## ==============
        for chain in xrange(self.n_chains_l):
            self.i_chain = chain
            #logging.info('Starting node %2i  chain %2i  of %2i' %(mpi_rank, chain+1, self.n_chains_l))
            P_cur = self.p.copy()
            prior_cur = asarray([p.prior(P_cur[i]) for i, p in enumerate(self.parameters)])
            X_cur = self.chifun(P_cur[:-1])

            at_test = np.zeros(self.n_parms, dtype=np.int)
 
            i_at = 0
            P_try = P_cur.copy()
            prior_try = prior_cur.copy()
            for i_s in xrange(self.n_steps):
                self.i_step = i_s
                for i_t in xrange(self.thinning):
                    for i_p, p in enumerate(self.parameters):
                        P_try[i_p] = p.draw(P_cur[i_p])
                        prior_try[i_p] = p.prior(P_try[i_p])
                        X_try =  self.chifun(P_try[:-1])

                        prior_ratio = prior_try.prod() / prior_cur.prod()
                        error_ratio = P_try[-1] / P_cur[-1]

                        self.result.accepted[chain, i_p, 0] += 1

                        if X_try < 1e17 and self.acceptStep(P_cur[-1]*X_cur, P_try[-1]*X_try, prior_ratio, error_ratio):
                            P_cur[i_p] = P_try[i_p]
                            prior_cur[i_p] = prior_try[i_p]
                            X_cur = X_try
                            self.result.accepted[chain, i_p, 1] += 1
                            at_test[i_p] += 1
                        else:
                            P_try[i_p] = P_cur[i_p]

                    ## Autotuning
                    ## ==========
                    if i_s*self.thinning + i_t < self.autotune_length and i_at == 25:
                        for i_p, p in enumerate(self.parameters):
                            accept_ratio  = at_test[i_p]/25. 
                            accept_adjust = (1. - self.autotune_strength) + self.autotune_strength*4.* accept_ratio
                            p._draw_method.sigma *= accept_adjust
                            #info("Autotune: %i %6.4f %6.2f %6.2f"%(i_p, p._draw_method.sigma, accept_ratio, accept_adjust), I1)
                        #info('', I1)
                        at_test[:] = 0.
                        i_at = 0
                    i_at  += 1

                self.result.steps[chain, i_s, :] = P_cur[:]
                self.result.chi[chain, i_s] = X_cur

                ## DEBUGGING CODE
                ## ==============
                if i_s!=0 and i_s%25 == 0:
                    #print "%2i %5i"%(mpi_rank, i_s), self.result.get_acceptance()
                    pl.figure(10, figsize=(20,20))
                    pl.clf()
                    for ip in range(self.n_parms):
                        ax1 = pl.subplot(self.n_parms,2,ip*2+1)
                        pl.text(0.02, 0.85, self.p_names[ip],transform = ax1.transAxes)
                        pl.plot(self.result.steps[0,:i_s,ip])
                        pl.subplot(self.n_parms,2,ip*2+2)
                        pl.hist(self.result.steps[0,:i_s,ip])
                        pl.savefig('mcmcdebug_n%i_%i.pdf'%(mpi_rank,chain))

                self.ui.update()

        ## MPI communication
        ## =================
        ## We do this the easy but slow way. Instead of using direct NumPy array transfer, we
        ## pass the data as generic Python objects. This should be ok for now.
        if self.use_mpi:
            result = MCMCResult(self.n_chains_g, self.n_steps, self.n_parms, self.p_names, self.p_descr)
            result.steps    = mpi_comm.gather(self.result.steps)
            result.chi      = mpi_comm.gather(self.result.chi)
            result.accepted = mpi_comm.gather(self.result.accepted)

            if is_root:
                result.steps    = np.concatenate(result.steps)
                result.chi      = np.concatenate(result.chi)
                result.accepted = np.concatenate(result.accepted)
                self.result = result
            
        return self.result

    def get_results(self, parameter=None, burn_in=None, cor_len=None, separate_chains=False):
        return self.result(parameter, burn_in, cor_len, separate_chains)

    def init_curses(self):
        self.ui = MCMCCursesUI(self)

class MCMCCursesUI(object):
    def __init__(self, mcmc):
        self.mcmc = mcmc
        
        self.screen = curses.initscr()
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

        #self.get_results = self.__call__

    def __call__(self, parameter=None, burn_in=None, cor_len=None, separate_chains=False):
        """Returns the results."""

        burn_in = burn_in if burn_in is not None else self.burn_in
        burn_in = int(burn_in*self.n_steps)
        cor_len = cor_len if cor_len is not None else self.cor_len
        
        if parameter is None:
            r = self.steps[:, burn_in::cor_len, :]
        else:
            i = self.p_names.index(parameter)
            r = self.steps[:, burn_in::cor_len, i]
       
        if parameter is not None:
            return r if separate_chains else r.ravel()
        else:
            if separate_chains:
                return r
            else:
                rt = np.split(r, self.n_chains, 0)
                return np.concatenate(rt, 1)[0, :, :]
    
    def get_acceptance(self):
        """Returns the acceptance ratio for the parameters."""
        t = self.accepted.sum(0)
        r = np.where(t[:,0]>0, t[:,1]/t[:,0], 0)
        return r

    
    def save(self, filename):
        f = open(filename, 'wb')
        dump(self, f)
        f.close()


def load_MCMCResult(filename):
    f = open(filename, 'rb')
    res = load(f)
    f.close()
    return res
