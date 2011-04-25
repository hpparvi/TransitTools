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
from numpy import pi, sin, arccos, asarray
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

        np.random.seed(self.seed)

        self.chifun = chifun
        self.fitpar = self.chifun.parm
        self.n_parms = self.fitpar.n_fitted_parameters

        self.parameters = []
        self.p_names = self.fitpar.get_parameter_names()
        self.p_descr = self.fitpar.get_parameter_descriptions()

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


    def _acceptStepChiLikelihood(self, X0, Xt, prior_ratio):
        if prior_ratio > 0.:
            if X0-Xt > 200 or np.random.random() < prior_ratio * exp(0.5*(X0 - Xt)):
                return True
            else:
                return False
        else:
            return False
    

    def __call__(self):
        """The main Markov Chain Monte Carlo routine in all its simplicity."""
        
        ## MCMC main loop
        ## ==============
        for chain in xrange(self.n_chains_l):
            logging.info('Starting node %2i  chain %2i  of %2i' %(mpi_rank, chain+1, self.n_chains_l))
            P_cur = self.p.copy()
            prior_cur = asarray([p.prior(P_cur[i]) for i, p in enumerate(self.parameters)])
            X_cur = self.chifun(P_cur)

            at_test = np.zeros(self.n_parms, dtype=np.int)
            
            i_at = 0
            P_try = P_cur.copy()
            prior_try = prior_cur.copy()
            for i_s in xrange(self.n_steps):
                for i_t in xrange(self.thinning):
                    for i_p, p in enumerate(self.parameters):
                        P_try[i_p] = p.draw(P_cur[i_p])
                        prior_try[i_p] = p.prior(P_try[i_p])
                        X_try =  self.chifun(P_try)

                        prior_ratio = prior_try.prod() / prior_cur.prod()

                        self.result.accepted[chain, i_p, 0] += 1

                        if self.acceptStep(X_cur, X_try, prior_ratio):
                            P_cur[i_p] = P_try[i_p]
                            prior_cur[i_p] = prior_try[i_p]
                            X_cur = X_try
                            self.result.accepted[chain, i_p, 1] += 1
                            at_test[i_p] += 1
                        else:
                            P_try[i_p] = P_cur[i_p]

                    ## Autotuning
                    ## ==========
                    if i_s*self.thinning + i_t < self.autotune_length and i_at == 100:
                        for i_p, p in enumerate(self.parameters):
                            accept_ratio  = at_test[i_p]/100. 
                            accept_adjust = (1. - self.autotune_strength) + self.autotune_strength*4.* accept_ratio
                            p._draw_method.sigma *= accept_adjust
                            info("Autotune: %i %6.4f %6.2f %6.2f"%(i_p, p._draw_method.sigma, accept_ratio, accept_adjust), I1)
                        info('', I1)
                        at_test[:] = 0.
                        i_at = 0
                    i_at  += 1

                self.result.steps[chain, i_s, :] = P_cur[:]
                self.result.chi[chain, i_s] = X_cur

                ## DEBUGGING CODE
                ## ==============
                if i_s!=0 and i_s%250 == 0:
                    print "%2i %5i"%(mpi_rank, i_s), self.result.get_acceptance()
                    pl.figure(10, figsize=(20,20))
                    pl.clf()
                    for ip in range(self.n_parms):
                        ax1 = pl.subplot(self.n_parms,2,ip*2+1)
                        pl.text(0.02, 0.85, self.p_names[ip],transform = ax1.transAxes)
                        pl.plot(self.result.steps[0,:i_s,ip])
                        pl.subplot(self.n_parms,2,ip*2+2)
                        pl.hist(self.result.steps[0,:i_s,ip])
                        pl.savefig('mcmcdebug_n%i_%i.pdf'%(mpi_rank,chain))

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
        r = t[:, 1] / t[:, 0]
        r[r!=r] = 0.
        
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
