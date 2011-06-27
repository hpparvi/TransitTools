"""
Markov Chain Monte Carlo
------------------------

A module to carry out parameter fitting error estimation using Markov Chain
Monte Carlo simulation. 
"""
from __future__ import division

import sys
import time
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import pylab as pl
import scipy as sp
import pyfits as pf

from copy import copy, deepcopy
from cPickle import dump, load

from math import exp, log, sqrt
from numpy import pi, sin, arccos, asarray, concatenate
from numpy.random import normal, random, randint

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


class PTMCMCChain(object):
    def __init__(self, id, n_parms, n_steps, n_points, p0, temperature, parameters, chifun, acceptStepFunc):
        self.n_parms  = n_parms
        self.n_steps  = n_steps
        self.n_points = n_points
        self.id       = id

        self.parameters = deepcopy(parameters)
        self.temperature = temperature

        self.P_cur = p0.copy()
        self.P_try = p0.copy()
        self.X_cur = chifun(p0[:-1])
        self.X_try = chifun(p0[:-1])
        self.prior_cur  = np.array([p.prior(vp) for p, vp in zip(parameters, p0)])
        self.prior_try  = self.prior_cur.copy()

        self.chifun     = chifun
        self.acceptStep = acceptStepFunc

        self.swaps_tried = 0
        self.swaps_succeeded = 0

        self.acceptance = np.zeros([self.n_parms, 2])
        self.autotune_test = np.zeros(self.n_parms, dtype=np.int)
        self.autotune_adjust = np.ones(self.n_parms)

    def draw_trial_step(self, ip):
        self.P_try[ip] = self.parameters[ip].draw(self.P_cur[ip])
        self.prior_try[ip] = self.parameters[ip].prior(self.P_try[ip])
        self.X_try =  self.chifun(self.P_try[:-1])
        self.acceptance[ip, 0] += 1

    def try_trial_step(self):
        if self.X_try < 1e17:
            prior_ratio = self.prior_try.prod() / self.prior_cur.prod()
            error_ratio = (self.P_try[-1] / self.P_cur[-1])
            error_ratio = error_ratio if error_ratio > 1e-18 else 1e-18
            return self.acceptStep(self.P_cur[-1]*self.X_cur, self.P_try[-1]*self.X_try, prior_ratio, error_ratio, self.temperature)
        else:
            return False
        
    def accept_trial_step(self, ip):
        self.P_cur[ip] = self.P_try[ip]
        self.prior_cur[ip] = self.prior_try[ip]
        self.X_cur = self.X_try
        self.acceptance[ip, 1] += 1
        self.autotune_test[ip] += 1

    def discard_trial_step(self, ip):
        self.P_try[ip] = self.P_cur[ip]
        self.prior_try[ip] = self.prior_cur[ip] 
        
    ## Parallel tempering swapping
    ## ===========================
    ## (Gregory, P., Bayesian Logical Data Analysis in Physical Sciences, 2005, p. 322)
    ## Note: The priors cancel out
    ##
    def try_swapping(self, other):
        self.swaps_tried += 1
        err0 = self.P_cur[-1]
        err1 = other.P_cur[-1]
        tmp0 = self.temperature
        tmp1 = other.temperature
        chi0 = self.X_cur
        chi1 = other.X_cur
        
        trm = 0.5*(tmp0-tmp1)*(self.n_points*log(err1/err0) + err0*chi0 - err1*chi1)

        if trm > 0 or random() < exp(trm):
            tmp = self.P_cur.copy()
            self.P_cur[:]  = other.P_cur
            self.P_try[:]  = other.P_cur
            other.P_cur[:] = tmp
            other.P_try[:] = tmp
            tmp = self.X_cur
            self.X_cur = other.X_cur
            other.X_cur = tmp
            self.swaps_succeeded += 1
            return True
        else:
            return False

    def get_acceptance(self):
        return np.where(self.acceptance[:,0]>0, self.acceptance[:,1]/self.acceptance[:,0], 0)


class PTMCMC(object):
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
 
        self.monitor  = kwargs.get('monitor', True)
        self.mfile    = kwargs.get('monitor_file', 'mcmc_monitor')
        self.minterval= kwargs.get('monitor_interval', 150)

        self.sinterval= kwargs.get("autosave_interval", 250)
        self.sname    = kwargs.get("autosave_filename", 'mcmc_autosave.pkl')

        self.autotune_length = kwargs.get('autotune_length', 2000)
        self.autotune_strength = kwargs.get('autotune_strength', 0.5)
        self.autotune_interval = kwargs.get('autotune_interval', 25)

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
        
        #temperatures = [1., 0.55, 0.35, 0.15, 0.085, 0.03]
        temperatures = [1., 0.65, 0.40, 0.20, 0.085, 0.03]
        self.chains = [PTMCMCChain(i, self.n_parms, self.n_steps, self.n_points, self.p0, temperatures[i], self.parameters,
                                 self.chifun, self._acceptStepChiLikelihood) for i in range(self.n_chains_l)]

        self.result = MCMCResult(self.n_chains_l, self.n_steps, self.n_parms, self.p_names, self.p_descr)

        acceptionTypes = {'ChiLikelihood':self._acceptStepChiLikelihood}
        self.acceptStep = acceptionTypes['ChiLikelihood']


    def _acceptStepChiLikelihood(self, X0, Xt, prior_ratio, error_ratio, temperature=1):
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
          The Chi^2-values are supposed to be premultiplied by the error scale factor b.
        """
        if prior_ratio < 0: 
            return False
        else:
            trm = 0.5*( self.n_points*log(error_ratio) + X0 - Xt)
            if trm > 200 or np.random.random() < prior_ratio*exp(trm)**temperature:
                return True
            else:
                return False


    def __call__(self):
        curses.wrapper(self._mcmc)
        #self._mcmc()
        return self.result

    def _mcmc(self, *args):
        """The main Markov Chain Monte Carlo routine in all its simplicity."""
        self.init_curses()
        self.swaps = []

        ## MCMC main loop
        ## ==============
        for i_s in xrange(self.n_steps):
            self.i_step = i_s

            for it in xrange(self.thinning):
                for ip, p in enumerate(self.parameters):
                    for ic, chain in enumerate(self.chains):
                        chain.draw_trial_step(ip)
                        if chain.try_trial_step():
                            chain.accept_trial_step(ip)
                        else:
                            chain.discard_trial_step(ip)

                ## State swapping
                #if i_s>self.autotune_length and random() < 1./30.:
                if random() < 0.1:
                    ic = randint(0, self.n_chains_l-1)
                    swap = self.chains[ic].try_swapping(self.chains[ic+1])
                    if swap: self.swaps.append(i_s)


            ## Autotuning
            ## ==========
            if i_s < self.autotune_length and (i_s+1)%25 == 0:
                for ic, c in enumerate(self.chains):
                    for ip, p in enumerate(c.parameters):
                        accept_ratio  = c.autotune_test[ip]/(25.*self.thinning)
                        accept_adjust = (1. - self.autotune_strength) + self.autotune_strength*4.* accept_ratio
                        p._draw_method.sigma *= accept_adjust
                    c.autotune_test[:] = 0.

            for ic, chain in enumerate(self.chains):
                self.result.steps[ic, i_s, :] = chain.P_cur
                self.result.chi[ic, i_s] = chain.X_cur

            ## MONITORING
            ## ==========
            if self.monitor and (i_s+1)%self.minterval == 0:
                self.plot_simulation_progress(i_s)

            if (i_s+1)%self.sinterval == 0:
                self.result.save(self.sname)

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

    def plot_simulation_progress(self, i_s):
        fig = self.fig_progress
        fig.clf()
        nrows = self.n_parms/2+1
        ncols = 6
        
        for ip in range(self.n_parms):
            ax_chain = fig.add_subplot(nrows,ncols,ip*3+1, axis_bgcolor='1.0')
            ax_hist  = fig.add_subplot(nrows,ncols,ip*3+2)
            ax_acorr = fig.add_subplot(nrows,ncols,ip*3+3)
            pl.text(0.035, 0.85, self.p_names[ip],transform = ax_chain.transAxes,
                    backgroundcolor='1.0', size='large')

            xmin = self.result.steps[:,:i_s,ip].min()
            xmax = self.result.steps[:,:i_s,ip].max()

            for ic in range(1,self.n_chains):
                d = self.result.steps[ic,:i_s,ip]
                ax_chain.plot(d, alpha=0.5)
                ax_hist.hist(d, bins=10, range=[xmin,xmax], histtype='stepfilled', alpha=0.25)

            d = self.result.steps[0,:i_s,ip]
            ax_chain.plot(d, color='0.0', lw=1.5)
            ax_acorr.acorr(d-d.mean(), maxlags=45, usevlines=True)
            ax_hist.hist(d, bins=10, range=[xmin,xmax], histtype='stepfilled', color='0', alpha=0.50)

            for swap in self.swaps:
                ax_chain.axvline(swap, c='0.5', ls=':')

            ax_acorr.axhline(1./np.e, ls='--', c='0.5')
            pl.setp([ax_chain, ax_hist, ax_acorr], yticks=[])
            pl.setp(ax_chain, xlim=[0,i_s])
            pl.setp(ax_acorr, xlim=[-45,45], ylim=[0,1])

        pl.subplots_adjust(top=0.99, bottom=0.02, left=0.01, right=0.99, hspace=0.2, wspace=0.04)
        pl.savefig('mcmcdebug_n%i.pdf'%(mpi_rank))


class MCMCCursesUI(object):
    def __init__(self, mcmc):
        self.mcmc = mcmc

        self.screen = curses.initscr()
        curses.use_default_colors()

        self.height, self.width = self.screen.getmaxyx()
        self.main_window = CTitleWindow('MCMC', self.width, self.height, 0, 0)
        self.main_window.refresh()

        self.vspace = 1

        self.iw_height = 8+mcmc.n_chains_l
        self.aw_height = 3+mcmc.n_chains_l
        self.sw_height = 3+mcmc.n_chains_l
        self.fw_height = 4

        self.iw_ypos   = 2
        self.aw_ypos   = self.iw_ypos + self.iw_height + self.vspace
        self.sw_ypos   = self.aw_ypos + self.aw_height + self.vspace
        self.fw_ypos   = self.sw_ypos + self.sw_height + self.vspace

        self.info_window = CTitleWindow('Information', 70, self.iw_height, 3, self.iw_ypos)
        self.iw = self.info_window        
        self.iw.addstr(1,1,'Chains')
        self.iw.addstr(2,1,'Steps')
        self.iw.addstr(4,1,'Autotune length {}'.format(self.mcmc.autotune_length))
        self.iw.addstr(6,1,'Swaps')

        self.accept_window = CTitleWindow('Accept ratio', self.width-6, self.aw_height, 3, self.aw_ypos)
        self.aw = self.accept_window
        self.aw.addstr(1,1,' '.join(['{0:^10s}'.format(n) for n in self.mcmc.p_names]))

        self.sigma_window = CTitleWindow('Parameter Sigma', self.width-6, self.sw_height, 3, self.sw_ypos)
        self.sw = self.sigma_window
        self.sw.addstr(1,1,' '.join(['{0:^10s}'.format(n) for n in self.mcmc.p_names]))

        self.fitted_p_window = CTitleWindow('Fitted parameters', self.width-6, self.fw_height, 3, self.fw_ypos)
        self.fw = self.fitted_p_window
        self.fw.refresh()

    def update(self):
        #self.iw.addstr(1,15,"{0:6d}/{1:6d}".format(self.mcmc.i_chain+1, self.mcmc.n_chains))
        self.iw.addstr(2,15,"{0:6d}/{1:6d}".format(self.mcmc.i_step+1, self.mcmc.n_steps))

        for i, chain in enumerate(self.mcmc.chains):
            self.aw.addstr(2+i,1,' '.join(['{0:^10.1%}'.format(a) for a in chain.get_acceptance()]))
            self.sw.addstr(2+i,1,' '.join(['{0:^10.6f}'.format(s) for s in [p._draw_method.sigma for p in chain.parameters]]))
            self.iw.addstr(7+i,3,'Tried: {}  Succeeded: {}  Succes rate: {}'.format(chain.swaps_tried,
                                                                                                chain.swaps_succeeded,
                                                                                                0 if chain.swaps_tried == 0 else chain.swaps_succeeded/chain.swaps_tried))

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

    def __call__(self, parameter, burn_in=None, thinning=None):
        """Returns the results."""
        burn_in  = burn_in or self.burn_in
        thinning = thinning or self.cor_len
        i = list(self.p_names).index(parameter)
        return self.steps[0, burn_in::thinning, i]

    
    def get_acceptance(self):
        """Returns the acceptance ratio for the parameters."""
        t = self.accepted.sum(0)
        r = np.where(t[:,0]>0, t[:,1]/t[:,0], 0)
        return r

    
    def save(self, filename):
        f = open(filename, 'wb')
        dump(self, f)
        f.close()

    def save_fits(self, filename):
        pass

    def plot(self, i_c, burn_in, thinning, fign=100, s_max=-1, c='b', alpha=1.0):
        fig = pl.figure(fign, figsize=(34,20), dpi=50)
        nrows = self.n_parms/2+1
        ncols = 6
        mask = np.abs(self.steps[i_c,:,:]).sum(1) > 0.
        mask[:burn_in] = False
        mask[s_max:] = False
        ns = mask.sum()/thinning
        for ip in range(self.n_parms):
            xmin = self.steps[:,mask,ip][:,::thinning].min()
            xmax = self.steps[:,mask,ip][:,::thinning].max()
            d = self.steps[i_c,mask,ip][::thinning]

            ax_chain = fig.add_subplot(nrows,ncols,ip*3+1)
            ax_hist  = fig.add_subplot(nrows,ncols,ip*3+2)
            ax_acorr = fig.add_subplot(nrows,ncols,ip*3+3)

            pl.text(0.035, 0.85, self.p_names[ip],transform = ax_chain.transAxes,
                    backgroundcolor='1.0', size='large')
            ax_chain.plot(d, c=c, alpha=1.0)
            ax_hist.hist(d, range=[xmin,xmax], fc=c, alpha=alpha)
            ax_acorr.acorr(d-d.mean(), maxlags=45, usevlines=True, color=c)
            ax_acorr.axhline(1./np.e, ls='--', c='0.5')
            pl.setp([ax_chain, ax_hist, ax_acorr], yticks=[])
            pl.setp(ax_chain, xlim=[0,ns])
            pl.setp(ax_acorr, xlim=[-45,45], ylim=[0,1])
        pl.subplots_adjust(top=0.99, bottom=0.02, left=0.01, right=0.99, hspace=0.2, wspace=0.04)
        #pl.savefig('%i_%i.pdf'%(mpi_rank,chain))


def load_MCMCResult(filename):
    f = open(filename, 'rb')
    res = load(f)
    f.close()
    return res
