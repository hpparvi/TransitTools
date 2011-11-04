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
    def __init__(self, loglfun, parameterization, **kwargs):
        
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

        self.loglfun = loglfun
        self.parameters = parameterization.fp_defs
        self.fitpar = parameterization
        self.n_parms = self.fitpar.n_fp

        self.n_points = concatenate(self.loglfun.times).size

        self.p0 = self.fitpar.fp_vect.copy()
        self.p  = self.p0.copy()
        
        self.p_names = self.fitpar.fp_names
        self.p_descr = [p.description for p in self.parameters if p.free == True]

        self.result = MCMCResult(self.n_chains_l, self.n_steps, self.n_parms, self.p_names, self.p_descr)

        acceptionTypes = {'ChiLikelihood':self._acceptStepChiLikelihood, 'LogLikelihood':self._acceptStepLogLikelihood}
#        self.acceptStep = acceptionTypes['ChiLikelihood']
        self.acceptStep = acceptionTypes['LogLikelihood']


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


    def _acceptStepLogLikelihood(self, logL0, logL1, prior_ratio):
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
            log_posterior_ratio = log(prior_ratio) + logL1 - logL0
            if log_posterior_ratio > 0 or np.random.random() < exp(log_posterior_ratio):
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

    def get_samples(self, burn, thinning, name=None):
        raise NotImplementedError

    def get_map(self, name, burn, thinning, method='fit'):
        from scipy.stats.mstats import mquantiles
        d = name if isinstance(name, np.ndarray) else self.get_samples(burn, thinning, name)
        method = method.lower()

        if method.lower() == 'fit':
             map, ep, em = fit_distribution(d)
        elif method.lower() == 'fit2':
            map, ep, em, map2, ep2, em2, x = fit_distribution2(d)
            tt = np.linspace(d.min(), d.max(), 500)
            ft = ((1-x)*asymmetric_gaussian(tt, map, ep, em)
                  + x*asymmetric_gaussian(tt, map2, ep2, em2))
            map = tt[ft.argmax()]
            em  = mquantiles(d[d<map],[1-2*0.341])[0]
            ep  = mquantiles(d[d>map],[2*0.341])[0]
        elif method.lower() == 'median':
            map, ep, em = mquantiles(d, [0.5, 0.5-0.341, 0.5+0.341])
        elif method == 'histogram':
            nb, vl = np.histogram(data, res)
            mid    = np.argmax(nb)
            map    = 0.5*(vl[mid]+vl[mid+1])
            em  = mquantiles(d[d<map],[1-2*0.341])[0]
            ep  = mquantiles(d[d>map],[2*0.341])[0]

        return map, ep, em


    #TODO: Replace with get_max_likelihood
    def get_chi_minimum(self):
        raise NotImplementedError

    def get_quantiles(self, burn, thinning, quantiles=[0.5-0.341, 0.5, 0.5+0.341]):
        from scipy.stats.mstats import mquantiles
        data = self.get_samples(burn, thinning)
        return {p: mquantiles(d, quantiles) for p, d in data.items()}

    def save(self, filename):
        f = open(filename, 'wb')
        dump(self, f)
        f.close()

    def save_fits(self, filename):
        raise NotImplementedError

    def plot(self, i_c, burn_in, thinning, fign=100, s_max=-1, c='b', alpha=1.0):
        raise NotImplementedError

    def plot_parameter_distribution(self, data, burn, thinning, ax=None, **kwargs):
        """
        
        Parameters
        ==========
          name
          burn
          thinning
          ax

        Keyword arguments
        =================
          nbins
          alpha
          fc
          xlim
          range
          title
          transformation
          median
          confidence_limits
          fit_distribution
          fit_n
          xlabel
          data
        """
        from scipy.stats.mstats import mquantiles

        nbins = kwargs.get('nbins', 50)
        fc    = kwargs.get('fc', '0.65')
        alpha = kwargs.get('alpha', 1)
        xlim  = kwargs.get('xlim', None)
        rnge  = kwargs.get('range', None)
        xlabel = kwargs.get('xlabel', None)
        verbose = kwargs.get('verbose', False)

        if isinstance(data,str):
            d = self.get_samples(burn, thinning, data)
            name = kwargs.get('name',None) or data
        elif isinstance(data, np.ndarray):
            d = data
            name = kwargs.get('name','')
        else:
            raise NotImplementedError

        transformation = kwargs.get('transformation', None)
        if transformation is not None:
        #    d = eval("{}()")
            if transformation == 'sqrt':
                d = np.sqrt(d)
            elif transformation == 'square':
                d = d**2

        ax = ax or pl.axes()
        xlim = xlim or mquantiles(d, [0.001, 0.999])
        rnge = rnge or mquantiles(d, [0.001, 0.999])

        if xlabel is None:
            try:
                xlabel = self.p_description[list(self.p_names).index(name)]
                if  transformation == 'sqrt':
                    xlabel = xlabel.lower().replace('squared','').strip().capitalize()
            except ValueError:
                xlabel = ''

        n,b,p = ax.hist(d, nbins, range=rnge, fc=fc, histtype='stepfilled', normed=True, lw=1, alpha=alpha)

        if kwargs.get('median', False):
            ax.axvline(np.median(d), c='0.25', lw=1.5)
        
        if kwargs.get('confidence_limits', False):
            qnt = mquantiles(d, [0.5-0.341, 0.5+0.341])
            ax.axvline(qnt[0], ls='--', c='0.5', lw=1.5)
            ax.axvline(qnt[1], ls='--', c='0.5', lw=1.5)
            ax.fill_between(qnt,0,1e8, facecolor='0.9', edgecolor='1.0')

        if kwargs.get('fit_distribution', False):
            tt = np.linspace(d.min(), d.max(), 500)
            if kwargs.get('fit_n', 1) == 1:
                map, ep, em = fit_distribution(d)
                #ax.plot(tt, asymmetric_gaussian(tt, map, ep, em), c='0.0', lw=2)
                if verbose: print "{name:10s} {map:12.6f} {psigma:+12.6f} {msigma:+12.6f}".format(name=name, map=map, psigma=ep-map, msigma=em-map)

                ax.axvline(map, c='0.25', lw=2.)
                ax.axvline(ep, ls='--', c='0.5', lw=1.5)
                ax.axvline(em, ls='--', c='0.5', lw=1.5)
                ax.fill_between([em,ep],0,1e8, facecolor='0.0', edgecolor='1.0', alpha=0.15)

            else:
                map, ep, em, map2, ep2, em2, x = fit_distribution2(d)
                ft = ((1-x)*asymmetric_gaussian(tt, map, ep, em)
                      + x*asymmetric_gaussian(tt, map2, ep2, em2))
                #ax.plot(tt, (1-x)*asymmetric_gaussian(tt, map, ep, em), c='0.25', ls='--', lw=1)
                #ax.plot(tt, x*asymmetric_gaussian(tt, map2, ep2, em2), c='0.25', ls='--', lw=1)
                #ax.plot(tt, ft, c='0.0', lw=2)

                diff = np.sign(np.diff(ft))
                mask = diff[:-1]>diff[1:]
                map = tt[mask]

                em  = [mquantiles(d[d<m],[1-2*0.341])[0] for m in map]
                ep  = [mquantiles(d[d>m],[2*0.341])[0] for m in map]

                c = ['r','b','g']
                c = ['0.25','0.25','0.25']

                for i, m in enumerate(map):
                    if verbose: print "{name:10s} {map:12.6f} {psigma:+12.6f} {msigma:+12.6f} ({i})".format(name=name, map=m, psigma=ep[i]-m, msigma=em[i]-m,i=i+1)
                
                    ax.axvline(m, c=c[i], lw=2.)
                    ax.axvline(ep[i], ls='--', c=c[i], lw=1.5)
                    ax.axvline(em[i], ls='--', c=c[i], lw=1.5)
                    ax.fill_between([em[i],ep[i]],0,1e8, facecolor=c[i], edgecolor='1.0', alpha=0.15)

        ymax = max(n.max(), ax.get_ylim()[1])
        ax.set_ylim([0, 1.05*ymax])
        
        ax.set_xlabel(xlabel, size='x-large')
        ax.set_title(kwargs.get('title', ''))
        ax.set_xlim(xlim)
        ax.set_yticks([])


    def plot_2d_distribution(self, names, burn, thinning, ax=None, **kwargs):
        from scipy.stats.mstats import mquantiles
        from numpy import sqrt, log
        from matplotlib.cm import gray_r

        transformations = kwargs.get('transformations', [None,None])
        gridsize = kwargs.get('gridsize',50)
        extent = kwargs.get('extent', None)
        xlim = kwargs.get('xlim', None)
        ylim = kwargs.get('ylim', None)
        xlabel = kwargs.get('xlabel', None)
        ylabel = kwargs.get('ylabel', None)

        ax = ax or pl.axes()
        d = []
        for name in names:
            if isinstance(name,str):
                d.append(self.get_samples(burn, thinning, name))
            elif isinstance(name, np.ndarray):
                d.append(name)
            else:
                raise NotImplementedError

        for i, t in enumerate(transformations):
            if t is not None: d[i] = eval(transformations[i].format('d[%i]'%i))

        xlim = xlim or mquantiles(d[0], [0.001, 0.999])
        ylim = ylim or mquantiles(d[1], [0.001, 0.999])

        if extent is None:
            extent = [xlim[0],xlim[1],ylim[0],ylim[1]]

        ax.hexbin(d[0],d[1], gridsize=gridsize, extent=extent, cmap=gray_r)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if xlabel != '': pl.setp(ax, xlabel=xlabel or names[0])
        if ylabel != '': pl.setp(ax, ylabel=ylabel or names[1])

        #pl.setp(ax, xlabel=xlabel or names[0], ylabel=ylabel or names[1])


def load_MCMCResult(filename):
    f = open(filename, 'rb')
    res = load(f)
    f.close()
    return res


## Distribution fitting
## ====================
def asymmetric_gaussian(x, center, sp, sm):
    from scipy.stats import norm
    t = np.zeros_like(x)
    Np = sqrt(2*pi*(sp-center)**2)/(sqrt(0.5*pi)*(sp-sm))
    Nm = sqrt(2*pi*(center-sm)**2)/(sqrt(0.5*pi)*(sp-sm))
    t[x>center]  = Np*norm.pdf(x[x>center], center, sp-center)
    t[x<=center] = Nm*norm.pdf(x[x<=center], center, center-sm)
    return t


def fit_distribution(data):
    from scipy.optimize import fmin
    from scipy.stats.mstats import mquantiles

    x = mquantiles(data, [0.5, 0.5+0.341, 0.5-0.341])
    bv, be = np.histogram(data, 150, normed=True)
    bc = 0.5*(be[:-1]+be[1:])
    def fitness(x):
        return ((bv - asymmetric_gaussian(bc, *x[:3]))**2).sum()

    res = fmin(fitness, x, xtol=1e-8, disp=False)
    return res

def fit_distribution2(data):
    from scipy.optimize import fmin
    from scipy.stats.mstats import mquantiles

    x = mquantiles(data, [0.35, 0.35+0.341, 0.35-0.341, 0.35, 0.65+0.341, 0.65-0.341, 0.5])
    bv, be = np.histogram(data, 150, normed=True)
    bc = 0.5*(be[:-1]+be[1:])
    x[6] = 0.5
    
    def fitness(x):
        return ((bv - (1-x[6])*asymmetric_gaussian(bc, *x[:3])
                 - x[6]*asymmetric_gaussian(bc, *x[3:6]))**2).sum()

    res = fmin(fitness, x, xtol=1e-8, maxiter=1000, disp=False)
    return res
