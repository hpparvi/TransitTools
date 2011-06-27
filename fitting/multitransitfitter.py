"""Estimation of the planetary parameters from lightcurves covering possibly multiple transits.
 
 Doesn't use binning nor folding over phase. Instead, the near-transit portions are
 selected from the lightcurve, each transit is corrected for a polynomial continuum
 trend, sigma-clipping is used to remove the outlier points, and a lightcurve model
 is fitted to the data using first a global optimization to find a rough global fit,
 which is then improved using a local optimization method.

 Author
   Hannu Parviainen <hpparvi@gmail.com>

 Date 
   15.01.2011
"""
from __future__ import division
import sys

from copy import copy, deepcopy
from time import time
from cPickle import dump, load
from math import acos, cos, sin, asin, sqrt

try:
    from scipy.optimize import fmin
except ImportError:
    do_local_fit = False
    
from numpy import array, linspace
from matplotlib import pyplot as pl

from transitLightCurve.core import *
from transitLightCurve.transitparameterization import TransitParameterization, parameters, parameterizations
from transitLightCurve.transitlightcurve import TransitLightcurve
from transitLightCurve.fitting.de import DiffEvol, ParallelDiffEvol
from transitLightCurve.lightcurvedata import MultiTransitLC

from fitparameterization import MTFitParameterization
from fitnessfunction import FitnessFunction

class MTFitResult(FitResult):
    def __init__(self, fit_prm, fitfun, res_ds, prm_ds, lcdata):
        self.n_channels = fit_prm.nch
        self.channel_names = None
        self.ephemeris  = prm_ds.pv
        self.limb_darkening = [[fit_prm.get_ldc(i, res_ds)] for i in range(fit_prm.nch)]
        self.zeropoint = [[fit_prm.get_zp(i, res_ds)] for i in range(fit_prm.nch)]

        self.chi_sqr = None
        self.de_population = None
        self.de_fitness    = None
        
        ## Shortcuts
        ## =========
        self.nch = self.n_channels
        self.e   = self.ephemeris
        self.ldc = self.limb_darkening
        self.zp  = self.zeropoint

        ## Save the parameterization
        ## =========================
        self.parameterization = copy(fit_prm)
        self.parameterization.get_b2 = None
        self.parameterization.get_ldc = None
        self.parameterization.get_zp = None
        self.parameterization.get_kipping = None
        self.parameterization.map_p_to_k = None
        self.parameterization.map_k_to_p = None

        ## Save the fitness function
        ## =========================
        self.fitness_function = fitfun.fitfun_src


    def get_fit(self, ch=0):
        return self.ephemeris, self.limb_darkening[ch], self.zeropoint[ch]

    def get_chi(self):
        return self.chi_sqr

    def get_lc(self):
        pass

    def save(self, filename):
        f = open(filename, 'wb')
        dump(self, f)
        f.close()


def load_MTFitResult(filename):
    f = open(filename, 'rb')
    res = load(f)
    f.close()
    return res

    
def fit_multitransit(lcdata, bounds, stellar_prm, **kwargs):
    """Fits a transit model to a lightcurve using both global and local optimization.

    First searches for a rough global minimum using the differential evolution method, and
    continues from the best de fit using a basic downhill simplex method (fmin in scipy.optimize).
    
    Uses the Kipping parameterization for fitting, but excludes the transit width parameter IT,
    which is calculated from the period, impact parameter, stellar mass and stellar radius. We
    calculate first the semi-major axis a from the period and the stellar parameters using Kepler's
    third law, and then the Kipping's transit width parameter using the period and squared impact
    parameter.
    """
    if is_root: info('Starting multitransit fitting',H1)

    #######################################################################################
    ##
    ## INITIALIZATION
    ## ==============
    de_pars = {'npop':50, 'ngen':5, 'C':0.9, 'F':0.25}
    ds_pars = {}

    de_pars.update(kwargs.get('de_pars',{}))
    ds_pars.update(kwargs.get('ds_pars',{}))

    do_local_fit = kwargs.get('do_local_fit', True)
    method       = kwargs.get('method', 'python')

    lcdata    = lcdata if isinstance(lcdata, list) else [lcdata]
    nchannels = len(lcdata)
    totpoints = sum([d.npts for d in lcdata]) 

    ## Generate the fitting parameterization
    ## -------------------------------------
    p = MTFitParameterization(bounds, stellar_prm, nchannels, lcdata[0].n_transits, **kwargs)
    ##FIXME: Having different number of transits for each channel breaks things up.

    if is_root:
        info('Fitting data with',I1)
        info('%6i free parameters'%p.p_cur.size, I2)
        info('%6i channels'%nchannels, I2)
        info('%6i datapoints'%totpoints, I2)
        info('%6i levels of freedom'%(totpoints-p.p_cur.size), I2)
        info('')

    ## Setup the minimization function
    ## -------------------------------
    minfun = FitnessFunction(p, lcdata, **kwargs)

    #######################################################################################
    ##
    ## MINIMIZATION
    ## ============
    ##
    ## Global fitting using differential evolution
    ## -------------------------------------------
    fitter_g = DiffEvol(minfun, array([p.p_min,p.p_max]).transpose(), **de_pars)
    #fitter_g = ParallelDiffEvol(minfun, array([p.p_min,p.p_max]).transpose(), **de_pars)

    r_de     = fitter_g()

    f_de     = r_de.get_fit()
    chi_de   = r_de.get_chi()
    r_fn     = None 
    f_fn     = f_de
    chi_fn   = chi_de

    ## Local fitting using downhill simplex 
    ## ------------------------------------
    if do_local_fit:
        r_fn    = fmin(minfun, f_de, full_output=1, **ds_pars)
        f_fn    = r_fn[0]
        chi_fn  = r_fn[1]

    ## Map the fits represented in Kipping parameterization to physical parameterization
    ## =================================================================================
    p_de   = TransitParameterization('physical', p.get_physical(0, f_de))
    p_fn   = TransitParameterization('physical', p.get_physical(0, f_fn))

    #######################################################################################
    ##
    ## OUTPUT
    ## ======
    ##
    ## Print parameters and statistics
    ## -------------------------------
    if is_root:
        info('Best-fit physical parameters',H2)
        info("%14s %14s"%("DiffEvol" ,"FMin"),I1)
        nc = nchannels if p.separate_k2_ch else 1 
        nt = lcdata[0].n_transits if p.separate_zp_tr else 1
        for chn in range(nc):
            info("%14.5f %14.5f  -  radius ratio"%(p.get_physical(chn, f_de)[0],
                                                   p.get_physical(chn, f_fn)[0]), I1)

        for i,k in enumerate(parameterizations['physical'][1:]):
            info("%14.5f %14.5f  -  %s" %(p_de[i+1], p_fn[i+1], parameters[k].description), I1)

        info("")
        for chn in range(nchannels):
            for tr in range(nt):
                info("%14.5f %14.5f  -  zeropoint ch %i"%(p.get_zp(chn, tr, f_de),
                                                          p.get_zp(chn, tr, f_fn), chn), I1)

        info("")
        nc = nchannels if p.separate_ld else 1 
        for chn in range(nc):
            ldc_de  = p.get_ldc(chn, f_de)
            ldc_fn  = p.get_ldc(chn, f_fn)
            if p.n_ldc == 1:
                info("%14.5f %14.5f  -  limb darkening ch %i u"%(ldc_de[0], ldc_fn[0], chn), I1)
            if p.n_ldc == 2:
                info(" % 5.3f % 5.3f  % 5.3f % 5.3f  -  limb darkening ch %i u v"%(ldc_de[0], ldc_de[1], ldc_fn[0], ldc_fn[1], chn), I1)

        if p.fit_ttv:
            p_ttv_de = p.get_ttv(f_fn)
            p_ttv_fn = p.get_ttv(f_fn)

            info("%14.5f %14.5f  -  TTV amplitude [min]"%(p_ttv_de[0]*1440, p_ttv_fn[0]*1440), I1)
            info("%14.5f %14.5f  -  TTV period [d]"%(p_ttv_de[1], p_ttv_fn[1]), I1)


        info('Best-fit fitting parameters',H2)
        for idx, pn in enumerate(p.fitted_parameter_names):
            info('%10s %14.5f'%(pn, p.fitted_parameters[idx]))

        info('Fit statistics',H2)
        info('Differential evolution minimum %10.2f'%chi_de,I1)
        info('Downhill simplex minimum       %10.2f'%chi_fn,I1)
        info('')
        info("Akaike's information criterion %10.2f" %(chi_fn + 2*p.p_cur.size), I1)
        info('')

    if is_root:
        return MTFitResult(p, minfun, f_fn, p_fn, lcdata)
