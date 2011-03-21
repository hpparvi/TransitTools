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

from time import time
from cPickle import dump, load
from math import acos, cos, sin, asin, sqrt

from scipy.optimize import fmin
from numpy import array, linspace
from matplotlib import pyplot as pl

from transitLightCurve.core import *
from transitLightCurve.transitparameterization import TransitParameterization, parameters, parameterizations
from transitLightCurve.transitlightcurve import TransitLightcurve
from transitLightCurve.fitting.de import DiffEvol, MPIDiffEvol
from transitLightCurve.lightcurvedata import MultiTransitLC

from fitparameterization import MTFitParameterization
from fitnessfunction import FitnessFunction

class MTFitResult(FitResult):
    def __init__(self, fit_prm, res_ds, prm_ds, lcdata):
        self.n_channels = {'value':fit_prm.nch, 'description':'Number of channels'}
        self.ephemeris  = {'value':prm_ds.pv, 'description':'Ephemeris'}
        self.limb_darkening =  {'value':[[fit_prm.get_ldc(i, res_ds)] for i in range(fit_prm.nch)],
                                'description':'Limb darkening coefficients per channel.'}

        self.nch = self.n_channels['value']
        self.e   = self.ephemeris['value']
        self.ldc = self.limb_darkening['value']

    def get_fit(self):
        return self.res_ds

    def get_chi(self):
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
    info('Starting multitransit fitting',H1)

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
    totpoints = sum([d.time.size for d in lcdata]) 

    ## Generate the fitting parameterization
    ## -------------------------------------
    p = MTFitParameterization(bounds, stellar_prm, nchannels, lcdata[0].n_transits, **kwargs)
    ##FIXME: Having different number of transits for each channel breaks things up.

    info('Fitting data with',I1)
    info('%6i free parameters'%p.p_cur.size, I2)
    info('%6i channels'%nchannels, I2)
    info('%6i datapoints'%totpoints, I2)
    info('%6i levels of freedom'%(totpoints-p.p_cur.size), I2)
    info('')

    ## Setup the minimization function
    ## -------------------------------
    minfun = FitnessFunction(p, lcdata)

    #######################################################################################
    ##
    ## MINIMIZATION
    ## ============
    ##
    ## Global fitting using differential evolution
    ## -------------------------------------------
    fitter_g = DiffEvol(minfun, array([p.p_min,p.p_max]).transpose(), **de_pars)
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
    info('Best-fit parameters',H2)
    info("%14s %14s"%("DiffEvol" ,"FMin"),I1)
    nc = nchannels if p.separate_k2_ch else 1 
    nt = lcdata[0].n_transits if p.separate_zp_tr else 1
    for chn in range(nc):
        info("%14.5f %14.5f  -  radius ratio"%(p.get_physical(chn, f_de)[0],
                                               p.get_physical(chn, f_fn)[0]), I1)

    for chn in range(nchannels):
        for tr in range(nt):
            info("%14.5f %14.5f  -  zeropoint"%(p.get_zp(chn, tr, f_de),
                                                p.get_zp(chn, tr, f_fn)), I1)

    for i,k in enumerate(parameterizations['physical'][1:]):
        info("%14.5f %14.5f  -  %s" %(p_de[i+1], p_fn[i+1], parameters[k].description), I1)
    nc = nchannels if p.separate_ld else 1 
    for chn in range(nc):
        info("%14.5f %14.5f  -  limb darkening"%(p.get_ldc(chn, f_de)[0], p.get_ldc(chn, f_fn)[0]), I1)

    if p.fit_ttv:
        p_ttv_de = p.get_ttv(f_fn)
        p_ttv_fn = p.get_ttv(f_fn)

        info("%14.5f %14.5f  -  TTV amplitude [min]"%(p_ttv_de[0]*1440, p_ttv_fn[0]*1440), I1)
        info("%14.5f %14.5f  -  TTV period [d]"%(p_ttv_de[1]/TWO_PI, p_ttv_fn[1]/TWO_PI), I1)

    info('Fit statistics',H2)
    info('Differential evolution minimum %10.2f'%chi_de,I1)
    info('Downhill simplex minimum       %10.2f'%chi_fn,I1)
    info('')
    info("Akaike's information criterion %10.2f" %(chi_fn + 2*p.p_cur.size), I1)
    info('')

    return MTFitResult(p, f_fn, p_fn, lcdata)































#    import pylab as pl
#    for chn, (time,flux,pntn,sls) in enumerate(zip(times,fluxes,pntns,slices)):
#        pl.subplot(3,1,chn+1)
#        for trn, sl in enumerate(sls):
#            pl.plot(pntn[sl], flux[sl]/p.get_zp(chn, trn) - 1, '.', c='0')
#            pl.plot(pntn[sl], lc(time[sl], p.get_kipping(chn, f_fn), p.get_ldc(chn, f_fn)), lw=5, c='0')
#            pl.plot(pntn[sl], lc(time[sl], p.get_kipping(chn, f_fn), p.get_ldc(chn, f_fn)), lw=2, c='.9')
#        pl.ylim(-0.04, 0.01)
#             t = linspace(time[sl][0], time[sl][-1],200)
#             dt = t[-1]-t[0]
#             x  = (t-t[0])/dt
#             pl.plot((1.-x)*pntn[sl][0] + x*pntn[sl][-1], lc(t, p.get_kipping(chn, f_fn), p.get_ldc(chn, f_fn)), c='r')
    
#    pl.show()
    #exit()

#    lc = TransitLightcurve(TransitParameterization("kipping", p.p_k_min),
#                           method=method, mode='phase', ldpar=bounds['ld'][0])
#    ph = TWO_PI*linspace(-0.03,0.03,500)
#    for chn in range(nchannels):
#        pl.plot(ph, lc(ph, p.get_kipping(chn, r_fn[0]), p.get_ldc(chn, r_fn[0])))
#    pl.show()
#    exit()


