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

from cPickle import dump, load

import pylab as pl

from numpy import abs, array, zeros, concatenate
from scipy.optimize import fmin
from math import acos, cos, sin, asin, sqrt
from sys import exit

from time import time

try:
    from scipy.constants import G
except ImportError:
    G = 6.67e-11

from transitLightCurve.core import *
from transitLightCurve.transitlightcurve import TransitLightcurve
from transitLightCurve.transitparameterization import TransitParameterization, generate_mapping, parameterizations, parameters
from transitLightCurve.fitting.de import DiffEvol
from transitLightCurve.lightcurvedata import MultiTransitLC

#TODO: parameter limits

class MTFitResult(FitResult):
    def __init__(self, res_ds, prm_ds, res_de, prm_de, obsdata, lc):
        self.res_ds  = res_ds[0]
        self.res_de  = res_de
        self.obsdata = obsdata

        self.ds_prm  = prm_ds
        self.ds_ldc  = res_ds[0][-1:]

        self.de_prm  = prm_de
        self.de_ldc  = res_de.get_fit()[-1:]

    def get_fit(self):
        return self.res_ds

    def get_chi(self):
        pass

    def save(self, filename):
        ds_t = self.ds_prm
        de_t = self.ds_prm
        of_t = self.obsdata.fit

        self.ds_prm = None
        self.de_prm = None
        self.obsdata.fit = None

        f = open(filename, 'wb')
        dump(self, f)
        f.close()

        self.ds_prm = ds_t
        self.de_prm = de_t
        self.obsdata.fit = of_t


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

    lcdata = lcdata if isinstance(lcdata, list) else [lcdata]

    nchannels = len(lcdata)

    totpoints = 0
    for d in lcdata:
        totpoints += d.time.size

    logging.info("Using %i points for fitting." %totpoints)

    method = 'python' if 'method' not in kwargs.keys() else kwargs['method'] 

    de_pars = {'npop':50, 'ngen':5, 'C':0.9, 'F':0.25}
    ds_pars = {}

    if 'de_pars' in kwargs.keys(): de_pars.update(kwargs['de_pars'])
    if 'ds_pars' in kwargs.keys(): ds_pars.update(kwargs['ds_pars'])

    ## Generate mappings
    ## =================
    map_p_to_k = generate_mapping("physical","kipping")
    map_k_to_p = generate_mapping("kipping","physical")

    ## Define differential evolution parameter boundaries
    ## ==================================================
    ## The boundaries are given as a dictionary with physical parameter boundaries. These are
    ## mapped to the Kipping parameterization with the relative semi-major axis set to unity.
    ##
    p_k_min  = map_p_to_k([bounds['k'][0], bounds['tc'][0], bounds['p'][0], 10, bounds['b'][0]])
    p_k_max  = map_p_to_k([bounds['k'][1], bounds['tc'][1], bounds['p'][1], 10, bounds['b'][1]])
    ##
    ## The final fitting parameter boundaries are obtained by excluding the transit width parameter
    ## from the fitting parameter set and adding the limb darkening parameters.
    ##
    n_ldc = len(bounds['ld'][0])
    p_min = concatenate([p_k_min[[0,1,2,4]],bounds['ld'][0]])
    p_max = concatenate([p_k_max[[0,1,2,4]],bounds['ld'][1]])

    ## Setup the fitting lightcurve
    ## ============================
    lc = TransitLightcurve(TransitParameterization("kipping", p_k_min), method=method, ldpar=bounds['ld'][0])

    ## Obtain the Kipping's transit width parameter and the semi-major axis using Kepler's third law
    ## =============================================================================================
    ac = ((G*stellar_prm['M']/TWO_PI**2)**(1/3)) / stellar_prm['R']
    def kipping_i(period, b2):
        a = ac * (d_to_s*period)**(2/3)
        it = TWO_PI/period/asin(sqrt(1-b2)/(a*sin(acos(sqrt(b2)/a))))
        return it

    ## The parameterization used in fitting differs from the normal Kipping parameterization,
    ## thus we need to translate between the fitting and Kipping parameterizations.
    ##
    def fitting_to_kipping(p_in, p_out=None):
        if p_out is None: p_out = zeros(5)
        p_out[[0,1,2,4]] = p_in[:-n_ldc]
        p_out[3] = kipping_i(p_in[2], p_in[3])
        return p_out

    ## Define the minimization function.
    ## =================================
    ## We use the normal Chi squared for convenience. Since we assume a constant 
    ## point-to-point scatter for the whole lightcurve, we can take the division 
    ## by the variance outside the sum.
    ##

    times  = [t.get_time() for t in lcdata]
    fluxes = [t.get_flux() for t in lcdata]
    ivars  = [t.ivar for t in lcdata]
    norms  = [1/(len(lcdata)*f.size) for f in fluxes]

    p_geom = zeros(5)
    def minfun(p_fit):
        if p_fit[0] < 0: return 1e18
        if p_fit[-n_ldc-1] < 0 or p_fit[-n_ldc-1] > 1: return 1e18
        if p_fit[-n_ldc] < 0 or p_fit[-n_ldc] > 1: return 1e18

        fitting_to_kipping(p_fit, p_geom)

        chi = 0.
        for time,flux,ivar,norm in zip(times,fluxes,ivars,norms):
            chi += ((flux - lc(time, p_geom, p_fit[-n_ldc:]))**2 * ivar).sum() * norm

        return chi

    ## Global fitting using differential evolution
    ## ============================================
    fitter_g = DiffEvol(minfun, array([p_min,p_max]).transpose(), **de_pars)
    r_de = fitter_g()

    ## Local fitting using downhill simplex 
    ## ====================================
    r_fn = fmin(minfun, r_de.get_fit(), full_output=1, **ds_pars)

    ## Map the fits represented in Kipping parameterization to physical parameterization
    ## =================================================================================
    p_de   = TransitParameterization('physical', map_k_to_p(fitting_to_kipping(r_de.get_fit())))
    p_fn   = TransitParameterization('physical', map_k_to_p(fitting_to_kipping(r_fn[0])))

    ## Update the multitransit lightcurve with the new transit solution
    ## ================================================================
    lc.update(fitting_to_kipping(r_fn[0]), r_fn[0][-n_ldc:])

    for d in lcdata:
        d.fit = {'parameterization':p_fn, 'ldc':r_fn[0][-n_ldc:]}
        d.tc = p_fn[1]
        d.p = p_fn[2]

    logging.info("%15s %15s"%("DiffEvol" ,"FMin"))
    for i,k in enumerate(parameterizations['physical']):
        logging.info("%15.5f %15.5f  -  %s" %(p_de[i], p_fn[i], parameters[k].description))
    logging.info("%15.5f %15.5f  -  Limb darkening"%(r_de.get_fit()[-1], r_fn[0][-1]))

    return MTFitResult(r_fn, p_fn, r_de, p_de, lcdata, lc)
