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

from numpy import abs, array, zeros, concatenate, any, tile, hstack, vstack, repeat, linspace
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

    

class MTFitParameterization(object):
    def __init__(self, bnds, stellar_prm, nch, **kwargs):
        logging.info('Initializing fitting parameterization')
        logging.info('-------------------------------------')
        self.nch = nch
        self.separate_k2 = kwargs.get('separate_k2', False)
        self.separate_ld = kwargs.get('separate_ld', False)

        logging.info('  Fitting %i channels' %nch)
        logging.info('    Separate transit depths per channel: %s' %self.separate_k2)
        logging.info('    Separate limb darkening coefficients per channel: %s' %self.separate_ld)

        self.n_k2  = nch if self.separate_k2 else 1
        self.n_lds = nch if self.separate_ld else 1
        self.n_ldc = len(bnds['ld'][0])

        ## [k2_ch0 k2_ch1 ... k2_chn tc p b2 ld1 ld2 ... ldn]
        ##
        self.id_k2 = self.nch*[0] if not self.separate_k2 else range(self.nch)
        self.id_tc = self.n_k2
        self.id_p  = self.n_k2 + 1
        self.id_b2 = self.n_k2 + 2
        self.id_ld = self.n_k2 + 3

        if self.separate_ld:
            self.id_ldc = [[self.id_ld+self.n_ldc*i, self.id_ld+self.n_ldc*(i+1)] for i in range(self.nch)]
        else:
            self.id_ldc = self.nch*[[self.id_ld, self.id_ld+self.n_ldc]]
        
        self.ac = ((G*stellar_prm['M']/TWO_PI**2)**(1/3)) / stellar_prm['R']

        ## Generate mappings
        ## =================
        self.map_p_to_k = generate_mapping("physical","kipping")
        self.map_k_to_p = generate_mapping("kipping","physical")

        ## Define differential evolution parameter boundaries
        ## ==================================================
        ## The boundaries are given as a dictionary with physical parameter boundaries. These are
        ## mapped to the Kipping parameterization with the relative semi-major axis set to unity.
        ##
        self.p_k_min  = self.map_p_to_k([bnds['k'][0], bnds['tc'][0], bnds['p'][0], 10, bnds['b'][0]])
        self.p_k_max  = self.map_p_to_k([bnds['k'][1], bnds['tc'][1], bnds['p'][1], 10, bnds['b'][1]])
        ##
        ## The final fitting parameter boundaries are obtained by excluding the transit width parameter
        ## from the fitting parameter set and adding the limb darkening parameters.
        ##
        self.p_min = concatenate([repeat(self.p_k_min[0], self.n_k2),
                                  self.p_k_min[[1,2,4]],
                                  tile(bnds['ld'][0], self.n_lds)])

        self.p_max = concatenate([repeat(self.p_k_max[0], self.n_k2),
                                  self.p_k_max[[1,2,4]],
                                  tile(bnds['ld'][1], self.n_lds)])

        self.p_cur = vstack([self.p_min, self.p_max]).mean(0)

        ## Setup hard parameter limits 
        ## ===========================
        self.l_min = concatenate([repeat(0.0, self.n_k2),
                                 repeat(0.0, 3),
                                 repeat(-2.0, self.n_lds*self.n_ldc)])

        self.l_max = concatenate([repeat(0.3**2, self.n_k2),
                                 array([1e18, 1e5, 0.95]),
                                 repeat(2.0, self.n_lds*self.n_ldc)])

        logging.info('  Total length of the parameter vector: %i' %self.p_cur.size)
        logging.info('')

    def update(self, pv):
        self.p_cur = pv

    def is_inside_limits(self):
        if any(self.p_cur < self.l_min) or any(self.p_cur > self.l_max):
            return False
        else:
            return True

    def kipping_i(self, period, b2):
        ## Obtain the Kipping's transit width parameter and the semi-major axis using Kepler's third law
        ## =============================================================================================
        a = self.ac * (d_to_s*period)**(2/3)
        it = TWO_PI/period/asin(sqrt(1-b2)/(a*sin(acos(sqrt(b2)/a))))
        return it

    def get_physical(self, ch=0, p_in=None):
        return self.map_k_to_p(self.get_kipping(ch, p_in))

    def get_kipping(self, ch=0, p_in=None, p_out=None):
        if p_in is not None: self.update(p_in)
        if p_out is None: p_out = zeros(5)
        p_out[0] = self.p_cur[self.id_k2[ch]]
        p_out[[1,2,4]] = self.p_cur[self.id_tc:self.id_b2+1]
        p_out[3] = self.kipping_i(self.p_cur[self.id_p], self.p_cur[self.id_b2])
        return p_out

    def get_ldc(self, ch=0, p_in=None):
        if p_in is not None: self.update(p_in)
        return self.p_cur[self.id_ldc[ch][0] : self.id_ldc[ch][1]]


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
    logging.info('Starting multitransit fitting')
    logging.info('=============================')

    de_pars = {'npop':50, 'ngen':5, 'C':0.9, 'F':0.25}
    ds_pars = {}

    de_pars.update(kwargs.get('de_pars',{}))
    ds_pars.update(kwargs.get('ds_pars',{}))

    method = kwargs.get('method', 'python')
    sep_ch = kwargs.get('separate_channels', False) 
    
    lcdata    = lcdata if isinstance(lcdata, list) else [lcdata]
    nchannels = len(lcdata)
    totpoints = sum([d.time.size for d in lcdata]) 

    ## Generate the fitting parameterization
    ## =====================================
    p = MTFitParameterization(bounds, stellar_prm, nchannels, separate_k2=sep_ch, separate_ld=True)

    logging.info('  Fitting data with')
    logging.info('    %6i free parameters'%p.p_cur.size)
    logging.info('    %6i channels'%nchannels)
    logging.info('    %6i datapoints'%totpoints)
    logging.info('    %6i levels of freedom'%(totpoints-p.p_cur.size))
    logging.info('')

    ## Setup the fitting lightcurve
    ## ============================
    lc = TransitLightcurve(TransitParameterization("kipping", p.p_k_min),
                           method=method, ldpar=bounds['ld'][0])

    ## Define the minimization function.
    ## =================================
    ## We use the normal Chi squared for convenience. Since we assume a constant 
    ## point-to-point scatter for the whole lightcurve, we can take the division 
    ## by the variance outside the sum.
    ##
    times  = [t.get_time() for t in lcdata]
    fluxes = [t.get_flux() for t in lcdata]
    ivars  = [t.ivar       for t in lcdata]
    norms  = [1/(len(lcdata)*f.size) for f in fluxes]

    p_geom = zeros(5)
    def minfun(p_fit):
        p.update(p_fit)
        if p.is_inside_limits():
            chi = 0.
            for chn, (time,flux,ivar,norm) in enumerate(zip(times,fluxes,ivars,norms)):
                chi += ((flux - lc(time, p.get_kipping(chn), p.get_ldc(chn)))**2 * ivar).sum()
            return chi
        else:
            return 1e18

    ## Global fitting using differential evolution
    ## ============================================
    fitter_g = DiffEvol(minfun, array([p.p_min,p.p_max]).transpose(), **de_pars)
    r_de = fitter_g()

    ## Local fitting using downhill simplex 
    ## ====================================
    r_fn = fmin(minfun, r_de.get_fit(), full_output=1, **ds_pars)

    ## Map the fits represented in Kipping parameterization to physical parameterization
    ## =================================================================================
    p_de   = TransitParameterization('physical', p.get_physical(0, r_de.get_fit()))
    p_fn   = TransitParameterization('physical', p.get_physical(0, r_fn[0]))

    ## Update the multitransit lightcurve with the new transit solution
    ## ================================================================
    lc.update(p.get_physical(0, r_fn[0]), p.get_ldc(0))

    chi_sqr = r_fn[1]

    for d in lcdata:
        d.fit = {'parameterization':p_fn, 'ldc':p.get_ldc(0, r_fn[0])}
        d.tc = p_fn[1]
        d.p = p_fn[2]

    logging.info("%15s %15s"%("DiffEvol" ,"FMin"))

    for chn in range(nchannels):
        logging.info("%15.5f %15.5f  -  radius ratio"%(p.get_physical(chn, r_de.get_fit())[0],
                                                       p.get_physical(chn, r_fn[0])[0]))
    for i,k in enumerate(parameterizations['physical'][1:]):
        logging.info("%15.5f %15.5f  -  %s" %(p_de[i+1], p_fn[i+1], parameters[k].description))

    for chn in range(nchannels):
        logging.info("%15.5f %15.5f  -  limb darkening"%(p.get_ldc(chn, r_de.get_fit())[0], p.get_ldc(chn, r_fn[0])[0]))


    logging.info('')
    logging.info("Akaike's information criterion %10.2f" %(chi_sqr + 2*p.p_cur.size))
    logging.info('')

#    import pylab as pl
#    lc = TransitLightcurve(TransitParameterization("kipping", p.p_k_min),
#                           method=method, mode='phase', ldpar=bounds['ld'][0])
#    ph = TWO_PI*linspace(-0.03,0.03,500)
#    for chn in range(nchannels):
#        pl.plot(ph, lc(ph, p.get_kipping(chn, r_fn[0]), p.get_ldc(chn, r_fn[0])))
#    pl.show()
#    exit()

    return MTFitResult(r_fn, p_fn, r_de, p_de, lcdata, lc)
