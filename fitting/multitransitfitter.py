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

from numpy import abs, array, zeros, ones, concatenate, any, tile, vstack, repeat, linspace, arange
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
from transitLightCurve.fitting.de import DiffEvol, MPIDiffEvol
from transitLightCurve.lightcurvedata import MultiTransitLC


class MTFitResult(FitResult):
    def __init__(self, fit_prm, res_ds, prm_ds, lcdata):
        self.nch = fit_prm.nch
        self.ephemerisis = prm_ds.pv
        self.limb_darkening = [[fit_prm.get_ldc(i, res_ds)] for i in range(self.nch)]
        self.lcdata = lcdata

        self.e   = self.ephemerisis
        self.ldc = self.limb_darkening

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

    

class MTFitParameterization(object):
    def __init__(self, bnds, stellar_prm, nch, ntr, **kwargs):
        info('Initializing fitting parameterization', H2)

        self.nch = nch

        self.fit_radius_ratio     = kwargs.get('fit_radius_ratio',     True)
        self.fit_transit_center   = kwargs.get('fit_transit_center',   True)
        self.fit_period           = kwargs.get('fit_period',           True)
        self.fit_impact_parameter = kwargs.get('fit_impact_parameter', True)
        self.fit_limb_darkening   = kwargs.get('fit_limb_darkening',   True)
        self.fit_zeropoint        = kwargs.get('fit_zeropoint',        True)
        self.fit_ttvs             = kwargs.get('fit_ttvs',            False)

        self.separate_k2_ch = kwargs.get('separate_k_per_channel',    False)
        self.separate_zp_tr = kwargs.get('separate_zp_per_transit',   False)
        self.separate_ld    = kwargs.get('separate_ld_per_channel',   False)

        self.constant_parameters = []
        self.fitted_parameters   = []

        for k in kwargs.keys():
            if 'fit_' in k:
                info(k.replace('_',' ')+': %s'%kwargs[k], I1)

        info('Separate radius ratio per channel: %s' %self.separate_k2_ch, I1)
        info('Separate zeropoint per transit: %s' %self.separate_zp_tr, I1)
        info('Separate limb darkening per channel: %s' %self.separate_ld, I1)
        info('Fit TTVs: %s' %self.fit_ttvs, I1)

        self.n_k2  = nch if self.separate_k2_ch else 1
        self.n_zp  = nch if not self.separate_zp_tr else nch*ntr
        self.n_lds = nch if self.separate_ld else 1
        self.n_ldc = len(bnds['ld'][0])
        
        ## Setup the parameter index tables
        ## ================================
        ##
        self.id_k2 = arange(nch) if self.separate_k2_ch else zeros(nch)

        if not self.separate_zp_tr:
            self.id_zp = repeat(arange(nch),ntr).reshape([nch,ntr]) + self.n_k2
        else:
            self.id_zp = arange(nch*ntr).reshape([nch,ntr]) + self.n_k2

        self.id_tc = self.id_zp.ravel()[-1] + 1 
        self.id_p  = self.id_tc + 1
        self.id_b2 = self.id_p  + 1
        self.id_ld = self.id_b2 + 1

        if self.separate_ld:
            self.id_ldc = [[self.id_ld+self.n_ldc*i, self.id_ld+self.n_ldc*(i+1)] for i in range(self.nch)]
        else:
            self.id_ldc = self.nch*[[self.id_ld, self.id_ld+self.n_ldc]]
        
        if self.fit_ttvs:
            self.id_ttv_a = self.id_ldc[-1][-1]
            self.id_ttv_p = self.id_ttv_a + 1

        ## Blah
        ## ====
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
                                  repeat(bnds['zp'][0], self.n_zp),
                                  self.p_k_min[[1,2,4]],
                                  tile(bnds['ld'][0], self.n_lds)])

        self.p_max = concatenate([repeat(self.p_k_max[0], self.n_k2),
                                  repeat(bnds['zp'][1], self.n_zp),
                                  self.p_k_max[[1,2,4]],
                                  tile(bnds['ld'][1], self.n_lds)])

        if self.fit_ttvs:
            self.p_min = concatenate([self.p_min,
                                      [bnds['ttv_amplitude'][0]],
                                      [bnds['ttv_period'][0]]])

            self.p_max = concatenate([self.p_max,
                                      [bnds['ttv_amplitude'][1]],
                                      [bnds['ttv_period'][1]]])
                                     

        self.p_cur = vstack([self.p_min, self.p_max]).mean(0)

        ## Setup hard parameter limits 
        ## ===========================
        self.l_min = concatenate([repeat(0.0, self.n_k2),
                                 repeat(0.5, self.n_zp),
                                 repeat(0.0, 3),
                                 repeat(0.0, self.n_lds*self.n_ldc)])

        self.l_max = concatenate([repeat(0.3**2, self.n_k2),
                                 repeat(1.5, self.n_zp),
                                 array([1e18, 1e5, 0.95]),
                                 repeat(1.0, self.n_lds*self.n_ldc)])


        if self.fit_ttvs:
            self.l_min = concatenate([self.l_min, [-1, 1]])
            self.l_max = concatenate([self.l_max, [ 1, 1e4]])


        info('Parameter vector length: %i' %self.p_cur.size, I1)
        info('')

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

    def get_zp(self, ch=0, tn=0, p_in=None):
        if p_in is not None: self.update(p_in)
        return self.p_cur[self.id_zp[ch,tn]]

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

    def get_ttv(self, p_in=None):
        if p_in is not None: self.update(p_in)
        return self.p_cur[self.id_ttv_a], self.p_cur[self.id_ttv_p]


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

    de_pars = {'npop':50, 'ngen':5, 'C':0.9, 'F':0.25}
    ds_pars = {}

    de_pars.update(kwargs.get('de_pars',{}))
    ds_pars.update(kwargs.get('ds_pars',{}))

    do_local_fit = kwargs.get('do_local_fit', True)
    method = kwargs.get('method', 'python')
    sep_k  = kwargs.get('separate_k_per_channel', False) 
    sep_zp = kwargs.get('separate_zp_per_transit', False) 
    sep_ld = kwargs.get('separate_ld_per_channel', False)
    fit_ttv= kwargs.get('fit_ttvs', False)

    lcdata    = lcdata if isinstance(lcdata, list) else [lcdata]
    nchannels = len(lcdata)
    totpoints = sum([d.time.size for d in lcdata]) 

    ## Generate the fitting parameterization
    ## =====================================
    p = MTFitParameterization(bounds, stellar_prm, nchannels, lcdata[0].n_transits, **kwargs)
    ##FIXME: Having different number of transits for each channels breaks things up.

    info('Fitting data with',I1)
    info('%6i free parameters'%p.p_cur.size, I2)
    info('%6i channels'%nchannels, I2)
    info('%6i datapoints'%totpoints, I2)
    info('%6i levels of freedom'%(totpoints-p.p_cur.size), I2)
    info('')

    ## Setup the fitting lightcurve
    ## ============================
    lc = TransitLightcurve(TransitParameterization("kipping", p.p_k_min),
                           method=method, ldpar=bounds['ld'][0],
                           zeropoint=0.)

    ## Define the minimization function
    ## ================================
    times  = [t.get_time() for t in lcdata]
    fluxes = [t.get_flux() for t in lcdata]
    ivars  = [t.ivar       for t in lcdata]
    pntns  = [t.pntn       for t in lcdata]
    slices = [t.get_transit_slices() for t in lcdata]
    p_geom = zeros(5)

    def minfun_per_transit(p_fit):
        p.update(p_fit)
        if p.is_inside_limits():
            chi = 0.
            for chn, (time,flux,ivar,sls) in enumerate(zip(times,fluxes,ivars,slices)):
                for trn, sl in enumerate(sls):
                    chi += ((flux[sl]/p.get_zp(chn,trn) - 1. -
                             lc(time[sl], p.get_kipping(chn), p.get_ldc(chn)))**2 * ivar[sl]).sum()
            return chi
        else:
            return 1e18


    def minfun_per_transit_ttv(p_fit):
        p.update(p_fit)
        if p.is_inside_limits():
            p_ttv = p.get_ttv()
            chi = 0.
            for chn, (time,flux,ivar,sls) in enumerate(zip(times,fluxes,ivars,slices)):
                for trn, sl in enumerate(sls):
                    tp = p.get_kipping(chn)
                    epoch = round((time[sl].mean() - tp[1])/tp[2])
                    t_center = epoch*tp[2]
                    t_ttv = p_ttv[0]*sin(p_ttv[0]*t_center)
                    chi += ((flux[sl]/p.get_zp(chn,0) - 1. -
                             lc(time[sl]+t_ttv, p.get_kipping(chn), p.get_ldc(chn)))**2 * ivar[sl]).sum()
            return chi
        else:
            return 1e18

    def minfun_basic(p_fit):
        p.update(p_fit)
        if p.is_inside_limits():
            chi = 0.
            for chn, (time,flux,ivar) in enumerate(zip(times,fluxes,ivars)):
                chi += ((flux/p.get_zp(chn,0) - 1. - lc(time, p.get_kipping(chn), p.get_ldc(chn)))**2 * ivar).sum()
            return chi
        else:
            return 1e18

    if fit_ttv:
        minfun = minfun_per_transit_ttv
    else:
        minfun = minfun_per_transit if sep_zp else minfun_basic

    ## Global fitting using differential evolution
    ## ============================================
    fitter_g = DiffEvol(minfun, array([p.p_min,p.p_max]).transpose(), **de_pars)
    r_de = fitter_g()
    f_de    = r_de.get_fit()
    chi_de  = r_de.get_chi()
    r_fn    = None 
    f_fn    = f_de
    chi_fn  = chi_de

    ## Local fitting using downhill simplex 
    ## ====================================
    if do_local_fit:
        r_fn    = fmin(minfun, r_de.get_fit(), full_output=1, **ds_pars)
        f_fn    = r_fn[0]
        chi_fn  = r_fn[1]

    ## Map the fits represented in Kipping parameterization to physical parameterization
    ## =================================================================================
    p_de   = TransitParameterization('physical', p.get_physical(0, f_de))
    p_fn   = TransitParameterization('physical', p.get_physical(0, f_fn))

    ## Update the multitransit lightcurve with the new transit solution
    ## ================================================================
    lc.update(p.get_physical(0, f_fn), p.get_ldc(0))


    ## Print parameters and statistics
    ## ===============================
    info('Best-fit parameters',H2)
    info("%14s %14s"%("DiffEvol" ,"FMin"),I1)
    nc = nchannels if sep_k else 1 
    nt = lcdata[0].n_transits if sep_zp else 1
    for chn in range(nc):
        info("%14.5f %14.5f  -  radius ratio"%(p.get_physical(chn, f_de)[0],
                                               p.get_physical(chn, f_fn)[0]), I1)

    for chn in range(nchannels):
        for tr in range(nt):
            info("%14.5f %14.5f  -  zeropoint"%(p.get_zp(chn, tr, f_de),
                                                p.get_zp(chn, tr, f_fn)), I1)

    for i,k in enumerate(parameterizations['physical'][1:]):
        info("%14.5f %14.5f  -  %s" %(p_de[i+1], p_fn[i+1], parameters[k].description), I1)
    nc = nchannels if sep_ld else 1 
    for chn in range(nc):
        info("%14.5f %14.5f  -  limb darkening"%(p.get_ldc(chn, f_de)[0], p.get_ldc(chn, f_fn)[0]), I1)

    if p.fit_ttvs:
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

    import pylab as pl
    for chn, (time,flux,pntn,sls) in enumerate(zip(times,fluxes,pntns,slices)):
        pl.subplot(3,1,chn+1)
        for trn, sl in enumerate(sls):
            pl.plot(pntn[sl], flux[sl]/p.get_zp(chn, trn) - 1, '.', c='0')
            pl.plot(pntn[sl], lc(time[sl], p.get_kipping(chn, f_fn), p.get_ldc(chn, f_fn)), lw=5, c='0')
            pl.plot(pntn[sl], lc(time[sl], p.get_kipping(chn, f_fn), p.get_ldc(chn, f_fn)), lw=2, c='.9')
        pl.ylim(-0.04, 0.01)
#             t = linspace(time[sl][0], time[sl][-1],200)
#             dt = t[-1]-t[0]
#             x  = (t-t[0])/dt
#             pl.plot((1.-x)*pntn[sl][0] + x*pntn[sl][-1], lc(t, p.get_kipping(chn, f_fn), p.get_ldc(chn, f_fn)), c='r')
    
    pl.show()
    #exit()

#    lc = TransitLightcurve(TransitParameterization("kipping", p.p_k_min),
#                           method=method, mode='phase', ldpar=bounds['ld'][0])
#    ph = TWO_PI*linspace(-0.03,0.03,500)
#    for chn in range(nchannels):
#        pl.plot(ph, lc(ph, p.get_kipping(chn, r_fn[0]), p.get_ldc(chn, r_fn[0])))
#    pl.show()
#    exit()

    return MTFitResult(p, f_fn, p_fn, lcdata)
