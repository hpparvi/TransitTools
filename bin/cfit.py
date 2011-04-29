"""Script to estimate the planetary parameters from multiple transit lightcurves over a variable star.
 
 A proof-of-concept script to simulate the CoRoT-7b lightcurve and 
 test the accuracy of the parameters estimated from the lightcurve.

 Author
   Hannu Parviainen <hpparvi@gmail.com>

 Date 
   15.01.2011
"""
from __future__ import division
import sys

from ConfigParser import ConfigParser
from optparse import OptionParser

import matplotlib.pyplot as pl
import numpy as np

from time import time
from matplotlib.backends.backend_pdf import PdfPages

from transitLightCurve.core import *
from transitLightCurve.utilities import bin, fold
from transitLightCurve.transitlightcurve import TransitLightcurve
from transitLightCurve.fitting.multitransitfitter import fit_multitransit, load_MTFitResult
from transitLightCurve.fitting.multitransitmcmc import MultiTransitMCMC
from transitLightCurve.transitparameterization import TransitParameterization

from transitLightCurve.fitting.fitparameterization import MTFitParameter
from transitLightCurve.fitting.mcmc import DrawGaussian
from transitLightCurve.fitting.mcmcprior import UniformPrior, JeffreysPrior

from transitLightCurve.io.corot import CoRoT_targets


from CoRoT import CoRoT
from CoRoT.CoRoT import import_as_MTLC, C01, C02, C05, C07, C08, C10, C11

def main():
    ##########################################################################################
    ##
    ## CONFIG AND OPTION PARSERS
    ## =========================

    op = OptionParser()
    cp = ConfigParser()

    op.add_option('-f', '--config-file', dest='config_file', type='str', default=None)
    op.add_option('', '--no-initial-fit', dest='do_initial_fit', action='store_false', default=True)
    op.add_option('', '--no-final-fit', dest='do_final_fit', action='store_false', default=True)
    op.add_option('', '--no-mcmc', dest='do_mcmc', action='store_false', default=True)

    opt, arg = op.parse_args()

    cp.add_section('General')
    cp.set('General','method',      'fortran')
    cp.set('General','n_threads',         '1')
    cp.set('General','do_initial_fit', 'True')
    cp.set('General','do_final_fit',   'True')
    cp.set('General','do_mcmc',        'True')

    cp.add_section('Target')
    cp.set('Target','name',    '')
    cp.set('Target','max_pts', '')

    cp.add_section('Initial DE')
    cp.set('Initial DE','npop',  '50')
    cp.set('Initial DE','ngen',  '50')
    cp.set('Initial DE','C',   '0.75')
    cp.set('Initial DE','F',   '0.25')

    cp.add_section('Final DE')
    cp.set('Final DE','npop',    '50')
    cp.set('Final DE','ngen',    '50')
    cp.set('Final DE','C',     '0.75')
    cp.set('Final DE','F',     '0.25')

    cp.add_section('Fitting')
    cp.set('Fitting','fit_radius_ratio',         'True')
    cp.set('Fitting','fit_transit_center',       'True')
    cp.set('Fitting','fit_period',               'True')
    cp.set('Fitting','fit_transit_width',       'False')
    cp.set('Fitting','fit_impact_parameter',     'True')
    cp.set('Fitting','fit_limb_darkening',       'True')
    cp.set('Fitting','fit_zeropoint',           'False')
    cp.set('Fitting','fit_ttv',                 'False')
    cp.set('Fitting','separate_k_per_channel',  'False')
    cp.set('Fitting','separate_zp_per_transit', 'False')
    cp.set('Fitting','separate_ld_per_channel',  'True')
    cp.set('Fitting','do_local_fit',             'True')

    cp.add_section('MCMC Parameters')
    cp.set('MCMC Parameters','k2','{"sigma":2e-4, "prior":UniformPrior(0.05**2, 0.15**2)}')
    cp.set('MCMC Parameters','tc','{"sigma":4e-4, "prior":UniformPrior(0.00, 1e8)}')
    cp.set('MCMC Parameters','p',' {"sigma":2e-5, "prior":UniformPrior(0.25, 1e2)}')
    cp.set('MCMC Parameters','b2','{"sigma":5e-3, "prior":UniformPrior(0.0, 0.99**2)}')
    cp.set('MCMC Parameters','u0','{"sigma":8e-2, "prior":UniformPrior(0.0, 1.0)}')
    cp.set('MCMC Parameters','u1','{"sigma":8e-2, "prior":UniformPrior(0.0, 1.0)}')
    cp.set('MCMC Parameters','u2','{"sigma":8e-2, "prior":UniformPrior(0.0, 1.0)}')

    cp.add_section('Fitting parameters')

    cp.read(opt.config_file)

    ct = C05

    ##########################################################################################
    ##
    ## INITIALIZATION
    ## ==============
    ##
    method = cp.get('General','method')
    maxpts = cp.getint('Target','max_pts') or None

    channels = ['w'] if ct in [C10] else ['r','g','b']

    de_pars = {'npop':cp.getint('Initial DE','npop'),
               'ngen':cp.getint('Initial DE','ngen'),
               'C':cp.getfloat('Initial DE','C'),
               'F':cp.getfloat('Initial DE','F')}

    ds_pars = {'ftol':1e-4, 'disp':False}

    do_fit  = cp.getboolean('General','do_initial_fit') and opt.do_initial_fit
    do_mcmc = cp.getboolean('General','do_mcmc') and opt.do_mcmc

    fit_pars = dict(cp.items('Fitting'))
    for p in fit_pars.items():
        fit_pars[p[0]] = p[1].lower() == 'true'

    initial_name = '%s_initial.pkl'%ct.basename
    np.set_printoptions(precision=5)
    np.random.seed(0)
    INF = 1e8

    ##########################################################################################
    ##
    ## READ IN COROT PLANET PARAMETERS
    ## ===============================
    ##

    fpars = {}
    for p, d in cp.items('Fitting parameters'):
        d = d.split(',')
        if p == 'tc':
            fpars[p] = MTFitParameter(p, [ct.tc+float(d[0]),ct.tc+float(d[1])])
        elif p == 'p':
            fpars[p] = MTFitParameter(p, [ct.p+float(d[0]), ct.p+float(d[1])])
        else:
            dsc = d[2].strip() if d[2].strip() != '-' else None
            uni = d[3].strip() if d[3].strip() != '-' else None
            fpars[p] = MTFitParameter(p, [float(d[0]),float(d[1])], description=dsc, units=uni)

    clean_pars = {'top':3, 'bottom':INF}
    w_period   = cp.getfloat('General','w_period')
    w_transit  = cp.getfloat('General','w_transit')
    phase_lim  = cp.getfloat('General','phase_lim')


    ##########################################################################################
    ##
    ## DEFINE HELPER FUNCTIONS
    ## =======================
    ##
    def load_corot_data(ct, chs=['r','g','b']):
        obsd = []
        for ch in chs:
            obsd.append(import_as_MTLC(ct, ch, w_period, w_transit, maxpts=maxpts,
                                       clean_pars=clean_pars,
                                       ps_period=CoRoT.orbit_period))

        return obsd

    def fit_corot(ct, data, fitparameters):
        return fit_multitransit(data, fitparameters,
                                ct.stellar_parameters,
                                de_pars=de_pars,
                                ds_pars=ds_pars,
                                method=method,
                                **fit_pars)

    ##########################################################################################
    ##
    ## LOAD DATA
    ## =========
    ##
    data = load_corot_data(ct, channels) if is_root else None
    if with_mpi:
        data = mpi_comm.bcast(data)
        
    ##########################################################################################
    ##
    ## MINIMIZATION
    ## ============
    ##
    if do_fit:
        mres = fit_corot(ct, data, fpars)
        if is_root:
            mres.save(initial_name)
        if with_mpi:
            mres = mpi_comm.bcast(mres)
    else:
        mres = load_MTFitResult(initial_name)

            
    ##########################################################################################
    ##
    ## MCMC
    ## ====
    ##
    if do_mcmc:
        mcmc_pars = {'n_steps':5000, 'n_chains': 1, 'seed':0, 'thinning':1, 'autotune_length':3000}
        initial_parameter_values = {'tc':mres.e[1], 'p':mres.e[2], 'ld':[0.568],
                                    'b2':mres.e[4]**2, 'zp':1.0}

        parameter_defs = {}
        for p, v in zip(mres.parameterization.fitted_parameter_names,
                        mres.parameterization.p_cur):

            pt = eval(cp.get('MCMC Parameters',p))
            
            parameter_defs[p] = {'start_value':v,
                                 'draw_function': DrawGaussian(pt['sigma']),
                                 'prior': pt['prior']}
  
        parameter_defs['error scale'] = {'start_value':1.0,
                                         'draw_function': DrawGaussian(0.01),
                                         'prior': JeffreysPrior(0.05, 2.0)}

        mcmc = MultiTransitMCMC(data, fpars, ct.stellar_parameters, parameter_defs, initial_parameter_values, mcmc_pars, **fit_pars)
        mcmcres = mcmc()

        pl.figure(1)
        for i in range(2):
            pl.subplot(3,1,i+1)
            pl.plot(np.sqrt(mcmcres.steps[0,::3,i]))
        pl.subplot(3,1,3)    
        pl.plot(mcmcres.steps[0,::3,2])

        pl.figure(2)
        for i in range(2):
            pl.subplot(2,1,i+1)
            pl.hist(np.sqrt(mcmcres.steps[0,::3,i]))

        pl.figure(3)
        for i in range(2):
            pl.subplot(2,1,i+1)
            pl.plot(np.sqrt(mcmcres.steps[0,::20,i]))

        pl.figure(4)
        for i in range(2):
            pl.subplot(2,1,i+1)
            pl.hist(np.sqrt(mcmcres.steps[0,::20,i]))

        pl.figure(5)
        for i in range(2):
            pl.subplot(2,1,i+1)
            d = np.sqrt(mcmcres.steps[0,:,i])
            pl.acorr(d-d.mean(), maxlags=100)

        print mcmcres.get_acceptance()
        pl.show()
        sys.exit()

    ##########################################################################################
    ##
    ## WRITE REPORT
    ## ============
    ##
    pp = PdfPages('%s_initial.pdf'%ct.basename)

    pl.figure(1)
    pl.figtext(0.5, 0.5, 'Transits', size='xx-large', ha='center')
    pp.savefig()

    figi = 2
    for i, d in enumerate(data):
        nt = d.n_transits
        npg = nt // 5
        for j in range(npg):
            d.plot(figi, j*5, (j+1)*5, pp)
            figi += 1
        d.plot(figi, npg*5, npg*5+nt%5, pp)
        figi += 1

    pl.figure(figi)
    pl.figtext(0.5, 0.5, 'Transits', size='xx-large', ha='center')
    pp.savefig()
    figi+=1

    tp = TransitParameterization('physical', mres.ephemeris)

    p  = [fold(d.get_time(), tp.pv[2], origo=tp.pv[1], shift=0.5)-0.5 for d in data]
    pm = [np.abs(tmp)<phase_lim for tmp in p]
    f  = [d.get_flux() for d in data]


    p  = [tmp[mask] for mask,tmp in zip(pm,p)] 
    f  = [tmp[mask] for mask,tmp in zip(pm,f)] 

    fmin = min([tmp.min() for tmp in f])
    fmax = max([tmp.max() for tmp in f])
    pmin = min([tmp.min() for tmp in p])
    pmax = max([tmp.max() for tmp in p])

    pb = []; fb = []; eb = []
    for i in range(len(data)):
        pbt, fbt, ebt = bin(p[i],f[i],200)
        pb.append(pbt)
        fb.append(fbt)
        eb.append(ebt)

    fbmin = min([tmp.min() for tmp in fb])
    fbmax = max([tmp.max() for tmp in fb])
    fbmrg = (fbmax - fbmin)*0.05

    for i, d in enumerate(data):
        pl.figure(figi)
        lc = TransitLightcurve(tp, ldpar=mres.ldc[i], method=method, mode='phase')
        p2 = np.linspace(p[i].min(),p[i].max(), 2000)
        pl.plot(p[i], f[i]*mres.zp[i],',', c='0')
        pl.plot(p2, lc(p2*2.*np.pi), lw=3, c='0')
        pl.plot(p2, lc(p2*2.*np.pi), lw=1, c='0.9')
        pl.ylim(fmin, fmax)
        pl.xlim(pmin, pmax)
        pp.savefig()
        figi+=1

    for i, d in enumerate(data):
        pl.figure(figi)
        lc = TransitLightcurve(tp, ldpar=mres.ldc[i], method=method, mode='phase')
        p2 = np.linspace(pb[i].min(),pb[i].max(), 2000)
        pl.errorbar(pb[i], fb[i]*mres.zp[i], eb[i], fmt='.', c='0')
        pl.plot(p2, lc(p2*2.*np.pi), lw=1.5, c='0')
        pl.plot(p2, lc(p2*2.*np.pi), lw=0.5, c='0.9')
        pl.ylim(fbmin-fbmrg, fbmax+fbmrg)
        pl.xlim(pmin, pmax)
        pp.savefig()
        figi+=1

    pp.close()

if __name__ == '__main__':
    main()
