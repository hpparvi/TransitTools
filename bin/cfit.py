#!/usr/bin/env python
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
import cPickle
import numpy as np
from time import time
from optparse import OptionParser
from os.path import join, exists

##########################################################################################
##
## OPTION PARSER
## ==============
op = OptionParser()
op.add_option('-f', '--config-file', dest='config_file', type='str', default=None)
op.add_option('', '--no-initial-fit', dest='do_initial_fit', action='store_false', default=True)
op.add_option('', '--no-final-fit', dest='do_final_fit', action='store_false', default=True)
op.add_option('', '--do-global-fit', dest='do_global_fit', action='store_true', default=False)
op.add_option('', '--no-mcmc', dest='do_mcmc', action='store_false', default=True)
op.add_option('', '--load-lcdata', dest='load_lcdata', action='store_true', default=False)
op.add_option('', '--save-lcdata', dest='save_lcdata', action='store_true', default=False)
op.add_option('', '--plot-transits', dest='plot_transits', action='store_true', default=False)
op.add_option('', '--no-gui', dest='use_curses', action='store_false', default=True)
op.add_option('', '--use-Agg', dest='use_agg', action='store_true', default=False)

opt, arg = op.parse_args()

import matplotlib as mpl
if opt.use_agg:
    mpl.use('Agg')
import matplotlib.pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages

from transitLightCurve.core import *
from transitLightCurve.utilities import bin, fold
from transitLightCurve.transitlightcurve import TransitLightcurve
from transitLightCurve.fitting.multitransitfitter import fit_multitransit, load_MTFitResult
from transitLightCurve.fitting.multitransitmcmc import MultiTransitMCMC
from transitLightCurve.transitparameterization import TransitParameterization

from transitLightCurve.fitting.fitparameterization import MTFitParameter
from transitLightCurve.fitting.mcmc import DrawGaussian
from transitLightCurve.fitting.mcmcprior import UniformPrior, GaussianPrior, JeffreysPrior, InverseSqrtPrior, LinCombPrior
from transitLightCurve.fitting.fitparameterization import MTFitParameterization

import transitLightCurve.io.corot as CoRoT
from transitLightCurve.io.corot import CoRoT_targets, import_as_MTLC

from config import cp

def main():
    ##########################################################################################
    ##
    ## CONFIG AND OPTION PARSERS
    ## =========================

    cp.read(opt.config_file)

    ##########################################################################################
    ##
    ## INITIALIZATION
    ## ==============
    ##
    ## General Options
    ## ---------------
    basename = cp.get('General','basename')
    method   = cp.get('General','method')
    targetid = CoRoT_targets[int(cp.get('General','name')[1:])]

    maxpts   = cp.getint('Target','max_pts') or None

    fn_data    = '{}_00_lc.pkl'.format(basename)
    fn_initial = '{}_01_if.pkl'.format(basename)
    fn_final   = '{}_02_ff.pkl'.format(basename)
    fn_mcmc    = '{}_03_mc.pkl'.format(basename)

    do_global_fit = not exists(fn_initial) or opt.do_global_fit

    de_init_pars = {'npop':cp.getint('Initial DE','npop'),
                    'ngen':cp.getint('Initial DE','ngen'),
                    'C':cp.getfloat('Initial DE','C'),
                    'F':cp.getfloat('Initial DE','F')}

    de_final_pars = {'npop':cp.getint('Final DE','npop'),
                     'ngen':cp.getint('Final DE','ngen'),
                     'C':cp.getfloat('Final DE','C'),
                     'F':cp.getfloat('Final DE','F')}

    ds_pars = {'ftol':1e-4, 'disp':False}

    combine_channels = cp.getboolean('General','combine_channels')
    do_initial_fit  = cp.getboolean('General','do_initial_fit') and opt.do_initial_fit
    do_final_fit  = cp.getboolean('General','do_final_fit') and opt.do_final_fit
    do_mcmc = cp.getboolean('General','do_mcmc') and opt.do_mcmc

    fit_pars = dict(cp.items('Fitting'))
    for p in fit_pars.items():
        if p[1].lower() == 'true': fit_pars[p[0]] = True
        elif p[1].lower() == 'false': fit_pars[p[0]] = False
        else:
            try:
                fit_pars[p[0]] = float(p[1])
            except ValueError:
                fit_pars[p[0]] = p[1]

    fit_pars['n_threads'] = cp.getint('General','n_threads')

    np.set_printoptions(precision=5)
    np.random.seed(0)
    INF = 1e8

    ##########################################################################################
    ##
    ## DEFINE HELPER FUNCTIONS
    ## =======================
    ##
    def load_corot_data(targetid):
        return import_as_MTLC(targetid, w_period, w_transit, maxpts=maxpts,
                              clean_pars=clean_pars,
                              ps_period=CoRoT.orbit_period,
                              combine_channels=combine_channels)


    def fit_corot(targetid, data, parameterization, de_pars):
        return fit_multitransit(data, parameterization,
                                de_pars=de_pars,
                                ds_pars=ds_pars,
                                method=method,
                                **fit_pars)

    ##########################################################################################
    ##
    ## LOAD DATA
    ## =========
    ##
    clean_pars = {'top':3, 'bottom':10}
    w_period   = cp.getfloat('General','w_period')
    w_transit  = cp.getfloat('General','w_transit')
    phase_lim  = cp.getfloat('General','phase_lim')

    if not opt.load_lcdata:
        data = load_corot_data(targetid) if is_root else None
        if opt.save_lcdata and is_root:
            f = open(fn_data, 'w')
            cPickle.dump(data, f)
            f.close()
    elif is_root:
        f = open(fn_data, 'r')
        data = cPickle.load(f)
        f.close()

    if with_mpi:
        data = mpi_comm.bcast(data)
       

    ##########################################################################################
    ##
    ## DEFINE PARAMETERS
    ## =================
    parameter_defs = {}
    for p in cp.items('MCMC Parameters'):
        parameter_defs[p[0]] = eval(p[1])

    def addpar(name, pdef):
        if name not in parameter_defs.keys():
            parameter_defs[name] = pdef

    for ch in range(len(data)):
        if 'error {}'.format(ch) not in parameter_defs.keys():
            std = data[ch].get_std().mean() 
            parameter_defs['error {}'.format(ch)] = {"sigma":1e-4, 'free':True, 'value':std, 'use':'om', "prior":UniformPrior(1e-1*std, 1e1*std), 'description':'Error {}'.format(ch), 'units':'-'}
            
        addpar('zp {}'.format(ch), {"sigma":1e-4, 'free':True, 'value':1., 'use':'om', "prior":UniformPrior(0.995, 1.005), 'description':'Zeropoint', 'units':'-'})

    addpar('b',  {"sigma":1e-2, 'free':True, 'value':0.5, 'use':'om', "prior":UniformPrior(0.0, 1.0), 'description':'Impact parameter', 'units':'-'})
    addpar('tc', {"sigma":1e-4, 'free':True, 'value':targetid.tc, 'use':'om', "prior":UniformPrior(targetid.tc-2e-1, targetid.tc+2e-1), 'description':'Transit center', 'units':'d'})
    addpar('p',  {"sigma":1e-4, 'free':True, 'value':targetid.p, 'use':'om', "prior":UniformPrior(targetid.p-1e-1, targetid.p+1e-1), 'description':'Period', 'units':'d'})

    mcmc_pars = {'n_steps':cp.getint('MCMC','nsteps'),
                 'n_chains':cp.getint('MCMC','nchains'),
                 'seed':cp.getint('MCMC','seed'),
                 'thinning':cp.getint('MCMC','thinning'),
                 'autotune_length':cp.getint('MCMC','autotune_length'),
                 'autotune_interval':cp.getint('MCMC','autotune_interval'),
                 'monitor_interval':cp.getint('MCMC','monitor_interval'),
                 'autosave_interval':cp.getint('MCMC','autosave_interval'),
                 'autosave_filename':cp.get('MCMC','autosave_filename'),
                 'use_curses':opt.use_curses}


    ##########################################################################################
    ##
    ## MINIMIZATION
    ## ============
    ##
    parameterization = MTFitParameterization(parameter_defs, len(data), data[0].n_transits,
                                             mode='o', **fit_pars)
    ## Initial fit
    ## ===========
    if do_global_fit:
        mres = fit_corot(targetid, data, parameterization, de_init_pars)
        if is_root:  mres.save(fn_initial)
        if with_mpi: mres = mpi_comm.bcast(mres)
    else:
        mres = load_MTFitResult(fn_initial)

    ## Clean the data using the initial fit
    ## ====================================
    if is_root:
        if opt.plot_transits: plot_transits(500, data, targetid, '%s_initial_transits.pdf'%targetid.basename)
        #tp = TransitParameterization('physical', mres.ephemeris)
        #for i, d in enumerate(data):
        #    lc = TransitLightcurve(tp, ldpar=mres.ldc[i], method='fortran', mode='time')
        #    d.clean_with_lc(lc, top=5., bottom=12.)
        #if opt.plot_transits: plot_transits(600, data, targetid, '%s_cleaned_transits.pdf'%targetid.basename, lc)

    #if with_mpi:
    #    data = mpi_comm.bcast(data)

    ## Final fit
    ## ==========
    #if do_final_fit:
    #    mres = fit_corot(targetid, data, parameterization, de_final_pars)
    #    if is_root: mres.save(fn_final)
    #    if with_mpi: mres = mpi_comm.bcast(mres)
    #else:
    #    mres = load_MTFitResult(fn_final)


    ##########################################################################################
    ##
    ## MCMC
    ## ====
    ##
    if do_mcmc:
        parameterization = parameterization.mcmc_from_opt(mres.pv)
        mcmc = MultiTransitMCMC(data, parameterization, mcmc_pars, **fit_pars)
        mcmcres = mcmc()
        mcmcres.save(cp.get('MCMC','file'))

        # pl.figure(1)
        # for i in range(2):
        #     pl.subplot(3,1,i+1)
        #     pl.plot(np.sqrt(mcmcres.steps[0,::3,i]))
        # pl.subplot(3,1,3)    
        # pl.plot(mcmcres.steps[0,::3,2])

        # pl.figure(2)
        # for i in range(2):
        #     pl.subplot(2,1,i+1)
        #     pl.hist(np.sqrt(mcmcres.steps[0,::3,i]))

        # pl.figure(3)
        # for i in range(2):
        #     pl.subplot(2,1,i+1)
        #     pl.plot(np.sqrt(mcmcres.steps[0,::20,i]))

        # pl.figure(4)
        # for i in range(2):
        #     pl.subplot(2,1,i+1)
        #     pl.hist(np.sqrt(mcmcres.steps[0,::20,i]))

        # pl.figure(5)
        # for i in range(2):
        #     pl.subplot(2,1,i+1)
        #     d = np.sqrt(mcmcres.steps[0,:,i])
        #     pl.acorr(d-d.mean(), maxlags=100)

        # print mcmcres.get_acceptance()
        # pl.show()
        # sys.exit()

    ##########################################################################################
    ##
    ## WRITE REPORT
    ## ============
    ##
    pp = PdfPages('%s_initial.pdf'%targetid.basename)

    figi = 10
    slices = [t.get_transit_slices()     for t in data]
    tnumbs = []
    for i, ch in enumerate(data):
        tnumbs.append(np.zeros(ch.get_time().size))
        for sl, tn in zip(slices[i], [t.number for t in ch.transits]):
            tnumbs[i][sl] = tn

    ttv_a    = mres.parameterization.parameter_view.get('ttv a', None)
    ttv_p    = mres.parameterization.parameter_view.get('ttv p', None)

    tp = TransitParameterization('physical', mres.ephemeris)

    if ttv_a is None:
        p  = [fold(d.get_time(), tp.pv[2], origo=tp.pv[1], shift=0.5)-0.5 for d in data]
    else:
        p  = [fold(d.get_time() + ttv_a*np.sin(TWO_PI*ttv_p * tp.pv[2]*tnumbs[i]), tp.pv[2], origo=tp.pv[1],
                   shift=0.5)-0.5 for i,d in enumerate(data)]

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
        pbt, fbt, ebt = bin(p[i],f[i],50)
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
        pl.plot(p[i], f[i]/mres.zp[i],',', c='0')
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
        pl.errorbar(pb[i], fb[i]/mres.zp[i], eb[i], fmt='.', c='0')
        pl.plot(p2, lc(p2*2.*np.pi), lw=1.5, c='0')
        pl.plot(p2, lc(p2*2.*np.pi), lw=0.5, c='0.9')
        #pl.ylim(fbmin-fbmrg, fbmax+fbmrg)
        #pl.ylim(0.9992, 1.0004)
        pl.xlim(pmin, pmax)
        pp.savefig()
        figi+=1

    pp.close()

def plot_transits(figi, data, targetid, filename, lc_solution=None):
    pp = PdfPages(filename)
    for i, d in enumerate(data):
        nt = d.n_transits
        npg = nt // 5
        for j in range(npg):
            d.plot(figi, j*5, (j+1)*5, pp, lc_solution)
            figi += 1
        d.plot(figi, npg*5, npg*5+nt%5, pp, lc_solution)
        figi += 1
    pp.close()

if __name__ == '__main__':
    main()
