import numpy as np
import pylab as pl

from scipy.interpolate import LSQUnivariateSpline as Spline
from numpy import abs, asarray, array, poly1d, polyfit
from core import *

class SingleTransit(object):
    def __init__(self, time, flux, ivar, tmask, bmask, number, t_center, g_range):
        self.time = time
        self.flux = flux
        self.ivar = ivar
        self.tmask = tmask
        self.number = number
        self.t_center = t_center
        self.g_range = g_range

        self.badpx_mask = bmask
        self.continuum_fit = None
        self.periodic_signal = None
        self.periodic_sig_p  = None
        self.zeropoint = None
        self.std = None 

        self.zeropoint = self.flux[self.tmask].mean()

    def get_std(self, clean=True):
        t,f = self.get_transit(mask_transit=True, cleaned=clean, normalize=True)
        return f.std()


    def get_flux(self, mask_transit=False, clean=None, normalize=False):
        """Returns the observed flux.

           Returns the observed flux datapoints.

           Options
             mask_transit  bool  
                           Should the transit be masked away.

             clean         bool or ['continuum','periodic'] 
                           Should the cleaning steps be applied.

             normalize     bool
                           Should the data be normalized.

        """
        flux = self.flux.copy()
        if clean is not None:
            stages = np.zeros(2, np.bool)
            if isinstance(clean,bool) and clean:
                stages[:] = True
            else:
                stages[0] = 'continuum' in clean
                stages[1] = 'periodic'  in clean

            if stages[0] and self.continuum_fit is not None:
                flux -= self.continuum_fit(self.time - self.t_center)
            if stages[1] and self.periodic_signal is not None:
                flux -= self.periodic_signal(self.time % self.periodic_sig_p)
        if normalize: flux /= self.zeropoint
        return flux if not mask_transit else flux[self.tmask]

    
    def get_time(self, mask_transit=False):
        return self.time.copy() if not mask_transit else self.time[self.tmask].copy()


    def get_transit(self, time=True, mask_transit=False, cleaned=None, normalize=False):
        tdata = self.get_time(mask_transit) if time else np.arange(self.g_range[0], self.g_range[1])
        fdata = self.get_flux(mask_transit, cleaned, normalize)
        return tdata, fdata


    def fit_continuum(self, n_iter=15, top=5.0, bottom=15.0):
        t, f = self.get_transit()
        t -= self.t_center

        bm = self.badpx_mask.copy()
        for j in range(n_iter):
            fit = poly1d(polyfit(t[self.tmask][bm[self.tmask]], f[self.tmask][bm[self.tmask]], 2))
            crf = f - fit(t)
            self.std = crf[self.tmask][bm[self.tmask]].std()
            bm = np.logical_and(bm, crf <   top    * self.std)
            bm = np.logical_and(bm, crf > - bottom * self.std)

        self.zeropoint = fit(t).mean()
        self.ivar[:] = 1./self.get_std()**2
        self.continuum_fit = fit
        self.badpx_mask[:] = bm


    def fit_periodic_signal(self, period, nknots=10, nper=150):
        t, f = self.get_transit(mask_transit=True, cleaned=True)
        p = t % period

        sid = np.argsort(p)
        p_ord = p[sid]
        f_ord = f[sid] - np.median(f)
        p_per = np.concatenate([p_ord[-nper:]-period,p_ord[:],p_ord[:nper]+period])
        f_per = np.concatenate([f_ord[-nper:],f_ord[:],f_ord[:nper]])
        
        s = Spline(p_per,f_per, t=np.linspace(p_ord[0],p_ord[-1], nknots))

        self.periodic_sig_p  = period
        self.periodic_signal = s

        self.ivar[:] = 1./self.get_std()**2

        #if self.number == 45: self.plot_periodic_signal()


    def plot_periodic_signal(self, fig=0):
        pl.figure(fig)

        t, f = self.get_transit(mask_transit=True, cleaned=['continuum'])
        p    = t % self.periodic_sig_p
        s    = self.periodic_signal
        ps   = np.linspace(p.min(), p.max(), 1000)

        zp   = np.median(f)

        ## Flux folded over the period of the CoRoT orbit with the spline fit.
        ##
        pl.subplot(2,2,1)
        pl.plot(p,f/zp,'.', c='0')
        pl.plot(ps,1.+s(ps)/zp,'-', c='0.9', lw=4)
        pl.plot(ps,1.+s(ps)/zp,'-', c='0.3', lw=2)
        pl.xlim([p.min(), p.max()])
        ylim = pl.ylim()
        pl.xticks([])

        ## Folded flux with the spline fit removed.
        ##
        pl.subplot(2,2,2)
        pl.plot(p, (f-s(p))/zp,'.', c='0')
        pl.xlim([p.min(), p.max()])
        pl.ylim(ylim)
        pl.xticks([])
        pl.yticks([])


        ## Plots with transit
        ## ==================
        t1,f1 = self.get_transit(mask_transit=False, cleaned=['continuum'])
        t2,f2 = self.get_transit(mask_transit=False, cleaned=['continuum','periodic'])
        p1    = t1 % self.periodic_sig_p
        st    = np.linspace(t1[0],t1[-1],500)
        sp    = st % self.periodic_sig_p

        pl.subplot(2,2,3)
        pl.plot(t1, f1/zp,'.', c='0')
        pl.plot(st, 1.+s(sp)/zp, '-', c='0.9', lw=4)
        pl.plot(st, 1.+s(sp)/zp, '-', c='0.3', lw=2)
        pl.xlim(t[0],t[-1])
        ylim = pl.ylim()

        pl.subplot(2,2,4)
        pl.plot(t2, f2/zp, '.', c='0')

        pl.xlim(t[0],t[-1])
        pl.ylim(ylim)
        pl.yticks([])

        pl.subplots_adjust(right=0.98, top=0.98, left=0.08, wspace=0.05, hspace=0.05)

        pl.show()
        


class MultiTransitLC(object):
    def __init__(self, time, flux, tc, p, s, t_width=0.1, mtime=True, **kwargs):
        logging.info('Initializing the lightcurve data')
        logging.info('--------------------------------')
        ## TODO: Store the removed points in a separate array
        otime = np.abs(((time-tc+0.5*p)%p) - 0.5*p)
        phase = np.abs(((time-tc+0.5*p)%p) / p - 0.5)
        mask = otime < s if mtime else phase < s

        clean_pars = {'n_iter':15, 'top':5.0, 'top':15.0}
        if 'clean_pars' in kwargs.keys(): clean_pars.update(kwargs['clean_pars'])
        
        self.phase = phase[mask]
        self.time  = time[mask]
        self.flux  = flux[mask]
        self.ivar  = np.ones(self.time.size) 
        self.pntn  = np.arange(time.size)
        self.badpx_mask = np.ones(self.time.size, np.bool)
        self.tmask = otime[mask] > t_width if mtime else self.phase > t_width

        self.fit = None
        self.tw = t_width
        self.tc = tc
        self.p  = p
        self.s  = s

        self.transits = []

        self.t_number = (self.time-tc)/p + 0.5
        self.t_number = np.round(self.t_number-self.t_number[0]).astype(np.int)
        self.n_transits = self.t_number[-1] + 1

        ## Separate transits
        ## =================
        ## Separate the lightcurve into transit-centered pieces of given width,
        ## each transit represented by a SingleTransit class.
        logging.info('  Separating transits')
        t = np.arange(self.time.size)
        for i in range(self.n_transits):
            b = self.t_number == i
            ## First check if we have any points corresponding the given transit
            ## number.
            if np.any(b):
                r1 = t[b][0]; r2 = t[b][-1]+1
                ## Next check if we actually have any in-transit points for the 
                ## transits, reject the transit if not.
                if np.any(~self.tmask[r1:r2]):
                    self.transits.append(SingleTransit(self.time[r1:r2],
                                                       self.flux[r1:r2],
                                                       self.ivar[r1:r2],
                                                       self.tmask[r1:r2],
                                                       self.badpx_mask[r1:r2],
                                                       i, self.tc+i*self.p, [r1,r2]))
                else:
                    self.n_transits -= 1
            else:
                self.n_transits -= 1
        logging.info('    Found %i good transits'%self.n_transits)

        for tr in self.transits:
            tr.ivar[:] = 1./tr.get_std()**2

        logging.info('  Cleaning the transit data')
        logging.info('    Fitting per-transit continuum level')
        for t in self.transits:
            t.fit_continuum(**clean_pars)

        logging.info('    Removing bad points')
        self.clean(self.badpx_mask)

        if 'ps_period' in kwargs.keys():
            logging.info('    Cleaning a periodic error signal')
            for t in self.transits:
                t.fit_periodic_signal(kwargs['ps_period'])

        #logging.debug('  Per-transit std')
        std = []
        for tr in self.transits:
            std.append(tr.get_std())
            #logging.debug('    raw  %3i %8.5f'%(tr.number, tr.get_std(None)))
            #logging.debug('    cont %3i %8.5f'%(tr.number, tr.get_std(['continuum'])))
            #logging.debug('    pers %3i %8.5f'%(tr.number, tr.get_std(True)))
            #logging.debug('')


        self.std = np.array(std)

        logging.info('  Created a lightcurve with %i points'%self.time.size)
        logging.info('  Mean   std %10.8f'%self.get_mean_std())
        logging.info('  Median std %10.8f'%np.median(self.std))

        logging.info('')
        #self.plot()
        #pl.show()
        #exit()

    def get_mean_std(self):
        return self.std.mean()


    def clean(self, mask):
        self.phase = self.phase[mask].copy()
        self.time  = self.time[mask].copy()
        self.flux  = self.flux[mask].copy()
        self.ivar  = self.ivar[mask].copy()
        self.pntn  = np.arange(self.time.size)
        self.badpx_mask = self.badpx_mask[mask].copy()
        self.tmask = self.tmask[mask].copy()
        self.t_number = self.t_number[mask].copy()

        n = np.arange(self.time.size)
        for t in self.transits:
            b = self.t_number == t.number
            r1 = n[b][0]
            r2 = n[b][-1]+1
            t.time = self.time[r1:r2]
            t.flux = self.flux[r1:r2]
            t.tmask = self.tmask[r1:r2]
            t.badpx_mask = self.badpx_mask[r1:r2]
            t.g_range = [r1,r2]

    def get_time(self):
        return self.time

    def get_flux(self):
        cleaned_lc = np.zeros(self.time.size)

        for tr in self.transits:
            t, f = tr.get_transit(cleaned=True)
            cleaned_lc[tr.g_range[0]:tr.g_range[1]] = f / tr.zeropoint

        return 1.+cleaned_lc

    def plot(self, fig=0):
        pl.figure(fig)
        pl.subplot(2,1,1)
        t_0 = 0.
        for tr in self.transits:
            t_ot, f_ot = tr.get_transit(time=True, mask_transit=True)
            t_tt, f_tt = tr.get_transit(time=True, mask_transit=False)

            t_it = t_tt[~tr.tmask]
            f_it = f_tt[~tr.tmask]

            t_it = t_it - t_tt[0] + t_0
            t_ot = t_ot - t_tt[0] + t_0
            t_tt = t_tt - t_tt[0] + t_0

            t_0 += 2.2*self.s
            
            pl.plot(t_ot,f_ot,'.', c='0.5')
            pl.plot(t_it,f_it,'.', c='0.0')
            pl.axvline(t_0-0.1*self.s, c='0.0')

            if tr.continuum_fit is not None:
                tt = np.linspace(tr.time[0], tr.time[-1], 200)
                tp = np.linspace(t_ot[0], t_ot[-1], 200)
                pl.plot(tp, tr.continuum_fit(tt-tr.t_center), c='0.9', lw=4)
                pl.plot(tp, tr.continuum_fit(tt-tr.t_center), c='0', lw=2)
                #pl.plot(t[tr.badpx_mask], f[tr.badpx_mask], '+', c='0')

        pl.xlim(0, self.n_transits*2.2*self.s)
        pl.xticks([])

        ## Plot the cleaned lightcurve
        ## ===========================
        pl.subplot(2,1,2)
        for tr in self.transits:
            c = '0.25' if tr.number % 2 == 0 else '0.0'
            t,f = tr.get_transit(time=False, mask_transit=False, cleaned=True, normalize=True)
            pl.plot(t, f, c=c)            
        pl.xlim(0,self.time.size)


    def plot_transit(self, fig=0, nbin=100):
        from transitLightCurve.transitlightcurve import TransitLightcurve
        from transitLightCurve.utilities import bin, fold

        phase = fold(self.time, self.p, self.tc, shift=0.5)
        flux  = self.get_flux()

        if nbin is not None:
            phase, flux, fe = bin(phase, flux, nbin)
            
        pl.figure(fig)
        pl.plot(phase, flux, '.')
        
        if self.fit is not None:
            lc = TransitLightcurve(self.fit['parameterization'], ldpar=self.fit['ldc'], mode='phase')
            phase = np.linspace(phase[0], phase[-1], 1000)
            pl.plot(phase, lc(TWO_PI*(phase-0.5)))
