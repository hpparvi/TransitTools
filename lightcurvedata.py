import numpy as np
import matplotlib.pyplot as pl

from numpy import abs, asarray, array, poly1d, polyfit, concatenate, repeat
from core import *

try:
    from scipy.interpolate import LSQUnivariateSpline as Spline
except ImportError:
    info("Couldn't import spline from scipy.interpolate.")

class SingleTransit(object):
    def __init__(self, time, flux, err, ivar, tmask, bmask, number, t_center):
        self.time = time.copy()
        self.flux = flux.copy()
        self.err  = err.copy()
        self.ivar = ivar.copy()
        self.tmask = tmask.copy()
        self.number = number
        self.t_center = t_center

        self.badpx_mask = bmask
        self.continuum_fit = None
        self.periodic_signal = None
        self.periodic_sig_p  = None
        self.zeropoint = None

        self.zeropoint = self.flux[self.tmask].mean()


    def get_npts(self, mask_transit=False):
        return self.flux[self.badpx_mask].size

    def get_std(self, clean=True, normalize=True):
        return self.get_flux(mask_transit=True, clean=clean, normalize=normalize).std()

    def get_flux(self, mask_transit=False, clean=True, normalize=True, apply_bad_mask=True, invert_bad_mask=False):
        """Returns the observed flux.
        """
        if apply_bad_mask:
            bad_mask = ~self.badpx_mask if invert_bad_mask else self.badpx_mask 
            mask = np.logical_and(bad_mask, self.tmask) if mask_transit else bad_mask
        else:
            mask = self.tmask if mask_transit else np.ones(self.time.size, dtype=np.bool)
            
        if clean and self.continuum_fit is not None:
            cf = self.continuum_fit(self.time - self.t_center)
            flux = self.flux/cf if normalize else self.flux/cf*cf.mean()
        else:
            flux = self.flux.copy() if not normalize else self.flux / np.median(self.flux[self.tmask])
                
        return flux[mask]

    
    def get_time(self, mask_transit=False):
        mask = np.logical_and(self.badpx_mask, self.tmask) if mask_transit else self.badpx_mask
        return self.time[mask].copy()


    def get_rejected_points(self, normalize=True):
        return self.time[~self.badpx_mask], self.get_flux(normalize=normalize, invert_bad_mask=True)

    def get_transit(self, time=True, mask_transit=False, cleaned=None, normalize=False):
        if not time: raise NotImplementedError
        tdata = self.get_time(mask_transit)
        fdata = self.get_flux(mask_transit, cleaned, normalize)
        return tdata, fdata


    def fit_continuum(self, n_iter=15, top=5.0, bottom=15.0):
        t, f = self.get_transit()
        t -= self.t_center

        tmask = self.tmask
        emask = self.err < 1e-5
        emask = self.err > -1e5
        bmask = self.badpx_mask.copy()

        ## First fit the background continuum flux with a second order polynomial,
        ## iterate n times and remove deviating points.
        for j in range(n_iter):
            mask = np.logical_and(bmask, tmask)
            fit = poly1d(polyfit(t[mask], f[mask], 2))
            crf = f / fit(t)
            std = crf[mask].std()
            bmask = np.logical_and(bmask, crf-1. < top * std)
            bmask[tmask] = np.logical_and(bmask[tmask], crf[tmask]-1. > - bottom * std)


        self.err[emask]    = (f/fit(t)*fit(t).mean())[mask].std()
        self.ivar[:]       = 1./self.err**2
        self.zeropoint     = fit(t).mean()
        self.continuum_fit = fit
        self.badpx_mask[:] = bmask        
        info("      Rejected %i outliers"%(~bmask).sum())


    def clean_with_lc(self, lc_solution, n_iter=5, top=3., bottom=6., **kwargs):
        info("Cleaning data with an light curve solution", H1)
        time = self.time
        flux = self.get_flux(apply_bad_mask=False)

        residuals = flux - lc_solution(time, **kwargs)
        badmask   = self.badpx_mask.copy()

        for i in range(n_iter):
            std = residuals[badmask].std()
            badmask = np.logical_and(badmask, residuals <  top*std)
            badmask = np.logical_and(badmask, residuals > -bottom*std)

        self.badpx_mask[:] = badmask
        info("Removed %i points"%(~badmask).sum())

    def fit_periodic_signal(self, period, nknots=10, nper=150):
        t, f = self.get_transit(mask_transit=True, cleaned=True, normalize=False)
        p    = t % period

        sid = np.argsort(p)
        p_ord = p[sid]
        f_ord = f[sid] - np.median(f)
        p_per = np.concatenate([p_ord[-nper:]-period,p_ord[:],p_ord[:nper]+period])
        f_per = np.concatenate([f_ord[-nper:],f_ord[:],f_ord[:nper]])
        
        s = Spline(p_per,f_per, t=np.linspace(p_ord[0],p_ord[-1], nknots))

        self.periodic_sig_p  = period
        self.periodic_signal = s

        self.flux -= s(self.time%period)

        self.err[:]  = self.get_std(normalize=False)
        self.ivar[:] = 1./self.err**2

        
    def get_slope(self):
        t, f = self.get_transit(mask_transit=True, cleaned=False, normalize=False)
        fit = poly1d(polyfit(t-self.t_center,f,1))
        self.dfdt = fit.coeffs[0] / fit.coeffs[1]
        #print "% 7.2f"%(1e3*self.dfdt)
        
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
    def __init__(self, name, time, flux, err, tc, p, s, t_width=0.1, mtime=True, **kwargs):
        info('Initializing lightcurve data', H1)
        ## TODO: Store the removed points in a separate array
        otime = np.abs(((time-tc+0.5*p)%p) - 0.5*p)
        phase = np.abs(((time-tc+0.5*p)%p) / p - 0.5)
        mask = otime < s if mtime else phase < s
        
        self.fit_continuum = kwargs.get('fit_continuum', True)
        clean_pars = {'n_iter':15, 'top':5.0, 'top':15.0}
        if 'clean_pars' in kwargs.keys(): clean_pars.update(kwargs['clean_pars'])
        
        self.name  = name
        phase = phase[mask]
        time  = time[mask]
        flux  = flux[mask]
        ferr  = err[mask]
        ivar  = np.ones(time.size) 
        pntn  = np.arange(time.size)
        badpx_mask = np.ones(time.size, np.bool)
        tmask = otime[mask] > t_width if mtime else self.phase > t_width

        self.fit = None
        self.tw = t_width
        self.tc = tc
        self.p  = p
        self.s  = s

        self.transits = []

        self.t_number = (time-tc)/p + 0.5
        self.t_number = np.round(self.t_number-self.t_number[0]).astype(np.int)
        self.n_transits = self.t_number[-1] + 1

        ## Separate transits
        ## =================
        ## Separate the lightcurve into transit-centered pieces of given width,
        ## each transit represented by a SingleTransit class.
        info('Separating transits',I1)
        t = np.arange(time.size)
        for i in range(self.n_transits):
            b = self.t_number == i
            ## First check if we have any points corresponding the given transit
            ## number.
            if np.any(b):
                sl = np.s_[t[b][0]: t[b][-1]+1]
                ## Check if we have any in-transit points for the transits, reject the transit if not.
                if np.any(~tmask[sl]):
                    self.transits.append(SingleTransit(time[sl], flux[sl], ferr[sl],
                                                       ivar[sl], tmask[sl], badpx_mask[sl],
                                                       i, time[sl].mean()))
                else:
                    self.n_transits -= 1
            else:
                self.n_transits -= 1
        info('Found %i good transits'%self.n_transits, I2)
        info('')

        for t in self.transits:
            t.get_slope()

        ## Fit the per-transit continuum level
        ## -----------------------------------
        if self.fit_continuum:
            logging.info('  Cleaning transit data')
            logging.info('    Fitting per-transit continuum level')
            for t in self.transits:
                t.fit_continuum(**clean_pars)

        ## Clean up a possible periodic signal if a period is given
        ## --------------------------------------------------------
        #if 'ps_period' in kwargs.keys():
        #    logging.info('    Cleaning a periodic error signal')
        #    for t in self.transits:
        #        t.fit_periodic_signal(kwargs['ps_period'])
        #logging.info('')

        ## Remove transits with a high point-to-point scatter
        ## --------------------------------------------------
        logging.info('  Removing bad transits')
        self.update_stds()
        #self.remove_bad_transits()
        self.update_stats()

        logging.info('')
        logging.info('  Created a lightcurve with {} points'.format(self.get_npts()))
        logging.info('')
        logging.info('  Mean   std %7.5f'%self.get_mean_std())
        logging.info('  Median std %7.5f'%self.get_median_std())
        logging.info('')

        #for i, tr in enumerate(self.transits):
        #    logging.info('%s %2i %+10.3f %+10.3f %+10.3f'%(self.name,i, (p0[i]-p0.mean())/p0e, (p1[i]- p1.mean())/p1e, (p2[i]-p2.mean())/p2e))


    def remove_bad_transits(self, sigma=5.):
        """Removes transits with point-to-point scatter higher thatn sigma*mean_std.
        """
        tt = np.arange(self.n_transits)
        bad_transits = tt[(self.ferr - self.ferr.mean()) >  sigma*self.ferr.std()]
        logging.info('    Found %i bad transit%s with std > %3.1f mean std' %(bad_transits.size,
                                                                            's' if bad_transits.size > 1 else '',
                                                                            sigma))
        i = 0
        for tn in bad_transits:
            logging.info('      Removing transit %i'%(tn-i))
            self.remove_transit(tn-i)
            i += 1

    def update_stds(self):
        std = []
        for tr in self.transits:
            tr.err[:] = tr.get_std(normalize=False)
            tr.ivar[:] = 1./tr.err**2

    def update_stats(self):
        self.npts = self.get_npts()
        flux_r = self.get_flux(normalize=False)
        flux_n = self.get_flux()

        self.flux_r_max  = flux_r.max()
        self.flux_r_min  = flux_r.min()
        self.flux_r_mean = flux_r.mean()
        self.flux_n_max  = flux_n.max()
        self.flux_n_min  = flux_n.min()
        self.flux_n_mean = flux_n.mean()

    def clean_with_lc(self, lc, n_iter=5, top=5., bottom=5., **kwargs):
        for t in self.transits:
            t.clean_with_lc(lc, n_iter, top, bottom, **kwargs)

    def remove_transit(self, tn):
        raise NotImplementedError

    def get_npts(self, mask_transits=False):
        return sum([tr.get_npts(mask_transits) for tr in self.transits])

    def get_time(self, mask_transits=False):
        return concatenate([tr.get_time(mask_transit=mask_transits) for tr in self.transits])

    def get_flux(self, mask_transits=False, normalize=True):
        return concatenate([tr.get_flux(clean=True, mask_transit=mask_transits, normalize=normalize) for tr in self.transits])

    def get_std(self, normalize=True):
        return concatenate([repeat(tr.get_std(normalize=normalize), tr.time[tr.badpx_mask].size) for tr in self.transits])

    def get_ivar(self, normalize=True):
        return 1./self.get_std(normalize=normalize)**2

    def get_mean_std(self, normalize=True):
        return array([tr.get_std(normalize=normalize) for tr in self.transits]).mean()

    def get_median_std(self, normalize=True):
        return np.median(array([tr.get_std(normalize=normalize) for tr in self.transits]))

    def get_transit_slices(self):
        slices = []
        start = 0
        for tr in self.transits:
            end = tr.get_npts()
            slices.append(np.s_[start:end])
            start = end
        return slices
    
    def plot(self, fig=0, tr_start=0, tr_stop=None, pdfpage=None, lc_solution=None):
        ##TODO: Plot excluded points
        fig = pl.figure(fig, figsize=[15,10])
    
        ntr = float(self.n_transits - tr_start if tr_stop is None else tr_stop-tr_start)
        upl = [fig.add_subplot(2,ntr,i+1) for i in range(int(ntr))]
        dpl = [fig.add_subplot(2,ntr,i+1+ntr) for i in range(int(ntr))]

        for i, tr in enumerate(self.transits[tr_start:tr_stop]):
            t_tt, f_tt = tr.get_transit(time=True, mask_transit=False)

            tmask = tr.tmask[tr.badpx_mask]
            mf = np.median(f_tt[tmask])
            df = 4*f_tt[tmask].std()
            tp = [t_tt[~tmask][0], t_tt[~tmask][-1], t_tt[~tmask][-1], t_tt[~tmask][0]]

            upl[i].plot(t_tt[tmask],f_tt[tmask],'o', c='0.5', ms=2.5)
            upl[i].plot(t_tt[~tmask],f_tt[~tmask],'o', c='0.0', ms=2.5)
            upl[i].fill(tp, [mf+df, mf+df, mf-df, mf-df], alpha=0.05)
            upl[i].plot([tr.t_center, tr.t_center], [mf-df, mf+df], c='0.5', ls='--', lw=1)

            tt = np.linspace(tr.time[0], tr.time[-1], 200)
            upl[i].plot(tt, tr.continuum_fit(tt-tr.t_center), c='0.9', lw=4)
            upl[i].plot(tt, tr.continuum_fit(tt-tr.t_center), c='0.0', lw=2)

            fc = tr.continuum_fit.c
            upl[i].text(0.015, 0.95, 'Transit {:3d}'.format(tr.number+1), transform = upl[i].transAxes)
            upl[i].text(0.015, 0.115, 'Normalized linear slope\n {:8.5f}'.format(tr.dfdt), transform = upl[i].transAxes, size='small')
            upl[i].text(0.015, 0.025, 'Normalized quadratic fit\n 1 {:+8.5f}x {:+8.5f}x^2'.format(fc[1]/fc[-1], fc[0]/fc[-1]), transform = upl[i].transAxes, size='small')
            upl[i].set_xlim(t_tt[0], t_tt[-1])

            std = tr.get_std()
            t_fn, f_fn = tr.get_transit(time=True, mask_transit=False, cleaned=True, normalize=True)
            dpl[i].plot(t_fn, f_fn, '-', c='0')
            dpl[i].plot(t_fn, f_fn, '.', c='0')

            if lc_solution is None:
                dpl[i].axhline(1+3*std, c='0', ls=':')
                dpl[i].axhline(1-6*std, c='0', ls=':')

            rtime, rflux = tr.get_rejected_points(normalize=True)
            dpl[i].plot(rtime, rflux, 'x', c='0', ms=3)
            for rt in rtime:
                dpl[i].axvline(rt, ymin=0, ymax=0.025, c='0')
                dpl[i].axvline(rt, ymin=0.975, ymax=1, c='0')

            if lc_solution is not None:
                t_lc = np.linspace(t_fn[0], t_fn[-1], 200)
                f_lc = lc_solution(t_lc)
                dpl[i].plot(t_lc, f_lc, c='b')
                dpl[i].plot(t_lc, f_lc+3*std, ':',c='0.5')
                dpl[i].plot(t_lc, f_lc-6*std, ':',c='0.5')


            dpl[i].set_xlim(t_fn[0], t_fn[-1])

        ymargin = 0.01*(self.flux_r_max - self.flux_r_min)

        pl.setp(upl, ylim=(self.flux_r_min-ymargin, self.flux_r_max+ymargin))
        pl.setp(dpl, ylim=((self.flux_r_min-ymargin)/self.flux_r_mean, (self.flux_r_max+ymargin)/self.flux_r_mean))
        pl.setp((upl[1:], dpl[1:]), yticks=[])
        pl.setp((upl, dpl), xticks=[])

        fig.subplots_adjust(left=0.05, right=0.99, bottom=0.03, top=0.97, wspace=0.0, hspace=0.1)
        if pdfpage is not None: pdfpage.savefig()


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
