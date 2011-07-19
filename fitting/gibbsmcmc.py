"""
Markov Chain Monte Carlo
------------------------

A module to carry out parameter fitting error estimation using Markov Chain
Monte Carlo simulation. 
"""

from mcmc import *

class GibbsMCMC(MCMC):
    """
    Markov Chain Monte Carlo with Gibbs sampling.
    """

    def __init__(self, chifun, parameter_defs, **kwargs):
        super(GibbsMCMC, self).__init__(chifun, parameter_defs, **kwargs)
        self.result = GibbsMCMCResult(self.n_chains_l, self.n_steps, self.n_parms, self.p_names, self.p_descr)

    def __call__(self):
        if self.use_curses:
            curses.wrapper(self._mcmc)
        else:
            self._mcmc()
        return self.result


    def _mcmc(self, *args):
        """The main Markov Chain Monte Carlo routine in all its simplicity."""
        if self.use_curses: self.init_curses()

        ## MCMC main loop
        ## ==============
        for chain in xrange(self.n_chains):
            self.i_chain = chain
            P_cur = self.p.copy()
            prior_cur = asarray([p.prior(P_cur[i]) for i, p in enumerate(self.parameters)])
            X_cur = self.chifun(P_cur)

            at_test = np.zeros(self.n_parms, dtype=np.int)
 
            i_at = 0
            P_try = P_cur.copy()
            prior_try = prior_cur.copy()
            for i_s in xrange(self.n_steps):
                self.i_step = i_s
                for i_t in xrange(self.thinning):
                    for i_p, p in enumerate(self.parameters):
                        P_try[i_p] = p.draw(P_cur[i_p])
                        prior_try[i_p] = p.prior(P_try[i_p])
                        X_try =  self.chifun(P_try)

                        prior_ratio = prior_try.prod() / prior_cur.prod()

                        err_t = self.fitpar.get_error_scale(P_try)
                        err_c = self.fitpar.get_error_scale(P_cur)
                        error_ratio = err_t / err_c

                        self.result.accepted[chain, i_p, 0] += 1

                        if X_try < 1e17 and self.acceptStep(err_c*X_cur, err_t*X_try, prior_ratio, error_ratio):
                            P_cur[i_p] = P_try[i_p]
                            prior_cur[i_p] = prior_try[i_p]
                            X_cur = X_try
                            self.result.accepted[chain, i_p, 1] += 1
                            at_test[i_p] += 1
                        else:
                            P_try[i_p] = P_cur[i_p]

                    ## Autotuning
                    ## ==========
                    if i_s*self.thinning + i_t < self.autotune_length and i_at == 25:
                        for i_p, p in enumerate(self.parameters):
                            accept_ratio  = at_test[i_p]/25. 
                            accept_adjust = (1. - self.autotune_strength) + self.autotune_strength*4.* accept_ratio
                            p.draw_function.sigma *= accept_adjust
                        at_test[:] = 0.
                        i_at = 0
                    i_at  += 1

                self.result.steps[chain, i_s, :] = P_cur[:]
                self.result.chi[chain, i_s] = X_cur

                ## MONITORING
                ## ==========
                if self.monitor and (i_s+1)%self.minterval == 0:
                    try:
                        self.plot_simulation_progress(i_s, chain)
                    except RuntimeError: pass
                        
                if (i_s+1)%self.sinterval == 0:
                    self.result.save(self.sname)

                if self.use_curses: self.ui.update()

        ## MPI communication
        ## =================
        ## We do this the easy but slow way. Instead of using direct NumPy array transfer, we
        ## pass the data as generic Python objects. This should be ok for now.
        if self.use_mpi:
            result = GibbsMCMCResult(self.n_chains_g, self.n_steps, self.n_parms, self.p_names, self.p_descr)
            result.steps    = mpi_comm.gather(self.result.steps)
            result.chi      = mpi_comm.gather(self.result.chi)
            result.accepted = mpi_comm.gather(self.result.accepted)

            if is_root:
                result.steps    = np.concatenate(result.steps)
                result.chi      = np.concatenate(result.chi)
                result.accepted = np.concatenate(result.accepted)
                self.result = result
            
        return self.result

    def init_curses(self):
        self.ui = MCMCCursesUI(self)

    def plot_simulation_progress(self, i_s, chain):
        fig = self.fig_progress
        fig.clf()
        nrows = self.n_parms/2+1
        ncols = 6
        for ip in range(self.n_parms):
            ax_chain = fig.add_subplot(nrows,ncols,ip*3+1)
            ax_hist  = fig.add_subplot(nrows,ncols,ip*3+2)
            ax_acorr = fig.add_subplot(nrows,ncols,ip*3+3)
            pl.text(0.035, 0.85, self.p_names[ip],transform = ax_chain.transAxes,
                    backgroundcolor='1.0', size='large')

            for ch in range(chain+1):
                d = self.result.steps[ch,:i_s,ip]
                ax_chain.plot(d)
                ax_hist.hist(d)
                ax_acorr.acorr(d-d.mean(), maxlags=40, usevlines=True)

            ax_acorr.axhline(1./np.e, ls='--', c='0.5')
            pl.setp([ax_chain, ax_hist, ax_acorr], yticks=[])
            pl.setp(ax_chain, xlim=[0,i_s])
            pl.setp(ax_acorr, xlim=[-40,40], ylim=[0,1])
        pl.subplots_adjust(top=0.99, bottom=0.02, left=0.01, right=0.99, hspace=0.2, wspace=0.04)
        pl.savefig('mcmcdebug_n%i_%i.pdf'%(mpi_rank,chain))


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
        self.sw.addstr(2,1,' '.join(['{0:^10.6f}'.format(s) for s in [p.draw_function.sigma for p in self.mcmc.parameters]]))


class GibbsMCMCResult(MCMCResult):
    def __init__(self, n_chains, n_steps, n_parms, p_names, p_descr, burn_in = 0.2, cor_len=1):
        super(GibbsMCMCResult, self).__init__(n_chains, n_steps, n_parms, p_names, p_descr, burn_in, cor_len)

    def __call__(self, parameter=None, burn_in=None, thinning=None):
        """Returns the results."""

        burn_in  = burn_in or self.burn_in
        thinning = thinning or self.cor_len
        i = list(self.p_names).index(parameter)
        return self.steps[:, burn_in::thinning, i].ravel()

    def get_acceptance(self):
        """Returns the acceptance ratio for the parameters."""
        t = self.accepted.sum(0)
        r = np.where(t[:,0]>0, t[:,1]/t[:,0], 0)
        return r

    def plot(self, i_c, burn_in, thinning, fign=100, s_max=-1, c='b', alpha=1.0):
        fig = pl.figure(fign, figsize=(34,20), dpi=50)
        nrows = self.n_parms/2+1
        ncols = 6
        mask = np.abs(self.steps[i_c,:,:]).sum(1) > 0.
        mask[:burn_in] = False
        mask[s_max:] = False
        ns = mask.sum()/thinning
        for i_p in range(self.n_parms):
            xmin = self.steps[:,mask,i_p][:,::thinning].min()
            xmax = self.steps[:,mask,i_p][:,::thinning].max()
            d = self.steps[i_c,mask,i_p][::thinning]

            ax_chain = fig.add_subplot(nrows,ncols,i_p*3+1)
            ax_hist  = fig.add_subplot(nrows,ncols,i_p*3+2)
            ax_acorr = fig.add_subplot(nrows,ncols,i_p*3+3)

            pl.text(0.035, 0.85, self.p_names[i_p],transform = ax_chain.transAxes,
                    backgroundcolor='1.0', size='large')
            ax_chain.plot(d, c=c, alpha=1.0)
            ax_hist.hist(d, range=[xmin,xmax], fc=c, alpha=alpha)
            ax_acorr.acorr(d-d.mean(), maxlags=75, usevlines=True, color=c)
            ax_acorr.axhline(1./np.e, ls='--', c='0.5')
            pl.setp([ax_chain, ax_hist, ax_acorr], yticks=[])
            pl.setp(ax_chain, xlim=[0,ns])
            pl.setp(ax_acorr, xlim=[-75,75], ylim=[0,1])
        pl.subplots_adjust(top=0.99, bottom=0.02, left=0.01, right=0.99, hspace=0.2, wspace=0.04)
        #pl.savefig('%i_%i.pdf'%(mpi_rank,chain))
