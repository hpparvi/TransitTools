import sys
import time
import numpy as np
import pylab as pl
import scipy as sp
import tables as tbl
import scipy.signal

from numpy import pi, sin, sqrt, arccos
from base import *

try:
    from mpi4py import MPI
    with_mpi = True
except ImportError:
    with_mpi = False

class MCMC(object):
    def __init__(self, chifun, 
                 p0, p_limits, p_free, p_sigma, p_names, p_descr, 
                 n_chains, n_steps, 
                 burn_in=0.2, cor_len=1, dtype='Cyclic', verbose=True, use_mpi=False):

        if with_mpi and use_mpi:
            self.cm = MPI.COMM_WORLD
            self.rank = self.cm.Get_rank()
            self.size = self.cm.Get_size()
            self.use_mpi = True
            
            n_chains_loc     = n_chains//self.size
            self.chain_start = self.rank * n_chains_loc
    
            if self.rank == self.size-1:
                self.chain_end = n_chains
                n_chains_loc = self.chain_end - self.chain_start
            else:
                self.chain_end = self.chain_start + n_chains_loc
                
            if self.rank != 0:
                n_chains = n_chains_loc
            
            logging.info('Created node %i with %i chains (%i--%i)'%(self.rank, n_chains, self.chain_start+1, self.chain_end))
        else:        
            self.use_mpi = False
            self.rank = 0
            self.size = 1
        
        self.chifun = chifun
        self.n_chains = n_chains
        self.n_steps = n_steps
        self.n_parms = len(p0)
        
        self.p = np.array(p0).copy()
        self.p_limits = np.array(p_limits)
        self.p_free = np.array(p_free)
        self.p_sigma = np.array(p_sigma)
        self.p_names = p_names
        self.p_description = p_descr

        self.result = MCMCResult(n_chains, n_steps, self.n_parms, burn_in, cor_len)
                
        self._pn = np.arange(self.n_parms)[self.p_free]
        self._pi = 0
        
        drawTypes = {'Cyclic':self._drawNewCyclic, 'Random':self._drawNewRandom, 'All':self._drawNewAll}
        acceptionTypes = {'Exp':self._acceptStepExp}
    
        self.drawNew = drawTypes[dtype]
        self.acceptStep = acceptionTypes['Exp']
        self.verbose = verbose

    def _drawNewAll(self, pCur):
        return pCur.copy() + self.pFree*self.pSigma*np.random.normal(self.p.size)
    
    def _drawNewCyclic(self, pCur):
        i = self._pn[self._pi]
        pNew = pCur.copy()
        pNew[i] += np.random.normal(0., self.p_sigma[i])
        self._pi = (self._pi + 1) % self._pn.size
        return pNew, i
        
    def _drawNewRandom(self, pCur):
        i = self._pn[int(np.floor(np.random.random()*self._pn.size))]
        pNew = pCur.copy()
        pNew[i] += np.random.normal(0., self.pSigma[i])
        self._pi = (self._pi + 1) % self._pn.size
        return pNew, i
    
    def _acceptStepExp(self, X0, Xt):
        P = np.exp(-0.5*(Xt - X0))
        if np.random.random() < P:
            return True
        else:
            return False
        
    def __call__(self):
        "The main Markov Chain Monte Carlo routine in all its simplicity."

        for chain in range(self.n_chains):
            logging.info('Starting node %2i  chain %2i  of %2i' %(self.rank, chain+1, self.n_chains))
            P_cur = self.p.copy()
            X_cur = self.chifun(P_cur)
    
            for i in range(self.n_steps):
                P_try, pi_try = self.drawNew(P_cur)
                X_try = self.chifun(P_try)
                self.result.accepted[chain, pi_try, 0] += 1
    
                if self.acceptStep(X_cur, X_try):
                    self.result.steps[chain, i,:] = P_try.copy()
                    self.result.chi[chain, i] = X_try
                    P_cur[:] = P_try[:]
                    X_cur = X_try
                    self.result.accepted[chain, pi_try, 1] += 1
                else:
                    self.result.steps[chain, i,:] = P_cur.copy()
                    self.result.chi[chain, i] = X_cur
                
        if self.use_mpi:
            if self.rank == 0:
                for node in range(1, self.size):
                    cs, ce = self.cm.recv(source=node, tag=11)
                    for chain in range(cs, ce):
                        logging.info('Master receiving node %i chain %i' %(node, chain+1))
                        self.cm.Recv([self.result.steps[chain, :, :], MPI.DOUBLE], source=node, tag=77)
                        self.cm.Recv([self.result.chi[chain, :], MPI.DOUBLE], source=node, tag=77)
                        self.cm.Recv([self.result.accepted[chain, :], MPI.DOUBLE], source=node, tag=77)
            else:
                self.cm.send((self.chain_start, self.chain_end), dest=0, tag=11)
                for chain in range(self.n_chains):
                    logging.info('Node %i sendind chain local:%3i  global:%3i' %(self.rank, chain+1, self.chain_start+chain+1))
                    self.cm.Send([self.result.steps[chain,:,:], MPI.DOUBLE], dest=0, tag=77)
                    self.cm.Send([self.result.chi[chain, :], MPI.DOUBLE], dest=0, tag=77)
                    self.cm.Send([self.result.accepted[chain, :], MPI.DOUBLE], dest=0, tag=77)

    def get_results(self, burn_in=None, cor_len=None):
        burn_in = burn_in if burn_in is not None else self.result.burn_in
        burn_in = int(burn_in*self.n_steps)
        cor_len = cor_len if cor_len is not None else self.result.cor_len
        return self.result.steps[:, burn_in::cor_len, :]

    def get_parameter(self, p_idx):
        result = np.zeros([self.n_steps, self.n_threads])
        for i, t in enumerate(self.simulation_thread):
            result[:,i] = t.psim[:,p_idx]
        return result

    def get_best_fit(self):
        minX = 1e30
        for t in self.simulation_thread:
            if t.x.min() < minX:
                pb = t.psim[np.argmin(t.x),  :]
                minX = t.x.min()
        return pb,  minX

    def get_parameter_median(self):
        return np.median(self.get_results(), 0)

    def get_parameter_mean(self):
        return self.get_results().mean(0)

    def print_statistics(self):
        for t in self.simulation_thread:
            print "Accept ratio: ",
            for i in range(self.np):
                print "%10.6f" % (t.accepted[i,1] / t.accepted[i,0]),
            print ""

        p_median = np.median(self.get_results(), 0)
        print "Parameter median: ",
        for i in range(self.np):
            print "%10.4e " %p_median[i],
        print ""

        X = np.zeros(self.n_threads)
        for i, t in enumerate(self.simulation_thread):
            X[i] = t.x.min()

        print "Minimum X^2: %12.6f" %X.min()

    def plot_correlation(self, figidx=0):
        pl.figure(figidx)
        for i, t in enumerate(self.simulation_thread):
            for pIdx in range(self.p_0.size):
                pl.subplot(self.p_0.size, 1, pIdx+1)
##                d = t.psim[int(self.burn_in_p*self.n_steps):, pIdx]
                d = t.psim[:, pIdx]
                pl.plot(sp.signal.correlate(d,d))
                #pl.acorr(d, normed=True, alpha=0.2)

    def plot_parameter(self, p=0, n=25, figidx=0):
        r = self.get_results()
        pl.figure(figidx)
        pl.hist(r[:,p], n, fc='0.95')
        for i, t in enumerate(self.simulation_thread):
            pl.hist(t.psim[int(self.burn_in_p*self.n_steps):, p], n, alpha=0.2)

    def plot_parameters(self, n=25, figidx=0):
        r = self.get_results()
        pl.figure(figidx)
        for i in range(self.np):
            pl.subplot(self.np, 1, i+1)
            pl.hist(r[:,i], n, fc='0.95', histtype='stepfilled')
            for t in self.simulation_thread:
                pl.hist(t.psim[int(self.burn_in_p*self.n_steps)::self.p_s, i], n, alpha=0.2, histtype='stepfilled')

    def save(self, filename, channel, name, description):
        f = tbl.openFile(filename, 'a', "")        
        
        if not f.__contains__('/mcmc'):
            g_mc = f.createGroup('/', 'mcmc', 'Markov Chain Monte Carlo')
        else:
            g_mc = f.root.mcmc

        if not g_mc.__contains__(channel):
            g_ch = f.createGroup(g_mc, channel, channel)
        else:
            g_ch = f.getNode(g_mc,channel)

        if not g_ch.__contains__(name):
            g_sim = f.createGroup(g_ch, name, description)
        else:
            g_sim = f.getNode(g_ch, name)
            
        g_sim._v_attrs.n_steps = self.n_steps
        g_sim._v_attrs.n_chains = self.n_threads
        g_sim._v_attrs.p_names = self.p_names
        g_sim._v_attrs.p_description = self.p_description
        g_sim._v_attrs.n_variables = self.np

        Chi_squared = np.zeros([self.n_steps, self.n_threads])

        for i, t in enumerate(self.simulation_thread):
            Chi_squared[:,i] = t.x

        if g_sim.__contains__('Chi_squared'):
            f.removeNode(g_sim, 'Chi_squared')
            
        f.createArray(g_sim, 'Chi_squared', Chi_squared.astype(float), 'Chi squared')

        for i in range(self.np):
            if g_sim.__contains__('p%i'%i):
                f.removeNode(g_sim, 'p%i'%i)
            
            f.createArray(g_sim, 'p%i' %i, self.get_parameter(i).astype(np.float), self.p_description[i])

        f.close()


def __test(n_jumps=500, n_threads=2, test_type='Gaussian'):
    """
    Function for the testing of the MCMC class.
    """
    a = 2.0
    mu = 0.5
    sigma = 0.2
    
    x = np.linspace(0.0,2.0, 225)
    
    def Gaussian(p):
        return p[0] * np.exp(-(x-p[1])**2 / (2.*p[2]**2))
        
    y = Gaussian([a, mu, sigma]) + np.random.normal(0, 0.05, x.size)
        
    def f(p):
        return ((y-Gaussian(p))**2).sum()
        
    p0 = np.array([a+1.1, mu+.1, sigma+.3])
    pL = [[0,4],[0,4],[0,2]]
    pF = np.array([True, True, True])
    pS = np.array([3e-1, 3e-1, 3e-1])
    
    mc = MCMC(f, p0, pL, pF, pS, n_jumps, n_threads)
    mc.run()

    return mc
    
class MCMCResult(FitResult):
    def __init__(self, n_chains, n_steps, n_parms, burn_in = 0.2, cor_len=1):
        self.n_chains = n_chains
        self.n_steps  = n_steps
        self.n_parms  = n_parms
        self.burn_in  = burn_in
        self.cor_len  = cor_len
        
        self.steps    = np.zeros([n_chains, n_steps, n_parms])
        self.chi      = np.zeros([n_chains, n_steps])
        self.accepted = np.zeros([n_chains, n_parms, 2])

    def plot_2d_param_distributions(self, figidx=0, axes=None, nb=25, params=[0,1], lims=None, interpolation='bicubic', label=None):

        pl.figure(figidx)
        if axes is not None:
            if isinstance(axes, int):
                ax = pl.subplot(axes)
            else:
                ax = pl.axes(axes)
        else:
            ax = pl.axes()
        
        if label is not None:
            pl.text(1.-0.025, 0.9, label, va='top', ha='right', transform = ax.transAxes)
                
        pl.gray()
        if lims is not None:
            i, x, y = np.histogram2d(self(params[0]), self(params[1]), nb, lims)
        else:
            i, x, y = np.histogram2d(self(params[0]), self(params[1]), nb)
            
        pl.imshow(i.max() - i, interpolation=interpolation, extent=(y[0], y[-1], x[0], x[-1]), origin='lower', aspect='auto')
        #pl.contour(i, 5, origin='lower', extent=(y[0], y[-1], x[0], x[-1]), colors='0.5')
        pl.ylabel(self.pnames[params[0]])
        pl.xlabel(self.pnames[params[1]])

        print lims

        fmt = pl.matplotlib.ticker.ScalarFormatter()
        fmt.set_powerlimits((-2,2))
        ax.xaxis.set_major_formatter(fmt)
        ax.yaxis.set_major_formatter(fmt)

    def plot_parameter(self, i, nb=50, polyorder=10, figidx=0):
        h  = np.histogram(self(i),nb, normed=True)
        hc =  0.5*(h[1][0:-1] + h[1][1:])
        
        pl.figure(figidx)
        pl.hist(self(i),nb, normed=True)
        #pl.plot(hc, sp.signal.medfilt(h[0],9))

    def plot_parameters(self, figidx=0, axes=None, nb=25, params=None, lims=None, label=None, plot_chains=True, color='0.85', alpha=1., label_position=1):

        if params is None:
            params = range(self.np)

        pl.figure(figidx)
        
        median, mean = self.get_statistics(params, lims)
        
        for i, pIdx in enumerate(params):

            if axes is not None:
                ax = pl.axes(axes[i])
            else:
                ax = pl.subplot(len(params), 1, i+1)
                
            if label is not None:
                pl.text(0.025, 0.15, label[i], va='top', ha='left', transform = ax.transAxes)
                
            pl.hist(self(pIdx), nb, lims[i], fc=color, histtype='stepfilled', alpha=alpha)

            pl.axvline(median[pIdx], c='0')
            #pl.axvline(mean[pIdx], c='0')

            #pl.text(1.-0.025, 0.9, 'med(p) = %4.2e' %median[pIdx], va='top', ha='right', transform = ax.transAxes)

            if plot_chains:
                r = self(pIdx, True)
                for j in range(self.n_chains):
                    pl.hist(r[:,j], nb, lims[i], alpha='0.05', histtype='stepfilled')
                    #pl.hist(r[:,j], nb, alpha='0.05', histtype='stepfilled')


            pl.yticks([])

            fmt = pl.matplotlib.ticker.ScalarFormatter()
            fmt.set_powerlimits((-2,2))
            ax.xaxis.set_major_formatter(fmt)

            if label_position < 0: 
                pass
            elif label_position == 0:
                pl.xlabel(self.pnames[pIdx])
            elif label_position == 1:
                pl.text(1.-0.025, 0.9, self.pnames[pIdx], va='top', ha='right', transform = ax.transAxes)


            if lims is not None:
                pl.xlim(lims[i])

    def print_best_fit(self, nb=50, params=None):
        if params is None:
            params = range(4)
        for i in params:
            h  = np.histogram(self(i),nb)
            hc =  0.5*(h[1][0:-1] + h[1][1:])
            m  = np.argmax(h[0])
            print "%s %6.2e" %(self.keys[i], hc[m])

    def get_minimum_Chi_squared(self):
        return self.Chi_squared.min()

    def get_statistics(self, idx=None, lims=None):
        median = np.zeros(self.np)
        mean   = np.zeros(self.np)
        
        if idx is None:
            idx = range(self.np)
        
        if lims is not None:
            for i, id in enumerate(idx):
                median[id] = np.median(self(id).compress((self(id) > lims[i][0]) & (self(id) < lims[i][1])) )
                mean[id] = self(id).compress((self(id) > lims[i][0]) & (self(id) < lims[i][1])).mean()
        else:
            for i in idx:
                median[i] = np.median(self(i))
                mean[i] = self(i).mean()

        return median, mean

    def get_best_fit(self):
        xp = np.argmin(self.Chi_squared)
        pb = np.zeros(self.np)

        for i, k in enumerate(self.keys):
            pb[i] = self.data[k].ravel()[xp]

        return pb
        
    def get_parameter_mean(self, method=np.mean):
        r = np.zeros(self.np, np.double)
        for i in range(self.np):
            r[i] = method(self(i))
        return r
            
    def __call__(self, burn_in=None, cor_len=None, separate_chains=False):
        burn_in = burn_in if burn_in is not None else self.burn_in
        burn_in = int(burn_in*self.n_steps)
        cor_len = cor_len if cor_len is not None else self.cor_len
        
        if not separate_chains:
            r = np.array([])
            for j in range(self.n_chains):
                r = np.concatenate((r, self.steps[burn_in::cor_len, j]))
        else:
            r = self.steps[burn_in::cor_len, :]
        return r
    
if __name__ == "__main__":
    p0 = [1., 2., 1.]
    pL = [[-1., 1.], [-1, 1], [-1, 1]]
    pf = [True, True, True]
    ps = [0.2, 0.2, 0.2]
    pn = ['x', 'y', 'z']

    mc = MCMC(lambda P: np.sum(P**2), p0, pL, pf, ps, pn, pn, 4, 1000, use_mpi=True)
    mc()
    if mc.rank == 0:
        r = mc.get_results(0, 10)
        for c in range(mc.n_chains):
            pl.plot(r[c, :, 0])
        pl.show()

#    print "Running MCMC test with %i jumps and %i threads." %(n_jumps, n_threads)
#
#    mc = __test(n_jumps, n_threads)

#    mc.print_statistics()
#    mc.plot_correlation()
#    mc.plot_parameter(0, figidx=100)
#    mc.plot_parameter(1, figidx=101)
#    mc.plot_parameter(2, figidx=102)
#    pl.show()


