import numpy as np
import pylab as pl
import scipy as sp
import sys
import scipy.signal
import time
import tables as tbl

from numpy import pi, sin, sqrt, arccos

class MCMCFitter(object):
    pass
    
class MCMCThread(threading.Thread):
    def __init__(self, f, p, pLims, pFree, pSigma, nSteps, type='Cyclic', verbose=True):
        self.f = f
        self.p = p[:]
        self.pLims = pLims[:]
        self.pFree = pFree[:]
        self.pSigma = pSigma[:]
        self.nSteps = nSteps

        self.np = p.size
        self.psim = np.zeros([nSteps, p.size])
        self.x = np.zeros(nSteps)
    
        self._pn = np.arange(p.size)[pFree]
        self._pi = 0
    
        self.accepted = np.zeros([self.np, 2], np.float)
    
        if type=='Cyclic':
            self.drawNew = self.drawNewCyclic
        elif type == 'Random':
            self.drawNew = self.drawNewRandom
        else:
            self.drawNew = self.drawNewAll

        self.acceptStep = self.acceptStepExp

        self.verbose = verbose

    def drawNewAll(self, pCur):
        return pCur.copy() + self.pFree*self.pSigma*np.random.normal(self.p.size)
    
    def drawNewCyclic(self, pCur):
        i = self._pn[self._pi]
        pNew = pCur.copy()
        pNew[i] += np.random.normal(0., self.pSigma[i])
        self._pi = (self._pi + 1) % self._pn.size

        return pNew, i
        
    def drawNewRandom(self, pCur):
        i = self._pn[int(np.floor(np.random.random()*self._pn.size))]
        pNew = pCur.copy()
        pNew[i] += np.random.normal(0., self.pSigma[i])
        self._pi = (self._pi + 1) % self._pn.size
        return pNew, i
    
    def acceptStepExp(self, X0, Xt):
        P = np.exp(-0.5*(Xt - X0))
        if np.random.random() < P:
            return True
        else:
            return False

    def acceptStepWiki(self, X0, Xt):
        a = X0 / Xt
        if a >= 1.0:
            return True
        else:
            if np.random.random() < a:
                return True
            else:
                return False

    def run(self):
        self.MCMC()
        
    def MCMC(self):
        P_cur = self.p.copy()
        X_cur = self.f(P_cur)

        for i in range(self.nSteps):
            P_try, i_try = self.drawNew(P_cur)
            X_try = self.f(P_try)
            self.accepted[i_try, 0] += 1

            if self.acceptStep(X_cur, X_try):
                self.psim[i,:] = P_try.copy()
                self.x[i] = X_try
                P_cur[:] = P_try[:]
                X_cur = X_try
                self.accepted[i_try, 1] += 1
            else:
                self.psim[i,:] = P_cur.copy()
                self.x[i] = X_cur
                
            if self.verbose and i % (self.nSteps / 20) == 0:
                print "%s %6.0f%%" %(self.getName(), float(i)/self.nSteps * 100.)
        
        if self.verbose: print "%s %6.0f%%" %(self.getName(), 100.)
            

class MCMC():
    def __init__(self, fun, p_0, p_limits, p_free, p_sigma, p_names, p_descr, n_steps, n_threads=1, burn_in_p=0.2, p_s=1, ptype='Cyclic', verbose=True):
        
        self.fun = fun
        self.p_0 = np.array(p_0)
        self.p_limits = np.array(p_limits)
        self.p_free = np.array(p_free)
        self.p_sigma = np.array(p_sigma)
        self.p_names = p_names
        self.p_description = p_descr
        self.n_steps = n_steps
        self.n_threads = n_threads
        self.burn_in_p = burn_in_p
        self.p_s = p_s
        
        self.np = self.p_0.size
        self.p_result = np.zeros([n_threads, n_steps, self.p_0.size])
        self.chi_squared = np.zeros([n_threads, n_steps])
        
        p_0 = self.p_limits[:,0] + np.random.random(self.np) * (self.p_limits[:,1] - self.p_limits[:,0])
        self.simulation_thread = MCMCThread(fun, p_0, self.p_limits, self.p_free, self.p_sigma, n_steps, ptype, verbose)

    def run(self):
        self.simulation_thread.run()
        self.p_result[0,:,:] = t.psim[:,:]
        
    def runMPI(self):
        try:
            from mpi4py import MPI
            cm = MPI.COMM_WORLD
            rank = cm.Get_rank()
            size = cm.Get_size()
        except:
            rank = 0
            size = 1
        
        if rank == 0:
            nChains = self.n_threads / size
        else:
            nChains = self.n_threads
            
        for t in self.simulation_thread[0:nChains]:
            t.start()
            t.join()


    def get_results(self, burn_in_p=None, s=None):
        
        if burn_in_p is not None:
            burn_in = int(burn_in_p * self.n_steps)
        else:
            burn_in = int(self.burn_in_p * self.n_steps)
        
        if s is None:
            s = self.p_s
        
        result = np.empty([0,self.p_0.size])
        
        for t in self.simulation_thread:
            result = np.concatenate((result, t.psim[burn_in::s,:]))

        return result

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
    
class MCMCResult():
    def __init__(self, filename, channel, name, burn_in_p = 0.2, s=1):
        
        self.data = {}
        
        f    = tbl.openFile(filename, 'r')
        g_mc = f.root.mcmc
                
        d = f.getNode(g_mc, '%s/%s' %(channel, name))
        
        self.keys = d._v_attrs.p_names
        self.pnames = d._v_attrs.p_description
        self.np = d._v_attrs.n_variables
        
        self.n_steps = d._v_attrs.n_steps
        self.n_chains = d._v_attrs.n_chains
        
        for i in range(self.np):
            self.data[self.keys[i]] = f.getNode(d, 'p%i' %i).read()
        
        self.Chi_squared = f.getNode(d, 'Chi_squared').read()

        f.close()
        
        self.burn_in = burn_in_p * self.n_steps
        self.s = s


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
            
    def __call__(self, i, separate_chains=False, s=None):
        if s is None:
            s = self.s
        
        if not separate_chains:
            r = np.array([])
            for j in range(self.n_chains):
                r = np.concatenate((r,self.data[self.keys[i]][self.burn_in::s,j]))
        else:
            r = self.data[self.keys[i]][self.burn_in::s,:]
        return r
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        n_jumps = 500
    else: 
        n_jumps = int(sys.argv[1])

    if len(sys.argv) < 3:
        n_threads = 2
    else: 
        n_threads = int(sys.argv[2])

    print "Running MCMC test with %i jumps and %i threads." %(n_jumps, n_threads)

    mc = __test(n_jumps, n_threads)

    mc.print_statistics()
    mc.plot_correlation()
    mc.plot_parameter(0, figidx=100)
    mc.plot_parameter(1, figidx=101)
    mc.plot_parameter(2, figidx=102)
    pl.show()


