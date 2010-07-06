import numpy as np
from numpy.random import random, randint
from numpy import asarray

from base import *

try:
    from mpi4py import MPI
    with_mpi = True
except ImportError:
    with_mpi = False

class DiffEvol(object):
    """
    Implements the differential evolution optimization method by Storn & Price
    (Storn, R., Price, K., Journal of Global Optimization 11: 341--359, 1997)
    
    Parameters:
      fun      the function to be minimized
      bounds   parameter bounds as [npar,2] array
      npop     the size of the population (5*D - 10*D)
      ngen     the number of generations to run
      F        the difference amplification factor. Values of 0.5-0.8 are good
               in most cases.
      C        the cross-over probability. Use 0.9 to test for fast convergence,
               and smaller values (~0.1) for a more elaborate search.
    
    N free parameters
    N population vectors (pv1 .. pvN)
    
    Population = [pv1_x1 pv1_x2 pv1_x3 ... pv1_xN]
                 [pv2_x1 pv2_x2 pv2_x3 ... pv2_xN]
                 .
                 .
                 .
                 [pvN_x1 pvN_x2 pvN_x3 ... pvN_xN]
    
    Population = [pv, parameter]
    """ 
    
    def __init__(self, fun, bounds, npop, ngen, F=0.5, C=0.5, id=0, size=1, seed=0, use_mpi=True, verbose=True):
        self.minfun = fun
        self.bounds = asarray(bounds)
        self.n_gen  = ngen
        self.n_pop  = npop
        self.n_parm = (self.bounds).shape[0]
        self.bl = np.tile(self.bounds[:,0],[npop,1])
        self.bw = np.tile(self.bounds[:,1]-self.bounds[:,0],[npop,1])
        
        self.seed = seed
        self.F = F
        self.C = C
        
        if with_mpi and use_mpi:
            self.cm = MPI.COMM_WORLD
            self.rank = self.cm.Get_rank()
            self.size = self.cm.Get_size()
            self.use_mpi = True
            self.seed += self.rank
            logging.info('Created node %i'%self.rank)
        else:        
            self.use_mpi = False
            self.rank = 0
            self.size = 1    

        np.random.seed(self.seed)
        self.result = DiffEvolResult(npop, self.n_parm, self.bl, self.bw)

    def __call__(self):
        """The differential evolution algorithm."""
        r = self.result
        t = np.zeros(3, np.int)
        
        for i in xrange(self.n_pop):
            r.fit[i] = self.minfun(r.pop[i,:])
            
        for j in xrange(self.n_gen):
            for i in xrange(self.n_pop):
                t[:] = i
                while  t[0] == i:
                    t[0] = randint(self.n_pop)
                while  t[1] == i or t[1] == t[0]:
                    t[1] = randint(self.n_pop)
                while  t[2] == i or t[2] == t[0] or t[2] == t[1]:
                    t[2] = randint(self.n_pop)
    
                v = r.pop[t[0],:] + self.F * (r.pop[t[1],:] - r.pop[t[2],:])
                crossover = random(self.n_parm) <= self.C
                u = r.pop[i,:].copy()
                u[crossover] = v[crossover]
                ri = randint(self.n_parm)
                u[ri] = v[ri]
                ufit = self.minfun(u)
    
                if ufit < r.fitness[i]:
                    r.pop[i,:] = u[:]
                    r.fit[i]   = ufit
                            
            logging.info('Node %i finished generation %4i/%4i  F = %7.5f'%(self.rank, j+1, self.n_gen, r.fit.min()))
        
        ## --- MPI communication ---
        ##
        if self.use_mpi:
            if self.rank == 0:
                tpop = np.zeros([self.size*self.n_pop,  self.n_parm])
                tfit = np.zeros(self.size*self.n_pop)
                tpop[:self.n_pop, :] = r.pop[:, :]
                tfit[:self.n_pop] = r.fit[:]
                for node in range(1, self.size):
                    s=node*self.n_pop; e=(node+1)*self.n_pop
                    logging.info('Master receiving node %i' %node)
                    self.cm.Recv([tpop[s:e, :], MPI.DOUBLE], source=node, tag=77)
                    self.cm.Recv([tfit[s:e], MPI.DOUBLE], source=node, tag=77)
                    logging.info('Master received node %i' %node)
                r.population = tpop; r.pop = tpop
                r.fitness = tfit; r.fit = tfit
            else:
                logging.info('Node %i sending results to master' %self.rank)
                self.cm.Send([r.pop[:, :], MPI.DOUBLE], dest=0, tag=77)
                self.cm.Send([r.fit[:], MPI.DOUBLE], dest=0, tag=77)
                
        self.result.minidx = np.argmin(r.fitness)
        return self.result
        #return r.get_chi(), r.get_fit()

class DiffEvolResult(FitResult):
    def __init__(self, npop, npar, bl, bw):
        self.population = bl + random([npop, npar]) * bw
        self.fitness    = np.zeros(npop)
        self.pop        = self.population
        self.fit        = self.fitness

    def get_chi(self):
        return self.fitness[self.minidx]
    
    def get_fit(self):
        return self.population[self.minidx,:]

if __name__ == '__main__':
    de = DiffEvol(lambda P: np.sum((P-1)**2), [[-3, 2], [-2, 3], [-4, 2]], 50, 200, seed=0)
    de()
        
    if de.rank == 0:
        print de.result.get_chi()
        print de.result.get_fit()