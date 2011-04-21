"""
Implements the differential evolution optimization method by Storn & Price
(Storn, R., Price, K., Journal of Global Optimization 11: 341--359, 1997)

.. moduleauthor:: Hannu Parviainen <hannu@iac.es>
"""
import numpy as np
from numpy import asarray, tile
from numpy.random import seed, random, randint

from transitLightCurve.core import *

try:
    from mpi4py import MPI
    with_mpi = True
except ImportError:
    with_mpi = False

class DiffEvol(object):
    """
    Implements the differential evolution optimization method by Storn & Price
    (Storn, R., Price, K., Journal of Global Optimization 11: 341--359, 1997)
    """
    def __init__(self, fun, bounds, npop, ngen, F=0.5, C=0.5, seed=0, use_mpi=True, verbose=True):
        """

        :param fun: the function to be minimized
        :param bounds: parameter bounds as [npar,2] array
        :param npop:   the size of the population (5*D - 10*D)
        :param  ngen:  the number of generations to run
        :param  F:     the difference amplification factor. Values of 0.5-0.8 are good
                       in most cases.
        :param C:      The cross-over probability. Use 0.9 to test for fast convergence,
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
        self.minfun = fun
        self.bounds = asarray(bounds)
        self.n_gen  = ngen
        self.n_pop  = npop
        self.n_parm = (self.bounds).shape[0]
        self.bl = tile(self.bounds[:,0],[npop,1])
        self.bw = tile(self.bounds[:,1]-self.bounds[:,0],[npop,1])
        
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

                ## --- CROSS OVER ---
                crossover = random(self.n_parm) <= self.C
                u = np.where(crossover, v, r.pop[i,:])

                ## --- FORCED CROSSING ---
                ri = randint(self.n_parm)
                u[ri] = v[ri].copy()

                ufit = self.minfun(u)
    
                if ufit < r.fitness[i]:
                    r.pop[i,:] = u[:].copy()
                    r.fit[i]   = ufit.copy()
                            
            logging.info('Node %i finished generation %4i/%4i  F = %7.5f'%(self.rank, j+1, self.n_gen, r.fit.min()))
        
        if self.use_mpi:
            self._gather_populations()

        return self.result


    def _gather_populations(self):
        if self.use_mpi:
            r = self.result
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


class ParallelDiffEvol(DiffEvol):
    """
    Implements the parallel differential evolution method by Tasoulis et al (Tasoulis , DK, et al.,
    Parallel differential evolution. In: Proceedings of the 2004 Congress on Evolutionary Computation
    (IEEE Cat. No.04TH8753). IEEE; 2004:2023-2029)
    """
    def __init__(self, fun, bounds, npop, ngen, F=0.5, C=0.5, migration_probability=0.1, seed=0, use_mpi=True, verbose=True):
        super(ParallelDiffEvol, self).__init__(fun, bounds, npop, ngen, F, C, seed, use_mpi, verbose)
        self.migration_probability = migration_probability


    def __call__(self):
        """The differential evolution algorithm."""
        r = self.result
        t = np.zeros(3, np.int)

        migrate = np.zeros(1, dtype=np.short)

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

                ## --- CROSS OVER ---
                crossover = random(self.n_parm) <= self.C
                u = np.where(crossover, v, r.pop[i,:])

                ## --- FORCED CROSSING ---
                ri = randint(self.n_parm)
                u[ri] = v[ri].copy()

                ufit = self.minfun(u)
    
                if ufit < r.fitness[i]:
                    r.pop[i,:] = u[:].copy()
                    r.fit[i]   = ufit.copy()
                    
            ## -- migration --
            if self.rank == 0:
                migrate[:] = self.migration_probability > random()
            self.cm.Bcast(migrate, 0)
            self.cm.Barrier()
            
            if migrate[0]:
                minidx = np.argmin(r.fitness)
                rndidx = minidx
                while rndidx == minidx:
                    rndidx = randint(self.n_pop)
                    
                sendid = (self.rank+1) % self.size
                recvid = self.size-1 if self.rank == 0 else self.rank-1
                
                self.cm.Send(r.pop[minidx, :], sendid, 10)
                self.cm.Recv(r.pop[rndidx, :], recvid, 10)
                self.cm.Send(r.fitness[minidx:minidx+1], sendid, 10)
                self.cm.Recv(r.fitness[rndidx:rndidx+1], recvid, 10)
                
            logging.info('Node %i finished generation %4i/%4i  F = %7.5f'%(self.rank, j+1, self.n_gen, r.fit.min()))

        if self.use_mpi:
            self._gather_populations()

        return self.result


class MPIDiffEvolFull(DiffEvol):
    """
    Implements the differential evolution optimization method by Storn & Price
    (Storn, R., Price, K., Journal of Global Optimization 11: 341--359, 1997)
    """
    def __call__(self):
        r = self.result
        t = np.zeros(3, np.int)

        pop_curr = r.pop.copy() if self.rank == 0 else np.zeros([self.n_pop, self.n_parm])
        pop_trial = np.zeros([self.n_pop, self.n_parm])
        fitness  = np.zeros(self.n_pop)

        from math import ceil
        self.cm.Barrier()

        npop_loc_max = int(ceil(float(self.n_pop)/self.size))
        npop_loc = npop_loc_max
        if (self.rank == self.size-1) and (self.n_pop%npop_loc_max != 0):
            npop_loc = self.n_pop%npop_loc_max

        pop_curr_loc  = np.zeros([npop_loc, self.n_parm])
        pop_trial_loc = np.zeros([npop_loc, self.n_parm])
        fitness_loc   = np.zeros(npop_loc)

        if self.rank == 0:
            for i in xrange(self.n_pop):
                fitness[i] = self.minfun(r.pop[i,:])
            
        for j in xrange(self.n_gen):
            if self.rank == 0:
                for i in xrange(self.n_pop):
                    t[:] = i
                    while  t[0] == i:
                        t[0] = randint(self.n_pop)
                    while  t[1] == i or t[1] == t[0]:
                        t[1] = randint(self.n_pop)
                    while  t[2] == i or t[2] == t[0] or t[2] == t[1]:
                        t[2] = randint(self.n_pop)

                    v = pop_curr[t[0],:] + self.F * (pop_curr[t[1],:] - pop_curr[t[2],:])

                    ## --- CROSS OVER ---
                    crossover = random(self.n_parm) <= self.C
                    pop_trial[i,:] = np.where(crossover, v, pop_curr[i,:])

                    ## --- FORCED CROSSING ---
                    ri = randint(self.n_parm)
                    pop_trial[i,ri] = v[ri].copy()

            self.cm.Scatter([pop_curr, npop_loc*self.n_parm, MPI.DOUBLE],
                            [pop_curr_loc, npop_loc*self.n_parm, MPI.DOUBLE])

            self.cm.Scatter([pop_trial, npop_loc*self.n_parm, MPI.DOUBLE],
                            [pop_trial_loc, npop_loc*self.n_parm, MPI.DOUBLE])

            self.cm.Scatter(fitness, fitness_loc)

            for i in xrange(npop_loc):
                ufit = self.minfun(pop_trial_loc[i,:])
                if ufit < fitness_loc[i]:
                    pop_curr_loc[i,:] = pop_trial_loc[i,:]
                    fitness_loc[i]    = ufit

            self.cm.Gather([pop_curr_loc, npop_loc*self.n_parm, MPI.DOUBLE],
                           [pop_curr, npop_loc*self.n_parm, MPI.DOUBLE])

            self.cm.Gather(fitness_loc, fitness)
                            
            if self.rank == 0:
                logging.info('Finished generation %4i/%4i  F = %7.5f'%(j+1,
                                                                       self.n_gen,
                                                                       fitness.min()))
        self.result.minidx = np.argmin(r.fitness)
        return self.result


class DiffEvolResult(FitResult):
    """
    Encapsulates the results from the differential evolution fitting.
    """
    def __init__(self, npop, npar, bl, bw):
        self.population = bl + random([npop, npar]) * bw
        self.fitness    = np.zeros(npop)
        self.pop        = self.population
        self.fit        = self.fitness

    def get_fitness(self):
        """Returns the best-fit value of the minimized function."""
        return self.fitness[self.minidx]

    def get_chi(self):
        return self.get_fitness()
    
    def get_fit(self):
        """Returns the best-fit solution."""
        return self.population[self.minidx,:]

if __name__ == '__main__':
    de = DiffEvol(lambda P: np.sum((P-2.2)**2), [[-3, 2], [-2, 3], [-4, 2]], 50, 50, seed=0)
    pde = ParallelDiffEvol(lambda P: np.sum((P-2.2)**2), [[-3, 2], [-2, 3], [-4, 2]], 50, 50, seed=0,
                           migration_probability=0.25)

    de()
    pde()
        
    if de.rank == 0:
        print de.result.get_chi(), pde.result.get_chi()
        print de.result.get_fit(), pde.result.get_fit()
