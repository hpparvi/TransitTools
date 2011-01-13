#!/usr/bin/env python
import numpy as np
from scipy.special import jacobi, gamma, gammaln

from transitLightCurve.core import *
from transitmodel import TransitModel

class Gimenez(TransitModel):
    """Implements the transit model by A. Gimenez (A&A 450, 1231--1237, 2006)."""

    def __init__(self, method='python', n_threads=0, float_t=np.float32):
        self.float_t = float_t
        self.n_threads = n_threads

        if method == 'python':
            self.shape   = self._shape_py
        elif method == 'fortran':
            import Gimenez_f
            self.shape = Gimenez_f.gimenez_f.gimenez


    def __call__(self, z, r, u=[], npol=100, n_threads=0):
        return self.shape(z, r, u, npol, n_threads)


    def _shape_py(self, z, r, u=[], npol=100, n_threads=0):
        """
        Transit light curve model by A. Gimenez (A&A 450, 1231--1237, 2006).
        """
 
        z = np.array(z, DOUBLE)
        u = np.array(u, DOUBLE)
        r = DOUBLE(r)

        def alpha(b, c, n, jn):

            nu = (n + 2.)/2.
            norm = b*b * (1. - c*c)**(1. + nu) / (nu * gamma(1. + nu))

            d = self.jacobi_Burkardt(jn, 0., 1. + nu, 1. - 2. * c**2)
            e = self.jacobi_Burkardt(jn, nu, 1.,      1. - 2. * (1. - b))

            sum = DOUBLE(0.0)

            for j in range(jn):
                nm = gammaln(nu+j+1.) - gammaln(j+2.)
                vl = (-1)**j * (2.+2.*j+nu)*np.exp(nm)
                nm = gammaln(j+1) + gammaln(nu+1.) - gammaln(j+1+nu)	

                e[j] *= np.exp(nm)
                vl   *= d[j] * e[j] * e[j]
                sum  += vl

            return norm * sum

        mask = z < (1.+r)

        a = np.zeros([u.size+1, mask.sum()], DOUBLE)

        b = r/(1.+r)
        c = z[mask]/(1.+r)

        n = np.arange(u.size+1)
        f = np.ones(z.size, DOUBLE)

        for i in n:
            a[i,:] = alpha(b, c, i, npol)

        Cn = np.ones(u.size+1, DOUBLE)

        if u.size > 0:
            Cn[0] = (1. - u[:].sum()) / (1. - (n[1:] * u[:] / (n[1:]+2.)).sum())
            Cn[1:] = u[:] / (1. - n[1:] * u[:] / (n[1:]+2.))

        f[mask] = 1. - (np.tile(Cn, (mask.sum(), 1)).transpose() * a).sum(0)

        return f

    def jacobi_Burkardt(self, n, alpha, beta, x, cx=None):

        if n < 0 or alpha < -1. or beta < -1.:
            print "Error in Jacobi_Burkardt: bad parameters."
            sys.exit()

        if cx is None:
            cx = np.zeros([n+1, x.size], DOUBLE)

        cx[0, :] = 1.

        if n > 0:
            cx[1, :] = ( 1. + 0.5 * ( alpha + beta ) ) * x + 0.5 * ( alpha - beta )

            for i in range(2,n):
                ri = DOUBLE(i)
                c1 = 2. * ri * ( ri + alpha + beta ) * ( 2. * ri - 2. + alpha + beta )
                c2 = ( 2.* ri - 1. + alpha + beta ) * ( 2. * ri  + alpha + beta ) * ( 2.* ri - 2. + alpha + beta )
                c3 = ( 2.* ri - 1. + alpha + beta ) * ( alpha + beta ) * ( alpha - beta )
                c4 = - 2. * ( ri - 1. + alpha ) * ( ri - 1. + beta )  * ( 2.* ri + alpha + beta )

                cx[i,:] = ( ( c3 + c2 * x ) * cx[i-1] + c4 * cx[i-2] ) / c1

        return cx
