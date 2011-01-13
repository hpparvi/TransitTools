import numpy    as np

try:
    import pyopencl as cl
    WITH_OPENCL = True
except ImportError:
    WITH_OPENCL = False

from numpy import sqrt, sin, arccos, pi

from base import transitShape


class MandelAgol(transitShape):

    def __init__(self, use_fortran=True, use_openCL=True):
        #super().__init__()

        if use_fortran:
            pass

        if use_openCL and WITH_OPENCL:
            self.cl_ctx   = cl.Context()
            self.cl_queue = cl.CommandQueue(self.cl_ctx)
            self.uniform_cl_kernel = cl.Program(self.cl_ctx, open("MandelAgol.cl").read()).build().uniform
            self.uniform = self._uniform_cl
            self.z = None
        else:
            self.uniform = self._uniform


    def __call__(self, z, r):
        return self.uniform(z, r)

    def _uniform(self, z, r):
        """
        The analytic transit light curve without limb darkening by Mandel and Algol (APJL 580, L171-L175, 2002).

        Parameters:
          z   normalized distance
          r   radius ratio. Rp/Rs for the primary transit, Rs/Rs for the secondary.
        """

        f = np.zeros(z.size, dtype=np.float64)

        m = (np.abs(1. - r) < z) * (z <= 1. + r)

        z2 = z[m]**2
        p2 = r**2

        k0 = arccos( (p2 + z2 - 1.)  /  (2. * r * z[m]) )
        k1 = arccos( (1. - p2 + z2)  /  (2. * z[m]) )
        f[m] = (1. / pi) * (p2 * k0 + k1 - sqrt(.25 * (4. * z2 - (1. + z2 - p2)**2 ) ))

        f[z <= 1. - r] = p2
        f[z <= r - 1.] = 1.

        return 1. - f

    def _uniform_cl(self, z, r):

        result = np.zeros(z.size, np.float32)

        if self.z is None:
            self.z = z
            self.z_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=z.astype(np.float32))
            self.s_buf = cl.Buffer(self.cl_ctx, cl.mem_flags.WRITE_ONLY, z.nbytes)

        self.uniform_cl_kernel(self.cl_queue, self.z.shape, self.z_buf, np.float32(r), self.s_buf)
        cl.enqueue_read_buffer(self.cl_queue, self.s_buf, result).wait()

        return result


if __name__ == "__main__":
    import pylab as pl
    import timeit

    nreps = 50000

    z = np.linspace(-2., 2., 200)
    r = 0.4

    ## Timeit
    ##
    tPy = timeit.Timer("sh(z,0.5*np.random.rand())", setup="import numpy as np; from MandelAgol import MandelAgol; sh = MandelAgol(use_openCL=False); z = 2. * np.sin(2.*np.pi*np.linspace(0,0.5,300)); r=%f;"%r)
    tCL = timeit.Timer("sh(z,0.5*np.random.rand())", setup="import numpy as np; from MandelAgol import MandelAgol; sh = MandelAgol(use_openCL=True); z = 2. * np.sin(2.*np.pi*np.linspace(0,0.5,300)); r=%f;"%r)

    print "\nTiming Mandel & Agol routines\n"
    print "  Python: ",
    tPy = tPy.timeit(nreps)
    print "%7.3f seconds %10.3f ms/call" %(tPy, 1e3*tPy/nreps)

    print "  OpenCL: ",
    tCL = tCL.timeit(nreps)
    print "%7.3f seconds %10.3f ms/call" %(tCL, 1e3*tCL/nreps)
    print "\n  Ratio:    %7.3f\n" %(tPy/tCL)


    ## Plot
    ##
    shPy = MandelAgol(use_openCL=False)
    shCL = MandelAgol(use_openCL=True)

    pl.plot(z, shPy(np.abs(z),r))
    pl.plot(z, shCL(np.abs(z),r))

    pl.show()
    
