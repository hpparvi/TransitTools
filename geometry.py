"""
Includes the classes and tools to define the geometry of the planet and its orbit.

Author
  Hannu Parviainen <hpparvi@gmail.com>

Date
  9.1.2011  
"""
from numpy import asarray, array, zeros, sin, cos, sqrt, sign

from transitparameterization import TransitParameterization
from core import *

class Geometry(object):
    """Defines the planetary orbit and the planet to star radius ratio.

    Geometry class defines the geometry of the planet and its orbit.

    The parameter vector divides into separate parts. First comes the
        [0] k  [Rs]  planet to star radius ratio

    next, the parameters for a circular orbit 
        [1] t0 [d]   transit center
        [2] P  [d]   period        
        [3] a  [Rs]  semi-major axis divided by the stellar radius
        [4] i  [rad] inclination [rad]

    and the parameters introduced by eccentricity
        [5] e  [ - ] eccentricity
        [6] w  [rad] argument of pericenter

    Finally, additional geometry parameters can be included with
        add_parm a numpy array
        add_name a list with parameter names
        add_desc a list of parameter descriptions
    """
    def __init__(self, p=None, eccentric=False, mode='time', add_parm=None, add_name=None, add_desc=None):
        self.mode = mode
        self.npar = 5 if not eccentric else 7
        self.eccentric = eccentric
        self.type = 'circular' if not eccentric else 'eccentric'

        pdfuncs = {'time':  {'circular':self._projected_distance_c_t,'eccentric':self._projected_distance_e_t}, 
                   'phase': {'circular':self._projected_distance_c_p,'eccentric':self._projected_distance_e_p}}
        
        try:
            self.projected_distance = pdfuncs[self.mode][self.type]
        except Keyerror:
            raise ValueError

        if p is None:
            self.pv = zeros(self.npar)
        else:
            assert p.size == self.npar
            self.pv = asarray(p)

        self.k  = self.pv[0:1]
        self.t0 = self.pv[1:2]
        self.p  = self.pv[2:3]
        self.a  = self.pv[3:4]
        self.i  = self.pv[4:5]

        self.e = None if not eccentric else self.pv[5:6]
        self.w = None if not eccentric else self.pv[6:7]

        if add_parm is not None:
            self.add_parm = add_parm
            self.add_name = add_name
            self.add_desc = add_desc


    def update(self, tp):
        self.pv[:] = tp.map_to_orbit().pv

    def get_transit_center(self):
        return self.t0
        
    def get_period(self):
        return self.p
        
    def get_radius_ratio(self):
        return self.a

    def get_semimajor_axis(self):
        return self.a
        
    def get_inclination(self):
        return self.i

    def get_transit_depth(self):
        return self.k**2

    #FIXME: impact parameter for eccentric orbit not implemented.
    def impact_parameter(self):
        if self.eccentric: raise NotImplementedError
        return self.a * np.cos(self.i)

    #FIXME: Transit duration not implemented.
    def transit_duration(self):
        raise NotImplementedError

    #FIXME: simple transit duration not implemented.
    def transit_duration_simple(self):
        raise NotImplementedError
        
    ##--- Projected distance functions ---
    ##
    def _projected_distance_c_t(self, t):
        dt = t - self.t0
        ph = dt*TWO_PI/self.p
        s = sign(-1*((dt/self.p + 0.25)%1-0.5))

        return s * self.a*sqrt(sin(ph)**2 + (cos(self.i)*cos(ph))**2)

    def _projected_distance_c_t_ftmpl(n=None, fname=None):
        dimstr = ", dimension(%i)" %n if n is not None else "" 
        fname  = fname if fname is not None else "projected_distance"
        src_str  = "pure function %s(t, t_c, p, a, i)\n\timplicit none\n"%fname
        src_str += "\treal(8), intent(in)%s :: t, t_c, p, a, i" %dimstr
        src_str += "\treal(8), intent(out)%s :: z\n\treal(8)%s :: dt,n" %(dimstr, dimstr)
        src_str += "\tdt = t - t_c\n"
        src_str += "\tn  = 2.*pi/P\n"
        src_str += "\tz  = a*sqrt(sin(n*dt)**2 + (cos(i)*cos(n*dt))**2)\n"
        src_str += "end function %s"%fname
        return src_str

    def _projected_distance_c_p(self,p):
            return self.a*sqrt(sin(p)**2 + (cos(self.i)*cos(p))**2)

    def _projected_distance_c_p_ftmpl(n=None, fname=None):
        dimstr = ", dimension(%i)" %n if n is not None else "" 
        fname  = fname if fname is not None else "projected_distance"
        src_str  = "pure function %s(p, a, i)\n\timplicit none\n" %fname
        src_str += "\treal(8), intent(in)%s :: p, a, i\n\treal(8), intent(out)%s :: z\n" %(dimstr, dimstr)
        src_str += "\tz  = a*sqrt(sin(p)**2 + (cos(i)*cos(p))**2)\n"
        src_str += "end function %s" %fname
        return src_str

    #FIXME: projected_distance_e_t for eccentric orbit is not implemented.
    def _projected_distance_e_t(self, t):
        raise NotImplementedError
        dt = t - t_c
        r  = a*(1.-e*e)/(1.+e*sin(w))
        n  = TWO_PI/P * (1.+e*sin(w))**2 / (1.-e*e)**1.5
        return a*sqrt(sin(n*dt)**2 + (cos(i)*cos(n*dt))**2)

    #FIXME: projected_distance_e_p for eccentric orbit is not implemented.
    def _projected_distance_e_p(self, p):
        raise NotImplementedError
 

if __name__ == '__main__':
    from math import radians
    
    p = array([0.1, 0.0, 2, 10, HALF_PI])
    g = Geometry(p)

    print g.projected_distance(0.0)
    
    #print projected_distance_c_t_fortran(np.random.random(10), fname='pd_t')
    #print projected_distance_c_p_fortran()
    
