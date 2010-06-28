import numpy as np
import unittest

from numpy.testing import assert_almost_equal as aeq

class Orbit(object):
    def __init__(self, p=None):
        self.p = p

    def update(self, tp):
        self.p = tp.mapped_to_orbit()

    def get_transit_center(self):
        return self.p[0]
        
    def get_period(self):
        return self.p[1]
        
    def get_radius_ratio(self):
        return self.p[2]

    def get_semimajor_axis(self):
        return self.p[3]
        
    def get_inclination(self):
        return self.p[4]

    def transit_depth(self):
        return self.p[2]**2


class CircularOrbit(Orbit):
    """
    Circular orbit parameterized by
        tc  transit center time [d]
        P    period [d]
        p    ratio of the planet radius to the stellar radius
        a    semi-major axis divided by the stellar radius
        i    inclination [rad]
    """
    def __init__(self, p=None, mode='time'):
        self.p = p
        modes = {'time': self.projected_distance_t, 
                 'phase': self.projected_distance_p}
                 
        try:
            self.projected_distance = modes[mode]
        except Keyerror:
            raise ValueError

    def transit_duration(self):
        #FIXME: transit_duration acts funny.
        return ((self.p[0] / np.pi) * 
                np.arcsin( (1./self.p[3] * 
                np.sqrt(((1.+self.p[2])**2 - (self.p[3] * np.cos(self.p[4]))**2)/(1.-np.cos(self.p[4])**2))) 
                ))

    def transit_duration_simple(self):
        return self.p[1]/np.pi * 1./self.p[3] * np.sqrt((1.+self.p[2])**2 - (self.p[3] * np.cos(self.p[4]))**2)
        
    def transit_depth(self):
        return self.p[2]**2
        
    def impact_parameter(self):
        return self.p[3] * np.cos(self.p[4])
        
    def projected_distance_t(self, t):
        return projected_distance_c_t(t, self.p[0], self.p[1], self.p[3], self.p[4])

    def projected_distance_p(self, p):
        return projected_distance_c_p(p, self.p[3], self.p[4])


class EccentricOrbit(Orbit):
    """
    Eccentric orbit parameterized by
        t_c  transit center time
        P    period
        i    inclination
        e    eccentricity
        w    argument of pericenter
        p    radius ratio of the planet and the star
        a    semi-major axis
    """
    def __init__(self, t_c, P, i, e, w, p, a):
        self.t_c = t_c
        self.P = P
        self.i = i
        self.e = e
        self.w = w
        self.p = p
        self.a = a

    def transit_duration(self):
        return ((self.P / np.pi) * 
                np.arcsin( (1./self.a * 
                np.sqrt(((1.+self.p)**2 - (self.a * np.cos(self.i))**2)/(1.-np.cos(self.i)**2))) 
                ))

    def transit_duration_simple(self):
        return self.P/np.pi * 1./self.a * np.sqrt((1.+self.p)**2 - (self.a * np.cos(self.i))**2)
        
    def impact_parameter(self):
        return self.a * np.cos(self.i)
        
    def projected_distance(self, t):
        return e_projected_distance(t, self.t_c, self.P, self.i, self.e, self.w, self.a)


def projected_distance_c_t(t, t_c, P, a, i):
        dt = t - t_c
        n  = 2.*np.pi/P
        return a*np.sqrt(np.sin(n*dt)**2 + (np.cos(i)*np.cos(n*dt))**2)

def projected_distance_c_p(p, a, i):
        return a*np.sqrt(np.sin(p)**2 + (np.cos(i)*np.cos(p))**2)

def projected_distance_e_t(t, t_c, P, i, e, w, a):
        #FIXME: projected_distance_e_t should be fixed.
        raise NotImplementedError
        dt = t - t_c
        r  = a*(1.-e*e)/(1.+e*np.sin(w))
        n  = 2.*np.pi/P * (1.+e*np.sin(w))**2 / (1.-e*e)**1.5
        return a*np.sqrt(np.sin(n*dt)**2 + (np.cos(i)*np.cos(n*dt))**2)

def projected_distance_e_p(t, t_c, P, i, e, w, a):
        #FIXME: projected_distance_e_p should be fixed.
        raise NotImplementedError
        dt = t - t_c
        r  = a*(1.-e*e)/(1.+e*np.sin(w))
        n  = 2.*np.pi/P * (1.+e*np.sin(w))**2 / (1.-e*e)**1.5
        return a*np.sqrt(np.sin(n*dt)**2 + (np.cos(i)*np.cos(n*dt))**2)


if __name__ == '__main__':
    from math import radians
    
    p = np.array([0.5, 2, 0.1, 10, 0.5*np.pi])
    o = CircularOrbit(p)
    
    print o.transit_duration()
    print o.transit_duration_simple()
