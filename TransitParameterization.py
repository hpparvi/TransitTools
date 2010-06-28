import numpy as np
import sys

from base import *

class TransitParameterization(object):
    """
    Base class for the different transit parameterization classes. Can't be used in fitting.
    """
    
    def __init__(self, p_init=None, p_low=None, p_high=None,  eccentric=False):
        self.p_names = ['tc', 'P']
        self.p_descr = ['transit center', 'period']
        self.p_units = ['HJD', 'd']

        if not eccentric:
            self.is_circular = True
            self.map_from_orbit = self.map_from_orbit_c
            self.mapped_to_orbit = self.mapped_to_orbit_c
        else:
            self.is_circular = False
            self.p_names.extend(['e', 'w'])
            self.p_descr.extend(['eccentricity', 'argument of pericenter'])
            self.map_from_orbit = self.map_from_orbit_e
            self.mapped_to_orbit = self.mapped_to_orbit_e

        if isinstance(p_init, TransitParameterization):
            p_l = self.mapped_from_orbit_c(p_init.mapped_to_orbit(p_init.p_low))
            p_h = self.mapped_from_orbit_c(p_init.mapped_to_orbit(p_init.p_high))
            p_b = np.array([p_l, p_h])
        
            self.p = self.mapped_from_orbit_c(p_init.mapped_to_orbit(p_init.p))
            self.p_low = p_b.min(0)
            self.p_high = p_b.max(0)
        else:
            self.p_low = p_low if p_low is not None else p_init
            self.p_high = p_high if p_high is not None else p_init
            
            if p_init is None:
                self.p = 0.5 * (p_low + p_high)
            else:
                self.p = p_init
    
    def __str__(self):
        s = ''
        for p in self.p:
            s += '%5.3f'%p
        return s
    
    def update(self, p):
        self.p = p.copy()
        

class OrbitTransitParameterization(TransitParameterization):
    """
    Physical transit parameterization with inclination instead of the impact parameter. 
    The transit is parameterized by
      the transit center t_c
      the period P
      the radius ratio p
      the semi-major axis divided by the stellar radius R
      the inclination i
    """
    
    def __init__(self, p_low, p_high,  p_init=None):
        super(OrbitTransitParameterization,  self).__init__(p_low, p_high,  p_init)
        self.p_names.extend(['p', 'a', 'i'])
        self.p_descr.extend(['radius ratio', 'semi-major axis [R_star]', 
                             'inclination'])
        self.npar = len(self.p_names)
    
    def mapped_from_orbit_c(self, p): return p
        
    def map_from_orbit_c(self, p): self.p = p
        
    def mapped_from_orbit_e(self, p): return p

    def map_from_orbit_e(self, p): self.p = p
    
    def mapped_to_orbit_c(self, p=None):
        return self.p if p is None else p
        
    def mapped_to_orbit_e(self, p=None):
        return self.p if p is None else p

class PhysicalTransitParameterization(TransitParameterization):
    """
    Physical transit parameterization. The transit is parameterized by
      the transit center t_c
      the period P
      the radius ratio p
      the semi-major axis divided by the stellar radius R
      the impact parameter b
      
    The problem with this parameterization is the large correlation between parameters,
    fitting will be highly uneficcient.
    """
    
    def __init__(self, p_init=None, p_low=None, p_high=None, eccentric=False):
        super(PhysicalTransitParameterization,  self).__init__(p_init, p_low, p_high,  eccentric)
        self.p_names.extend(['p', 'a', 'b'])
        self.p_descr.extend(['radius ratio', 'semi-major axis', 
                             'impact parameter'])
        self.p_units.extend(['', 'R_star', ''])
        self.npar = len(self.p_names)

    def __str__(self):
        parfmt = "  %-18.18s %8.3f %s\n"
        result = "Transit parameters:\n"
        for i, p in enumerate(self.p):
            result += parfmt %(self.p_descr[i], p, self.p_units[i])
        
        return result

    def mapped_from_orbit_c(self, p):
        b = p[3]*np.cos(p[4])
        self.p = np.concatenate((p[:4],  [b]))
        return self.p
        
    def map_from_orbit_c(self, p):
        self.p = mapped_from_orbit(p)
    
    def mapped_from_orbit_e(self, p):
        raise NotImplementedError
        
    def map_from_orbit_e(self, p):
        raise NotImplementedError
    
    def mapped_to_orbit_c(self, p=None):
        if p is None: p = self.p
        return np.concatenate((p[:4],  [np.arccos(p[4]/p[3])]))

    def mapped_to_orbit_e(self, p=None):
        raise NotImplementedError

class CarterTransitParameterization(TransitParameterization):
    """
    Transit parameterization by Carter et al. (2008) with modifications by Kipping (2010).
    The transit is parameterized by
        the transit center t_c
        the squared radius ratio p2 (a.k.a. the transit depth)
        the transit width W_1 defined as the average of the flat transit width t_F and total transit width t_T
        the ingress/egress duration t_1
    """
    
    def __init__(self, t_c, p2, W_1, t_1):
        raise NotImplementedError

    def map(self):
        raise NotImplementedError


class KippingTransitParameterization(TransitParameterization):
    """
    Transit parameterization by by Kipping (2010). The transit is parameterized by
        tc  transit center
        P   period
        p2  squared radius ratio p2 (a.k.a. the transit depth)
        iT  two divided by the transit width parameter T_1
        b2  squared impact parameter
    """
    
    def __init__(self, p_init=None, p_low=None, p_high=None, eccentric=False):
        super(KippingTransitParameterization,  self).__init__(p_init, p_low, p_high, eccentric)  
        self.p_names.extend(['p2', 'it', 'b2'])
        self.p_descr.extend(['transit depth', 'transit duration parameter', 
                             'squared impact parameter'])
        self.npar = len(self.p_names)

    def mapped_from_orbit_c(self,  p):
        """
        Maps the physical parameterization to the Kipping parameterization for circular orbits.
        """
        tc = p[0]
        P  = p[1]
        p2 = p[2]*p[2]
        a  = p[3]
        i  = p[4]
        iT = TWO_PI/P / np.arcsin( np.sqrt(1.-p[3]*p[3]*np.cos(i)**2) / (p[3]*np.sin(i)) )
        b2 = (a*np.cos(i))**2
        
        return np.array([tc, P, p2, iT, b2])

    def map_from_orbit_c(self, p):     
        self.p = self.mapped_from_orbit_c(p)
 
    def mapped_from_orbit_e(self, p):
        raise NotImplementedError
 
    def map_from_orbit_e(self,  p):
        self.p = self.get_mapped_from_orbit_e(p)
 
    def mapped_to_orbit_c(self, p=None):
        """
        Maps the Kipping parameterization to the physical parameterization for circular orbits.
        """
        if p is None: p = self.p
        tc = p[0]
        P  = p[1]
        p2 = p[2]
        iT = p[3]
        b2 = p[4]
        a = np.sqrt( (1.-b2) / np.sin(TWO_PI/(P*iT))**2 + b2)
        i = np.arccos(np.sqrt(b2)/a)
        
        return np.array([tc, P, np.sqrt(p2), a, i])
         
    def mapped_to_orbit_e(self,  p=None):
        if p is None: p = self.p
        b2 = p[4]
        P  = p[1]
        iT = p[3]
        e  = p[5]
        w  = p[6]
        e2 = 1 #FIXME: Calculate e2 for eccentric orbits
        a = np.sqrt((1.-b2) / e2 / 
                    np.sin(TWO_PI * np.sqrt(1.-e**2) /
                           (P*iT*e2))**2 + b2/e2)
        i = np.arccos(np.sqrt(b2)/a * (1.+e*np.sin(w)/(1.-e*e)))
            
        return np.array([self.t_c, np.sqrt(self.p2), a, i, e, w])
