import numpy as np
import sys
from string import Template
from math import cos, acos, sqrt
from base import *

class Mapping(object):
    __slots__ = ['name','dependencies','mapping']

    def __init__(self, name, dependencies, mapping):
        self.name = name
        self.dependencies = dependencies
        self.mapping = Template(mapping)

    def is_mappable(self, parameterization):
        return set(self.dependencies).issubset(set(parameterization))

    def get_mapping(self, p_from, p_to):
        mapdict = {}
        mapdict[self.name] = '    v_to[%i]' %p_to.index(self.name)
        for p in self.dependencies:
            mapdict[p] = 'v_from[%i]' %p_from.index(p)
        return self.mapping.substitute(**mapdict)


class TransitParameter(object):
    __slots__ = ['name','description','unit','mappings']

    def __init__(self, name, description, unit):
        self.name = name
        self.description = description
        self.unit = unit
        

class TransitParameterization(object):

    parameters = {'tc' : TransitParameter('tc',  'transit center',       'HJD'),
                  'P'  : TransitParameter( 'P',          'period',         'd'),
                  'p'  : TransitParameter( 'p',    'radius ratio',    'R_star'),
                  'a'  : TransitParameter( 'a', 'semi-major axis',    'R_star'),
                  'i'  : TransitParameter( 'i',     'inclination',          ''),
                  'b'  : TransitParameter( 'b', 'impact parameter',         ''),
                  'p2' : TransitParameter('p2', 'squared radius ratio',     ''),
                  'it' : TransitParameter('it', 'transit width parameter',  ''),
                  'b2' : TransitParameter('b2', 'squared impact parameter', '')}

    parameterizations   = {'orbit'    : ['tc', 'P', 'p',  'a',  'i'],
                           'physical' : ['tc', 'P', 'p',  'a',  'b'],
                           'kipping'  : ['tc', 'P', 'p2', 'it', 'b2']}

    mappings = { 'p' : [Mapping('p',           ['p2'], '$p = sqrt($p2)'    )],
                 'a' : [Mapping('a',  ['P','it','b2'], '$a = sqrt( (1.-$b2) / sin(TWO_PI/($P*$it))**2 + $b2)')],
                 'b' : [Mapping('b',        ['a','i'], '$b = $a*cos($i)'   ),
                        Mapping('b',           ['b2'], '$b = sqrt($b2)'    )],
                 'i' : [Mapping('i',        ['a','b'], '$i = acos($b/$a)'  )],
                'b2' : [Mapping('b2',           ['b'], '$b2 = $b*$b'       ),
                        Mapping('b2',       ['a','i'], '$b2 = ($a*cos($i))**2')],
                'it' : [Mapping('it',   ['a','i','P'], '$it = TWO_PI/$P/asin(sqrt(1.-$a*$a*cos($i)**2)/($a*sin($i)))')],
                'p2' : [Mapping('p2',           ['p'], '$p2 = $p*$p'       )]}


    def __init__(self, type, init=None):
        if type in self.parameterizations.keys():
            self.type = type
            self.parameter_set = self.parameterizations[type]
            self.parameter_vector = (np.asarray(init) if init is not None and np.asarray(init).size == len(self.parameter_set) 
                                     else np.zeros(len(self.parameter_set)))
            self.tp = self.type
            self.ps = self.parameter_set
            self.pv = self.parameter_vector
        else:
            logging.error("Bad parameterization type %s!"%type)
            sys.exit()

    def __str__(self):
        istr = '\n%s transit parameterization\n\n' %self.type.capitalize()
        for p,v in zip(self.ps, self.pv):
            istr += '    %-20s %6.3f\n' %(self.parameters[p].description, v)
        return istr

    def update(p):
        self.parameter_vector = p.copy()
        self.pv = self.parameter_vector

    def generate_mapping(self, p_from, p_to):
        s_from = self.parameterizations[p_from]
        s_to   = self.parameterizations[p_to]

TP = TransitParameterization

def generate_mapping(p_from, p_to):
    s_from  = TransitParameterization.parameterizations[p_from]
    s_to    = TransitParameterization.parameterizations[p_to]

    map_str  = 'def mapping(p_from):\n'
    map_str += '    if isinstance(p_from, TransitParameterization):\n'
    map_str += '        v_from = p_from.pv\n'
    map_str += '    else:\n'
    map_str += '        v_from = p_from\n\n'
    map_str += '    v_to = np.zeros(v_from.shape)\n\n'

    for i, p in enumerate(s_to):
        if p in s_from:
            map_str += '    v_to[%i] = v_from[%i]\n' %(i, s_from.index(p))
        else:
            for mapping in TP.mappings[p]:
                if mapping.is_mappable(s_from): break
            else:
                logging.error("Couldn't find suitable mapping for %s:%s." %(p,str(mapping.dependencies)))
                sys.exit()

            map_str += mapping.get_mapping(s_from, s_to)+'\n'

    map_str += '\n    if isinstance(p_from, TransitParameterization):\n'
    map_str += '        return TransitParameterization("%s", v_to)\n' %p_to
    map_str += '    else:\n'
    map_str += '        return v_to\n'

    exec map_str
    return mapping


print TP.parameterizations['orbit']
print TP.parameterizations['physical']

pp = TransitParameterization('physical',[1,2,3,4,0.9])
po = TransitParameterization('orbit')

o2p = generate_mapping('orbit','physical')
p2o = generate_mapping('physical','orbit')

po = p2o(pp)

print pp
print po


# class TransitParameterization(object):
#     """
#     Base class for the different transit parameterization classes. Can't be used in fitting.
#     """
    
#     def __init__(self, p_init=None, p_low=None, p_high=None, p_sigma=None, eccentric=False):
#         self.p_names = ['tc', 'P']
#         self.p_descr = ['transit center', 'period']
#         self.p_units = ['HJD', 'd']

#         if not eccentric:
#             self.is_circular = True
#             self.map_from_orbit = self.map_from_orbit_c
#             self.mapped_to_orbit = self.mapped_to_orbit_c
#         else:
#             self.is_circular = False
#             self.p_names.extend(['e', 'w'])
#             self.p_descr.extend(['eccentricity', 'argument of pericenter'])
#             self.map_from_orbit = self.map_from_orbit_e
#             self.mapped_to_orbit = self.mapped_to_orbit_e

#         if isinstance(p_init, TransitParameterization):
#             p_l = self.mapped_from_orbit_c(p_init.mapped_to_orbit(p_init.p_low))
#             p_h = self.mapped_from_orbit_c(p_init.mapped_to_orbit(p_init.p_high))
#             p_b = np.array([p_l, p_h])
        
#             self.p = self.mapped_from_orbit_c(p_init.mapped_to_orbit(p_init.p))
#             #self.p_sigma = self.mapped_from_orbit_c(p_init.mapped_to_orbit(p_init.p_sigma))
#             self.p_sigma = p_init.p_sigma
#             self.p_low = p_b.min(0)
#             self.p_high = p_b.max(0)
#         else:
#             self.p_low = p_low if p_low is not None else p_init
#             self.p_high = p_high if p_high is not None else p_init
            
#             self.p_sigma = p_sigma
            
#             if p_init is None:
#                 self.p = 0.5 * (p_low + p_high)
#             else:
#                 self.p = p_init
    
#     def __str__(self):
#         s = ''
#         for p in self.p:
#             s += '%5.3f'%p
#         return s
    
#     def update(self, p):
#         self.p = p.copy()
        

# class OrbitTransitParameterization(TransitParameterization):
#     """
#     Physical transit parameterization with inclination instead of the impact parameter. 
#     The transit is parameterized by
#       the transit center t_c
#       the period P
#       the radius ratio p
#       the semi-major axis divided by the stellar radius R
#       the inclination i
#     """
    
#     def __init__(self, p_init=None, p_low=None, p_high=None, p_sigma=None):
#         super(OrbitTransitParameterization,  self).__init__(p_init, p_low, p_high, p_sigma)
#         self.p_names.extend(['p', 'a', 'i'])
#         self.p_descr.extend(['radius ratio', 'semi-major axis [R_star]', 
#                              'inclination'])
#         self.npar = len(self.p_names)
    
#     def mapped_from_orbit_c(self, p): return p
        
#     def map_from_orbit_c(self, p): self.p = p
        
#     def mapped_from_orbit_e(self, p): return p

#     def map_from_orbit_e(self, p): self.p = p
    
#     def mapped_to_orbit_c(self, p=None):
#         return self.p if p is None else p
        
#     def mapped_to_orbit_e(self, p=None):
#         return self.p if p is None else p

# class PhysicalTransitParameterization(TransitParameterization):
#     """
#     Physical transit parameterization. The transit is parameterized by
#       the transit center t_c
#       the period P
#       the radius ratio p
#       the semi-major axis divided by the stellar radius R
#       the impact parameter b
      
#     The problem with this parameterization is the large correlation between parameters,
#     fitting will be highly uneficcient.
#     """
    
#     def __init__(self, p_init=None, p_low=None, p_high=None, p_sigma=None, eccentric=False):
#         super(PhysicalTransitParameterization,  self).__init__(p_init, p_low, p_high, p_sigma, eccentric)
#         self.p_names.extend(['p', 'a', 'b'])
#         self.p_descr.extend(['radius ratio', 'semi-major axis', 
#                              'impact parameter'])
#         self.p_units.extend(['', 'R_star', ''])
#         self.npar = len(self.p_names)

#     def __str__(self):
#         parfmt = "  %-18.18s %8.3f %s\n"
#         result = "Transit parameters:\n"
#         for i, p in enumerate(self.p):
#             result += parfmt %(self.p_descr[i], p, self.p_units[i])
        
#         return result

#     def mapped_from_orbit_c(self, p):
#         b = p[3]*np.cos(p[4])
#         self.p = np.concatenate((p[:4],  [b]))
#         return self.p
        
#     def map_from_orbit_c(self, p):
#         self.p = mapped_from_orbit(p)
    
#     def mapped_from_orbit_e(self, p):
#         raise NotImplementedError
        
#     def map_from_orbit_e(self, p):
#         raise NotImplementedError
    
#     def mapped_to_orbit_c(self, p=None):
#         if p is None: p = self.p
#         return np.concatenate((p[:4],  [np.arccos(p[4]/p[3])]))

#     def mapped_to_orbit_e(self, p=None):
#         raise NotImplementedError

# class CarterTransitParameterization(TransitParameterization):
#     """
#     Transit parameterization by Carter et al. (2008) with modifications by Kipping (2010).
#     The transit is parameterized by
#         the transit center t_c
#         the squared radius ratio p2 (a.k.a. the transit depth)
#         the transit width W_1 defined as the average of the flat transit width t_F and total transit width t_T
#         the ingress/egress duration t_1
#     """
    
#     def __init__(self, t_c, p2, W_1, t_1):
#         raise NotImplementedError

#     def map(self):
#         raise NotImplementedError


# class KippingTransitParameterization(TransitParameterization):
#     """
#     Transit parameterization by by Kipping (2010). The transit is parameterized by
#         tc  transit center
#         P   period
#         p2  squared radius ratio p2 (a.k.a. the transit depth)
#         iT  two divided by the transit width parameter T_1
#         b2  squared impact parameter
#     """
    
#     def __init__(self, p_init=None, p_low=None, p_high=None, p_sigma=None, eccentric=False):
#         super(KippingTransitParameterization,  self).__init__(p_init, p_low, p_high, p_sigma, eccentric)  
#         self.p_names.extend(['p2', 'it', 'b2'])
#         self.p_descr.extend(['transit depth', 'transit duration parameter', 
#                              'squared impact parameter'])
#         self.npar = len(self.p_names)

#     def mapped_from_orbit_c(self,  p):
#         """
#         Maps the physical parameterization to the Kipping parameterization for circular orbits.
#         """
#         tc = p[0]
#         P  = p[1]
#         p2 = p[2]*p[2]
#         a  = p[3]
#         i  = p[4]
#         iT = TWO_PI/P / np.arcsin( np.sqrt(1.-p[3]*p[3]*np.cos(i)**2) / (p[3]*np.sin(i)) )
#         b2 = (a*np.cos(i))**2
        
#         return np.array([tc, P, p2, iT, b2])

#     def map_from_orbit_c(self, p):     
#         self.p = self.mapped_from_orbit_c(p)
 
#     def mapped_from_orbit_e(self, p):
#         raise NotImplementedError
 
#     def map_from_orbit_e(self,  p):
#         self.p = self.get_mapped_from_orbit_e(p)
 
#     def mapped_to_orbit_c(self, p=None):
#         """
#         Maps the Kipping parameterization to the physical parameterization for circular orbits.
#         """
#         if p is None: p = self.p
#         tc = p[0]
#         P  = p[1]
#         p2 = p[2]
#         iT = p[3]
#         b2 = p[4]
#         a = np.sqrt( (1.-b2) / np.sin(TWO_PI/(P*iT))**2 + b2)
#         i = np.arccos(np.sqrt(b2)/a)
        
#         return np.array([tc, P, np.sqrt(p2), a, i])
         
#     def mapped_to_orbit_e(self,  p=None):
#         if p is None: p = self.p
#         b2 = p[4]
#         P  = p[1]
#         iT = p[3]
#         e  = p[5]
#         w  = p[6]
#         e2 = 1 #FIXME: Calculate e2 for eccentric orbits
#         a = np.sqrt((1.-b2) / e2 / 
#                     np.sin(TWO_PI * np.sqrt(1.-e**2) /
#                            (P*iT*e2))**2 + b2/e2)
#         i = np.arccos(np.sqrt(b2)/a * (1.+e*np.sin(w)/(1.-e*e)))
            
#         return np.array([self.t_c, np.sqrt(self.p2), a, i, e, w])

# def map_parameters(p, p_from, p_to):
#     if p.ndim > 1:
#         tp = p.copy()
#         for i in xrange(p.shape[0]):
#             tp[i, :5] = p_to.mapped_from_orbit_c(p_from.mapped_to_orbit_c(p[i, :5]))
#         return tp
    
    
