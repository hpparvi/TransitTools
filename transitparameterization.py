"""
A module to parameterize the transit and map between different parameterizations.

Mappings
  Mappings between parameterizations are created with the generate_mapping function. 

Author
  Hannu Parviainen <hpparvi@gmail.com>

Date
  19.12.2010

Modified
  13.01.2011
"""
import sys
import numpy as np
from types import MethodType
from string import Template
from math import sin, cos, asin, acos, sqrt
from numpy import asarray, array

from core import *

##--- CLASSES ---
##
class Mapping(object):
    """Mapping to derive a parameter from a parameter set.
    
    Describes a way to derive a parameter A depending on parameter 
    set [b, c, d, ...]
    """

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
    """A Physical parameter with name, description and unit."""

    def __init__(self, name, description, unit, limits):
        self.name = name
        self.description = description
        self.unit = unit
        self.limits = limits


class TransitParameterization(object):
    """A set of parameters completely parameterizing a planetary transit."""

    def __init__(self, type, init=None):
        if type in parameterizations.keys():
            self.type = type
            self.parameter_set = parameterizations[type]
            self.parameter_vector = (np.asarray(init) if init is not None and np.asarray(init).size == len(self.parameter_set) 
                                     else np.zeros(len(self.parameter_set)))
            self.npar = self.parameter_vector.size
            
            self.parameter_definitions = []
            self.parameter_units = []
            self.parameter_limits = []

            for p in self.parameter_set:
                self.parameter_limits.append(parameters[p].limits)

            self.parameter_limits = array(self.parameter_limits)

            self.tp = self.type
            self.ps = self.parameter_set
            self.pv = self.parameter_vector
        else:
            logging.error("Bad parameterization type %s!"%type)
            sys.exit()

        self.map_to_orbit = MethodType(generate_mapping(self.type, 'orbit'), self, TransitParameterization)

    def __str__(self):
        istr = '\n%s transit parameterization\n\n' %self.type.capitalize()
        for p,v in zip(self.ps, self.pv):
            istr += '    %-20s %6.3f\n' %(parameters[p].description, v)
        return istr
 
    def __getitem__(self, i):
        return self.pv[i]

    def update(self, p):
        self.parameter_vector = p.copy()
        self.pv = self.parameter_vector

    def generate_mapping(self, p_from, p_to):
        s_from = parameterizations[p_from]
        s_to   = parameterizations[p_to]


##--- TABLES ---
##
## Format:                        [parameter]   [long name]        [units]  [limits]
parameters = {'tc' : TransitParameter('tc',  'transit center',       'HJD', [-1e18, 1e18]),
              'p'  : TransitParameter( 'p',          'period',         'd', [0, 1e18]),
              'k'  : TransitParameter( 'k',    'radius ratio',    'R_star', [1e-3, 5e-1]),
              'a'  : TransitParameter( 'a', 'semi-major axis',    'R_star', [1e00, 1e05]),
              'i'  : TransitParameter( 'i',     'inclination',   'radians', [-HALF_PI, HALF_PI]),
              'b'  : TransitParameter( 'b', 'impact parameter',         '', [0, 1]),
              'k2' : TransitParameter('k2', 'squared radius ratio',     '', [1e-6, 5e-2]),
              'it' : TransitParameter('it', 'transit width parameter',  '', [0,10]),
              'b2' : TransitParameter('b2', 'squared impact parameter', '', [0, 1])}

parameterizations   = {'orbit'    : ['k',  'tc', 'p',  'a',  'i'],
                       'physical' : ['k',  'tc', 'p',  'a',  'b'],
                       'kipping'  : ['k2', 'tc', 'p', 'it', 'b2'],
                       'btest'    : ['k2', 'tc', 'p', 'it',  'b']}

mappings = { 'k' : [Mapping('k',           ['k2'], '$k = sqrt($k2)'    )],
             'a' : [Mapping('a',  ['p','it','b2'], '$a = sqrt( (1.-$b2) / sin(TWO_PI/($p*$it))**2 + $b2)'),
                    Mapping('a',   ['p','it','b'], '$a = sqrt( (1.-$b**2) / sin(TWO_PI/($p*$it))**2 + $b**2)')],
             'b' : [Mapping('b',        ['a','i'], '$b = $a*cos($i)'   ),
                    Mapping('b',           ['b2'], '$b = sqrt($b2)'    )],
             'i' : [Mapping('i',        ['a','b'], '$i = acos($b/$a)'  ),
                    Mapping('i',       ['a','b2'], '$i = acos(sqrt($b2)/$a)'  ),
                    Mapping('i',  ['p','it','b2'], '$i = acos(sqrt($b2)/sqrt((1.-$b2)/sin(TWO_PI/($it*$p))**2+$b2))'  ),
                    Mapping('i',   ['p','it','b'], '$i = acos(sqrt($b**2)/sqrt((1.-$b**2)/sin(TWO_PI/($it*$p))**2+$b**2))' )],
            'b2' : [Mapping('b2',           ['b'], '$b2 = $b*$b'       ),
                    Mapping('b2',       ['a','i'], '$b2 = ($a*cos($i))**2')],
            'it' : [Mapping('it',   ['a','i','p'], '$it = TWO_PI/$p/asin(sqrt(1.-$a*$a*cos($i)**2)/($a*sin($i)))'),
                    Mapping('it',   ['a','b','p'], '$it = TWO_PI/$p/asin(sqrt(1.-$b**2)/($a*sin(acos($b/$a))))')],
            'k2' : [Mapping('k2',           ['k'], '$k2 = $k*$k'       )]}


##--- FUNCTIONS ---
##
def generate_mapping(p_from, p_to):
    s_from  = parameterizations[p_from]
    s_to    = parameterizations[p_to]

    map_str  = 'def mapping(p_from):\n'
    map_str += '    if isinstance(p_from, TransitParameterization):\n'
    map_str += '        v_from = p_from.pv\n'
    map_str += '    else:\n'
    map_str += '        v_from = asarray(p_from)\n\n'
    map_str += '    v_to = np.zeros(v_from.shape)\n\n'

    for i, p in enumerate(s_to):
        if p in s_from:
            map_str += '    v_to[%i] = v_from[%i]\n' %(i, s_from.index(p))
        else:
            for mapping in mappings[p]:
                if mapping.is_mappable(s_from): break
            else:
                logging.error("Couldn't find suitable mapping %s -> %s for %s:%s." %(p_from, p_to, p,str(mapping.dependencies)))
                sys.exit()

            map_str += mapping.get_mapping(s_from, s_to)+'\n'

    map_str += '\n    if isinstance(p_from, TransitParameterization):\n'
    map_str += '        return TransitParameterization("%s", v_to)\n' %p_to
    map_str += '    else:\n'
    map_str += '        return v_to\n'

    exec map_str
    return mapping


def generate_fortran_mapping(p_from, p_to, fname=None):
    s_from  = parameterizations[p_from]
    s_to    = parameterizations[p_to]

    ## Generate the function name and the input argument list
    ##
    fn_name = fname if fname is not None else "map_%s_to_%s" %(p_from, p_to)
    args_in = ", ".join(s_from)

    ## Generate the function definition
    ##
    map_str = ""
    map_str += "!! Maps the '%s' parameter set (%s) to the \n!! '%s' set (%s)\n" %(p_from, ", ".join(s_from), p_to, ", ".join(s_to))
    map_str += "function %s(v_from) result(v_to)\n" %fn_name    
    map_str += "  use iso_c_binding\n  implicit none\n"
    map_str += "  real(kind=8), dimension(0:%i), intent(in) :: v_from\n" %(len(s_from)-1)
    map_str += "  real(kind=8), dimension(0:%i) :: v_to\n" %(len(s_to)-1)
    map_str += "  real(kind=8), parameter :: two_pi = %10.8f\n" %TWO_PI

    for i, p in enumerate(s_to):
        if p in s_from:
            map_str += '    v_to(%i) = v_from(%i)\n' %(i, s_from.index(p))
        else:
            for mapping in mappings[p]:
                if mapping.is_mappable(s_from): break
            else:
                logging.error("Couldn't find suitable mapping %s -> %s for %s:%s." %(p_from, p_to, p,str(mapping.dependencies)))
                sys.exit()

            map_str += mapping.get_mapping(s_from, s_to).replace('[','(').replace(']',')')+'\n'
    
    map_str += "end function %s\n" %fn_name
    return map_str

def generate_fortran_module(mappings):
    from numpy.f2py import compile

    #fn_name = "map_%s_to_%s" %(p_from, p_to)
    map_str  = "module tp_mappings\ncontains\n"
    for mapping in mappings:
        map_str += generate_fortran_mapping(mapping[0], mapping[1]) + '\n'
        map_str += generate_fortran_mapping(mapping[1], mapping[0]) + '\n'
    map_str += "end module tp_mappings"
    print map_str
    #compile(map_str, modulename='mapping', source_fn='%s.f90'%fn_name)

if __name__ == '__main__':
    pass
    #generate_fortran_module([['kipping','physical']])
    #from mapping import pmap
    #a = pmap.map_kipping_to_physical(np.array([1,2,3,4,1], dtype=np.double))
    #print a

# print parameterizations['orbit']
# print parameterizations['physical']

# pp = TransitParameterization('physical',[1,2,3,4,0.9])
# po = TransitParameterization('orbit')

# o2p = generate_mapping('orbit','physical')
# p2o = generate_mapping('physical','orbit')

# po = p2o(pp)

# print pp
# print po
