from __future__ import division
import sys
from types import MethodType
from copy import deepcopy

import numpy as np
from numpy import array, zeros, ones, concatenate, any, tile, vstack, repeat, linspace, arange, s_
from numpy import asarray
from math import acos, sin, asin, sqrt

from transitLightCurve.core import *
from transitLightCurve.transitparameterization import TransitParameterization, generate_mapping
from transitLightCurve.transitparameterization import parameterizations, parameters
from mcmc import DrawGaussian

try:
    from scipy.constants import G
except ImportError:
    G = 6.67e-11
 

class MTFitParameterSet(object):
    def __init__(self):
        self.parameters = {}

class MTFitParameter(object):
    def __init__(self, name, free, value, draw_function, prior, flags='om', description=None, units=None):
        self.name  = name
        self.value = value
        self.free  = free
        self.prior = prior
        self.draw_function = draw_function

        self.flags = flags
        self.description = description or ''
        self.units = units or ''

    def draw(self, x):
        return self.draw_function(x)

    def prior(self, x):
        return self.prior(x)
        
    def limits(self): return self.prior.limits()

    def min(self): return self.prior.min()
    
    def max(self): return self.prior.max()

    def __str__(self):
        return ("{} {} [{}]\n".format(self.description.capitalize(), self.name, self.units) +
                "\t Limits {} -- {}\n".format(self.min(), self.max()) +
                "\t Current value {}\n".format(self.value) )


class MTFitParameterization(object):
    def __init__(self, parameters, nch, ntr, **kwargs):
        info('Initializing fitting parameterization', H2)

        self.pset = parameters
        self.nch  = nch
        self.ntr  = ntr
        
        self.mode                 = kwargs.get('mode', 'om')
        self.n_ldc                = kwargs.get('n_ldc', 1)
        self.n_ld_groups          = kwargs.get('n_ld_groups', nch)
        self.ld_group_idx         = kwargs.get('ld_group_idx', range(nch))

        self.fit_radius_ratio     = kwargs.get('fit_radius_ratio',     True)
        self.fit_transit_center   = kwargs.get('fit_transit_center',   True)
        self.fit_period           = kwargs.get('fit_period',           True)
        self.fit_transit_width    = kwargs.get('fit_transit_width',    True)
        self.fit_impact_parameter = kwargs.get('fit_impact_parameter', True)
        self.fit_limb_darkening   = kwargs.get('fit_limb_darkening',   True)
        self.fit_zeropoint        = kwargs.get('fit_zeropoint',        True)
        self.fit_error            = kwargs.get('fit_error',            True)
        self.include_ttv          = kwargs.get('include_ttv',         False)
        self.fit_ttv              = kwargs.get('fit_ttv',             False)

        self.separate_k2_ch = kwargs.get('separate_k_per_channel',    False)
        self.separate_zp_tr = kwargs.get('separate_zp_per_transit',   False)
        self.separate_ld    = kwargs.get('separate_ld_per_channel',   False)

        self.p_defs = []; self.p_indices = {}

        self.cp_names = [];  self.cp_priors = []
        self.cp_vect  = [];  self.cp_defs   = [];  self.cp_index = {}

        self.fp_names = [];  self.fp_priors = [];  self.fp_draws = []
        self.fp_vect  = [];  self.fp_defs   = [];  self.fp_index = {}

        self.p_min = []; self.p_max = []
        self.kp_tmp = zeros(5, np.double)

        self.parameter_view = {}; self.pv = self.parameter_view

        p = parameters

        for k in kwargs.keys():
            if 'fit_' in k:
                info(k.replace('_',' ')+': %s'%kwargs[k], I1)

        info('Separate radius ratio per channel: %s' %self.separate_k2_ch, I1)
        info('Separate zeropoint per transit: %s' %self.separate_zp_tr, I1)
        info('Separate limb darkening per channel: %s' %self.separate_ld, I1)
        info('Fit TTVs: %s' %self.fit_ttv, I1)

        self.n_lds = self.n_ld_groups if self.separate_ld else 1
        self.n_k2  = nch if self.separate_k2_ch else 1
        self.n_zp  = nch if not self.separate_zp_tr else nch*ntr

        ## Add parameters
        ## ==============
        def add_parameter(fit, name, flags='om'):
            self.p_defs.append(MTFitParameter(name,
                                              p[name]['free'],
                                              p[name].get('value', 0.0),
                                              DrawGaussian(p[name].get('sigma',1.0)),
                                              p[name]['prior'],
                                              flags=flags,
                                              description = p[name].get('description', None),
                                              units = p[name].get('units',None)))
            ppp = self.p_defs[-1]
            print ppp.name, ppp.value


        ## --- Radius ratio ---
        if self.separate_k2_ch:
            for i in range(self.n_k2):
                add_parameter(self.fit_radius_ratio, 'k^2 %i'%i)
        else:
            add_parameter(self.fit_radius_ratio, 'k^2')

        ## --- Zeropoint ---
        ##
        for i in range(self.n_zp):
            add_parameter(self.fit_zeropoint, 'zp %i'%i) 

        ## --- Geometric parameters ---
        add_parameter(self.fit_transit_center, 'tc') 
        add_parameter(self.fit_period, 'p') 
        add_parameter(self.fit_impact_parameter, 'b')

        ## --- Transit width ---
        ## We can either fit the (inverse) transit width or use the estimated stellar
        ## mass and radius to compute it. If we don't fit the inverse width, we need
        ## the mass and radius distributions.
        if self.fit_transit_width:
            add_parameter(p['it']['free'], 'it')
        else:
            add_parameter(False, 'Sm')
            add_parameter(False, 'Sr')

        ## --- Limb darkening ---
        ##
        if self.n_ldc == 1:
            if self.separate_ld:
                for i in range(self.n_ld_groups):
                    add_parameter(self.fit_limb_darkening, 'u %i'%i)
            else:
                add_parameter(self.fit_limb_darkening, 'u 0')

        elif self.n_ldc == 2:
            if self.separate_ld:
                for i in range(self.n_ld_groups):
                    add_parameter(self.fit_limb_darkening, 'u %i + v %i'%(i,i))
                    add_parameter(self.fit_limb_darkening, 'u %i - v %i'%(i,i))
            else:
                add_parameter(self.fit_limb_darkening, 'u 0 + v 0')
                add_parameter(self.fit_limb_darkening, 'u 0 - v 0')

        ## --- Transit timing variations ---
        ##
        if self.include_ttv:
            add_parameter(self.fit_ttv, 'ttv a')
            add_parameter(self.fit_ttv, 'ttv p')

        ## --- Contamination factor ---
        ##
        if 'contamination' in parameters.keys():
            add_parameter(False, 'contamination', 'om')

        ## --- Stellar density ---
        ##
        if 'density' in parameters.keys():
            add_parameter(False, 'density', 'om')

        ## --- Error scale ---
        ##
        for i in range(self.nch):
            add_parameter(self.fit_error, 'error %i'%i) 

        self._generate()


    def _generate(self):

        self.p_names   = [p.name for p in self.p_defs if self.mode in p.flags]
        self.p_indices = {name:self.p_names.index(name) for name in self.p_names}

        self.p_min = array([p.min() for p in self.p_defs if p.free == True and self.mode in p.flags])
        self.p_max = array([p.max() for p in self.p_defs if p.free == True and self.mode in p.flags])

        self.fp_defs  = [p for p in self.p_defs if p.free == True and self.mode in p.flags] 
        self.fp_names = array([p.name for p in self.p_defs if p.free == True and self.mode in p.flags])
        self.n_fp     = self.fp_names.size
        self.fp_vect  = zeros(self.n_fp)

        self.cp_defs  = [p for p in self.p_defs if p.free == False and self.mode in p.flags] 
        self.cp_names = array([p.name for p in self.p_defs if p.free == False and self.mode in p.flags])
        self.n_cp     = self.cp_names.size
        self.cp_vect  = zeros(self.n_cp)

        for i, name in enumerate(self.fp_names):
            self.fp_vect[i] = self.p_defs[self.p_indices[name]].value

        self.parameter_string = {}
        
        ## Get views
        ## =========
        for idx, name in enumerate(self.fp_names):
            self.parameter_view[name] = self.fp_vect[idx:idx+1]
            self.fp_index[name] = idx
            self.parameter_string[name] = 'p', idx

        for idx, name in enumerate(self.cp_names):
            self.parameter_view[name] = self.cp_vect[idx:idx+1]
            self.cp_index[name] = idx
            self.parameter_string[name] = 'self.cp_vect', idx
            
        ## Generate getters
        ## ================
        self._generate_b_getter()
        self._generate_ld_getter()
        self._generate_zp_getter()
        self._generate_p_getter()
        self._generate_kipping_getter()
        self._generate_contamination_getter()
        #self._generate_error_scale_getter()
        self._generate_error_getter()

        ## Generate mappings
        ## =================
        self.map_p_to_k = generate_mapping("physical","kipping")
        #self.map_k_to_p = generate_mapping("kipping","physical")
        self.map_k_to_p = generate_mapping("btest","physical")

        info('Parameter vector length: %i' %self.fp_vect.size, I1)
        info('')

        
    def mcmc_from_opt(self, p):
        mcmcp = deepcopy(self)
        mcmcp.mode = 'm'
        mcmcp._generate()

        for n, v in zip(self.fp_names, p):
            mcmcp.p_defs[mcmcp.p_indices[n]].value = v
            mcmcp.fp_vect[self.fp_index[n]] = v

        return mcmcp

    def is_inside_limits(self, p):
        if any(p < self.p_min) or any(p > self.p_max):
            return False
        else:
            return True

    ## Obtain the Kipping's transit width parameter and the semi-major axis using Kepler's third law
    ## =============================================================================================
    def inv_half_twidth_b2(self, period, b2):
        ac = ((G*self.pv['Sm']/TWO_PI**2)**(1/3)) / self.pv['Sr']
        a = ac * (d_to_s*period)**(2/3)
        it = TWO_PI/period/asin(sqrt(1-b2)/(a*sin(acos(sqrt(b2)/a))))
        return it

    def inv_half_twidth_b(self, period, b):
        ac = ((G*self.pv['Sm']/TWO_PI**2)**(1/3)) / self.pv['Sr']
        a = ac * (d_to_s*period)**(2/3)
        it = TWO_PI/period/asin(sqrt(1-b**2)/(a*sin(acos(b/a))))
        return it

    ## Impact parameter
    ## ========================
    def _generate_b_str(self):
        ps = self.parameter_string['b']
        if self.fit_impact_parameter:
            return "{}[{}]".format(ps[0], ps[1])
        else:
            return self._generate_cp_str('b')

    def _generate_b_getter(self):
        src  = "def get_b(self, p, ch=0, tn=0):\n"
        src += "  return {}\n".format(self._generate_b_str())

        exec(src)
        self.get_b_src = src
        self.get_b = MethodType(get_b, self, MTFitParameterization)


    ## Squared impact parameter
    ## ========================
    def _generate_b2_str(self):
        ps = self.parameter_string['b^2']
        if self.fit_impact_parameter:
            return "{}[{}]".format(ps[0], ps[1])
        else:
            return self._generate_cp_str('b^2')

    def _generate_b2_getter(self):
        src  = "def get_b2(self, p, ch=0, tn=0):\n"
        src += "  return {}\n".format(self._generate_b2_str())
        
        exec(src)
        self.get_b2_src = src
        self.get_b2 = MethodType(get_b2, self, MTFitParameterization)


        
    ## Period
    ## ======
    def _generate_p_str(self):
        ps = self.parameter_string['p']
        if self.fit_period:
            return "{}[{}]".format(ps[0], ps[1])
        else:
            return self._generate_cp_str('p')
        
    def _generate_p_getter(self):
        src  = "def get_p(self, p):\n"
        src += "  return {}\n".format(self._generate_p_str())

        exec(src)
        self.get_p_src = src
        self.get_p = MethodType(get_p, self, MTFitParameterization)

        
    ## Limb Darkening
    ## ==============
    def _generate_ld_str(self):
        ## Don't fit limb darkening
        if not self.fit_limb_darkening:
            if self.n_ldc == 1:
                return "[{}]".format(self._generate_cp_str('u 0'))
            else:
                return "[{},{}]".format(self._generate_cp_str('u 0'),self._generate_cp_str('v 0'))

        ## Fit limb darkening with constant impact parameter
        else:
            if self.n_ldc == 1:
                ch_str = '+self.ld_group_idx[ch]' if self.separate_ld else ''
                ps = self.parameter_string['u 0']
                return "[{}[{}{}]]".format(ps[0], ps[1], ch_str)
            else:
                ch_str = '+2*self.ld_group_idx[ch]' if self.separate_ld else ''
                ps1 = self.parameter_string['u 0 + v 0']
                ps2 = self.parameter_string['u 0 - v 0']

                n1 = "{}[{}{}]".format(ps1[0], ps1[1], ch_str)
                n2 = "{}[{}{}]".format(ps2[0], ps2[1], ch_str)
                return "[0.5*({}+{}), 0.5*({}-{})]".format(n1, n2, n1, n2)

    def _generate_ld_getter(self):
        src  = "def get_ldc(self, p, ch=0, tn=0):\n"
        src += "  return {}\n".format(self._generate_ld_str())

        exec(src)
        self.get_ldc_src = src
        self.get_ldc = MethodType(get_ldc, self, MTFitParameterization)
        

    ## Zeropoint
    ## =========
    def _generate_zp_getter(self):
        src  = "def get_zp(self, p, ch=0, tn=0):\n"
        if self.fit_zeropoint:
            ps = self.parameter_string['zp 0']
            if not self.separate_zp_tr:
                src += "  return {}[{}+ch]\n".format(*ps)
            else:
                src += "  return {}[{} + tn*self.nch + ch]\n".format(*ps)
        else:
            if 'm' in self.mode:
                src += "  return self.p_defs[{}+ch].prior.random()".format(self.p_indices['zp 0'])
            else:
                src += "  return self.p_defs[{}+ch].value".format(self.p_indices['zp 0'])

        exec(src)
        self.get_zp_src = src
        self.get_zp = MethodType(get_zp, self, MTFitParameterization)

        
    #FIXME: Inverse half-width should be handled differently.
    def _generate_inverse_transit_halfwidth_str(self):
       if self.fit_transit_width:
           if self.p_defs[self.p_indices['it']].free:
               return "%s[%i]\n" %self.parameter_string['it']
           else:
               return self._generate_cp_str('it')
       else:
           return "self.inv_half_twidth_b(%s[%i], %s)\n"%(ps['p']+(self._generate_b_str(),))


    ## Kipping parameterization
    ## ========================
    def _generate_kipping_getter(self):
        ps = self.parameter_string

        k21, k22 = ps['k^2']
        k22 = str(k22) if not self.separate_k2_ch else str(k22)+'+ch'
        src  = "def get_kipping(self, p, ch=0, tn=0, _p_out=None):\n"
        src += "  p_out = _p_out or self.kp_tmp\n"
        src += "  p_out[0] = %s[%s]\n"%(k21, k22)
        src += "  p_out[1] = %s[%i]\n"%ps['tc']
        src += "  p_out[2] = %s\n"%(self._generate_p_str())
        src += "  p_out[3] = %s\n"%(self._generate_inverse_transit_halfwidth_str())
        #src += "  p_out[4] = %s\n"%self._generate_b2_str()
        src += "  p_out[4] = %s\n"%self._generate_b_str()
        src += "  return p_out\n"

        exec(src)
        self.get_kipping_src = src
        self.get_kipping = MethodType(get_kipping, self, MTFitParameterization)


    ## Contamination
    ## =============
    def _generate_contamination_getter(self):
        src="def get_contamination(self, p, ch=0):\n"
        if 'contamination' in self.p_names:
            src += "  return "+self._generate_cp_str('contamination')
        else:
            src += "  return 0."

        exec(src)
        self.get_contamination_src = src
        self.get_contamination = MethodType(get_contamination, self, MTFitParameterization)


    ## Stellar density
    ## ===============
    def _generate_density_getter(self):
        src="def get_density(self, p, ch=0):\n"
        if 'density' in self.p_names:
            src += "  return "+self._generate_cp_str('density')
        else:
            src += "  return 0."

        exec(src)
        self.get_contamination_src = src
        self.get_contamination = MethodType(get_contamination, self, MTFitParameterization)

    #def get_density(self, p):
    #    return 1e-3 * 3*pi*semimajor_axis(d)**3 / (G*(d['p']*24*60*60)**2)


    ## Error scale
    ## ===========
    def _generate_error_str(self):
        ps = self.parameter_string['error 0']
        if self.fit_error:
            return "{}[{}+ch]".format(ps[0], ps[1])
        else:
            return self._generate_cp_str('error 0')
        
    def _generate_error_getter(self):
        src  = "def get_error(self, p, ch=0):\n"
        src += "  return {}\n".format(self._generate_error_str())

        exec(src)
        self.get_error_src = src
        self.get_error = MethodType(get_error, self, MTFitParameterization)


    def _generate_error_scale_getter(self):
        src="def get_error_scale(self, p, ch=0):\n"
        if 'error scale' in self.p_names:
            ps = self.parameter_string['error scale']
            if self.p_defs[self.p_indices['error scale']].free:
                src += "  return {}[{}]".format(*ps)
            else:
                src += "  return "+self._generate_cp_str('error scale')
        else:
            src += "  return 1."

        exec(src)
        self.get_error_scale_src = src
        self.get_error_scale = MethodType(get_error_scale, self, MTFitParameterization)


    ## Simple getters
    ## ==============
    def get_physical(self, p, ch=0):
        return self.map_k_to_p(self.get_kipping(p, ch))

    def get_ttv(self, p):
        return [self.parameter_view['ttv a'], self.parameter_view['ttv p']] 

    def get_parameter_names(self):
        return self.fp_names

    def get_parameter_descriptions(self):
        return self.fp_names

    @property
    def parameter_descriptions(self):
        return self.fp_names

    def _generate_cp_str(self, name):
        if 'm' in self.mode:
            return "self.p_defs[{}].prior.random()".format(self.p_indices[name])
        else:
            return "self.p_defs[{}].value".format(self.p_indices[name])
