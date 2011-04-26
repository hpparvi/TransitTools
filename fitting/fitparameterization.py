from __future__ import division
import sys
from types import MethodType

from numpy import array, zeros, ones, concatenate, any, tile, vstack, repeat, linspace, arange, s_
from numpy import asarray
from math import acos, sin, asin, sqrt

from transitLightCurve.core import *
from transitLightCurve.transitparameterization import TransitParameterization, generate_mapping
from transitLightCurve.transitparameterization import parameterizations, parameters
    
try:
    from scipy.constants import G
except ImportError:
    G = 6.67e-11
 

class MTFitParameterSet(object):
    def __init__(self):
        self.parameters = {}

class MTFitParameter(object):
    def __init__(self, name, soft_limits, value=None, hard_limits=None, description=None, units=None, draw_function=None, prior=None):
        self.name = name
        self.soft_limits = asarray(soft_limits)
        self.description = description or parameters[name].description
        self.units = units or parameters[name].unit
        self.value = value if value is not None else self.soft_limits.mean()
        self.hard_limits = hard_limits or self.soft_limits
        self.draw_function = draw_function
        self.prior = prior

        self.min_s = self.soft_limits[0]
        self.max_s = self.soft_limits[1]


    def __str__(self):
        return "%10s %2s [%s]\n\tInitial value %f\n\tSoft limits [%f %f]"%(self.description.capitalize(), self.name, self.units,
                                                    self.value, self.soft_limits[0], self.soft_limits[1])


class MTFitParameterization(object):
    def __init__(self, parameters, stellar_prm, nch, ntr, **kwargs):
        info('Initializing fitting parameterization', H2)

        self.pset = parameters
        self.nch  = nch
        self.ntr  = ntr

        self.fit_radius_ratio     = kwargs.get('fit_radius_ratio',     True)
        self.fit_transit_center   = kwargs.get('fit_transit_center',   True)
        self.fit_period           = kwargs.get('fit_period',          False)
        self.fit_transit_width    = kwargs.get('fit_transit_width',   False)
        self.fit_impact_parameter = kwargs.get('fit_impact_parameter', True)
        self.fit_limb_darkening   = kwargs.get('fit_limb_darkening',   True)
        self.fit_zeropoint        = kwargs.get('fit_zeropoint',        True)
        self.fit_ttv              = kwargs.get('fit_ttv',             False)

        self.separate_k2_ch = kwargs.get('separate_k_per_channel',    False)
        self.separate_zp_tr = kwargs.get('separate_zp_per_transit',   False)
        self.separate_ld    = kwargs.get('separate_ld_per_channel',   False)

        self.constant_parameters = None
        self.constant_parameter_names = []
        self.constant_parameter_index = {}

        self.fitted_parameters = None
        self.fitted_parameter_names = []
        self.fitted_parameter_index = {}

        self.parameter_view = {}

        self.pview = {}

        p = parameters

        p_init = kwargs.get('initial_parameter_values', {})

        for k in kwargs.keys():
            if 'fit_' in k:
                info(k.replace('_',' ')+': %s'%kwargs[k], I1)

        info('Separate radius ratio per channel: %s' %self.separate_k2_ch, I1)
        info('Separate zeropoint per transit: %s' %self.separate_zp_tr, I1)
        info('Separate limb darkening per channel: %s' %self.separate_ld, I1)
        info('Fit TTVs: %s' %self.fit_ttv, I1)

        self.n_lds = nch if self.separate_ld else 1
        self.n_ldc = 2 if 'v' in parameters.keys() else 1
        self.n_k2  = nch if self.separate_k2_ch else 1
        self.n_zp  = nch if not self.separate_zp_tr else nch*ntr


        ## Add parameters
        ## ==============
        def count_parameter(name):
            return sum([name == n.split(' ')[0] for n in self.fitted_parameter_names])
                                
        def add_parameter(fit, name, count=1):
            if fit:
                self.fitted_parameter_names.extend(count*[name])
            else:
                self.constant_parameter_names.extend(count*[name])

        for i in range(self.n_k2):
            add_parameter(self.fit_radius_ratio, 'k2')

        for i in range(self.n_zp):
            add_parameter(self.fit_zeropoint, 'zp %i'%i) 

        add_parameter(self.fit_transit_center, 'tc') 
        add_parameter(self.fit_period, 'p') 

        if self.fit_transit_width:
            add_parameter(True, 'it')

        ## Impact parameter and limb darkening
        ## -----------------------------------
        ## The impact parameter and limb darkening are correlated. This correlation
        ## can be reduced by fitting their linear combinations, but since we allow
        ## for both linear and quadratic limb darkening, and for separate limb 
        ## darkening per channel, things get a bit messy.
        ##
        ## We can try to clear this a bit by considering these separate cases:
        ##
        ## p1 = b2 + u         b2 = 0.5 (p1 + p2)
        ## p2 = b2 - u          u = 0.5 (p1 - p2)
        ##
        ## p1 =  u + v          u = 0.5 (p1 + p2)
        ## p2 =  u - v          v = 0.5 (p1 + p2)
        ##
        ## p1 = b2 + u + v     b2 = 0.5 (p1 + p2)
        ## p2 = b2 - u - v      u = 0.5 (p1 - p3)
        ## p3 = b2 - u + v      v = 0.5 (p3 - p2)
        ##
        ## p2 = b2 - u - v      u = 0.5 (0.5(p1 - p2) + p3)
        ## p3 =      u - v      v = 0.5 (0.5(p1 - p2) - p3)
        ##
        ## p1 = b2 + <u>       b2 = 0.5 (p1 + p2)
        ## p2 = b2 - <u>      <u> = 0.5 (p1 - p2)
        ## p3 = u1 - <u>       u1 = p3 + <u>
        ## pn = un - <u>       un = pn + <u>
        ##
        ##
        ##
        ## p1 = u1 + v1
        ## p2 = u1 - v1 
        ## p1 = u2 + v2
        ## p2 = u2 - v2 
        ## pn = un + vn
        ## pn = un - vn 
        ##
        ## If we fit different limb darkening for each channel, we replace the u and v
        ## with their means <u> and <v>, and fit the differences from the means...
        ##
        ##

        ## Case 1: Don't fit limb darkening
        ldcn = ('u','v')
        if not self.fit_limb_darkening:
            add_parameter(self.fit_impact_parameter, 'b2')
            if self.fit_impact_parameter:
                ldp_min = [p['b'].min_s**2]
                ldp_max = [p['b'].max_s**2]

            for i in range(self.n_ldc):
                add_parameter(self.fit_limb_darkening, '%s'%ldcn[i])

        ## Case 2: fit limb darkening with constant impact parameter
        elif not self.fit_impact_parameter:
            add_parameter(self.fit_impact_parameter, 'b2')

            ## Case 2.1: don't separate limb darkening
            if not self.separate_ld:
                ## Case 2.1a: linear limb darkening
                if self.n_ldc == 1:
                    add_parameter(True, 'u')
                    ldp_min = [p['u'].min_s]
                    ldp_max = [p['u'].max_s]

                ## Case 2.1b: quadratic limb darkening
                else:
                    add_parameter(True, 'u + v')
                    add_parameter(True, 'u - v')
                    ldp_min = [p['u'].min_s + p['v'].min_s, p['u'].min_s - p['v'].min_s]
                    ldp_max = [p['u'].max_s + p['v'].max_s, p['u'].max_s - p['v'].max_s]

            ## Case 2.2: separate limb darkening
            else:
                ## Case 2.2a: linear limb darkening
                if self.n_ldc == 1:
                    for i in range(self.nch):
                        add_parameter(True, 'u%i'%i)
                    ldp_min = self.nch*[p['u'].min_s]
                    ldp_max = self.nch*[p['u'].max_s]

                ## Case 2.2b: quadratic limb darkening
                else:
                    for i in range(self.nch):
                        add_parameter(True, 'u%i + v%i'%(i,i))
                        add_parameter(True, 'u%i - v%i'%(i,i))

                    s_min = p['u'].min_s + p['v'].min_s
                    s_min = s_min if s_min > 0. else 0.
                    s_max = p['u'].max_s + p['v'].max_s
                    s_max = s_max if s_max < 1. else 1.
                    ldp_min = self.nch*[s_min, p['u'].min_s - p['v'].max_s]
                    ldp_max = self.nch*[s_max, p['u'].max_s - p['v'].min_s]

        ## Case 3: fit both limb darkening and impact parameter
        else:
            ## Case 3.1: don't separate limb darkening coefficients
            if not self.separate_ld:
                ## Case 3.1a: linear limb darkening
                if self.n_ldc == 1:
                    add_parameter(True, 'b2 + u')
                    add_parameter(True, 'b2 - u')
                    ldp_min = [p['b'].min_s**2 + p['u'].min_s, p['b'].min_s**2 - p['u'].max_s]
                    ldp_max = [p['b'].max_s**2 + p['u'].max_s, p['b'].max_s**2 - p['u'].min_s]

                ## Case 3.1b: quadratic limb darkening
                else:
                    add_parameter(True, 'b2 + u + v')
                    add_parameter(True, 'b2 - u - v')
                    add_parameter(True, 'b2 - u + v')

                    ldp_min = [p['b'].min_s**2 + p['u'].min_s + p['v'].min_s,
                               p['b'].min_s**2 - p['u'].max_s - p['v'].max_s,
                               p['b'].min_s**2 - p['u'].max_s + p['v'].min_s]

                    ldp_max = [p['b'].max_s**2 + p['u'].max_s + p['v'].max_s,
                               p['b'].max_s**2 - p['u'].min_s - p['v'].min_s,
                               p['b'].max_s**2 - p['u'].min_s + p['v'].max_s]

            ## Case 3.2: separate limb darkening coefficients
            ## Currently the same as 2.2, we just add the b2 to the fitted parameters
            else:
                add_parameter(True, 'b2')
                ldp_min = [p['b'].min_s**2]
                ldp_max = [p['b'].max_s**2]

                ## Case 3.2a: linear limb darkening
                if self.n_ldc == 1:
                    for i in range(self.nch):
                        add_parameter(True, 'u%i'%i)
                    ldp_min = concatenate([ldp_min, self.nch*[p['u'].min_s]])
                    ldp_max = concatenate([ldp_max, self.nch*[p['u'].max_s]])

                ## Case 3.2b: quadratic limb darkening
                else:
                    for i in range(self.nch):
                        add_parameter(True, 'u%i + v%i'%(i,i))
                        add_parameter(True, 'u%i - v%i'%(i,i))

                    s_min = p['u'].min_s + p['v'].min_s
                    s_min = s_min if s_min > 0. else 0.
                    s_max = p['u'].max_s + p['v'].max_s
                    s_max = s_max if s_max < 1. else 1.
                    ldp_min = concatenate([ldp_min, self.nch*[s_min, p['u'].min_s - p['v'].max_s]])
                    ldp_max = concatenate([ldp_max, self.nch*[s_max, p['u'].max_s - p['v'].min_s]])

        ldp_min = asarray(ldp_min) 
        ldp_max = asarray(ldp_max) 

        ## TTV
        ## ===
        add_parameter(self.fit_ttv, 'ttv_a')
        add_parameter(self.fit_ttv, 'ttv_p')

        self.fitted_parameter_names = array(self.fitted_parameter_names)
        self.fitted_parameters = zeros(self.fitted_parameter_names.size)

        self.constant_parameter_names = array(self.constant_parameter_names)
        self.constant_parameters = zeros(self.constant_parameter_names.size)

        self.n_fitted_parameters = self.fitted_parameters.size
        self.n_constant_parameters = self.fitted_parameters.size

        self.p_cur = self.fitted_parameters

        self.parameter_string = {}

        ## Get views
        ## =========
        for idx, name in enumerate(self.fitted_parameter_names):
            self.parameter_view[name] = self.fitted_parameters[idx:idx+1]
            self.fitted_parameter_index[name] = idx
            self.parameter_string[name] = 'self.fitted_parameters', idx

        for idx, name in enumerate(self.constant_parameter_names):
            self.parameter_view[name] = self.constant_parameters[idx:idx+1]
            self.constant_parameter_index[name] = idx
            self.parameter_string[name] = 'self.constant_parameters', idx

            
        ## Generate getters
        ## ================
        self._generate_b2_getter()
        self._generate_ld_getter()
        self._generate_zp_getter()
        self._generate_kipping_getter()

        ## Blah
        ## ====
        self.ac = ((G*stellar_prm['M']/TWO_PI**2)**(1/3)) / stellar_prm['R']

        ## Generate mappings
        ## =================
        self.map_p_to_k = generate_mapping("physical","kipping")
        self.map_k_to_p = generate_mapping("kipping","physical")

        ## Define differential evolution parameter boundaries
        ## ==================================================
        ## The boundaries are given as a dictionary with physical parameter boundaries. These are
        ## mapped to the Kipping parameterization with the relative semi-major axis set to unity.
        ##
        self.p_k_min  = self.map_p_to_k([p['k'].soft_limits[0], p['tc'].soft_limits[0], p['p'].soft_limits[0], 10, p['b'].soft_limits[0]])
        self.p_k_max  = self.map_p_to_k([p['k'].soft_limits[1], p['tc'].soft_limits[1], p['p'].soft_limits[1], 10, p['b'].soft_limits[1]])

        
        ## Setup constant parameters
        ## =========================
        if 'k2' in self.constant_parameter_names:
            self.constant_parameters[self.constant_parameter_names=='k2'] = self.pset['k'].value**2

        if 'b2' in self.constant_parameter_names:
            self.constant_parameters[self.constant_parameter_names=='b2'] = self.pset['b'].value**2

        for tp in ['tc','p','a']:
            if tp in self.constant_parameter_names:
                self.constant_parameters[self.constant_parameter_names==tp] = self.pset[tp].value

        if not self.fit_zeropoint:
            self.constant_parameters[self.constant_parameter_names=='zp'] = 1.


        ## Setup the soft parameter limits
        ## ===============================
        self.p_min = concatenate([repeat(self.p_k_min[0], count_parameter('k2')),
                                  repeat(p['zp'].min_s,   count_parameter('zp')),
                                  [self.p_k_min[1]]     * count_parameter('tc'),
                                  [self.p_k_min[2]]     * count_parameter('p'),
                                  [p['it'].min_s] if self.fit_transit_width else [],
                                  ldp_min,
                                  [p['ttv a'].min_s] * count_parameter('ttv a'),
                                  [p['ttv p'].min_s] * count_parameter('ttv p')])


        self.p_max = concatenate([repeat(self.p_k_max[0], count_parameter('k2')),
                                  repeat(p['zp'].max_s,   count_parameter('zp')),
                                  [self.p_k_max[1]]     * count_parameter('tc'),
                                  [self.p_k_max[2]]     * count_parameter('p'),
                                  [p['it'].max_s] if self.fit_transit_width else [],
                                  ldp_max,
                                  [p['ttv a'].max_s]    * count_parameter('ttv a'),
                                  [p['ttv p'].max_s]    * count_parameter('ttv p')])

        ## Setup the hard parameter limits 
        ## ================================
        self.l_min = concatenate([repeat(0.0, count_parameter('k2')),
                                  repeat(0.9, count_parameter('zp')),
                                  [0.0]    *  count_parameter('tc'),
                                  [0.0]    *  count_parameter('p'),
                                  [0.0] if self.fit_transit_width else [],
                                  ldp_min,
                                  [-1.]    * count_parameter('ttv a'),
                                  [1.]     * count_parameter('ttv p')])

        self.l_max = concatenate([repeat(0.1, count_parameter('k2')),
                                  repeat(1.1, count_parameter('zp')),
                                  [1e10]   *  count_parameter('tc'),
                                  [1e10]   *  count_parameter('p'),
                                  [5e04] if self.fit_transit_width else [],
                                  ldp_max,
                                  [1.]     * count_parameter('ttv a'),
                                  [1e10]   * count_parameter('ttv p')])

        self.p_cur[:] = vstack([self.p_min, self.p_max]).mean(0)

        info('Parameter vector length: %i' %self.p_cur.size, I1)
        info('')

    def update(self, pv):
        self.p_cur[:] = pv

    def is_inside_limits(self):
        if any(self.p_cur < self.l_min) or any(self.p_cur > self.l_max):
            return False
        else:
            return True

    ## Obtain the Kipping's transit width parameter and the semi-major axis using Kepler's third law
    ## =============================================================================================
    def kipping_i(self, period, b2):
        a = self.ac * (d_to_s*period)**(2/3)
        it = TWO_PI/period/asin(sqrt(1-b2)/(a*sin(acos(sqrt(b2)/a))))
        return it


    ## Squared impact parameter
    ## ========================
    def _generate_b2_str(self):
        fpi = self.fitted_parameter_index
        cpi = self.constant_parameter_index
        fp  = "self.fitted_parameters"

        if not self.fit_impact_parameter:
            return "self.constant_parameters[%i]" %cpi['b2']
        elif not self.fit_limb_darkening or self.separate_ld:
            return "self.fitted_parameters[%i]" %fpi['b2']
        else:
            n1 = 'b2 + u' if self.n_ldc == 1 else 'b2 + u + v'
            n2 = 'b2 - u' if self.n_ldc == 1 else 'b2 - u - v'
            return "0.5 * (%s[%i] + %s[%i])"%(fp, fpi[n1], fp, fpi[n2])

    def _generate_b2_getter(self):
        src  = "def get_b2(self, ch=0, tn=0, p_in=None):\n"
        src += "  if p_in is not None: self.update(p_in)\n"
        src += "  return %s\n" %self._generate_b2_str()

        exec(src)
        self.get_b2 = MethodType(get_b2, self, MTFitParameterization)

        
    ## Limb Darkening
    ## ==============
    def _generate_ld_str(self):
        fpi = self.fitted_parameter_index
        cpi = self.constant_parameter_index
        fp  = "self.fitted_parameters"

        ## Don't fit limb darkening
        if not self.fit_limb_darkening:
            return "self.constant_parameters[%i:%i]"%(cpi['u'], cpi['u']+self.n_ldc)

        ## Fit limb darkening with constant impact parameter
        elif not self.fit_impact_parameter:
            if not self.separate_ld:
                if self.n_ldc == 1:
                    return "self.fitted_parameters[%i]"%fpi['u']
                else:
                    n1 = "self.fitted_parameters[%i]"%fpi['u + v']
                    n2 = "self.fitted_parameters[%i]"%fpi['u - v']
                    return "[0.5*(%s+%s), 0.5*(%s-%s)]" %(n1, n2, n1, n2)
            else:
                if self.n_ldc == 1:
                    return "self.fitted_parameters[%i+ch]"%fpi['u0']
                else:
                    n1 = "self.fitted_parameters[%i+2*ch]"%fpi['u0 + v0']
                    n2 = "self.fitted_parameters[%i+2*ch]"%fpi['u0 - v0']
                    return "[0.5*(%s+%s), 0.5*(%s-%s)]" %(n1, n2, n1, n2)

        ## Fit both limb darkening and impact parameter
        else:
            if not self.separate_ld:
                if self.n_ldc == 1:
                    id1 = "%s[%i]"%(fp, fpi['b2 + u'])
                    id2 = "%s[%i]"%(fp, fpi['b2 - u'])
                    return "[0.5*(%s - %s)]"%(id1, id2)
                else:
                    id1 = "%s[%i]"%(fp, fpi['b2 + u + v']) 
                    id2 = "%s[%i]"%(fp, fpi['b2 - u - v']) 
                    id3 = "%s[%i]"%(fp, fpi['b2 - u + v']) 
                    return "[0.5*(%s - %s), 0.5*(%s - %s)]"%(id1, id3, id2, id3)
            else:
                if self.n_ldc == 1:
                    return "self.fitted_parameters[%i+ch]"%fpi['u0']
                else:
                    n1 = "self.fitted_parameters[%i+2*ch]"%fpi['u0 + v0']
                    n2 = "self.fitted_parameters[%i+2*ch]"%fpi['u0 - v0']
                    return "[0.5*(%s+%s), 0.5*(%s-%s)]" %(n1, n2, n1, n2)

    def _generate_ld_getter(self):
        src  = "def get_ldc(self, ch=0, tn=0, p_in=None):\n"
        src += "  if p_in is not None: self.update(p_in)\n"
        #src += "  print self.fitted_parameters[7], self.fitted_parameters[8]\n"
        #src += "  print %s\n  print\n" %self._generate_ld_str()        
        src += "  return %s\n" %self._generate_ld_str()        
        exec(src)
        self.get_ldc = MethodType(get_ldc, self, MTFitParameterization)
        

    ## Zeropoint
    ## =========
    def _generate_zp_getter(self):
        src  = "def get_zp(self, ch=0, tn=0, p_in=None):\n"
        if self.fit_zeropoint:
            src += "  if p_in is not None: self.update(p_in)\n"
            if not self.separate_zp_tr:
                src += "  return self.fitted_parameters[%i+ch]\n"%self.fitted_parameter_index['zp 0']
            else:
                src += "  return self.fitted_parameters[%i + tn*self.nch + ch]\n"%self.fitted_parameter_index['zp 0']
        else:
            src += "  return 1."
        exec(src)
        self.get_zp = MethodType(get_zp, self, MTFitParameterization)

        
    ## Kipping parameterization
    ## ========================
    def _generate_kipping_getter(self):
        ps = self.parameter_string

        k21, k22 = ps['k2']
        k22 = str(k22) if not self.separate_k2_ch else str(k22)+'+ch'
        src  = "def get_kipping(self, ch=0, tn=0, p_in=None, p_out=None):\n"
        src += "  if p_in is not None: self.update(p_in)\n"
        src += "  if p_out is None: p_out = zeros(5)\n"
        src += "  p_out[0] = %s[%s]\n"%(k21, k22)
        src += "  p_out[1] = %s[%i]\n"%ps['tc']
        src += "  p_out[2] = %s[%i]\n"%ps['p']
        if self.fit_transit_width:
            src += "  p_out[3] = %s[%i]\n" %ps['it']
        else:
            src += "  p_out[3] = self.kipping_i(%s[%i], %s)\n"%(ps['p']+(self._generate_b2_str(),))
        src += "  p_out[4] = %s\n"%self._generate_b2_str()
        src += "  return p_out\n"

        exec(src)
        self.get_kipping = MethodType(get_kipping, self, MTFitParameterization)


    ## Simple getters
    ## ==============
    def get_physical(self, ch=0, p_in=None):
        return self.map_k_to_p(self.get_kipping(ch, p_in))

    def get_ttv(self, p_in=None):
        if p_in is not None: self.update(p_in)
        return [self.v_ttv_a, self.v_ttv_b] 
   
    def get_parameter_names(self):
        return self.fitted_parameter_names

    def get_parameter_descriptions(self):
        return self.fitted_parameter_names

    def get_hard_limits(self):
        return array([self.l_min, self.l_max])
