from __future__ import division
import sys

from numpy import array, zeros, ones, concatenate, any, tile, vstack, repeat, linspace, arange, s_
from math import acos, sin, asin, sqrt

from transitLightCurve.core import *
from transitLightCurve.transitparameterization import TransitParameterization, generate_mapping
from transitLightCurve.transitparameterization import parameterizations, parameters
    
try:
    from scipy.constants import G
except ImportError:
    G = 6.67e-11

class MTFitParameterization(object):
    def __init__(self, bnds, stellar_prm, nch, ntr, **kwargs):
        info('Initializing fitting parameterization', H2)

        self.nch = nch
        self.ntr = ntr

        self.fit_radius_ratio     = kwargs.get('fit_radius_ratio',     True)
        self.fit_transit_center   = kwargs.get('fit_transit_center',   True)
        self.fit_period           = kwargs.get('fit_period',          False)
        self.fit_impact_parameter = kwargs.get('fit_impact_parameter', True)
        self.fit_limb_darkening   = kwargs.get('fit_limb_darkening',   True)
        self.fit_zeropoint        = kwargs.get('fit_zeropoint',        True)
        self.fit_ttv              = kwargs.get('fit_ttv',             False)

        self.separate_k2_ch = kwargs.get('separate_k_per_channel',    False)
        self.separate_zp_tr = kwargs.get('separate_zp_per_transit',   False)
        self.separate_ld    = kwargs.get('separate_ld_per_channel',   False)

        self.constant_parameter_names = []
        self.fitted_parameter_names   = []

        self.constant_parameters = None
        self.fitted_parameters = None

        p_init = kwargs.get('P0', {})

        for k in kwargs.keys():
            if 'fit_' in k:
                info(k.replace('_',' ')+': %s'%kwargs[k], I1)

        info('Separate radius ratio per channel: %s' %self.separate_k2_ch, I1)
        info('Separate zeropoint per transit: %s' %self.separate_zp_tr, I1)
        info('Separate limb darkening per channel: %s' %self.separate_ld, I1)
        info('Fit TTVs: %s' %self.fit_ttv, I1)

        self.n_lds = nch if self.separate_ld else 1
        self.n_ldc = len(bnds['ld'][0])
        
        def count_parameter(name):
            return (self.fitted_parameter_names == name).sum()

        def add_parameter(fit, name, count=1):
            if fit:
                self.fitted_parameter_names.extend(count*[name])
            else:
                self.constant_parameter_names.extend(count*[name])

        self.n_k2  = nch if self.separate_k2_ch else 1
        self.n_zp  = nch if not self.separate_zp_tr else nch*ntr

        add_parameter(self.fit_radius_ratio, 'k2', self.n_k2) 
        add_parameter(self.fit_zeropoint, 'zp', self.n_zp) 
        add_parameter(self.fit_transit_center, 'tc') 
        add_parameter(self.fit_period, 'p') 
        add_parameter(self.fit_impact_parameter, 'b2') 
        add_parameter(self.fit_limb_darkening, 'ld', self.n_lds*self.n_ldc) 
        add_parameter(self.fit_ttv, 'ttv', 2)

        self.fitted_parameter_names = array(self.fitted_parameter_names)
        self.constant_parameter_names = array(self.constant_parameter_names)

        self.fitted_parameters = zeros(self.fitted_parameter_names.size)
        self.constant_parameters = zeros(self.constant_parameter_names.size)

        self.n_fitted_parameters = self.fitted_parameters.size
        self.n_constant_parameters = self.fitted_parameters.size

        self.p_cur = self.fitted_parameters

        i_f = arange(self.fitted_parameters.size)
        i_c = arange(self.constant_parameters.size)
        def get_view(fit, name):
            if fit:
                t = i_f[self.fitted_parameter_names == name]
                s = s_[t[0]:t[-1]+1]
                return self.fitted_parameters[s]
            else:
                t = i_c[self.constant_parameter_names == name]
                s = s_[t[0]:t[-1]+1]
                return self.constant_parameters[s]

        self.v_k2  = get_view(self.fit_radius_ratio,     'k2')
        self.v_zp  = get_view(self.fit_zeropoint,        'zp')
        self.v_tc  = get_view(self.fit_transit_center,   'tc')
        self.v_p   = get_view(self.fit_period,            'p')
        self.v_b2  = get_view(self.fit_impact_parameter, 'b2')
        self.v_ld  = get_view(self.fit_limb_darkening,   'ld')
        self.v_ttv = get_view(self.fit_ttv,             'ttv')

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
        self.p_k_min  = self.map_p_to_k([bnds['k'][0], bnds['tc'][0], bnds['p'][0], 10, bnds['b'][0]])
        self.p_k_max  = self.map_p_to_k([bnds['k'][1], bnds['tc'][1], bnds['p'][1], 10, bnds['b'][1]])
        ##
        ## The final fitting parameter boundaries are obtained by excluding the transit width parameter
        ## from the fitting parameter set and adding the limb darkening parameters.
        ##
        self.p_min = concatenate([repeat(self.p_k_min[0], self.n_k2),
                                  repeat(bnds['zp'][0], self.n_zp),
                                  self.p_k_min[[1,2,4]],
                                  tile(bnds['ld'][0], self.n_lds)])
        
        ## Setup constant parameters
        ## =========================
        for i,p in enumerate(['k2','tc','p','a','b2']):
            if p in self.constant_parameter_names:
                self.constant_parameters[self.constant_parameter_names==p] = self.p_k_min[i]

        if not self.fit_zeropoint:
            self.constant_parameters[self.constant_parameter_names=='zp'] = 1.

        for name in p_init:
            if name in self.constant_parameter_names:
                self.constant_parameters[self.constant_parameter_names==name] = p_init[name]
            elif name in self.fitted_parameter_names:
                self.fitted_parameters[self.fitted_parameter_names==name] = p_init[name]


        ## Setup the DE parameter boundaries
        ## =====================================
        self.p_min = concatenate([repeat(self.p_k_min[0], count_parameter('k2')),
                                  repeat(bnds['zp'][0],   count_parameter('zp')),
                                  [self.p_k_min[1]]     * count_parameter('tc'),
                                  [self.p_k_min[2]]     * count_parameter('p'),
                                  [self.p_k_min[4]]     * count_parameter('b2'),
                                  tile(bnds['ld'][0],     count_parameter('ld')),
                                  [bnds['ttv_amplitude'][0]] * count_parameter('ttv'),
                                  [bnds['ttv_period'][0]] * count_parameter('ttv')])

        self.p_max = concatenate([repeat(self.p_k_max[0], count_parameter('k2')),
                                  repeat(bnds['zp'][1],   count_parameter('zp')),
                                  [self.p_k_max[1]]     * count_parameter('tc'),
                                  [self.p_k_max[2]]     * count_parameter('p'),
                                  [self.p_k_max[4]]     * count_parameter('b2'),
                                  tile(bnds['ld'][1],     count_parameter('ld')),
                                  [bnds['ttv_amplitude'][1]] * count_parameter('ttv'),
                                  [bnds['ttv_period'][1]]    * count_parameter('ttv')])

        ## Setup the hard parameter limits 
        ## ================================
        self.l_min = concatenate([repeat(0.0, count_parameter('k2')),
                                  repeat(0.9, count_parameter('zp')),
                                  [0.0]    *  count_parameter('tc'),
                                  [0.0]    *  count_parameter('p'),
                                  [0.0]    *  count_parameter('b2'),
                                  tile(0.0,   count_parameter('ld')),
                                  [-1.]    * count_parameter('ttv'),
                                  [1.]     * count_parameter('ttv')])

        self.l_max = concatenate([repeat(0.1, count_parameter('k2')),
                                  repeat(1.1, count_parameter('zp')),
                                  [1e10]   *  count_parameter('tc'),
                                  [1e10]   *  count_parameter('p'),
                                  [0.99]   *  count_parameter('b2'),
                                  tile(1.0,   count_parameter('ld')),
                                  [1.]     * count_parameter('ttv'),
                                  [1e10]   * count_parameter('ttv')])

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

    def kipping_i(self, period, b2):
        ## Obtain the Kipping's transit width parameter and the semi-major axis using Kepler's third law
        ## =============================================================================================
        a = self.ac * (d_to_s*period)**(2/3)
        it = TWO_PI/period/asin(sqrt(1-b2)/(a*sin(acos(sqrt(b2)/a))))
        return it

    def get_zp(self, ch=0, tn=0, p_in=None):
        if p_in is not None: self.update(p_in)
        return self.v_zp[ch] if not self.separate_zp_tr else self.v_zp[tn*self.nch + ch]

    def get_physical(self, ch=0, p_in=None):
        return self.map_k_to_p(self.get_kipping(ch, p_in))

    def get_kipping(self, ch=0, p_in=None, p_out=None):
        if p_in is not None: self.update(p_in)
        if p_out is None: p_out = zeros(5)
        p_out[0] = self.v_k2[0] if not self.separate_k2_ch else self.v_k2[ch]
        p_out[1] = self.v_tc[0]
        p_out[2] = self.v_p[0]
        p_out[3] = self.kipping_i(self.v_p[0], self.v_b2[0])
        p_out[4] = self.v_b2[0]

        return p_out

    def get_ldc(self, ch=0, p_in=None):
        if p_in is not None: self.update(p_in)
        return self.v_ld[:] if not self.separate_ld else self.v_ld[ch*self.nch:(ch+1)*self.nch]

    def get_ttv(self, p_in=None):
        if p_in is not None: self.update(p_in)
        return self.v_ttv[:] 

    
    def get_parameter_names(self):
        return self.fitted_parameter_names

    def get_parameter_descriptions(self):
        return self.fitted_parameter_names

    def get_hard_limits(self):
        return array([self.l_min, self.l_max])
