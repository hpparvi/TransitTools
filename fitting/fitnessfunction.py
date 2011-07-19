import sys
from types import MethodType
from math import sin
from timeit import timeit

import numpy as np
from numpy import asarray, array, concatenate, zeros, zeros_like, add

try:
    import numexpr as ne
    with_numexpr = False #True
except ImportError:
    with_numexpr = False

from transitLightCurve.core import *
from transitLightCurve.transitlightcurve import TransitLightcurve
from transitLightCurve.transitparameterization import TransitParameterization

import fitnessfunction_f as ff
calculate_time_with_ttv = ff.fitnessfunction.calculate_time_with_ttv
chi_sqr = ff.fitnessfunction.chi_sqr
apply_zeropoints = ff.fitnessfunction.apply_zeropoints

def addl(code, cdl):
    code.append(cdl+'\n')

class FitnessFunction(object):
    def __init__(self, parm, data, **kwargs):
        self.parm = parm
        self.data = data
        self.nch  = len(data)
        self.chid = range(self.nch)

        self.contamination = kwargs.get('contamination', 0.)
        self.eccentric = kwargs.get('eccentric',False)
        if self.eccentric:
            self.eccentricity = kwargs['eccentricity']
            self.omega = kwargs['omega']

        self.gk = parm.get_kipping
        self.gz = parm.get_zp
        self.gl = parm.get_ldc
        self.gt = parm.get_ttv
        self.gb = parm.get_b2
        self.gc = parm.get_contamination

        self.times  = [t.get_time()               for t in data]
        self.fluxes = [t.get_flux(normalize=True) for t in data]
        self.ivars  = [t.get_ivar(normalize=True) for t in data]
        self.slices = [t.get_transit_slices()     for t in data]
        self.lengths= array([t.size for t in self.times]).astype(np.int32)

        self.channel_slices = []
        i_s = 0; i_e = 0
        for i in self.chid:
            i_e += self.lengths[i]
            self.channel_slices.append(np.s_[i_s:i_e])
            i_s = i_e

        self.tnumbs = []
        for i, ch in enumerate(self.data):
            self.tnumbs.append(np.zeros(self.times[i].size))
            for sl, tn in zip(self.slices[i], [t.number for t in ch.transits]):
                self.tnumbs[i][sl] = tn
                
        self.atimes  = concatenate(self.times)
        self.afluxes = concatenate(self.fluxes)
        self.aivars  = concatenate(self.ivars)
        self.atnumbs = concatenate(self.tnumbs)

        self.ttv     = zeros_like(self.atimes) 
        self.atmp    = zeros_like(self.atimes) 
        self.mtmp    = zeros_like(self.atimes)

        for i in range(len(self.ivars)):
            self.ivars[i] *= kwargs.get('ivar_multiplier', 1.)

        method   = kwargs.get('method', 'fortran')
        npol     = kwargs.get('n_pol', 500)
        nthreads = kwargs.get('n_threads', 1)

        self.lc = TransitLightcurve(TransitParameterization('kipping', [0.1,0,5,10,0]),
                                    method=method, ldpar=[0], n_threads=nthreads, npol=npol,
                                    eccentric=self.eccentric)

        self.fitfun_code = ""
        self.generate_fitfun()
 

    def basic_model(self, time, ch, tr):
        return self.gz(ch,tr) * self.lc(time, self.gk(ch), self.gl(ch))

    def basic_model_str(self, time, ch, tr):
        return 'gz({ch},{tr}) * lc({time}, kp, ld))'.format(ch=ch, tr=tr, time=time)

    def ttv_model_str(self, time, ch, tr):
        src = []
        src.append('    tc  = kp[2] * self.data[{ch}].transits[{tr}].number'.format(ch=ch,tr=tr))
        src.append('    ttv = p_ttv[0]*sin(p_ttv[1]*tc*TWO_PI)')
        src.append('    model = gz({ch},{tr}) * lc({time} + ttv, kp, ld)'.format(ch=ch,tr=tr, time=time))
        return '\n'.join(src)

    def ttv_model(self, time, ch, tr, p_ttv, ldc):
        pk  = self.gk(ch)
        tc  = pk[2] * self.data[ch].transits[tr].number
        ttv = p_ttv[0]*sin(p_ttv[1]*tc*TWO_PI)
        return self.gz(ch,0) * self.lc(time + ttv, pk, ldc) 


    def generate_fitfun(self):
        """Generates a fitness function based on given fit parameterization.
        Note: As anyone can see, this is a bit of a mess at the moment. Needs a serious cleanup!
        """
        nch = len(self.data)
        channels = range(nch)
        c = []
        addl(c, "def fitfun(self, p_fit):")
        addl(c, "  if self.parm.is_inside_limits(p_fit):")
        ecc_str = ', e=self.eccentricity, w=self.omega' if self.eccentric else ''

        ## Limb-darkening
        ## ==============
        if self.parm.separate_ld:
            for ch in channels:
                addl(c, "    ld{ch} = self.gl(p_fit, {ch})".format(ch=ch))
                addl(c, "    if ld{ch}[0] < 0. or ld{ch}[0] > 1.: return 1e18".format(ch=ch))
        else:
            addl(c, "    ld = self.gl(p_fit)")
            addl(c, "    if ld[0] < 0. or ld[0] > 1.: return 1e18")

        if not self.parm.separate_k2_ch and not self.parm.separate_zp_tr and not self.parm.fit_ttv:
            addl(c, "    kp = self.gk(p_fit)")
            addl(c, "    chi = 0.")
            #addl(c, '    zp = array([self.gz(i_ch) for i_ch in range(self.nch)])')
            for ch in channels:
                addl(c, '    model = self.gz(p_fit, {ch}) * self.lc(self.times[{ch}], kp, ld{ch}{ecc_str}, contamination=self.gc(p_fit))'.format(ch=ch,ecc_str=ecc_str))
                addl(c, '    chi += chi_sqr(self.fluxes[{ch}], model, self.ivars[{ch}])'.format(ch=ch))

                #addl(c, '    self.atmp[{ls}:{le}] = self.lc(self.times, self.gk({ch}), self.gl({ch}))'.format(ch=ch, ls=0 if ch==0 else self.lengths[:ch].sum(), le=-1 if ch==len(self.data) else self.lengths[:ch+1].sum()))
                #addl(c, '    self.atmp[:] = apply_zeropoints(zp, self.lengths, self.atmp)')
                #addl(c, '    chi = chi_sqr(self.afluxes, self.atmp, self.aivars)')
            #addl(c, "    import pylab as pl; pl.plot(self.times[1], model); pl.show(); exit()")    
            addl(c, "    return chi")
        else:
            #addl(c, "    b2 = self.gb()")
            #addl(c, "    if b2 < 0. or b2 > 1.: return 1e18\n")

            if self.parm.fit_ttv:
                addl(c, "    p_ttv = self.gt(p_fit); ttv_a = p_ttv[0]; ttv_p = p_ttv[1]")

            if not self.parm.separate_k2_ch:
                addl(c, '    kp=self.gk(p_fit); period=kp[2]')
            else:
                raise NotImplementedError

            addl(c,'')
            if self.parm.fit_ttv:
                addl(c, '    zp = array([self.gz(p_fit, i_ch) for i_ch in range(self.nch)])')
                addl(c, '    self.atmp[:] = calculate_time_with_ttv(ttv_a, ttv_p, period, self.atimes, self.atnumbs)')
                if not self.parm.separate_ld:
                    addl(c, '    self.atmp[:] = self.lc(self.atmp, kp, ld)')
                else:
                    for ch in range(len(self.data)):
                        addl(c, '    self.atmp[self.channel_slices[{ch}]] = self.lc(self.atmp[self.channel_slices[{ch}]], kp, ld{ch}, {ecc_str})'.format(ch=ch, ecc_str=ecc_str))
                addl(c, '    self.atmp[:] = apply_zeropoints(zp, self.lengths, self.atmp)')
                addl(c, '    chi = chi_sqr(self.afluxes, self.atmp, self.aivars)') # ne.evaluate("sum((fl - mtmp)**2 * iv)")')
            else:
                raise NotImplementedError
                #     else:
                #         model_str = self.basic_model_str('tm[{}]'.format(str(sl.start)+':'+str(sl.stop)), i, tr)
                #         addl(c, "    chi += ((fl[{1}] - {0})**2 * iv[{1}]).sum()".format(model_str, str(sl.start)+':'+str(sl.stop)))
            addl(c, "\n    return chi")
        addl(c, "  else:")
        addl(c, "    return 1e18")
        code = ""
        for line in c: code += line
        print code#; exit()
        exec(code)
        self.fitfun_src = code
        self.fitfun = MethodType(fitfun, self, FitnessFunction)

    def __call__(self, p_fit):
        return self.fitfun(p_fit)
