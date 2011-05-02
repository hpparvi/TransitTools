import sys
from types import MethodType
from math import sin
from timeit import timeit

import numpy as np
from numpy import asarray, array

try:
    import numexpr as ne
    with_numexpr = False #True
except ImportError:
    with_numexpr = False

from transitLightCurve.core import *
from transitLightCurve.transitlightcurve import TransitLightcurve
from transitLightCurve.transitparameterization import TransitParameterization

def addl(code, cdl):
    code.append(cdl+'\n')

class FitnessFunction(object):
    def __init__(self, parm, data, **kwargs):
        self.parm = parm
        self.data = data

        self.gk = parm.get_kipping
        self.gz = parm.get_zp
        self.gl = parm.get_ldc
        self.gt = parm.get_ttv
        self.gb = parm.get_b2

        self.times  = [t.get_time()               for t in data]
        self.fluxes = [t.get_flux(normalize=True) for t in data]
        self.ivars  = [t.get_ivar(normalize=True) for t in data]
        self.pntns  = [t.pntn                     for t in data]
        self.slices = [t.get_transit_slices()     for t in data]
 
        self.tnumbs = []
        for i, ch in enumerate(self.data):
            self.tnumbs.append(np.zeros(ch.time.size))
            for sl, tn in zip(self.slices[i], [t.number for t in ch.transits]):
                self.tnumbs[i][sl] = tn

        for i in range(len(self.ivars)):
            self.ivars[i] *= kwargs.get('ivar_multiplier', 1.)

        method   = kwargs.get('method', 'fortran')
        npol     = kwargs.get('n_pol', 250)
        nthreads = kwargs.get('n_threads', 1)

        self.lc = TransitLightcurve(TransitParameterization('kipping', [0.1,0,5,10,0]),
                                    method=method, ldpar=[0], zeropoint=1., n_threads=nthreads, npol=npol)

        self.fitfun_code = ""

        if kwargs.get('unroll_loops', True):
            self.generate_fitfun_unroll()
        else:
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


    def generate_fitfun_unroll(self):
        """Generates a fitness function based on given fit parameterization.

        """
        c = []
        addl(c, "def fitfun(self, p_fit):")
        addl(c, "  self.parm.update(p_fit)")
        addl(c, "  if self.parm.is_inside_limits():")
        addl(c, "    gz=self.gz; gk=self.gk; gl=self.gl; gb=self.gb; lc=self.lc")
        addl(c, "    fl=self.fluxes; tm=self.times; iv=self.ivars; tn=self.tnumbs")
        if with_numexpr:
            for i in range(len(self.data)):
                addl(c, '    fl_{ch}=fl[{ch}]; tm_{ch}=tm[{ch}]; iv_{ch} = iv[{ch}]; tn_{ch} = tn[{ch}]'.format(ch=i))

        addl(c, "    chi=0.")

        if not self.parm.separate_k2_ch and not self.parm.separate_zp_tr and not self.parm.fit_ttv:
            if self.parm.separate_ld:
                for i in range(len(self.data)):
                    if i>0: addl(c, "    if not np.all(asarray(self.gl({0})) > asarray(self.gl({1}))): return 1e18".format(i, i-1))

            for i in range(len(self.data)):
                addl(c, "    chi += ((fl[{0}] - self.basic_model(tm[{0}], {0}, 0))**2 * iv[{0}]).sum()".format(i))
            addl(c, "    return chi")
        else:
            if self.parm.separate_ld:
                for i in range(len(self.data)):
                    if i>0: addl(c, "    if not np.all(asarray(self.gl({0})) > asarray(self.gl({1}))): return 1e18".format(i, i-1))
            else:
                addl(c, "    ld = gl(); b2 = gb()")
                addl(c, "    if ld[0] < 0. or ld[0] > 1.: return 1e18")
                addl(c, "    if b2 < 0. or b2 > 1.: return 1e18")

            if self.parm.fit_ttv:
                addl(c, "    p_ttv = self.gt(); ttv_a = p_ttv[0]; ttv_p = p_ttv[1]")

            if not self.parm.separate_k2_ch:
                addl(c, '    kp=gk(); period=kp[2]')
            else:
                raise NotImplementedError

            ## Loop over all channels
            for ch in range(len(self.data)):
                addl(c,'')
                if not self.parm.separate_zp_tr:
                    zp_str = 'gz({})'.format(ch)

                if self.parm.separate_k2_ch: addl(c, '    kp=gk({0})'.format(ch))
                if self.parm.separate_ld: addl(c, '    ld=gl({0})'.format(ch))

                if self.parm.fit_ttv:
                    if with_numexpr:
                        zp_str = 'zp'
                        addl(c, '    tn_{ch:d} = tn[{ch:d}]; zp = gz({ch})'.format(ch=ch))
                        addl(c, '    tm_t  = ne.evaluate("tm_{ch} + ttv_a*sin( TWO_PI*ttv_p * period*tn_{ch:d} )")'.format(ch=ch))
                        addl(c, '    model = lc(tm_t, kp, ld)'.format(ch=ch))
                        addl(c, '    chi  += ne.evaluate("sum((fl_{ch} - {zp}*model)**2 * iv_{ch})")'.format(ch=ch,zp=zp_str))
                    else:
                        addl(c, '    ttv = ttv_a*np.sin( TWO_PI*ttv_p * period*tn[{ch}] )'.format(ch=ch))
                        addl(c, '    chi += ((fl[{ch}] - {zp}*lc(tm[{ch}] + ttv, kp, ld))**2 * iv[{ch}]).sum()'.format(ch=ch,zp=zp_str))
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
        print code; #exit()
        exec(code)
        self.fitfun_src = code
        self.fitfun = MethodType(fitfun, self, FitnessFunction)

    def generate_fitfun(self):
        """Generates a fitness function based on given fit parameterization.

        """
        c = []
        addl(c, "def fitfun(self, p_fit):")
        addl(c, "  self.parm.update(p_fit)")
        addl(c, "  if self.parm.is_inside_limits():")
        addl(c, "    chi = 0.")
        if not self.parm.separate_k2_ch and not self.parm.separate_zp_tr and not self.parm.fit_ttv:
            addl(c, "    for ch, (time,flux,ivar) in enumerate(zip(self.times,self.fluxes,self.ivars)):")
            addl(c, "        if ch > 0 and not np.all(asarray(self.gl(ch)) > asarray(self.gl(ch-1))): return 1e18\n")
            addl(c, "        chi += ((flux - self.basic_model(time, ch, 0))**2 * ivar).sum()")
            addl(c, "    return chi")
        else:
            if self.parm.fit_ttv:
                addl(c, "    p_ttv = self.gt()")
            addl(c, "    for ch, (time,flux,ivar,sls) in enumerate(zip(self.times,self.fluxes,self.ivars,self.slices)):")
            addl(c, "        if ch > 0 and not np.all(asarray(self.gl(ch)) > asarray(self.gl(ch-1))): return 1e18\n")
            addl(c, "        for tr, sl in enumerate(sls):")
            if self.parm.fit_ttv:
                addl(c, "            chi += ((flux[sl] - self.ttv_model(time[sl], ch, tr, p_ttv))**2 * ivar[sl]).sum()")
            else:
                addl(c, "            chi += ((flux[sl] - self.basic_model(time[sl], ch, tr))**2 * ivar[sl]).sum()")
            addl(c, "    return chi")
        addl(c, "  else:")
        addl(c, "    return 1e18")

        code = ""
        for line in c: code += line

        self.fitfun_code = code
        exec(code)
        self.fitfun = MethodType(fitfun, self, FitnessFunction)


    def __call__(self, p_fit):
        return self.fitfun(p_fit)
