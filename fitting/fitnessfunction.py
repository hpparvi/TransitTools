import sys
from types import MethodType
from math import sin

import numpy as np

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

        self.times  = [t.get_time() for t in data]
        self.fluxes = [t.get_flux() for t in data]
        self.ivars  = [t.ivar       for t in data]
        self.pntns  = [t.pntn       for t in data]
        self.slices = [t.get_transit_slices() for t in data]

        method = kwargs.get('method', 'fortran')

        self.lc = TransitLightcurve(TransitParameterization('kipping', [0.1,0,5,10,0]),
                                    method=method, ldpar=[0], zeropoint=1., n_threads=2, npol=100)

        self.fitfun_code = ""
        self.generate_fitfun()
 

    def basic_model(self, time, ch, tr):
        return self.gz(ch,0) * self.lc(time, self.gk(ch), self.gl(ch))


    def ttv_model(self, time, ch, tr, p_ttv):
        pk  = self.gk(ch)
        tc  = pk[2] * self.data[ch].transits[tr].number
        ttv = p_ttv[0]*sin(p_ttv[1]*tc)
        return self.gz(ch,0) * self.lc(time + ttv, pk, self.gl(ch)) 


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
            addl(c, "        chi += ((flux - self.basic_model(time, ch, 0))**2 * ivar).sum()")
            addl(c, "    return chi")
        else:
            if self.parm.fit_ttv:
                addl(c, "    p_ttv = self.gt()")
            addl(c, "    for ch, (time,flux,ivar,sls) in enumerate(zip(self.times,self.fluxes,self.ivars,self.slices)):")
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
