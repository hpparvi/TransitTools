#!/usr/bin/env python2.5
"""
Base
====
"""
import numpy as np
import logging

TWO_PI = 2.*np.pi
HALF_PI = 0.5*np.pi

logging.basicConfig(filename=None, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-6s %(message)s',
                    datefmt='%m-%d %H:%M')

class InvalidTransitShapeError(NotImplementedError): pass

class FitResult(object):
    def __init__(self): pass
    def get_fit(self): raise NotImplementedError
    def get_chi(self): raise NotImplementedError

class transitLightCurve(object):

    def __init__(self, primary_shape=None, secondary_shape=None):

        self.float_t = np.float32

        if primary_shape is None:
            self.do_primary_eclipse   = False
        elif isinstance(primary_shape, transitShape):
            self.do_primary_eclipse   = True
            self.primary_shape        = primary_shape
        else:
            raise InvalidTransitShapeError()

        if secondary_shape is None:
            self.do_secondary_eclipse = False
        elif isinstance(secondary_shape, transitShape):
            self.do_secondary_eclipse   = True
            self.secondary_shape        = secondary_shape
        else:
            raise InvalidTransitShapeError()
        
            

    def __eval__(self):
        pass

    def __str__(self):
        return "transitLightCurve:\n\tPrimary   %s\n\tSecondary %s"%(str(self.do_primary_eclipse), str(self.do_secondary_eclipse))

class transitShape(object):

    def __init__(self, use_Fortran=True, use_openCL=True):
        pass

    def __eval__(self):
        pass
