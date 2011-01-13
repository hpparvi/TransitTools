"""Transitpack core.

Includes the definitions and functions common to all the transitpack modules.

Author
  Hannu Parviainen <hpparvi@gmail.com>

Date
  13.01.2011

"""
import logging

from numpy import pi as PI
from numpy import double as DOUBLE

TWO_PI  = 2.*PI
HALF_PI = 0.5*PI

logging.basicConfig(filename=None, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-6s %(message)s',
                    datefmt='%m-%d %H:%M')

class InvalidTransitShapeError(NotImplementedError): pass

class FitResult(object):
    def __init__(self): pass
    def get_fit(self): raise NotImplementedError
    def get_chi(self): raise NotImplementedError
