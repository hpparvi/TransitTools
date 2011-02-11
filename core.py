"""Transitpack core.

Includes the definitions and functions common to all the transitpack modules.

Author
  Hannu Parviainen <hpparvi@gmail.com>

Date
  13.01.2011

"""
from __future__ import division

import logging

from numpy import pi as PI
from numpy import double as DOUBLE

TWO_PI  = 2.*PI
HALF_PI = 0.5*PI

d_to_s = float(24*3600)

logging.basicConfig(filename=None, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-6s %(message)s',
                    datefmt='%m-%d %H:%M')

try:
    import pyopencl as cl
    with_opencl = True
except ImportError:
    logging.warning("Failed to load pyopencl: cannot use OpenCL routines.")
    with_opencl = False

try:
    from mpi4py import MPI
    with_mpi = True
except ImportError:
    logging.warning("Failed to load mpi4py: cannot use MPI.")
    with_mpi = False

class FitResult(object):
    def __init__(self): pass
    def get_fit(self): raise NotImplementedError
    def get_chi(self): raise NotImplementedError


H1 = 0
H2 = 1
I1 = 2
I2 = 3
I3 = 4

def info(msg, style=10):
    if style == H1:
        logging.info('')
        logging.info(msg)
        logging.info(len(msg)*"=")
    elif style == H2:
        logging.info('')
        logging.info(msg)
        logging.info(len(msg)*"-")
    elif style == I1:
        logging.info(2*' '+msg)
    elif style == I2:
        logging.info(4*' '+msg)
    else:
        logging.info(msg)
