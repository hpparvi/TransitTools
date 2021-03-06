"""Transitpack core.

Includes the definitions and functions common to all the transitpack modules.

Author
  Hannu Parviainen <hpparvi@gmail.com>

Date
  13.01.2011

"""
from __future__ import division

import logging
import curses

from numpy import pi as PI
from numpy import double as DOUBLE

## - Astronomical constants -
##
solar_radius  = 6.955e8   # [m]
solar_mass    = 1.9891e30 # [kg]
solar_density = 1.408e3   # [kg/m^3]

## - Mathematical constants -
##
HALF_PI = 0.5 * PI
TWO_PI  = 2.0 * PI

## - Time conversion -
##
min_to_d = 1./(60.*24.)
h_to_d   = 1./24.
d_to_s   = 24.*3600.

## - Setup logging -
##
logging.basicConfig(filename=None, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-6s %(message)s',
                    datefmt='%m-%d %H:%M')

## - PyOpenCL -
##
try:
    import pyopencl as cl
    with_opencl = True
except ImportError:
    logging.warning("Failed to load pyopencl: cannot use OpenCL routines.")
    with_opencl = False

## - MPI4PY -
##
try:
    from mpi4py import MPI
    with_mpi = True
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    is_root  = mpi_rank == 0
except ImportError:
    logging.warning("Failed to load mpi4py: cannot use MPI.")
    with_mpi = False
    mpi_rank = 0
    mpi_size = 1
    is_root = True

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

class CTitleWindow(object):
    def __init__(self, title, width, height, xpos, ypos):
        self.window = curses.newwin(height, width, ypos, xpos)
        self.window.box()
        self.window.addstr(0,2, ' %s '%title)
        self.window.refresh()
        self.title = title
        self.widht = width
        self.height = height

    def addstr(self, *args):
        self.window.addstr(*args)
        self.window.refresh()
        
    def refresh(self) : self.window.refresh()
