"""Contains CoRoT specific constants."""
import sys
import os.path
import numpy as np
import pyfits as pf

from transitLightCurve.core import *
from transitLightCurve.lightcurvedata import MultiTransitLC

time_origin = 2400000

exp_long_s   = 1800
exp_long_min = exp_long_s / 60.
exp_long_h   = exp_long_min / 60.
exp_long_d   = exp_long_h / 24.

def import_as_MTLC(fname, period, center, width, twidth, maxpts=None, **kwargs):
    hdu_d = pf.open(fname)
    dat_d = hdu_d[1].data

    date = dat_d.field('barytime') + time_origin
    flux = dat_d.field('ap_corr_flux')
    fdev = dat_d.field('ap_corr_err')

    maxpts = maxpts or -1
    
    mtlc = MultiTransitLC('k', date[:maxpts], flux[:maxpts], fdev[:maxpts],
                          center, period, width, twidth, channel='k', **kwargs)

    return mtlc
