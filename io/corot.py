"""Contains CoRoT specific constants."""
import sys
import os.path
import numpy as np
import scipy.constants as c
import pyfits as pf

from transitLightCurve.core import *
from transitLightCurve.lightcurvedata import MultiTransitLC

## --- CoRoT specific constants ---
##

orbit_period = 0.071458  # The period around the Earth
time_origin  = 2451545.  # The epoch from where the CoRoT time starts

channel_center = {'r':0.800e-6, 'g':0.650e-6, 'b':0.550e-6, 'w':0.700e-6}
channel_fits_names = {'r':'red', 'g':'green', 'b':'blue', 'w':'white'}

try:
    corot_dir = os.environ['COROT_DATA_PATH']
except:
    corot_dir = None
    logging.warning("COROT_DATA_PATH is not set")

class CoRoT_target:
    def __init__(self, fname, name, period, transit_center, transit_width, ld_coeffs=None, stellar_parameters=None, contamination=0.0):
        self.file = os.path.join(corot_dir, fname) if corot_dir else fname
        self.name = name
        self.basename = name.replace('-','_')
        self.planet_period = period
        self.transit_center = transit_center
        self.transit_width = transit_width
        self.limb_darkening_coeffs = ld_coeffs
        self.stellar_parameters = stellar_parameters
        self.contamination = contamination

        self.colors = '_CHR_' in fname
        
        self.p = self.planet_period
        self.tc = self.transit_center
        self.sp = self.stellar_parameters


def import_as_MTLC(ctarget, width=0.2, twidth=None, maxpts=None, **kwargs):
    hdu_d = pf.open(ctarget.file)
    dat_d = hdu_d[1].data

    remove_contamination = kwargs.get('remove_contamination', True)

    stat = dat_d.field('STATUS')
    maskOutOfRange = np.bitwise_and(stat, 1) == 0
    maskOverSAA    = np.bitwise_and(stat, 4) == 0
    mask           = (np.logical_and(maskOverSAA, maskOutOfRange))

    date = dat_d.field('DATEJD')[mask].copy().astype(np.float64)

    flux = []; fdev = []
    if not ctarget.colors:
        flux.append(dat_d.field('whiteflux')[mask].copy().astype(np.float64))
        fdev.append(dat_d.field('whitefluxdev')[mask].copy().astype(np.float64))
    else:
        for ch in ['r','g','b']:
            flux.append(dat_d.field( channel_fits_names[ch]+'flux')[mask].copy().astype(np.float64))
            fdev.append(dat_d.field( channel_fits_names[ch]+'fluxdev')[mask].copy().astype(np.float64))

    #if remove_contamination:
    #    for i in range(len(flux)):
    #        flux[i] -= ctarget.contamination * np.median(flux[i])

    twidth = twidth or ctarget.transit_width
    maxpts = maxpts or -1 
    
    mtlc = []
    for i, ch in zip(range(len(flux)), ['r','g','b']):
        mtlc.append(MultiTransitLC(channel_fits_names[ch], date[:maxpts]+time_origin,
                                   flux[i][:maxpts], fdev[i][:maxpts],
                                   ctarget.transit_center, ctarget.planet_period,
                                   width, twidth, channel=ch, **kwargs))

    return mtlc


## --- The CoRoT Planets ---
##
C01 = CoRoT_target('EN2_STAR_CHR_0102890318_20070206T133547_20070402T070302.fits',
                   name = 'CoRoT-1b',
                   period = 1.5089557,
                   transit_center = 2454159.4532,
                   transit_width  = 0.1)

C02 = CoRoT_target('EN2_STAR_CHR_0101206560_20070516T060226_20071005T074409.fits',
                   name = 'CoRoT-2b',
                   period = 1.743,
                   transit_center = 2454237.53562,
                   transit_width = 75*min_to_d,
                   stellar_parameters={'T':5625., 'logg':4.3, 'M/H':0.0,
                                       'M':1.93e30, 'R':6.27e8})

C03 = CoRoT_target('EN2_STAR_CHR_0101368192_20070516T060050_20071015T062306.fits',
                   name = 'CoRoT-3b',
                   period = 0,
                   transit_center = 0,
                   transit_width  = 0*min_to_d,
                   stellar_parameters={'T':0., 'logg':0, 'M/H':0.0,
                                       'M':0, 'R':0})

C04 = CoRoT_target('EN2_STAR_CHR_0102912369_20070203T130553_20070402T070126.fits',
                   name = 'CoRoT-4b',
                   period = 0,
                   transit_center = 0,
                   transit_width  = 0*min_to_d,
                   stellar_parameters={'T':0., 'logg':0, 'M/H':0.0,
                                       'M':0, 'R':0})

C05 = CoRoT_target('EN2_STAR_CHR_0102764809_20071023T223035_20080303T093502.fits',
                   name = 'CoRoT-5b',
                   period = 4.0378962,
                   transit_center = 2454400.19885,
                   transit_width  = 0.12,
                   stellar_parameters={'T':6100., 'logg':4.19, 'M/H':0.25,
                                       'M':solar_mass, 'R':1.186*solar_radius})


C07 = CoRoT_target('EN2_STAR_CHR_0102708694_20071024T123523_20080303T093502.fits',
                   name = 'CoRoT-7b',
                   period = 0.853585,
                   transit_center = 2454398.0767,
                   transit_width  = 75*min_to_d,
                   stellar_parameters={'T':5250., 'vsini':1.1e3, 'logg':4.47, 'M/H':0.12,
                                       'M':1.81e30, 'R':5.63e8})

C08 = CoRoT_target('EN2_STAR_CHR_0101086161_20070516T060226_20071005T074409.fits',
                   name = 'CoRoT-8b',
                   period = 6.21229,
                   transit_center = 2454238.9743,
                   transit_width  = 2.74*h_to_d,
                   stellar_parameters={'T':5080., 'vsini':2e3, 'logg':4.58, 'M/H':0.3,
                                       'M':1.75e30, 'R':5.36e8})

C10 = CoRoT_target('EN2_STAR_MON_0100725706_20070516T060226_20071005T074409.fits',
                   name = 'CoRoT-10b',
                   period = 13.2406,
                   transit_center = 2454273.3436,
                   transit_width  = 2.98*h_to_d,
                   stellar_parameters={'T':5075., 'vsini':2e3, 'logg':4.65, 'M/H':0.26,
                                       'M':0.89*solar_mass, 'R':0.79*solar_radius})

C11 = CoRoT_target('EN2_STAR_CHR_0105833549_20080415T231048_20080907T224903.fits',
                   name = 'CoRoT-11b',
                   period = 2.99433,
                   transit_center = 2454597.6790,
                   transit_width  = 0.0348,
                   #{'r':[0.333, 0.208], 'g':[0.521, 0.148], 'b':[0.714, 0.064], 'w':[0.924, -0.278]},
                   stellar_parameters={'T':6440., 'vsini':40.0e3, 'logg': 4.26, 'M':2.525e30, 'R':9.53e8},
                   contamination = 0.13)

C23 = CoRoT_target('EN2_STAR_MON_0105228856_20100408T223049_20100705T044435.fits',
                   name = 'CoRoT-23b',
                   period = 3.6307667,
                   transit_center = 2455308.9488,
                   transit_width  = 3.823*h_to_d,
                   stellar_parameters={'T':6440., 'vsini':40.0e3, 'logg': 4.26, 'M':1.098*solar_mass, 'R':9.53e8},
                   contamination = 0.045)

CoRoT_targets = {1:C01, 2:C02, 3:C03, 4:C04, 5:C05, 7:C07, 8:C08, 10:C10, 11:C11, 23:C23}
