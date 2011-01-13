"""A set of utility functions including binning and folding."""

from core import *
import numpy as np

try:
    from utilities_f import *
except ImportError:
    logging.warning("Couldn't load Fortran routines, using Python fallback.")

    from numpy import mean

    def running_mean(x, width, res):
        nx  = x.size
        iw  = 1./real(width,8)
        hw  = (width-1)/2
        np  = nx - width + 1
        res = np.zeros(nx)
        
        for i in range(1+hw, nx-hw):
            res[i] = np.sum(x[i-hw:i+hw]) * iw

        return res


    def bin(x, y, nbins): 
        """
        Generic routine for binning. Takes the limits of the binned space and 
        either the number of bins or the width of a single bin.

        Parameters:
            bn Number of bins
            bw Width of a single bin
            lim Limits of the binned space
        """

        lim = np.array([x.min(), x.max()])
        sw = lim[1] - lim[0] + 1e-8

        xb = (np.arange(bn) + 0.5) * bw + lim[0]
        yb = np.zeros(bn)
        ys = np.zeros(bn)
        ye = np.zeros(bn)
        wb = np.zeros(bn)

        bi = np.floor((x - lim[0]) / bw).astype(np.int)

        for id in range(bn):            
            m = bi == id
            n = m.sum()
            ws = weights[m].sum()
            yb[id] = (y[m] * weights[m]).sum() / ws
            ys[id] = y[m].std()
            ye[id] = np.sqrt((weights[m] * (y[m] - yb[id])**2).sum() / (ws * (n-1.)))
            wb[id] = n

        return xb[wb>minn], yb[wb>minn], ye[wb>minn]


    def chi_squared(O, M, E):
        return ((O-M)**2 * E).sum()



def fold(x, period, origo=0.0, shift=0.0, normalize=True,  clip_range=None):
    """
    Folds the given data over a given period.
    """
    xf = ((x - origo)/period + shift) % 1.
    if clip_range is not None:
        mask = np.logical_and(clip_range[0]<xf, xf<clip_range[1])
        xf = xf[mask], mask
    return xf


def bin_old(x, y, weights=None, bn=None, bw=None, lim=None, minn=3, method=1): 
    """
    Generic routine for binning. Takes the limits of the binned space and 
    either the number of bins or the width of a single bin.

    Parameters:
        bn Number of bins
        bw Width of a single bin
        lim Limits of the binned space
    """

    if bw is not None and bn is not None:
        print "Error in binning: give either the number of bins or the width of a bin."

    if lim is None:
        lim = np.array([x.min(), x.max()])
        sw = lim[1] - lim[0] + 1e-8
    else:
        sw = lim[1] - lim[0]

    if bw is None:
        bw = sw / float(bn)
    else:
        bn = int(np.ceil(sw / bw))

    if weights is None:
        weights = np.ones(x.size, np.double) / np.double(x.size)

    xb = (np.arange(bn) + 0.5) * bw + lim[0]
    yb = np.zeros(bn)
    ys = np.zeros(bn)
    ye = np.zeros(bn)
    wb = np.zeros(bn)

    bi = np.floor((x - lim[0]) / bw).astype(np.int)

    # Two different binning methods for different situations. The other loops through 
    # the given data, and other through the bins. The latter is probably faster when
    # binning a large amount of data to relatively small number of bins.
    #
    # TODO: Binning - Test the efficiencies of the two binning methods.
    # TODO: Binning - Weighted errorbars may still need some checking 
    if method == 0:
        for i, id in enumerate(bi):
            yb[id] += y[i]
            wb[id] += 1.
        yb[wb>minn] /= wb[wb>minn]
    else:   
        for id in range(bn):            
            m = bi == id
            n = m.sum()
            ws = weights[m].sum()
            yb[id] = (y[m] * weights[m]).sum() / ws  # yb[id], ws = np.average(y[m], 0, weights[m], True)
            ys[id] = y[m].std()
            ye[id] = np.sqrt((weights[m] * (y[m] - yb[id])**2).sum() / (ws * (n-1.)))
            wb[id] = n

    return xb[wb>minn], yb[wb>minn], ye[wb>minn]
 
