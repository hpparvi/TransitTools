#!/usr/bin/env python
import numpy as np
from scipy.special import jacobi, gamma, gammaln
from numpy import exp

from transitLightCurve.core import *
from transitmodel import TransitModel

def _Planck_v(v,T):
    return 2. * c.h * v**3 / c.c**2 / (np.exp(c.h*v/(c.k*T)) - 1.)

def _Planck(l,T):
    return 2. * c.h * c.c**2 / l**5 / (np.exp(c.h*c.c/(l*c.k*T)) - 1.)


class Barnes(TransitModel):

    def luminosity(x, y, Ms, Rs, Os, Tp, gp, f, phi, beta, c, ot=0.51e-6):
        npt = x.size

        T  = np.zeros([npt, 3], double)
        Dg = np.zeros([npt, 3], double)
        Dc = np.zeros([npt, 3], double)

        z  = Barnes_z(x, y, Rs, f, phi)
        mu = z / sqrt((x**2 + y**2 + z**2))

        ## Direction vectors
        ##
        Dg[:,0] =  x
        Dg[:,1] =  y * cos(phi) + z * sin(phi)
        Dg[:,2] = -y * sin(phi) + z * cos(phi)

        Dc[:,0] = Dg[:,0]
        Dc[:,2] = Dg[:,2] 

        ## Lengths of the direction vectors
        ##
        lg2 = (Dg**2).sum(1)
        lg  = sqrt(lg2)
        lc  = sqrt((Dc**2).sum(1))

        ## Normalize the direction vectors
        ##
        Dg /= np.swapaxes(np.tile(lg,[3,1]),0,1)
        Dc /= np.swapaxes(np.tile(lc,[3,1]),0,1)

        gg = - G*Ms/lg2 
        gc =   Os*Os*lc

        Dgg =  np.swapaxes(np.tile(gg,[3,1]),0,1) * Dg + np.swapaxes(np.tile(gc,[3,1]),0,1) * Dc
        g   = sqrt((Dgg**2).sum(1))
        T   = Tp * g**beta / gp**beta

        return Planck(ot, T) * ( 1. - c[0] * (1. - mu) - c[1] * (1. - mu)**2)


    def _z(x, y, R, f, phi):
        """
        ("Transit Lightcurves of Extrasolar Planets Orbiting Rapidly-Rotating Stars", Barnes, J.)
        """

        sp  = sin(phi)
        cp  = cos(phi)
        sp2 = sp*sp
        cp2 = cp*cp

        x2  = x*x
        y2  = y*y
        f2  = f*f

        A   = 1. / (1.-f)**2

        d =   4. * y2 * sp2 * cp2  * (A-1.)**2 \
            - 4. * (cp2 + A*sp2) * (x2 + y2 * (A*cp2 + sp2) - R*R)

        z = (-2. * y * cp * sp * (A-1.) +sqrt(d)) / (2. * (cp2 + A*sp2))

        return z

 
    def _stellar_flux(
        f=0.0, phi=0.0, Rs = 2.029, Ms = 1.8, Ps=8.64, Tp = 8450., beta=0.2, c = [0.5, 0.1] , 
        l=0.51e-6, res_s=256, method='brute'):
        """
        Implements the transit model for fast rotating stars by J. Barnes (Barnes,, J., PASP 119, pp. 986--993, 2007 Sep.)
        with corrected equations for the shape of the star.

        Parameters:
            f       Stellar oblateness
            phi     Stellar obliquity [radians]

            Rs      stellar equator radius [R_sun]
            Ms      stellar mass [M_sun]
            Ps      stellar rotation period [h]
            Tp      temperature at the pole [K]
            beta    gravity-darkening parameter
            c       limb-darkening parameters

            l       wavelength [m]

            res_s   resolution for the star
        """

        Rs = Rs * double(6.955e8)       # the stellar equator radius [R_sun] -> [m]
        Ms = Ms * double(1.9891e30)     # the stellar mas [M_sun] -> [kg]
        Os = 2.*pi / (3600.*Ps)         # the stellar rotation rate [rad/s] 
        gp = G * Ms / (Rs*(1-f))**2     # the surface gravity at the pole [m/s^2]

        f_eff  = 1. - sqrt((1. - f)**2 * cos(phi)**2 + sin(phi)**2)

        if method == 'brute':
            xs, ys = np.meshgrid(np.linspace(-Rs, Rs, res_s), np.linspace(-Rs, Rs, res_s))
            mask_s = sqrt(xs**2 + (ys/(1.-f_eff))**2) < Rs

            Is = Barnes_luminosity(xs[mask_s], ys[mask_s], Ms, Rs, Os, Tp, gp, f, phi, beta, c, 0.51e-6) 
            Is_total = Is.sum() * (pi * Rs**2 *(1.-f_eff) / mask_s.sum()) 

            print "npts: ", mask_s.sum()

        elif method == 'Peirce':
            xs, ys, ws = dc.generate_Peirce_points(7, r=Rs)
            ys *= 1.-f_eff

            Is = Barnes_luminosity(xs.ravel(), ys.ravel(), Ms, Rs, Os, Tp, gp, f, phi, beta, c, 0.51e-6)
            Is = np.reshape(Is, xs.shape)

            Is_total = (ws[0,:]*Is.sum(0)).sum() * (Rs**2 *(1.-f_eff))
        else: 
            return 0

        return Is_total

    def transit_Barnes(
        f=0.0, phi=0.0, 
        Ap = 1., Rp = 0.1, 
        Rs = 2.029, Ms = 1.8, Ps=8.64, Tp = 8450., a=0., b=0., beta=0.2, c = [0.5, 0.1] , l=0.51e-6,  
        res_s=256, res_p=4, res_lc=4, 
        method='Peirce', phase=None, partial_results=None):
        """
        Implements the transit model for fast rotating stars by J. Barnes (Barnes,, J., PASP 119, pp. 986--993, 2007 Sep.)
        with corrected equations for the shape of the star.

        Parameters:
            f       Stellar oblateness
            phi     Stellar obliquity [radians]

            Ap      semimajor axis [AU]
            Rp      planet radius [R_sun]

            Rs      stellar equator radius [R_sun]
            Ms      stellar mass [M_sun]
            Ps      stellar rotation period [h]
            Tp      temperature at the pole [K]
            beta    gravity-darkening parameter
            c       limb-darkening parameters

            a       azimuthal angle [radians]
            b       impact parameter

            l       wavelength [m]

            res_s   resolution for the star
            res_p   resolution for the planet
            res_lc  resolution for the computed light curve

        """

        Ap = Ap * double(149597870691.) # the semimajor axis [AU] -> [m]
        Rp = Rp * double(6.955e8)       # the planet radius 
        Rs = Rs * double(6.955e8)       # the stellar equator radius [R_sun] -> [m]
        Ms = Ms * double(1.9891e30)     # the stellar mas [M_sun] -> [kg]
        Os = 2.*pi / (3600.*double(Ps)) # the stellar rotation rate [rad/s] 
        gp = G * Ms / (Rs*(1-f))**2     # the surface gravity at the pole [m/s^2]
        f = double(f)
        phi = double(phi)
        Tp = double(Tp)
        c  = np.array(c, double)

        if method == 'Peirce':
            global _XS, _YS, _CS, _XP, _YP, _CP
            if _XS is None:
                _XS, _YS, _CS = dc.generate_Peirce_points(res_s)
            if _XP is None:
                #xp1, yp1, cp1 = dc.generate_Peirce_points(res_p, rotation=0.0)
                #xp2, yp2, cp2 = dc.generate_Peirce_points(res_p, rotation=0.5)

                _XP, _YP, _CP = dc.generate_Peirce_points(res_p)

                #_XP = np.concatenate((xp1.ravel(),xp2.ravel()))
                #_YP = np.concatenate((yp1.ravel(),yp2.ravel()))
                #_CP = 0.5 * np.concatenate((cp1.ravel(),cp2.ravel()))

        if partial_results is None:
            partial_results = [None]

        if partial_results[0] is None:
            f_eff  = 1. - sqrt((1. - f)**2 * cos(phi)**2 + sin(phi)**2)
            if method == 'brute':
                xs, ys = np.meshgrid(np.linspace(-Rs, Rs, res_s), np.linspace(-Rs, Rs, res_s))
                mask_s = sqrt(xs**2 + (ys/(1.-f_eff))**2) < Rs

                Is = Barnes_luminosity(xs[mask_s], ys[mask_s], Ms, Rs, Os, Tp, gp, f, phi, beta, c,  l) 
                Is_total = Is.sum() * (pi * Rs**2 *(1.-f_eff) / mask_s.sum())

            elif method == 'Peirce':                            
                xs = Rs * _XS
                ys = Rs * _YS * (1.-f_eff)
                Is = np.reshape(Barnes_luminosity(xs.ravel(), ys.ravel(), Ms, Rs, Os, Tp, gp, f, phi, beta, c,  l), xs.shape)

                Is_total = (_CS[0,:]*Is.sum(0)).sum() * (Rs**2 *(1.-f_eff))

            partial_results[0] = Is_total
        else:
            Is_total = partial_results[0]

    
        if method == 'brute':
            xp_c, yp_c = np.meshgrid(np.linspace(-Rp, Rp, res_p), np.linspace(-Rp, Rp, res_p))
            mask_p     = sqrt(xp_c**2 + yp_c**2) < Rp
            xp_c       = xp_c[mask_p]
            yp_c       = yp_c[mask_p]

            xp = xp_c
            yp = yp_c

        else:
            xp = Rp * _XP
            yp = Rp * _YP

        if phase is not None:
            t = Ap * sin(phase)
        else:
            t = np.linspace(-Rs*1.1, Rs*1.1, res_lc)

        P = np.array([t*cos(a) - b*Rs*sin(a), -t*sin(a) - b*Rs*cos(a)], np.double)

        dist = np.sqrt((P[:,:]**2).sum(0))
        mask_t = np.abs(dist) <= Rp + Rs

        P = P[:,mask_t]
        dist = dist[mask_t]

        I = np.ones(t.size)
        Il = np.zeros(P.size)

        if method == 'brute':
            npt_p = double(mask_p.sum())
            for i in range(t[mask_t].size):
                Ip    = Barnes_luminosity(xp+P[0,i], yp+P[1,i], Ms, Rs, Os, Tp, gp, f, phi, beta, c, l) 
                area_f = ((Ip==Ip).sum() / npt_p) * pi *Rp**2 / npt_p
                Il[i] = (Is_total - Ip[Ip==Ip].sum() * area_f) / Is_total

        ## Compute the transit light curve using the cubature method by William H. Peirce. We separate
        ## the computation of the ingress and egress parts from the flat part for efficiency.
        ##
        elif method == 'Peirce':
            area = Rp**2

            mask_ingegr = np.logical_and(dist <  Rs+Rp, dist > Rs-Rp)

            ## Ingress and egress
            ##
            Itemp = Il[mask_ingegr]
            for i, Pp in enumerate(P[:,mask_ingegr].transpose()):
                Ip = Barnes_luminosity(xp.ravel()+Pp[0], yp.ravel()+Pp[1], Ms, Rs, Os, Tp, gp, f, phi, beta, c, l)
                Ip[Ip != Ip] = 0.

                s = dist[mask_ingegr][i]

                if s > Rs:
                    theta = 2. * np.arccos((s-Rs)/Rp)
                    a_seg = .5 * Rp**2 * (theta - sin(theta))
                    area_f = a_seg / pi
                else:
                    theta = 2. * np.arccos((np.abs(s-Rs))/Rp)
                    a_seg = .5 * Rp**2 * (theta - sin(theta))
                    area_f = area - a_seg / pi

                Itemp[i] = (Is_total - area_f * (_CP.ravel()*Ip).sum()) / Is_total
            Il[mask_ingegr] = Itemp

            ## Flat part
            ##
            ## FIXME: We shouldn't need to check for NaNs here, must fix the algorithm for obliquity
            Itemp = Il[~mask_ingegr]
            for i, Pp in enumerate(P[:,~mask_ingegr].transpose()):
                Ip = Barnes_luminosity(xp.ravel()+Pp[0], yp.ravel()+Pp[1], Ms, Rs, Os, Tp, gp, f, phi, beta, c, l)
                Itemp[i] = (Is_total - area * (_CP.ravel()[Ip==Ip]*Ip[Ip==Ip]).sum()) / Is_total
            Il[~mask_ingegr] = Itemp

        I[mask_t] = Il

        return I, partial_results
