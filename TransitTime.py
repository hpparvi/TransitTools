from numpy import pi, cos, sin, arcsin, sqrt

def transit_time_sm(P, a_R, i):
    """
    Implements the transit time equation by Seager & Mallén-Ornelas (2003)
    """
    
    return P/pi * arcsin(sqrt(1. - a_R**2 * cos(i)**2) / (a_R*sin(i)))
    
def transit_time_ts(P, a_R, i, e):
    pass
