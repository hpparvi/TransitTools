import sys
import unittest

from math import acos, asin, sqrt, pi
from numpy.testing import assert_almost_equal as aeq

sys.path.append('..')
from transitparameterization import *

k  = 1.0
k2 = k**2
tc = 3.5
p  = 2.4
a  = 13.0
b  = 0.56
b2 = b**2
i  = acos(b/a)
tw = (2*pi)/p/asin(sqrt(1.-b2)/(a*sin(i)))

p_orbit = np.array([k,  tc, p,  a,  i])
p_physi = np.array([k,  tc, p,  a,  b])
p_kippi = np.array([k2, tc, p, tw, b2])


class TestOrbitTransitParameterization(unittest.TestCase):
    def setUp(self):
        self.p_o = TransitParameterization('orbit', p_orbit)
        self.p_p = TransitParameterization('physical', p_physi)
        self.p_k = TransitParameterization('kipping', p_kippi)

    def test_generate_empty(self):
        "Generate empty orbit parameterization"
        a = TransitParameterization('orbit')

    def test_mapping_to_orbit(self):
        "Map the orbit parameterization to itself"
        p_o = TransitParameterization('orbit', p_orbit)
        aeq(p_o.map_to_orbit().pv, p_o.pv)

    def test_mapping_to_kipping(self):
        "Test mapping the physical parameterization to the Kipping parameterization."
        map = generate_mapping('orbit', 'kipping')
        aeq(map(self.p_o).pv, self.p_k.pv)


class TestPhysicalTransitParameterization(unittest.TestCase):
    def setUp(self):
        self.p_o = TransitParameterization('orbit', p_orbit)
        self.p_p = TransitParameterization('physical', p_physi)
        self.p_k = TransitParameterization('kipping', p_kippi)

    def test_generate_empty(self):
        "Generate empty physical parameterization"
        a = TransitParameterization('physical')

    def test_mapping_to_orbit(self):
        "Test mapping the physical parameterization to the orbit parameterization."
        aeq(self.p_p.map_to_orbit().pv, self.p_o.pv)

    def test_mapping_to_kipping(self):
        "Test mapping the physical parameterization to the Kipping parameterization."
        map = generate_mapping('physical', 'kipping')
        aeq(map(self.p_p).pv, self.p_k.pv)


class TestKippingTransitParameterization(unittest.TestCase):
    def setUp(self):
        self.p_o = TransitParameterization('orbit', p_orbit)
        self.p_p = TransitParameterization('physical', p_physi)
        self.p_k = TransitParameterization('kipping', p_kippi)

    def test_generate_empty(self):
        "Generate empty Kipping parameterization"
        a = TransitParameterization('kipping')

    def test_mapping_to_orbit(self):
        "Test mapping to the orbit parameterization."
        aeq(self.p_k.map_to_orbit().pv, self.p_o.pv)

    def test_mapping_to_physical(self):
        "Test mapping to the physical parameterization."
        map = generate_mapping('kipping', 'physical')
        aeq(map(self.p_k).pv, self.p_p.pv)
