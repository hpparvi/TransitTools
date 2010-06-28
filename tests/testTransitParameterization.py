import sys
import unittest
from numpy.testing import assert_almost_equal as aeq

sys.path.append('..')
from TransitParameterization import *

p_orbit_1 = np.array([1.0, 2.5, 0.10, 10., HALF_PI])
p_orbit_2 = np.array([1.0, 2.5, 0.05, 10., HALF_PI-0.08])
p_orbit_l = np.array([1.0, 2.5, 0.10, 10., HALF_PI])
p_orbit_h = np.array([1.0, 2.5, 0.10, 10., HALF_PI])

class TestOrbitTransitParameterization(unittest.TestCase):
    def setUp(self):
        self.p_o1 = OrbitTransitParameterization(p_orbit_1, p_orbit_l, p_orbit_h)
        self.p_o2 = OrbitTransitParameterization(p_orbit_2, p_orbit_l, p_orbit_h)
        
    def test_mapping_to_orbit(self):
        "Test mapping to the orbit parameterization."
        aeq(self.p_o1.mapped_to_orbit(), self.p_o1.p)
        aeq(self.p_o2.mapped_to_orbit(), self.p_o2.p)


class TestPhysicalTransitParameterization(unittest.TestCase):
    def setUp(self):
        self.p_o1 = OrbitTransitParameterization(p_orbit_1, p_orbit_l, p_orbit_h)
        self.p_o2 = OrbitTransitParameterization(p_orbit_2, p_orbit_l, p_orbit_h)
        self.p_p1 = PhysicalTransitParameterization(self.p_o1)
        self.p_p2 = PhysicalTransitParameterization(self.p_o2)
        
    def test_mapping_to_orbit(self):
        "Test mapping to the orbit parameterization."
        aeq(self.p_p1.mapped_to_orbit(), self.p_o1.p)
        aeq(self.p_p2.mapped_to_orbit(), self.p_o2.p)


class TestKippingTransitParameterization(unittest.TestCase):
    def setUp(self):
        self.p_o1 = OrbitTransitParameterization(p_orbit_1, p_orbit_l, p_orbit_h)
        self.p_o2 = OrbitTransitParameterization(p_orbit_2, p_orbit_l, p_orbit_h)
        self.p_k1 = KippingTransitParameterization(self.p_o1)
        self.p_k2 = KippingTransitParameterization(self.p_o2)

    def test_mapping_to_orbit(self):
        "Test mapping to the orbit parameterization."
        aeq(self.p_k1.mapped_to_orbit(), self.p_o1.p)
        aeq(self.p_k2.mapped_to_orbit(), self.p_o2.p)


class TestParameterizationMappings(unittest.TestCase):
    def setUp(self):
        self.p_o1 = OrbitTransitParameterization(p_orbit_1, p_orbit_l, p_orbit_h)
        self.p_p1 = PhysicalTransitParameterization(self.p_o1)
        self.p_k1 = KippingTransitParameterization(self.p_o1)

        self.p_o2 = OrbitTransitParameterization(p_orbit_2, p_orbit_l, p_orbit_h)
        self.p_p2 = PhysicalTransitParameterization(self.p_o2)
        self.p_k2 = KippingTransitParameterization(self.p_o2)
        
    def assert_par_equal(self, p1, p2):
        aeq(p1.p, p2.p)
        aeq(p1.p_low, p2.p_low)
        aeq(p1.p_high, p2.p_high)
        
    def test_kipping_to_physical(self):
        "Kipping -> physical."
        self.assert_par_equal(self.p_k1, KippingTransitParameterization(PhysicalTransitParameterization(self.p_k1)))
        self.assert_par_equal(self.p_k2, KippingTransitParameterization(PhysicalTransitParameterization(self.p_k2)))
        
    def test_physical_to_kipping(self):
        "Physical -> Kipping."
        self.assert_par_equal(self.p_p1, PhysicalTransitParameterization(KippingTransitParameterization(self.p_p1)))
        self.assert_par_equal(self.p_p2, PhysicalTransitParameterization(KippingTransitParameterization(self.p_p2)))
