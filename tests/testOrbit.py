import sys
import unittest
import numpy as np

sys.path.append('..')
from Orbit import *

class TestCircularOrbit(unittest.TestCase):
    def setUp(self):
        p = np.array([0.5, 2, 0.1, 10, 0.5*np.pi])
        self.orbit = CircularOrbit(p)
    
    def test_pdp(self):
        "Projected distance (phase)"
        aeq(self.orbit.projected_distance_p(0.0), 0.00)
        aeq(self.orbit.projected_distance_p(0.5*np.pi), 10.0)
        
    def test_pdt(self):
        "Projected distance (time)"
        aeq(self.orbit.projected_distance_t(0.5), 0.00)
        aeq(self.orbit.projected_distance_t(1.0), 10.0)
        
    def test_transit_depth(self):
        "Transit depth"
        aeq(0.01, self.orbit.transit_depth())
        
    def test_impact_parameter(self):
        "Impact parameter"
        aeq(0.00, self.orbit.impact_parameter())
        
    def test_simple_transit_duration(self):
        "Simple transit duration"
        self.fail()
        
    def test_transit_duration(self):
        "Transit duration"
        self.fail()
        
        
class TestEccentricOrbit(unittest.TestCase):
    def setUp(self): 
        pc  = np.array([0.5, 2, 0.1, 0.0, 0.0, 10, 0.5*np.pi])
        ple = np.array([0.5, 2, 0.1, 0.1, 0.1, 10, 0.5*np.pi])
        
        self.orbit_c  = EccentricOrbit(*pc)
        self.orbit_le = EccentricOrbit(*ple)
 
    def test_pdp_c(self):
        "Projected distance (phase) - no eccentricity"
        aeq(self.orbit_c.projected_distance_p(0.0), 0.00)
        aeq(self.orbit_c.projected_distance_p(0.5*np.pi), 10.0)
        
    def test_pdp_le(self):
        "Projected distance (phase) - low eccentricity"
        aeq(self.orbit.projected_distance_p(0.0), 0.00)
        aeq(self.orbit.projected_distance_p(0.5*np.pi), 10.0)
        
    def test_pdt_c(self):
        "Projected distance (time) - no eccentricity"
        aeq(self.orbit_c.projected_distance_t(0.5), 0.00)
        aeq(self.orbit_c.projected_distance_t(1.0), 10.0)
        
    def test_pdt_le(self):
        "Projected distance (time) - low eccentricity"
        aeq(self.orbit.projected_distance_t(0.5), 0.00)
        aeq(self.orbit.projected_distance_t(1.0), 10.0)
        
    def test_transit_depth(self):
        "Transit depth"
        aeq(0.01, self.orbit.transit_depth())
        
    def test_impact_parameter(self):
        "Impact parameter"
        aeq(0.00, self.orbit.impact_parameter())
        

