import sys
import unittest
from numpy.testing import assert_almost_equal as aeq

sys.path.append('..')
from DifferentialEvolution import *

class TestDiffEvol(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_minimization(self):
        de = DiffEvol(lambda P:np.sum((P-1)**2), [[-2, 2], [-2, 2], [-2, 2]], 40, 100)
        aeq(de()[1], np.ones(3), 7, 'DiffEvol fitter fails to find the minimum.')
