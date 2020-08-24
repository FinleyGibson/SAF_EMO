import numpy as np
import unittest
import unittest.mock as mock
from testsuite.scalarisers import ClassMethods



class TestBaseOptimiserClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.y = np.array([[0.1, 0.95], [0.2, 0.5], [0.35, 0.3], [0.44, 0.25],
                          [0.5, 0.15], [0.78, 0.12], [1.1, 0.05]])
        cls.ref_vector = np.array([1.5, 1.5])

        cls.saf = ClassMethods.saf

    def test_saf(self):
        # synthetic points which increase hypervolume
        y_put0 = np.ones_like(self.y[0:1])/10   # increases slightly
        y_put1 = np.ones_like(self.y[0:1])/100  # increases more
        hv0 = self.saf(y_put=y_put0)
        hv1 = self.saf(y_put=y_put1)
        hv2 = self.saf(y_put=y_put0, invert=True)
        hv3 = self.saf(y_put=y_put1, invert=True)

        self.assertLess(hv0, hv1)
        self.assertGreater(hv2, hv3)

if __name__ == "__main__":
    unittest.main()