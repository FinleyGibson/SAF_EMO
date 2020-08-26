import unittest
from parameterized import parameterized_class
from testsuite.surrogates import MonoSurrogate, MultiSurrogate, GP, RF
import numpy as np

@parameterized_class([
    {"name": "GP_unscaled", "surrogate": GP, "args": [], "kwargs": {"scaled": False}},
    {"name": "GP_scaled", "surrogate": GP, "args": [], "kwargs": {"scaled": True}},
    {"name": "GP_unscaled", "surrogate": RF, "args": [], "kwargs": {"extra_trees": False}},
    {"name": "GP_unscaled", "surrogate": RF, "args": [], "kwargs": {"extra_trees": True}}
])
class TestMonoSurrogate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        def test_function(x):
            try:
                return np.array([[np.sum([np.sin(xii) for xii in xi]) for xi in x],
                                 [np.sum([np.cos(xii) for xii in xi]) for xi in x],
                                 [np.sum([np.cos(xii) ** 2 for xii in xi]) for xi in x]]).T
            except:
                return np.array([np.sum([np.sin(xii) for xii in x]),
                                 np.sum([np.cos(xii) for xii in x]),
                                 np.sum([np.cos(xii) ** 2 for xii in x])])

        x = np.random.randn(10,5)
        y = test_function(x)
        cls.x = x
        cls.y = y

        cls.surr_0 = cls.surrogate(*cls.args, **cls.kwargs)
        cls.surr_0.update(x, y)

    def test_scaling(self):
        """
        test scaling and unscaling of x and y in MonoSurrogate
        :return:
        """
        if self.surr_0.scaled is True:
            # tests specific to scaled surrogates
            np.testing.assert_array_almost_equal(self.x.mean(axis=0), self.surr_0.mean_x)
            np.testing.assert_array_almost_equal(self.y.mean(axis=0), self.surr_0.mean_y)
            np.testing.assert_array_almost_equal(self.x.std(axis=0), self.surr_0.std_x)
            np.testing.assert_array_almost_equal(self.y.std(axis=0), self.surr_0.std_y)
        else:
            # tests specific to non-scaled surrogates
            pass

        # tests applicable to all surrogates
        np.testing.assert_array_almost_equal(self.x, self.surr_0.descale_x(self.surr_0.x))
        np.testing.assert_array_almost_equal(self.y, self.surr_0.descale_y(self.surr_0.y))

        np.testing.assert_array_almost_equal(self.x, self.surr_0.get_x())
        np.testing.assert_array_almost_equal(self.y, self.surr_0.get_y())

    def test_predict(self):
        """
        test predictions for surrogate.
        :return:
        """
        #predict single point
        x_new = np.ones_like(self.surr_0.x[0])
        prediction = self.surr_0.predict(x_new)
        (y_new_mean, y_new_var) = prediction
        self.assertIsInstance(prediction, tuple)
        self.assertIsInstance(y_new_mean, np.ndarray)
        self.assertIsInstance(y_new_var, np.ndarray)
        self.assertEqual(y_new_mean.shape, self.surr_0.y[0:1].shape)
        self.assertEqual(y_new_var.shape[1], self.y.shape[1])
        self.assertEqual(y_new_var.shape[0], 1)

        #predict single point 2 dim
        x_new = np.ones_like(self.surr_0.x[0:1])
        prediction = self.surr_0.predict(x_new)
        (y_new_mean, y_new_var) = prediction
        self.assertIsInstance(prediction, tuple)
        self.assertIsInstance(y_new_mean, np.ndarray)
        self.assertIsInstance(y_new_var, np.ndarray)
        self.assertEqual(y_new_mean.shape, self.surr_0.y[0:1].shape)
        self.assertEqual(y_new_var.shape[1], self.y.shape[1])
        self.assertEqual(y_new_var.shape[0], 1)

        # predict multiple points
        x_new = np.ones_like(self.surr_0.x[0:5])
        prediction = self.surr_0.predict(x_new)
        (y_new_mean, y_new_var) = prediction
        self.assertIsInstance(prediction, tuple)
        self.assertIsInstance(y_new_mean, np.ndarray)
        self.assertIsInstance(y_new_var, np.ndarray)
        self.assertEqual(y_new_mean.shape, y_new_var.shape)
        self.assertEqual(y_new_mean.shape, self.surr_0.y[0:5].shape)
        self.assertEqual(y_new_var.shape, self.surr_0.y[0:5].shape)
        self.assertEqual(y_new_mean.shape[0], 5)
        self.assertEqual(y_new_mean.shape[1], 3)


@parameterized_class([
    {"name": "Multi_GP_unscaled", "surrogate": GP, "args": [],
     "kwargs": {}},
    {"name": "Multi_RF_unscaled", "surrogate": RF, "args": [],
     "kwargs": {"extra_trees": True}}
])
class TestMultiSurrogate(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        def test_function(x):
            try:
                return np.array([[np.sum([np.sin(xii) for xii in xi]) for xi in x],
                                 [np.sum([np.cos(xii) for xii in xi]) for xi in x],
                                 [np.sum([np.cos(xii) ** 2 for xii in xi]) for xi in x]]).T
            except:
                return np.array([np.sum([np.sin(xii) for xii in x]),
                                 np.sum([np.cos(xii) for xii in x]),
                                 np.sum([np.cos(xii) ** 2 for xii in x])])

        x = np.random.randn(10,5)
        y = test_function(x)
        cls.x = x
        cls.y = y

        cls.surr_0 = MultiSurrogate(cls.surrogate, *cls.args, **cls.kwargs)
        cls.surr_0.update(x, y)

    def test_predict(self):
        """
        test predictions for surrogate.
        :return:
        """
        #predict single point
        x_new = np.ones_like(self.surr_0.x[0])
        prediction = self.surr_0.predict(x_new)
        (y_new_mean, y_new_var) = prediction
        self.assertIsInstance(prediction, tuple)
        self.assertIsInstance(y_new_mean, np.ndarray)
        self.assertIsInstance(y_new_var, np.ndarray)
        self.assertEqual(y_new_mean.shape, self.surr_0.y[0:1].shape)
        self.assertEqual(y_new_var.shape[1], self.y.shape[1])
        self.assertEqual(y_new_var.shape[0], 1)

        #predict single point 2 dim
        x_new = np.ones_like(self.surr_0.x[0:1])
        prediction = self.surr_0.predict(x_new)
        (y_new_mean, y_new_var) = prediction
        self.assertIsInstance(prediction, tuple)
        self.assertIsInstance(y_new_mean, np.ndarray)
        self.assertIsInstance(y_new_var, np.ndarray)
        self.assertEqual(y_new_mean.shape, self.surr_0.y[0:1].shape)
        self.assertEqual(y_new_var.shape[1], self.y.shape[1])
        self.assertEqual(y_new_var.shape[0], 1)

        # predict multiple points
        x_new = np.ones_like(self.surr_0.x[0:5])
        prediction = self.surr_0.predict(x_new)
        (y_new_mean, y_new_var) = prediction
        self.assertIsInstance(prediction, tuple)
        self.assertIsInstance(y_new_mean, np.ndarray)
        self.assertIsInstance(y_new_var, np.ndarray)
        self.assertEqual(y_new_mean.shape, y_new_var.shape)
        self.assertEqual(y_new_mean.shape, self.surr_0.y[0:5].shape)
        self.assertEqual(y_new_var.shape, self.surr_0.y[0:5].shape)
        self.assertEqual(y_new_mean.shape[0], 5)
        self.assertEqual(y_new_mean.shape[1], 3)
