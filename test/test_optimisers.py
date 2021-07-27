import unittest
from unittest.mock import MagicMock, Mock

from parameterized import parameterized_class
from testsuite.optimisers import *
from testsuite.surrogates import GP, RF


def test_function(x):
    try:
        return np.array([[np.sum([np.sin(xii) for xii in xi]) for xi in x],
                         [np.sum([np.cos(xii) for xii in xi]) for xi in x],
                         [np.sum([np.cos(xii) ** 2 for xii in xi]) for xi in
                          x]]).T
    except TypeError:
        return np.array([np.sum([np.sin(xii) for xii in x]),
                         np.sum([np.cos(xii) for xii in x]),
                         np.sum([np.cos(xii) ** 2 for xii in x])])


class TestBaseOptimiserClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        limits = np.array([[0, 0], [10, 10]])
        cls.opt = Optimiser(objective_function=test_function, limits=limits,
                            n_initial=10, budget=30, seed=None,
                            log_dir="./log_data")

        cls.opt.initial_evaluations(n_samples=10, x_dims=np.shape(limits)[1],
                                    limits=limits)

        # mock missing function calls
        cls.opt.get_next_x = MagicMock(
            side_effect=lambda : np.random.randn(1, cls.opt.n_dims)
        )
        # mock logging as mocked functions cannot be pickled
        cls.opt.write_log = MagicMock()

    def tearDown(self):
        # check x and y are accurate
        self.assertEqual(self.opt.x.shape[1], self.opt.n_dims)
        self.assertEqual(self.opt.y.shape[1], self.opt.n_objectives)
        # check Pareto split
        np.testing.assert_array_almost_equal(np.unique(self.opt.y),
                                             np.unique(np.vstack((self.opt.p,
                                                                  self.opt.d)))
                                             )

    def test_init(self):
        self.assertEqual(self.opt.x.shape[0],self.opt.n_initial)
        self.assertEqual(self.opt.y.shape[0], self.opt.n_initial)

    def test_step(self):
        # check step adds one evaluation
        n_pri = self.opt.n_evaluations
        self.opt.step()
        self.assertEqual(n_pri+1, self.opt.n_evaluations)

    def test_get_loggables(self):
        loggables = self.opt._get_loggables()
        for k, v in loggables.items():
            try:
                # handle basics
                self.assertEqual(v, getattr(self.opt, k))
            except ValueError:
                # handle np arrays
                np.testing.assert_array_almost_equal(v, getattr(self.opt, k))
            except AssertionError:
                # handle functions pickled as name only
                self.assertEqual(v, getattr(self.opt, k).__name__)

    def test_optimise(self):
        # check optimise(n) adds n evaluations
        n_pri = self.opt.n_evaluations
        n_step = np.random.randint(0, 10)
        self.opt.optimise(n_step)
        self.assertEqual(n_pri+n_step, self.opt.n_evaluations)

    def test_already_evaluated(self):
        # generate test data
        x_evaluated = self.opt.x[np.random.randint(len(self.opt.x))]
        x_not_evaluated = np.ones_like(x_evaluated)*11

        # check test data
        self.assertIn(x_evaluated, self.opt.x)
        self.assertNotIn(x_not_evaluated, self.opt.x)

        # test _already_evaluated method
        self.assertTrue(self.opt._already_evaluated(x_evaluated))
        self.assertFalse(self.opt._already_evaluated(x_not_evaluated))

@parameterized_class([
    {"name": "GP_unscaled", "surrogate": GP, "args": [],
     "kwargs": {"scaled": False}},
    {"name": "GP_scaled", "surrogate": GP, "args": [],
     "kwargs": {"scaled": True}},
    {"name": "GP_unscaled", "surrogate": RF, "args": [],
     "kwargs": {"extra_trees": False}},
    {"name": "GP_unscaled", "surrogate": RF, "args": [],
     "kwargs": {"extra_trees": True}}
])
class TestBayesianOptimiserClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        limits = np.array([[0], [10]])
        surr = cls.surrogate(*cls.args, **cls.kwargs)
        cls.opt = BayesianOptimiser(
            objective_function=test_function, limits=limits, surrogate=surr,
            n_initial=10, seed=None, cmaes_restarts=0)

        # mock missing function calls
        cls.opt.get_next_x = MagicMock(
            side_effect=lambda: np.random.randn(1, cls.opt.n_dims)
        )
        # mock logging as mocked functions cannot be pickled
        cls.opt.write_log = MagicMock()

    def test_step(self):
        """
        test the optimisation step
        :return:
        """
        n_optimsation_steps = self.opt.x.shape[0]
        self.assertEqual(self.opt.n_evaluations, n_optimsation_steps)
        self.opt.step()
        self.assertEqual(self.opt.n_evaluations, n_optimsation_steps+1)
        self.assertEqual(self.opt.x.shape[0], n_optimsation_steps+1)
        self.assertEqual(self.opt.y.shape[0], n_optimsation_steps+1)

    def test_optimise(self):
        n_optimsation_steps = self.opt.x.shape[0]
        self.assertEqual(self.opt.n_evaluations, n_optimsation_steps)
        self.opt.optimise(n_steps=2)
        self.assertEqual(self.opt.n_evaluations, n_optimsation_steps+2)
        self.assertEqual(self.opt.x.shape[0], n_optimsation_steps+2)
        self.assertEqual(self.opt.y.shape[0], n_optimsation_steps+2)


if __name__ == '__main__':
    unittest.main()