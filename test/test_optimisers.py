import unittest
from parameterized import parameterized_class
from testsuite.surrogates import GP, RF
from testsuite.optimisers import *
from testsuite.surrogates import GP


class TestBaseOptimiserClass(unittest.TestCase):

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

        limits = np.array([[0], [10]])
        cls.opt = Optimiser(objective_function=test_function, limits=limits,
                            n_initial=10, budget=30, seed=None,
                            log_dir="./log_data")

        cls.opt.initial_evaluations(n_samples=10, x_dims=np.shape(limits)[1], limits=limits)

    def test_already_evaluated(self):
        # generate test data
        x_evaluated = self.opt.x[np.random.randint(len(self.opt.x))]
        x_not_evaluated = np.ones_like(x_evaluated)*11

        # check test data
        assert(x_evaluated in self.opt.x)
        assert(x_not_evaluated not in self.opt.x)

        # test _already_evaluated method
        assert(self.opt._already_evaluated(x_evaluated))
        assert(not self.opt._already_evaluated(x_not_evaluated))

        # condition where x_new much larger
        x_not_evaluated = np.ones_like(x_evaluated)*100001
        assert(self.opt._already_evaluated(x_evaluated))
        assert(not self.opt._already_evaluated(x_not_evaluated))


# @parameterized_class([
#     {"name": "GP_unscaled", "surrogate": GP, "args": [], "kwargs": {"scaled": False}},
#     {"name": "GP_scaled", "surrogate": GP, "args": [], "kwargs": {"scaled": True}},
#     {"name": "GP_unscaled", "surrogate": RF, "args": [], "kwargs": {"extra_trees": False}},
#     {"name": "GP_unscaled", "surrogate": RF, "args": [], "kwargs": {"extra_trees": True}}
# ])
# class TestBayesianOptimiserClass(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#
#         def test_function(x):
#             try:
#                 return np.array([[np.sum([np.sin(xii) for xii in xi]) for xi in x],
#                                  [np.sum([np.cos(xii) for xii in xi]) for xi in x],
#                                  [np.sum([np.cos(xii) ** 2 for xii in xi]) for xi in x]]).T
#             except:
#                 return np.array([np.sum([np.sin(xii) for xii in x]),
#                                  np.sum([np.cos(xii) for xii in x]),
#                                  np.sum([np.cos(xii) ** 2 for xii in x])])
#
#         limits = np.array([[0], [10]])
#         surr = cls.surrogate(*cls.args, **cls.kwargs)
#         cls.opt = BayesianOptimiser(objective_function=test_function, limits=limits, surrogate=surr, n_initial=10,
#                                     seed=None, acquisition_function="saf_mu", cmaes_restarts=0)
#
#     def test_step(self):
#         """
#         test the optimisation step
#         :return:
#         """
#         n_optimsation_steps = self.opt.x.shape[0]
#         self.assertEqual(self.opt.n_evaluations, n_optimsation_steps)
#         self.opt.step()
#         self.assertEqual(self.opt.n_evaluations, n_optimsation_steps+1)
#         self.assertEqual(self.opt.x.shape[0], n_optimsation_steps+1)
#         self.assertEqual(self.opt.y.shape[0], n_optimsation_steps+1)
#
#     def test_optimise(self):
#         n_optimsation_steps = self.opt.x.shape[0]
#         self.assertEqual(self.opt.n_evaluations, n_optimsation_steps)
#         self.opt.optimise(n_steps=2)
#         self.assertEqual(self.opt.n_evaluations, n_optimsation_steps+2)
#         self.assertEqual(self.opt.x.shape[0], n_optimsation_steps+2)
#         self.assertEqual(self.opt.y.shape[0], n_optimsation_steps+2)
#

if __name__ == '__main__':
    unittest.main()