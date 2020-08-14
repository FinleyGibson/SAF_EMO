import unittest
from testsuite.optimisers import *
from testsuite.surrogates import Surrogate


class TestOptimiserClass(unittest.TestCase):

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
        cls.opt = Optimiser(objective_function=test_function, limits=limits, n_initial=10, budget=30, seed=None,
                             ref_vector=None, log_dir="./log_data")

        cls.opt.initial_evaluations(n_samples=10, x_dims=np.shape(limits)[1], limits=limits)

    def test_already_evaluated(self):
        # generate test data
        x_evaluated = self.opt.x[np.random.randint(len(self.opt.x))]
        x_notevaluated = np.ones_like(x_evaluated)*11

        # check test data
        assert(x_evaluated in self.opt.x)
        assert(x_notevaluated not in self.opt.x)

        # test _already_evaluated method
        assert(self.opt._already_evaluated(x_evaluated))
        assert(not self.opt._already_evaluated(x_notevaluated))
        # condition where x_new much larger
        x_notevaluated = np.ones_like(x_evaluated)*100001
        assert(self.opt._already_evaluated(x_evaluated))
        assert(not self.opt._already_evaluated(x_notevaluated))

    def test_compute_hypervolume(self):
        a = self.opt._compute_hypervolume()
        print(a)
        print(self.opt.x)


class TestBayesianOptimiserClass(unittest.TestCase):

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
        surr = Surrogate(x=, y, "GP", multi_surrogate=True)
        cls.opt = BayesianOptimiser(objective_function=test_function, limits=limits, surrogate=surr, n_initial=10,
                                    seed=None, acquisition_function="saf_mu", cmaes_restarts=0)

    def test_one(self):
        print(self.opt.x)
        print(self.opt.surrogate.x)


if __name__ == '__main__':
    unittest.main()