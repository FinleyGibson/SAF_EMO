import numpy as np
from optimisers import Saf
from testsuite.utilities import Pareto_split, optional_inversion

class DirectedSaf(Saf):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    @optional_inversion
    def osaf(y: np.ndarray, p: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @optional_inversion
    def osaf(y: np.ndarray, p: np.ndarray) -> np.ndarray:
        """
        Calculates summary attainment front distances.
        Calculates the distance of n, m-dimensional points X from the
        summary attainment front defined by points P

        :param np.array p: points in the pareto front, shape[?,m]
        :param np.array y: points for which the distance to the summary
        attainment front is to be calculated, shape[n,m]

        :return np.array: numpy array of saf distances between points in
        X and saf defined by P, shape[X.shape]
        """

        D = np.zeros((y.shape[0], p.shape[0]))

        for i, p in enumerate(p):
            D[:, i] = np.min(p - y, axis=1).reshape(-1)
        Dq = np.max(D, axis=1)
        return Dq

    @optional_inversion
    def _scalarise_y(self, y_put, std_put):
        if y_put.ndim<2:
            y_put = y_put.reshape(1,-1)
            std_put = std_put.reshape(1,-1)

        assert y_put.shape[0]==1
        assert std_put.shape[0]==1

        if self.ei:
            return float(self.osaf_ei(y_put, std_put, n_samples=3000*self.n_objectives,
                                     invert=False))
        else:
            return float(self.osaf(y_put, self.apply_weighting(self.p),
                                  invert=False))

#     def optimistic_saf_dist(P, X):
#         """Find the distance that identifies the optimistic summary attainment surface"""
#         assert P.shape[1] == X.shape[1]
#         D = zeros((X.shape[0], P.shape[0]))
#         for i, p in enumerate(P):
#             D[:, i] = np.min(p - X, axis=1)
#         return np.max(D, axis=1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import wfg
    from testsuite.surrogates import GP, MultiSurrogate

    # setup function
    n_obj = 2  # Number of objectives
    kfactor = 4
    lfactor = 4
    k = kfactor * (n_obj - 1)  # position related params
    l = lfactor * 2  # distance related params
    n_dim = k + l
    limits = np.zeros((2, n_dim))
    limits[1] = np.array(range(1, n_dim + 1)) * 2

    func = wfg.WFG6
    gp_surr_multi = MultiSurrogate(GP, scaled=True)
    # gp_surr_mono = GP(scaled=True)

    def test_function(x):
        if x.ndim < 2:
            x = x.reshape(1, -1)
        return np.array([func(xi, k, n_obj) for xi in x])

    opt = DirectedSaf(objective_function=test_function, ei=False,  limits=limits, surrogate=gp_surr_multi, n_initial=10, budget=20, seed=None, log_models=True, log_interval=1)
    opt.optimise(10)
