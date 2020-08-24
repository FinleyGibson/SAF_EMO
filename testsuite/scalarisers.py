import numpy as np
import _csupport as cs
from testsuite.utilities import Pareto_split


def optional_inversion(f):
    """decorator to invert the value of a function, and turn maximisation
    problem to minimization problem. Invoke by passing a keyword argument
    invert=True to the decorated function"""
    def wrapper(*args, **kwargs):
        try:
            if kwargs["invert"] is True:
                del(kwargs["invert"])
                return -f(*args, **kwargs)
            else:
                del(kwargs["invert"])
                return f(*args, **kwargs)
        except KeyError:
            return f(*args, **kwargs)
    return wrapper


@optional_inversion
def saf(y: np.ndarray, p: np.ndarray) -> np.ndarray:
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
    # check dimensionality of P is the same as that for X
    assert p.shape[1] == y.shape[1]

    D = np.zeros((y.shape[0], p.shape[0]))

    for i, p in enumerate(p):
        D[:, i] = np.max(p - y, axis=1).reshape(-1)
    Dq = np.min(D, axis=1)
    return Dq

# class ClassMethods:
#     """methods to be used by Optimiser classes, referencing self"""
#
#     def saf(self, y_put) -> np.ndarray:
#         """
#         Calculates summary attainment front distances.
#         Calculates the distance of n, m-dimensional points X from the
#         summary attainment front defined by points P
#
#         :param np.array y_put: putative solution in objective space
#         shape[n_queries, n_objectives]
#
#         :return np.array: numpy array of saf distances between points in
#         X and saf defined by P, shape[X.shape]
#         """
#         # check dimensionality of P is the same as that for X
#         p = Pareto_split(self.y)[0]
#         return _saf(y_put, p)
#
#     @optional_inversion
#     def sms_ego(self, y_put, y_put_std):
#         # split y into dominated and non-dominated points
#         p_inds, d_inds = Pareto_split(self.y, return_indices=True)
#         p, d = self.y[p_inds], self.y[d_inds]
#
#         current_hv = self._compute_hypervolume(p, self.ref_vector)
#
#         n_pfr = len(p)
#         c = 1 - (1 / 2 ** self.y.shape[1])
#         b_count = 10
#         epsilon = (np.max(self.y, axis=0) - np.min(y, axis=0)) / (
#                     n_pfr + (c * b_count))
#
#         # lower confidence bounds
#         lcb = y_put - (self.gain * np.multiply(self.obj_sense, y_put_std))
#         p_inds, d_inds = Pareto_split(self.y, return_indices=True)
#
#         # calculate penalty
#         c = 1 - (1 / 2 ** self.y.shape[1])
#         b_count = self.budget - self.y.shape[0] - 1  # remaining budget
#
#         epsilon = (np.max(self.y, axis=0) - np.min(self.y, axis=0)) / (
#                     n_pfr + (c * b_count))
#
#         yt = y_put - (epsilon * self.obj_sense)
#         l = [-1 + np.prod(1 + y_put - self.y[i]) if
#              cs.compare_solutions(self.y[i], yt, self.obj_sense) == 0
#              else 0 for i in range(self.y.shape[0])]
#         penalty = (max([0, max(l)]))
#
#         if penalty > 0:
#             return np.array([-penalty])
#
#         # new front
#         new_hv = self._compute_hypervolume(np.vstack((self.y, lcb)))
#         return np.array([new_hv - current_hv])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from testsuite.utilities import Pareto_split
    from evoalgos.performance import FonsecaHyperVolume
    from scipy.stats import norm


    def image_2obj(acq, *args, **kwargs):
        M, N = 100, 100
        x = np.linspace(0, 5, M)
        y = np.linspace(0, 5, N)
        xx, yy = np.meshgrid(x, y)
        xy = np.vstack((xx.flat, yy.flat)).T

        # observed points
        np.random.seed(6)
        ytr = np.random.uniform(1, 5, size=(15, 2))
        p, d = Pareto_split(ytr)

        # acqusition function
        zz = acq(xy, p, *args, **kwargs).reshape(N,M)

        # generate figure
        fig = plt.figure(figsize=[5., 4.])
        ax = fig.gca()
        ax.set_xlim([0, 5])
        ax.set_ylim([0, 5])
        pcol = ax.pcolor(x, y, zz)
        ax.contour(x, y, zz, np.linspace(zz.min(), zz.max(), 10),
                   colors="white", linewidths= 0.6)
        ax.contour(x, y, zz, [0.], colors="C3")
        plt.scatter(p[:, 0], p[:, 1], c="C3")
        plt.scatter(d[:, 0], d[:, 1], c="C0")
        plt.colorbar(pcol)
        return fig

    saf_fig = image_2obj(saf, invert=False)
    saf_fig_i = image_2obj(saf, invert=True)
    plt.show()

