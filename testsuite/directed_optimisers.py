import numpy as np
from testsuite.optimisers import Saf, BayesianOptimiser
from testsuite.utilities import Pareto_split, optional_inversion, sigmoid
from scipy.linalg import svd
from itertools import combinations
import time


class DirectedSaf(Saf):

    def __init__(self, *args, targets, w, **kwargs):
        self.targets = np.asarray(targets)
        self.targets = self.targets.reshape(1, -1) if self.targets.ndim ==1 \
            else self.targets
        self.w = w
        super().__init__(*args, **kwargs)
        self.target_history = {self.n_initial: self.targets}

    def _generate_filename(self, **kwargs):
        return super()._generate_filename(target=np.round(self.targets,2),
                                          w=np.round(self.w,2), **kwargs)

    @staticmethod
    def optimistic_saf(T, X):
        """
        Calculates optimmistic summary attainment front distances
        Calculates the osaf distance between the points in X and the targets defined by T

        :param T [np.array]: targetd points, shape[n,m]
        :param X [np.array]: points for which the distance to the summary attainment front is to be calculated, shape[]
        :param beta [float]: if not None, the saf distance is passed through sigmoid function with beta=squashed
        :param normalized [Bool]: if not None, the saf distance for points in X is normalized to a range from 0-1

        :return [np.array]: numpy array of saf distances between points in X and saf defined by T, shape[X.shape]
        """
        if T is None:
            Dq = np.ones(len(X))
        else:
            assert T.shape[1] == X.shape[1], "shape missmatch, {} and {}".format(T.shape, X.shape)
            D = np.zeros((X.shape[0], T.shape[0]))
            for i, p in enumerate(T):
                D[:, i] = np.min(p - X, axis=1)

            Dq = np.max(D, axis=1)
        return Dq

    @optional_inversion
    def osaf(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
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
        D_osaf = self.optimistic_saf(self.apply_weighting(self.targets), y)
        D_saf = self.saf(y, p)

        beta = 1.
        Dq = self.w*sigmoid(D_saf, beta) + (1-self.w)*sigmoid(D_osaf, beta)
        return Dq

    @optional_inversion
    def _scalarise_y(self, y_put, std_put):
        if y_put.ndim<2:
            y_put = y_put.reshape(1,-1)
            std_put = std_put.reshape(1,-1)

        assert y_put.shape[0] == 1
        assert std_put.shape[0] == 1

        if self.ei:
            return float(self.osaf_ei(y_put, std_put, n_samples=3000*self.n_objectives,
                                     invert=False))
        else:
            return float(self.osaf(y_put, self.apply_weighting(self.p),
                                  invert=False))

    def update_targets(self, new_targets):
        self.targets = np.asarray(new_targets)
        self.target_history[self.n_evaluations] = new_targets

    def _get_loggables(self, **kwargs):
        log_data = {'targets': self.targets,
                    'target_history': self.target_history
                    }
        return super()._get_loggables(**log_data, **kwargs)


class DmVector(DirectedSaf):

    def __init__(self, *args, w, dmv, **kwargs):
        self.dmv = dmv/np.linalg.norm(dmv)
        super().__init__(*args, targets=None, w=w, **kwargs)
        self.dm_times = []
        self.update_targets()

    def step(self):
        super().step()

        # update targets only if the latest addition is non-dominated.
        if np.all(self.y[-1] == self.p[-1]):
            # # only consider combinations of point which include the latest
            # # addition and the current set of targets
            # cone_combos = [np.vstack((self.p[-1], c)) for c in
            #                combinations(self.p[:-1], self.n_objectives-1)]
            #
            # # only include combinations which reduce the target-vector ssd
            # t_ssd = self._ssd_from_vector(self.targets, self.dmv)
            # cone_ssd = np.array([self._ssd_from_vector(c, self.dmv) for c in
            #                      cone_combos])
            #
            # # include current targets
            # cone_combos = np.vstack((cone_combos[cone_ssd<t_ssd], self.targets))
            #
            # # update targets to reflect intersected combinations with smallest
            # # ssd_to_vector
            self.update_targets()

    def update_targets(self, cone_combos=None):
        tic = time.time()
        existing_targets = self.targets

        # TODO: update to work with selected combinations only.
        # # if no points are passed in, use all cone_combos of non-dominated
        # if cone_combos is None:
        #     # get all M point cone_combos of points
        #     cone_combos = np.array(list(combinations(self.p, self.n_objectives)))
        #
        # assert cone_combos.ndim == 3
        # assert cone_combos.shape[-1] == self.n_objectives

        # # find which combos intersect the dmv
        # intersected_combos = cone_combos[self._in_cones(cone_combos, self.dmv)]
        #
        # # find ssd to the dm vector of considered point combinations
        # intersected_ssds = self._ssd_from_vector(intersected_combos, self.dmv)
        #
        # # select new target with minimum vector ssd
        # new_targets = intersected_combos[np.argmin(intersected_ssds)]

        cones = self._in_cones(self.p, self.dmv)

        if len(cones) > 1 :
            S = np.asarray([self._ssd_from_vector(self.dmv, self.p[list(cone)])
                            for cone in cones])
            k = np.argmin(S)

            new_targets = cones[k]
            new_targets = np.vstack([self.p[t] for t in new_targets])
        else:
            # if no enclosing cones are found
            new_targets = self.p[np.argmin([self._ssd_from_vector(self.dmv, pi.reshape(1,-1)) for pi in self.p])].reshape(1, -1)

        # update target_history with new targets if they change
        if not np.all(existing_targets == new_targets):
            super().update_targets(new_targets)
        self.dm_times.append(time.time()-tic)

    @staticmethod
    def _same_side(V, a, b):
        """Test whether points are on the same side of a plane.

        The plane is defined by the points forming the _rows_ of V and the origin.

        Parameters
        ----------
        V : ndarray of shape (d-1, d)
            d-1 points that together with the origin define a plane
            in d dimensions

        a : ndarray of shape (m,d)
        b : ndarray of shape (m,d)
            a[i] and b[i] are compared, for i = 1,...,m
            a and b must be 2d arrays


        Returns
        -------
        same : Bool ndarray of shape (m,)
            same[i] is True iff a[i] and b[i] are on the same side of the plane.
        """
        assert a.ndim == 2 and b.ndim == 2
        assert a.shape == b.shape
        m, d = a.shape
        assert V.shape == (d - 1, d)

        # Get the normal to the plane by via the SVD because the plane
        # passes through the origin, so its distance to the origin is zero
        # The normal to the plane is the null space of the matrix of the points
        # defining the plane, including the orgin.

        A = np.vstack((V, np.zeros(d)))
        U, s, Vt = np.linalg.svd(A.T)
        assert s[-1] < 1e-10
        n = np.squeeze(U[:, -1])

        # a and b on same side of V if dot product with normal has same sign
        same = np.sign(a @ n) == np.sign(b @ n)
        return same

    @classmethod
    def _in_cones(cls, P, x):
        """
        Determine which cones formed from the origin and points in P
        enclose the vector x

        Parameters
        ----------

        P : ndarray, shape (n_combinations, m, d)
            Array of m d-dimensional points.
            Cones are formed from combinations of d of these and the origin.

        x : ndarray, shape (d,)
            The ray to be enclosed.

        Returns
        -------

        cones : list of d-dimensional tuples.
            Each tuple is the indices of the rows of P which,
            with the origin, define the cone enclosing x.
            cones is empty if no combination of the points encloses x.
        """
        if x.ndim > 1:
            x = x.reshape(-1)

        m, d = P.shape
        assert x.ndim == 1 and x.shape[0] == d

        X = np.tile(x, (m - (d - 1), 1))  # Replicate x for same_side

        # For all possible d-dimensional planes defined by P and the origin,
        # find whether x is on the same side as each of the remaining points in
        # P. Store these in a dictionary whose key is a d-tuple of the point
        # (first element)and the indices of the points defining the plane.
        side = {}
        allpoints = set(range(m))
        for plane in combinations(allpoints, d - 1):
            Iplane = list(plane)
            pts = allpoints.difference(set(plane))
            Ipts = list(pts)
            S = cls._same_side(P[Iplane], P[Ipts], X)

            for p, same in zip(Ipts, S):
                if same:
                    side[(p, *Iplane)] = 1

        # For all possible combinations of d points that, together with the
        # origin define a cone, find the ones that contain x.
        cones = []
        for simplex in combinations(allpoints, d):
            simplex = set(simplex)
            for i in simplex:
                plane = sorted(simplex.difference((i,)))
                try:
                    _ = side[(i, *plane)]
                except:
                    break
            else:
                # x was on the same side of all the planes
                cones.append(tuple(simplex))
        return cones

    @staticmethod
    def _ssd_from_vector(x, V):
        """
        Find the sum of squared distances from the
        ray defined by x to the points in V

        Parameters
        ----------
        x : ndarray, shape (d,)
            Point defining a ray from the origin

        V : ndarray, shape (M, d)
            Rows of V define the points

        Returns
        -------

        S : float
            The sum of squared distances from the ray to V
        """
        if x.ndim > 1:
            x = x.reshape(-1)
        assert V.shape[1] == x.shape[0]
        xhat = x / np.linalg.norm(x)

        S = 0
        for v in V:
            lenv = np.linalg.norm(v)
            vhat = v / lenv
            cs = vhat @ xhat
            S += np.sqrt(1 - cs * cs) * lenv
        return S

    def _generate_filename(self, **kwargs):
        return Saf._generate_filename(self, dmv=np.round(self.dmv, 2), **kwargs)

    def _get_loggables(self, **kwargs):
        log_data = {'dmv': self.dmv,
                    'dm_times': self.dm_times}
        return super()._get_loggables(**log_data, **kwargs)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.cm import viridis
    import wfg
    from testsuite.surrogates import GP, RF, MultiSurrogate
    import sys
    import os
    import rootpath
    from testsuite.surrogates import  MultiSurrogate, GP

    problem_path = os.path.join(rootpath.detect(), 'experiments/directed/data/wfg6_3obj_8dim')
    sys.path.append(problem_path)

    from problem_setup import func, limits, objective_function

    t = [0.561, 4.037, 2.013]
    seed = 5
    surrogate = MultiSurrogate(GP, scaled=True)

    opt = DirectedSaf(objective_function=objective_function, ei=False,  targets=t,
                w=0.5, limits=limits, surrogate=surrogate, n_initial=10,
                budget=150, log_dir="./log_dir", seed=seed)

    opt.optimise()



