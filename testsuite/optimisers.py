import lhsmdu
import numpy as np
import cma
import time
import os
import pickle
import copy
import _csupport as cs
import uuid
import itertools
from typing import Union
from scipy.stats import norm
from scipy.special import erf
from evoalgos.performance import FonsecaHyperVolume

from testsuite.utilities import Pareto_split, optional_inversion, dominates
from testsuite.surrogates import GP, MultiSurrogate, MonoSurrogate
from testsuite.acquisition_functions import scalar_expected_improvement


def increment_evaluation_count(f):
    # TODO move this into Optimiser class.
    """decorator to increment the number of evaluations with each
    function call and log the process"""
    def wrapper(self, *args, **kwargs):
        self.n_evaluations += 1
        return_value = f(self, *args, **kwargs)
        if self.n_evaluations%self.log_interval == 0 \
                or self.n_evaluations == self.budget:
            self.log_optimisation(save=True)
        else:
            self.log_optimisation(save=False)

        return return_value
    return wrapper


class Optimiser:
    def __init__(self, objective_function, limits,
                 n_initial=10, budget=30, of_args=[], seed=None,
                 log_dir="./log_data", log_interval=None):

        self.unique_code = str(uuid.uuid1())
        self.objective_function = objective_function
        self.of_args = of_args
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        self.limits = [np.array(limits[0]), np.array(limits[1])]
        self.n_initial = n_initial
        self.budget = budget
        self.n_evaluations = 0
        self.x_dims = np.shape(limits)[1]
        self.log_interval = log_interval if log_interval else budget
        self.train_time = 0.

        # generate initial samples
        self.x, self.y = self.initial_evaluations(n_initial,
                                                  self.x_dims,
                                                  self.limits)
        self.n_objectives = self.y.shape[1]
        # TODO obj_sense currently only allows minimisation of objectives.
        # self.obj_sense = [-1]*self.n_objectives

        # computed once and stored for efficiency.
        # TODO possibly more efficient to compute dominance matrix and
        #  build on that with each new point
        self.Pareto_indices = [*Pareto_split(self.y, return_indices=True)]
        self.p = self.y[self.Pareto_indices[0]]
        self.d = self.y[self.Pareto_indices[1]]
        self.obj_weights, self.obj_offset = self.get_obj_weighting()

        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(self.log_dir)
        self.log_filename = self._generate_filename()
        self.log_data = None
        self.log_optimisation()

    def initial_evaluations(self, n_samples, x_dims, limits):
        """
        Makes the initial evaluations of the parameter space and
        evaluates the objective function at these locaitons
        :param n_samples: number of initial samples to make
        """
        np.random.seed(self.seed)
        x = np.array(lhsmdu.sample(x_dims, n_samples, randomSeed=self.seed)).T\
            * (limits[1]-limits[0])+limits[0]
        # try:
        y = np.array([self.objective_function(xi, *self.of_args).flatten() for xi in x])
        # except :
        #     y = self.objective_function(x)

        # update evaluation number
        self.n_evaluations += n_samples

        return x, y

    def get_next_x(self, excluded_indices: list):
        """
        Implementation required in child: method to find next parameter
        to evaluate in optimisation sequence.
        :return: x_new [np.array] shape (1, x_dims)
        """
        raise NotImplementedError

    def optimise(self, n_steps=None):
        # unless specified exhaust budget
        if n_steps is None:
            n_steps = self.budget - self.n_evaluations

        tic = time.time()
        for i in range(n_steps):
            self.step()

        self.train_time += time.time()-tic

    @increment_evaluation_count
    def step(self):
        """takes one step in the optimisation, getting the next decision
        vector by calling the get_next_x method"""

        self.obj_weights, self.obj_offset = self.get_obj_weighting()
        # ensures unique evaluation
        x_new = self.get_next_x()

        try_count = 0
        while self._already_evaluated(x_new) and try_count < 3:
            # repeats optimisation of the acquisition function up to
            # three times to try and find a unique solution.

            # # logs models which produced errors.
            # # TODO: Remove this once problem solved.
            # try:
            #     try:
            #         # self.log_data["error models"].append(
            #         #     self.surrogate.model.copy())
            #         self.log_data["error models"].append(
            #             self.surrogate.model.copy())
            #     except KeyError:
            #         self.log_data["error models"] =\
            #             [self.surrogate.model.copy()]
            # except AttributeError:
            #     # multi-surrogate case
            #     try:
            #         self.log_data["error models"].append(
            #             [surrogate.model.copy() for surrogate in
            #              self.surrogate.mono_surrogates])
            #     except KeyError:
            #         self.log_data["error models"] = [
            #             [surrogate.model.copy() for surrogate in
            #              self.surrogate.mono_surrogates]]
            #     except AttributeError:
            #         pass

            x_new = self.get_next_x()
            try_count += 1

        if try_count > 1 and self._already_evaluated(x_new):
            # TODO error handling like this to be removed. Dev use only
            # Error#01 -> failed to find a unique solution after 3 optimisations
            # of the acquisition function
            self.log_data["errors"].append(
                "Error#01: Failed to find unique new solution at eval {} "
                "seed:{}".format(self.n_evaluations+1, self.seed))
        elif try_count > 1 and not self._already_evaluated(x_new):
            # Error#02 -> required repeats to find a unique solution
            self.log_data["errors"].append(
                "Error#02: Took {} attempts to find unique solution at eval "
                "{} seed:{}".format(try_count, self.n_evaluations+1, self.seed))

        try_count = 0
        while self._already_evaluated(x_new):
            # if repeat optimisation of the acquisition function does not
            # produce an x_new which has not already been evaluated
            excluded_indices = np.argsort(
                np.linalg.norm((self.x-x_new), axis=1))[:try_count+1]

            x_new = self.get_next_x(excluded_indices)
            try_count += 1
            if try_count > 5:
                # This should not occur multiple times. Something is
                # wrong here.
                self.log_optimisation(save=True, failed=True)

                raise RuntimeError("Optimsation of acquisition function"
                                   "failed to find a unique new "
                                   "evaluation parameter even with the "
                                   "first 5 duplicate solutions removed"
                                   " from the model")

        if try_count > 1:
            # Error#03 -> required point removal to find a unique solution
            self.log_data["errors"].append(
                "Error#03: Took removal of {} points to find unique solution "
                "at eval {}".format(try_count-1, self.n_evaluations+1))

        # get objective function evaluation for the new point
        y_new = self.objective_function(x_new, *self.of_args)

        # TODO if self.y dominates y_new then can be appended to Pareto
        #  indices without calling Pareto split. This would be more efficnet.
        # update observations with new observed point
        self.x = np.vstack((self.x, x_new))
        self.y = np.vstack((self.y, y_new))

        self.Pareto_indices = [*Pareto_split(self.y, return_indices=True)]
        self.p = self.y[self.Pareto_indices[0]]
        self.d = self.y[self.Pareto_indices[1]]

    def get_obj_weighting(self):
        """set the weighting equal to the range observed in the non-dominated
        set.
        """
        p = self.p
        if p.shape[0] < 2:
            # handles edge case where only one non-dominated point
            p2 = Pareto_split(self.d)[0]
            p = np.vstack((p.reshape(1,-1), p2))
        offset = p.min(axis=0)
        weighting = p.max(axis=0) - offset

        assert weighting.shape == (self.n_objectives,)
        assert offset.shape == (self.n_objectives,)

        return 1/weighting, p.min(axis=0)

    def apply_weighting(self, y):
        """ apply objective weighting to y"""
        if y.ndim == 1:
            assert y.shape[0] == self.n_objectives
        else:
            assert y.shape[1] == self.n_objectives
        return (y-self.obj_offset)*self.obj_weights

    def log_optimisation(self, save=False, **kwargs):
        """
        updates dictionary of saved optimisation information and saves
        to disk, including keyword arguments pased by the child class
        in the log_data dict.

        :param bool save: Save to disk if True
        :param kwargs: dictionary of keyword arguments to include in
        log_data
        :return: N/A
        """
        try:
            # log modifications each self.log_interval steps. Called in
            # increment_evaluation_count decorator.
            self.log_data["x"] = self.x
            self.log_data["y"] = self.y
            self.log_data["n_evaluations"] = self.n_evaluations
            self.log_data["train_time"] = self.train_time
            for key, value in kwargs.items():
                self.log_data[key] = value

        except TypeError:
            # initial information logging called by Optimiser __init__
            log_data = {"objective_function": self.objective_function.__name__,
                        "limits": self.limits,
                        "n_initial": self.n_initial,
                        "seed": self.seed,
                        "x": self.x,
                        "y": self.y,
                        "log_dir": self.log_dir,
                        "log_filename": self.log_filename,
                        "n_evaluations": self.n_evaluations,
                        "budget": self.budget,
                        "errors": [],
                        "train_time": self.train_time
                        }
            log_data.update(kwargs)
            self.log_data = log_data

        # save log_data and model to file.
        if save:
            log_filepath = os.path.join(self.log_dir, self.log_filename)
            with open(log_filepath+"_results.pkl", 'wb') as handle:
                pickle.dump(self.log_data, handle, protocol=2)
            with open(log_filepath+"_model.pkl", 'wb') as handle:
                pickle.dump(self, handle, protocol=2)

    def _generate_filename(self, *args):
        """
        generates a filename from optimiser parameters, and ensures
        uniqueness. Creates a directory under that filename within
        self.log_dir if it does not already exist.

        :param args: optional arguments to also include in filename.
        used by function call in child class.
        :return str: unique file name
        """
        # get information to include in filename
        objective_function = str(self.objective_function.__name__)
        optimiser = self.__class__.__name__
        initial_samples = self.n_initial

        # generate a sub dir to contain similar optimisations. Dir name
        # is the same as file name without the repeat number.
        file_dir = "{}_{}_init{}" + "_{}" * len(args)

        file_dir = file_dir.format(objective_function, optimiser,
                                   initial_samples, *args)

        # update location to log data to incorporate sub dir
        self.log_dir = os.path.join(self.log_dir, file_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # generate unique filename, accommodating repeat optimisations
        filename = 'seed_{:02d}_'.format(self.seed)+file_dir +"_"+ self.unique_code
        return filename

    def _already_evaluated(self, x_put, thresh=1e-9):
        """
        returns True if x_new is in the existing set of evaluated
        solutions stored in self.x. Similarity defined by difference of
        thresh
        :param x_put: putative new parameter to be queried
        :param float thresh: threshold for similarity comparison

        :return bool: Has this point already been evaluated?
        """
        # TODO: do this better. Some meaningful value for thresh instead
        #  of arbitrary. Use unit hypercube to calculate.
        difference_matrix = (self.x-x_put)**2
        evaluated = np.any(np.all((difference_matrix<thresh), axis=1))
        return evaluated


class ParEgo(Optimiser):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1583627 
    """
    def __init__(self, *args, s=5, rho=0.05, ei=True, surrogate=None, 
                 cmaes_restarts=0, **kwargs):
        self.s = s
        self.rho = rho
        self.ei = ei
        self.l_set = None
        self.surrogate = surrogate if surrogate else GP(scaled=True)
        self.cmaes_restarts = cmaes_restarts
        super().__init__(*args, **kwargs)

        # check user has not tried to use a MultiSurrogate
        assert(not isinstance(self.surrogate, MultiSurrogate)), \
            "Cannot use multi-surrogate with {} optimiser.".format(__class__)

    def get_next_x(self, excluded_indices=None):
        """
        Implementation required in child: method to find next parameter
        to evaluate in optimisation sequence.
        :return: x_new [np.array] shape (1, x_dims)
        """
        y = self.apply_weighting(self.y)
        lambda_i = self._get_lambda(self.s, self.n_objectives)
        self._dace(self.x, y, lambda_i)
        # update model with new f_lambda

        seed = np.random.uniform(self.limits[0], self.limits[1],
                                 self.surrogate.x_dims)
        res = cma.fmin(self.alpha, seed,
                       sigma0=0.25,
                       options={'bounds': [self.limits[0], self.limits[1]],
                                'maxfevals': 1e5},
                       restarts=self.cmaes_restarts)

        x_new = res[0]
        return x_new

    def _dace(self, x, y, lambda_i):
        y_norm = self._normalise(y)
        # TODO verify use of y_norm instead of self.y in dot product here.
        #  Alma differs from paper paper instructions used.
        f_lambda = np.max(y_norm*lambda_i, axis=1)+(self.rho*np.dot(y_norm, lambda_i))
        self.surrogate.update(x, f_lambda)
        pass

    def _normalise(self, y):
        """
        Normalise cost functions. Here we use estimated limits from data in
        normalisation as suggested by Knowles (2006).

        Parameters.
        -----------
        y (np.array): matrix of function values.

        Returns normalised function values.
        """
        min_y = np.min(y, axis=0)
        max_y = np.max(y, axis=0)
        return (y - min_y) / (max_y - min_y)

    def _get_lambda(self, s, n_obj):
        """
        Select a lambda vector. See Knowles(2006) for full details.

        Parameters.
        -----------
        s (int): determine total number of vectors: from (s+k-1) choose (k-1)
                    vectors.
        n_obj (int): number of objectvies.

        Returns a selected lambda vector.
        """
        if self.l_set is None:
            l = [np.arange(s + 1, dtype=int) for i in range(n_obj)]
            self.l_set = np.array([np.array(i) \
                                   for i in itertools.product(*l) if
                                   np.sum(i) == s]) / s
            print("Number of scalarising vectors: ", self.l_set.shape[0])
        ind = np.random.choice(np.arange(self.l_set.shape[0], dtype=int))
        return self.l_set[ind]

    def alpha(self, x):
        mu, var = self.surrogate.predict(x)
        if self.ei:
            # use ei in accordance with knowles 2006
            # get_y() to use unscaled f_lambda values in ei
            # -ei becasue cmaes minimises only. We want max ei.
            # invert becasue it is a minimisation problem
            efficacy = -scalar_expected_improvement(mu, var,
                                               y=self.surrogate.get_y(),
                                               invert=True)
        else:
            # use mean prediction instead
            efficacy = self.surrogate.predict(x)[0]
        return float(efficacy)


class BayesianOptimiser(Optimiser):
    def __init__(self, objective_function, limits, surrogate, cmaes_restarts=0,
                 log_models=False, **kwargs):

        self.raw_surrogate = surrogate
        self.surrogate = copy.deepcopy(self.raw_surrogate)
        self.log_models = log_models
        if isinstance(surrogate, MultiSurrogate):
            self.multi_surrogate = True
        else:
            self.multi_surrogate = False

        if self.log_models:
            self.model_log = []

        super().__init__(objective_function, limits, **kwargs)
        self.cmaes_restarts = cmaes_restarts

    def _generate_filename(self, *args):
        """include surrogate and acquisition function for logging"""
        try:
            # multi-surrogate
            return super()._generate_filename(
                self.surrogate.__class__.__name__,
                self.surrogate.surrogate.__name__, *args)
        except AttributeError:
            # mono-surrogate
            return super()._generate_filename(
                self.surrogate.__class__.__name__, *args)

    def get_next_x(self, excluded_indices=None):
        """
        Gets the next point to be evaluated by the objective function.
        Decided by optimisation of the acquisition function in
                                'maxfevals': 1e5},
        self.alpha.

        :param excluded_indices: list of indices for self.x and self.y
        to exclude when building the surrogate model. This allows BO to
        navigate niche cases where the optimisation of the acquisition
        function returns a point which has already been evaluated.
        Re-evaluating this point would incur cost without advancing the
        optimisation. Excluding the point being repeated is the method
        to avoid this.

        :return np.ndarray: next point to be evaluated by the objective
        function and added to self.x
        """
        if excluded_indices is not None:
            # handle observations to be excluded from the model.
            x = self.x[[i for i in range(len(self.x))
                        if i not in excluded_indices]]
            y = self.y[[i for i in range(len(self.x))
                        if i not in excluded_indices]]
        else:
            x = self.x
            y = self.y

        y = self.apply_weighting(y)

        # update surrogate
        self.surrogate.update(x, y)

        # optimise the acquisition function using cm-aes algorithm for
        # parameter space with dimensions >1, else use random search.
        if self.surrogate.x_dims > 1:
            # optimise acquisition function using random search
            seed = np.random.uniform(self.limits[0], self.limits[1],
                                     self.surrogate.x_dims)
            res = cma.fmin(self.alpha, seed,
                           sigma0=0.25,
                           options={'bounds':[self.limits[0], self.limits[1]],
                                    'maxfevals': 1e5},
                           restarts=self.cmaes_restarts)

            x_new = res[0]
        else:
            # TODO fix use of argmin to accommodate maximisation cases.
            # optimise acquisition function using cma-es
            n_search_points = 1000
            search_points = np.random.uniform(self.limits[0], self.limits[1],
                                              n_search_points)
            res_index = np.argmin([self.alpha(search_point)
                                   for search_point in search_points])

            x_new = np.array(search_points[res_index:res_index + 1])

        return x_new.reshape(1, -1)

    def alpha(self, x_put):
        put_y, put_var = self.surrogate.predict(x_put)
        return self._scalarise_y(put_y, put_var**0.5, invert=True)

    def log_optimisation(self, save=False):
        if save:
            self.surrogate = copy.deepcopy(self.raw_surrogate)
        super().log_optimisation(save=save)

    @optional_inversion
    def _scalarise_y(self, put_y, put_std):
        assert NotImplementedError


class Saf(BayesianOptimiser):
    def __init__(self, *args, ei=True,  **kwargs):

        self.ei = ei
        super().__init__(*args, **kwargs)

    def _generate_filename(self, *args):
        if self.ei:
            return super()._generate_filename("ei", *args)
        else:
            return super()._generate_filename("mean", *args)

    @staticmethod
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

        D = np.zeros((y.shape[0], p.shape[0]))

        for i, p in enumerate(p):
            D[:, i] = np.max(p - y, axis=1).reshape(-1)
        Dq = np.min(D, axis=1)
        return Dq

    @optional_inversion
    def saf_ei(self, y_put: np.ndarray, std_put, n_samples: int = 1000,
               return_samples=False) -> Union[np.ndarray, tuple]:
        p = self.apply_weighting(self.p)
        if y_put.ndim < 2:
            y_put = y_put.reshape(1, -1)
        if std_put.ndim < 2:
            std_put = std_put.reshape(1, -1)

        # TODO implement multiple point saf_ei computation simultaneously.

        # best saf value observed so far.
        # f_star = np.max(self.saf(y, p))  # should be 0 in current cases
        f_star = 0.

        # sample from surrogate
        samples = np.random.normal(0, 1, size=(n_samples, y_put.shape[1])) \
                  * std_put + y_put

        saf_samples = self.saf(samples, p, invert=False)
        saf_samples[saf_samples < f_star] = f_star

        if return_samples:
            return samples, saf_samples
        else:
            return np.mean(saf_samples)

    @optional_inversion
    def _scalarise_y(self, y_put, std_put):
        if y_put.ndim<2:
            y_put = y_put.reshape(1,-1)
            std_put = std_put.reshape(1,-1)

        assert y_put.shape[0]==1
        assert std_put.shape[0]==1

        if self.ei:
            return float(self.saf_ei(y_put, std_put, n_samples=3000*self.n_objectives,
                invert=False))
        else:
            return float(self.saf(y_put, self.apply_weighting(self.p),
                                  invert=False))

    def alpha(self, x_put):
            y_put, var_put = self.surrogate.predict(x_put)
            efficacy_put = self._scalarise_y(y_put, var_put**0.5, invert=True)
            return efficacy_put


class SmsEgo(BayesianOptimiser):

    def __init__(self, *args, ref_vector=None, **kwargs):

        super().__init__(*args, **kwargs, )
        self.gain = -norm.ppf(0.5 * (0.5 ** (1 / self.n_objectives)))

        # set dynamic reference vector if not specified
        if ref_vector is None:
            self.dynamic_ref = True
            self.ref_vector = self.apply_weighting(self.p).max(axis=0)+1
        else:
            self.dynamic_ref = False
            try:
                self.ref_vector = np.array(self.apply_weighting(ref_vector)).reshape(-1)
                assert(self.ref_vector.shape == (self.n_objectives,))
            except AssertionError:
                raise AssertionError("Supplied reference vector is not"
                                     "formatted correctly. should be 1d array "
                                     "of size {}.".format(self.n_objectives))

        self.hpv = FonsecaHyperVolume(reference_point=self.ref_vector)
        self.chv = self._compute_hypervolume()
        # self.current_hv = self._compute_hypervolume()

    def _generate_filename(self):
        return super()._generate_filename()

    def _compute_hypervolume(self, p=None):
        """
        Calcualte the current hypervolume, or that of the provided y.
        """
        if p is None:
            p = self.apply_weighting(self.p)

        if self.n_objectives > 1:
            assert (p.ndim <= 2), "error in attainment front shape."
            if p.ndim == 1:
                p = p.reshape(1, self.n_objectives)
            assert(p.shape[1] == self.n_objectives)
            volume = self.hpv.assess_non_dom_front(p)
            return volume
        else:
            return np.min(p)

    def step(self):
        super().step()
        if self.dynamic_ref:
            # update ref vector to max observed if it was initially unspecified
            ref_vector = self.apply_weighting(self.p).max(axis=0)+1
            if np.any(ref_vector != self.ref_vector):
                # update hypervolume calc to include new ref and update ref
                self.hpv=FonsecaHyperVolume(reference_point=ref_vector)
                self.ref_vector = ref_vector
                self.chv = self._compute_hypervolume()
        # update hypervolume
        # self.current_hv = self._compute_hypervolume()

    def _compute_epsilon(self, p_scaled):
        n_pfr = len(p_scaled)
        c = 1 - (1 / 2 ** self.n_objectives)

        # TODO is b_count supposed to be the remaining budget?
        b_count = self.budget - self.n_evaluations - 1
        epsilon = (p_scaled.max(axis=0) - p_scaled.min(axis=0)) \
                  / (n_pfr + (c*b_count))
        return epsilon

    def _compute_penalty(self, lcb, p):
        pt = p + self._compute_epsilon(p)
        # pt = p
        # yt = lcb + self._compute_epsilon(p)
        yt = lcb
        if np.all(Pareto_split(np.vstack((yt, pt)))[0] == pt):
            assert lcb.ndim == 2
            return np.max([-1+np.prod(1+lcb-pi) for pi in p])
        else:
            return 0

    @optional_inversion
    def _scalarise_y(self, y_put, std_put):
        p = self.apply_weighting(self.p)

        if y_put.ndim < 2:
            y_put = y_put.reshape(1, -1)
            std_put = std_put.reshape(1, -1)

        assert y_put.shape[0] == 1
        assert std_put.shape[0] == 1

        # lower confidence bounds
        lcb = y_put + (self.gain * std_put)

        yt = lcb + (self._compute_epsilon(p))
        l = [-1 + np.prod(1 + lcb - p_i)
             if cs.compare_solutions(p_i, yt, [-1, -1]) == 0
             else 0 for p_i in p]

        penalty = (max([0, max(l)]))

        # penalty = self._compute_penalty(lcb, p)

        if penalty > 0:
            return -penalty
        else:
            # compute and update hypervolumes
            current_hv = self.chv
            # we use vstack(self.p, lcb) here without Pareto_split becasue
            # it is more efficient and gives the same answer. Verified
            # TODO create temporary class variable to store best hv so that
            #  it does not have to be recomputed in self.step
            put_hv = self._compute_hypervolume(np.vstack((p, y_put)))
            return put_hv - current_hv

    def alpha(self, x_put):
        y_put, var_put = self.surrogate.predict(x_put)
        return float(self._scalarise_y(y_put, var_put**0.5, invert=True))


class SmsEgoMu(SmsEgo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _compute_epsilon(self, p_scaled):
        epsilon = np.zeros(self.n_objectives)
        return epsilon

    def alpha(self, x_put):
        y_put, var_put = self.surrogate.predict(x_put)
        efficacy_put = self._scalarise_y(y_put, np.zeros_like(var_put),
                                         invert=True)
        return float(efficacy_put)


class Mpoi(BayesianOptimiser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _generate_filename(self):
        return super()._generate_filename("Mpoi")

    @optional_inversion
    def _scalarise_y(self, y_put, std_put):
        '''
        Calculate the minimum probability of improvement compared to current
        Pareto front. Refer to the paper for full details.

        parameters:
        -----------
        x (np.array): decision vectors.
        cfunc (function): cheap constraint function.
        cargs (tuple): argument for constraint function.
        ckwargs (dict): keyword arguments for constraint function.

        Returns scalarised cost.
        '''
        p = self.apply_weighting(self.p)

        if y_put.ndim<2:
            y_put = y_put.reshape(1, -1)
            std_put = std_put.reshape(1, -1)

        assert y_put.shape[0] == 1
        assert std_put.shape[0] == 1

        res = np.zeros((y_put.shape[0], 1))
        for i in range(y_put.shape[0]):
            m = (y_put[i] - p) / (np.sqrt(2) * std_put[i])
            pdom = 1 - np.prod(0.5 * (1 + erf(m)), axis=1)
            res[i] = np.min(pdom)
        return res

    def alpha(self, x_put):
        y_put, var_put = self.surrogate.predict(x_put)
        efficacy_put = self._scalarise_y(y_put, var_put**0.5, invert=True)
        return float(efficacy_put)


class Saf_Sms(SmsEgo):
    """
    saf in the non-dominated region, sms-ego in the dominated region.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @optional_inversion
    def _scalarise_y(self, y_put, std_put):
        if not self.ei:
            std_put = np.zeros_like(std_put)

        if y_put.ndim < 2:
            y_put = y_put.reshape(1, -1)
            std_put = std_put.reshape(1, -1)

        assert y_put.shape[0] == 1
        assert std_put.shape[0] == 1

        saf_v = float(Saf.saf(y_put.reshape(1,-1), self.p))
        if saf_v>0:
            return saf_v
        else:
            return super()._scalarise_y(y_put, std_put)


class Sms_Saf(SmsEgo):
    """
    saf in the non-dominated region, sms-ego in the dominated region.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @optional_inversion
    def _scalarise_y(self, y_put, std_put):
        y = self.y * self.obj_weights
        p = self.p * self.obj_weights
        if not self.ei:
            std_put = np.zeros_like(std_put)

        if y_put.ndim<2:
            y_put = y_put.reshape(1,-1)
            std_put = std_put.reshape(1,-1)

        assert y_put.shape[0] == 1
        assert std_put.shape[0] == 1

        # lower confidence bounds
        lcb = y_put - (self.gain * np.multiply(self.obj_sense, std_put))

        # calculate penalty
        n_pfr = len(p)
        c = 1 - (1 / 2 ** self.n_objectives)

        # TODO is b_count supposed to be the remaining budget?
        b_count = self.budget - self.n_evaluations -1
        epsilon = (np.max(y, axis=0) - np.min(y, axis=0)) / (
                n_pfr + (c * b_count))

        yt = lcb - (epsilon * self.obj_sense)
        l = [-1 + np.prod(1 + lcb - p_i) if cs.compare_solutions(p_i, yt, self.obj_sense) == 0 else 0 for p_i in p]
        penalty = (max([0, max(l)]))

        # if penalty > 0:

        saf_v = float(Saf.saf(y_put.reshape(1, -1), p))
        if saf_v < 0:
            return saf_v
        else:
            # compute and update hypervolumes
            current_hv = self._compute_hypervolume()
            put_hv = self._compute_hypervolume(
                Pareto_split(np.vstack((p, lcb)))[0]
            )
            return put_hv - current_hv


class Saf_Saf(Saf):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @optional_inversion
    def _scalarise_y(self, y_put, std_put):
        if y_put.ndim<2:
            y_put = y_put.reshape(1,-1)
            std_put = std_put.reshape(1,-1)

        assert y_put.shape[0]==1
        assert std_put.shape[0]==1
        saf_v = float(self.saf(y_put, self.p, invert=False))
        if saf_v<0:
            return saf_v
        else:
            return float(self.saf_ei(y_put, std_put, invert=False))

class Lhs():
    def __init__(self, objective_function, limits, n_initial=10, budget=30, 
                 of_args=[], seed=None, log_dir="./log_data"):

        self.objective_function = objective_function
        self.of_args = of_args
        self.log_data = {}
        self.train_time = 0
        self.n_initial = n_initial
        self.budget = budget
        self.n_inital = n_initial
        self.x_dims = np.shape(limits)[1]
        self.seed = seed if seed is not None else np.random.randint(0, 10000)
        self.limits = limits
        self.x = [self.lhs_sample(n_initial)]
        self.y = [self.objective_function(self.x[0], *of_args)]
        self.n_objectives = self.y[0].shape[1]
        self.n_evaluations = n_initial
        
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(self.log_dir)
        self.log_filename = self._generate_filename()
        self.log_data = None

    def lhs_sample(self, n_samples):
        x = lhsmdu.sample(self.x_dims, n_samples, randomSeed=self.seed)
        return np.array(x).T*(self.limits[1]-self.limits[0])+self.limits[0]

    def step(self):
        n_samples = self.n_evaluations+1
        x = self.lhs_sample(n_samples)
        y = self.objective_function(x, *self.of_args)
        self.x.append(x)
        self.y.append(y)
        self.n_evaluations = n_samples

    def optimise(self, n_steps=None):
        # unless specified exhaust budget
        if n_steps is None:
            n_steps = self.budget - self.n_evaluations

        tic = time.time()
        for i in range(n_steps):
            self.step()
        self.train_time+=time.time()-tic

        self.log_optimisation(save=True)

    def _generate_filename(self):
        file_dir = "{}_{}_init{}"
        # get information to include in filename
        objective_function = str(self.objective_function.__name__)
        optimiser = "lhs"
        initial_samples = self.n_initial

        file_dir = file_dir.format(objective_function, optimiser,
                                   initial_samples)

        # update location to log data to incorporate sub dir
        self.log_dir = os.path.join(self.log_dir, file_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        # generate unique filename, accommodating repeat optimisations
        unique_code = str(uuid.uuid1())
        filename = file_dir+'seed_{}_'.format(self.seed)+unique_code+"_results"
        return filename

    def log_optimisation(self, save=False, **kwargs):
        """
        updates dictionary of saved optimisation information and saves
        to disk, including keyword arguments pased by the child class
        in the log_data dict.

        :param bool save: Save to disk if True
        :param kwargs: dictionary of keyword arguments to include in
        log_data
        :return: N/A
        """
        try:
            # log modifications each self.log_interval steps. Called in
            # increment_evaluation_count decorator.
            self.log_data["x"] = self.x
            self.log_data["y"] = self.y
            self.log_data["n_evaluations"] = self.n_evaluations
            self.log_data["train_time"] = self.train_time
            for key, value in kwargs.items():
                self.log_data[key] = value

        except TypeError:
            # initial information logging called by Optimiser __init__
            log_data = {"objective_function": self.objective_function.__name__,
                        "limits": self.limits,
                        "n_initial": self.n_initial,
                        "seed": self.seed,
                        "x": self.x,
                        "y": self.y,
                        "log_dir": self.log_dir,
                        "log_filename": self.log_filename,
                        "n_evaluations": self.n_evaluations,
                        "budget": self.budget,
                        "errors": [],
                        "train_time": self.train_time
                        }
            log_data.update(kwargs)
            self.log_data = log_data

        # save log_data to file.
        
        if save:
            log_filepath = os.path.join(self.log_dir, self.log_filename)
            with open(log_filepath+".pkl", 'wb') as handle:
                pickle.dump(self.log_data, handle, protocol=2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import wfg

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

    # opt = Mpoi(objective_function=test_function, limits=limits, surrogate=gp_surr_multi, n_initial=10, seed=None)
    # opt = Saf(objective_function=test_function, ei=True,  limits=limits, surrogate=gp_surr_multi, n_initial=10, budget=12, seed=None)
    # opt = Saf(objective_function=test_function, ei=False,  limits=limits, surrogate=gp_surr_multi, n_initial=10, budget=20, seed=None, log_models=True, log_interval=1)
    opt = SmsEgo(objective_function=test_function, limits=limits, surrogate=gp_surr_multi, n_initial=10, budget =50, seed=None)
    opt2 = SmsEgoMu(objective_function=test_function, limits=limits, surrogate=gp_surr_multi, n_initial=10, budget =50, seed=None)
    # opt = SmsEgo(objective_function=test_function, ei=False,  limits=limits, surrogate=gp_surr_multi, n_initial=10, seed=17, budget=100)
    # opt = ParEgo(objective_function=test_function, limits=limits, surrogate=GP(), n_initial=10, s=5, rho=0.5)
    # opt = Lhs(objective_function = test_function, limits=limits, n_initial=10, budget=20, seed=None)

    opt.optimise(10)
    # opt.optimise(10)
    pass
