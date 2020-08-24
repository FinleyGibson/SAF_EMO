import lhsmdu
import numpy as np
import cma
import inspect
import os
import pickle
import _csupport as cs
from scipy.stats import norm

from testsuite.utilities import Pareto_split
import testsuite.acquisition_functions as acq_f
from testsuite.surrogates import GP, MultiSurrogate
from evoalgos.performance import FonsecaHyperVolume


def increment_evaluation_count(f):
    # TODO move this into Optimiser class.
    """decorator to increment the number of evaluations with each
    function call and log the process"""
    def wrapper(self, *args, **kwargs):
        self.n_evaluations += 1
        return_value = f(self, *args, **kwargs)
        if self.n_evaluations%self.log_interval == 0:
            self.log_optimisation(save=True)
        else:
            self.log_optimisation(save=False)

        return return_value
    return wrapper


class Optimiser:
    def __init__(self, objective_function, limits, n_initial=10, budget=30,
                 seed=None, ref_vector=None, log_dir="./log_data",
                 log_interval=None):
        self.objective_function = objective_function
        self.seed = seed if seed else np.random.randint(0, 10000)
        self.limits = [np.array(limits[0]), np.array(limits[1])]
        self.n_initial = n_initial
        self.budget = budget
        self.n_evaluations = 0
        self.x_dims = np.shape(limits)[1]
        self.log_interval = log_interval if log_interval else budget

        # generate initial samples
        self.x, self.y = self.initial_evaluations(n_initial,
                                                  self.x_dims,
                                                  self.limits)
        self.n_objectives = self.y.shape[1]
        # TODO obj_sense currently only allows minimisation of objectives.
        self.obj_sense = [-1]*self.n_objectives

        # set up logging
        # if no ref_vector provided, use max of initial evaluations
        self.ref_vector = ref_vector if ref_vector else self.y.max(axis=0)
        self.hpv = FonsecaHyperVolume(self.ref_vector) # used to find hypervolume
        self.log_dir = log_dir
        self.log_filename = self._generate_filename()
        if not os.path.exists(log_dir):
            os.makedirs(self.log_dir)
        self.log_data = None
        self.log_optimisation()

    def initial_evaluations(self, n_samples, x_dims, limits):
        """
        Makes the initial evaluations of the parameter space and
        evaluates the objective function at these locaitons
        :param n_samples: number of initial samples to make
        """
        x = np.array(lhsmdu.sample(x_dims, n_samples, randomSeed=self.seed)).T\
            * (limits[1]-limits[0])+limits[0]
        y = self.objective_function(x)

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
            n_steps = self.budget - self.n_initial

        for i in range(n_steps):
            self.step()

    @increment_evaluation_count
    def step(self):
        """takes one step in the optimisation, getting the next decision
        vector by calling the get_next_x method from the child class"""

        # ensures unique evaluation
        x_new = self.get_next_x()
        try_count = 0
        aa = self._already_evaluated(x_new)
        while self._already_evaluated(x_new) and try_count < 3:
            # repeats optimisation of the acquisition function up to
            # three times to try and find a unique solution.
            x_new = self.get_next_x()
            try_count += 1

        if try_count > 1 and self._already_evaluated(x_new):
            # TODO error handling like this to be removed. Dev work only
            # Error#01 -> failed to find a unique solution after 3 optimsations
            # of the acquisition function
            self.log_data["errors"].append(
                "Error#01: Failed to find unique new solution at eval {}"
                .format(self.n_evaluations+1))
        elif try_count > 1 and not self._already_evaluated(x_new):
            # Error#02 -> required repeats to find a unique solution
            self.log_data["errors"].append(
                "Error#02: Took {} attempts to find unique solution at eval {}"
                .format(try_count, self.n_evaluations+1))

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
        y_new = self.objective_function(x_new)

        # update observations with new observed point
        self.x = np.vstack((self.x, x_new))
        self.y = np.vstack((self.y, y_new))
        # self.log_optimisation()

    def log_optimisation(self, save=False):
        """
        updates dictionary of saved optimisation information and saves
        to disk.
        """
        try:
            # log modifications each self.log_interval steps. Called in
            # increment_evaluation_count decorator.
            self.log_data["x"] = self.x
            self.log_data["y"] = self.y
            self.log_data["hypervolume"].append(self._compute_hypervolume())
            self.log_data["n_evaluations"] += self.log_interval
        except TypeError:
            # initial information logging called by Optimiser __init__
            log_data = {"objective_function":
                        inspect.getsource(self.objective_function),
                        "limits": self.limits,
                        "n_initial": self.n_initial,
                        "seed": self.seed,
                        "hypervolume": [self._compute_hypervolume()],
                        "x": self.x,
                        "y": self.y,
                        "log_dir": self.log_dir,
                        "log_filename": self.log_filename,
                        "n_evaluations": self.n_evaluations,
                        "budget": self.budget,
                        "errors": []
                        }
            self.log_data = log_data

        # save log_data to file.
        if save:
            log_filepath = os.path.join(self.log_dir, self.log_filename)
            with open(log_filepath+".pkl", 'wb') as handle:
                pickle.dump(self.log_data, handle, protocol=2)

    def _compute_hypervolume(self, y=None):
        """
        Calcualte the current hypervolume, or that of the provided y.
        """
        if y is None:
            y = self.y

        if self.n_objectives > 1:
            front = Pareto_split(y, return_indices=False)[0]
            volume = self.hpv.assess_non_dom_front(front)
            return volume
        else:
            return np.min(y)

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
        repeat = -1
        filename = None
        while filename is None or os.path.exists(os.path.join(self.log_dir,
                                                              filename+".pkl")):
            repeat += 1
            filename = file_dir+"_{:03d}".format(repeat)

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


class BayesianOptimiser(Optimiser):
    def __init__(self, objective_function, limits, surrogate,
                 n_initial=10, budget=30, seed=None,
                 acquisition_function="saf_mu", ref_vector=None,
                 log_dir="./log_data", log_interval=None,
                 cmaes_restarts=0):

        self.surrogate = surrogate
        self.acquisition_function = getattr(acq_f, acquisition_function)

        super().__init__(objective_function=objective_function, limits=limits,
                         n_initial=n_initial, budget=budget, seed=seed,
                         ref_vector=ref_vector, log_dir=log_dir,
                         log_interval=log_interval)

        self.cmaes_restarts = cmaes_restarts

    def _generate_filename(self):
        """include surrogate and acquisition function for logging"""
        try:
            # multi-surrogate
            file_name = super()._generate_filename(
                self.surrogate.__class__.__name__,
                self.surrogate.surrogate.__name__,
                self.acquisition_function.__name__.replace("_", ""))
        except AttributeError:
            # mono-surrogate
            file_name = super()._generate_filename(
                self.surrogate.__class__.__name__,
                self.acquisition_function.__name__.replace("_", ""))
        return file_name

    def get_next_x(self, excluded_indices=None):
        """
        Gets the next point to be evaluated by the objective function.
        Decided by optimisation of the acquisition function in
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

        # update surrogate
        self.surrogate.update(x, y)

        # optimise the acquisition function using cm-aes algorithm for
        # parameter space with dimensions >1, else use random search.
        alpha_arguments = [self.surrogate, y]
        if self.surrogate.x_dims > 1:
            # optimise acquisition function using random search
            seed = np.random.uniform(self.limits[0], self.limits[1],
                                     self.surrogate.x_dims)
            res = cma.fmin(self.alpha, seed,
                           sigma0=0.25,
                           options={'bounds':
                                    [self.limits[0], self.limits[1]]},
                           restarts=self.cmaes_restarts,
                           args=alpha_arguments)
            x_new = res[0]
        else:
            # TODO fix use of argmin to accommodate maximisation cases.
            # optimise acquisition function using cma-es
            n_search_points = 1000
            search_points = np.random.uniform(self.limits[0], self.limits[1],
                                              n_search_points)
            res_index = np.argmin([self.alpha(search_point, *alpha_arguments)
                                   for search_point in search_points])

            x_new = np.array(search_points[res_index:res_index + 1])

        return x_new.reshape(1, -1)

    def alpha(self, x_put, *args, **kwargs):
        """
        evaluates the predicted efficacy of the putative solution x_put,
        based on predictions from the surrogate and the acquisition
        function.
        :param np.ndarray x_put: putative solution to the objective
        function (1, x_dims)
        :param args: args to acquisition_function
        :param kwargs: kwargs to acquisition_function
        :return float:  assessment of x_put efficacy as a solution to
        the objective function.
        """
        efficacy_put = self.acquisition_function(x_put, *args, **kwargs)
        return efficacy_put


class SmsEgo(Optimiser):

    def __init__(self, objective_function, limits, surrogate, n_initial=10,
                 budget=30, seed=None, ref_vector=None, log_dir="./log_data",
                 log_interval=None, cmaes_restarts=0):

        self.surrogate = surrogate
        super().__init__(objective_function=objective_function, limits=limits,
                         n_initial=n_initial, budget=budget, seed=seed,
                         ref_vector=ref_vector, log_dir=log_dir,
                         log_interval=log_interval)

        self.gain = -norm.ppf(0.5 * (0.5 ** (1 / self.n_objectives)))
        self.cmaes_restarts = cmaes_restarts

    def _generate_filename(self):
        """include surrogate and acquisition function for logging"""
        try:
            # multi-surrogate
            file_name = super()._generate_filename(
                self.surrogate.__class__.__name__,
                self.surrogate.surrogate.__name__)
        except AttributeError:
            # mono-surrogate
            file_name = super()._generate_filename(
                self.surrogate.__class__.__name__)
        return file_name

    def get_next_x(self, excluded_indices=None):
        """
        Gets the next point to be evaluated by the objective function.

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

        # update surrogate
        self.surrogate.update(x, y)

        # optimise the acquisition function using cm-aes algorithm for
        # parameter space with dimensions >1, else use random search.
        alpha_arguments = [self.surrogate, y]
        if self.surrogate.x_dims > 1:
            # optimise acquisition function using random search
            seed = np.random.uniform(self.limits[0], self.limits[1],
                                     self.surrogate.x_dims)
            # ans = self.alpha(seed, *alpha_arguments)
            res = cma.fmin(self.alpha, seed,
                           sigma0=0.25,
                           options={'bounds':
                                        [self.limits[0], self.limits[1]]},
                           restarts=self.cmaes_restarts,
                           args=alpha_arguments)
            x_new = res[0]
        else:
            # TODO fix use of argmin to accommodate maximisation cases.
            # optimise acquisition function using cma-es
            n_search_points = 1000
            search_points = np.random.uniform(self.limits[0], self.limits[1],
                                              n_search_points)
            res_index = np.argmin([self.alpha(search_point, *alpha_arguments)
                                   for search_point in search_points])

            x_new = np.array(search_points[res_index:res_index + 1])

        return x_new.reshape(1, -1)

    def _penalty(self, y, y_test):
        '''
        Penalty mechanism in the infill criterion. Penalise if dominated by the
        current front.

        Parameters.
        -----------
        y (np.array): current Pareto front elements.
        y_test (np.array): tentative solution.

        Returns a penalty value.
        '''
        # TODO check this maths against paper.
        p_inds, d_inds = Pareto_split(self.y)
        n_pfr = len(p_inds)
        c = 1 - (1/2**self.n_objectives)
        b_count = self.budget - len(self.x) - 1

        epsilon = (np.max(self.y, axis=0) - np.min(self.y, axis=0))/\
                  (n_pfr + (c * b_count))

        yt = y_test - (epsilon * self.obj_sense)
        l = [-1 + np.prod(1 + y_test - y[i]) if
             cs.compare_solutions(y[i], yt, self.obj_sense) == 0
             else 0 for i in range(y.shape[0])]
        return (max([0, max(l)]))

    def _compare_add_solution(self, y, ytest, obj_sense):
        '''
        Compare and add a solution to the data set given its not dominated.

        Parameters.
        -----------
        y (np.array): current Pareto front objective vectors.
        y_test (np.array): candidate for adding to the archive.

        Returns latest Pareto front.
        '''
        result = np.ones(y.shape[0])
        for i in range(y.shape[0]):
            result[i] = cs.compare_solutions(y[i], ytest, self.obj_sense)
            if result[i] == 0:
                return y
        inds = np.where(result == 3)[0]
        try:
            return np.concatenate([y[inds], ytest])
        except ValueError:
            print("Likely error in y: ", y[inds])
            return ytest

    def _scalarise_y(self, y_put, y_put_std):
        # split y into dominated and non-dominated points
        p_inds, d_inds = Pareto_split(self.y, return_indices=True)
        p, d = self.y[p_inds], self.y[d_inds]

        current_hv = self._compute_hypervolume()

        # lower confidence bounds
        lcb = y_put - (self.gain * np.multiply(self.obj_sense, y_put_std))

        # calculate penalty
        n_pfr = len(p_inds)
        c = 1 - (1 / 2 ** self.n_objectives)
        b_count = self.budget - self.n_evaluations -1 # TODO is this supposed to be the remaining budget
        epsilon = (np.max(self.y, axis=0) - np.min(self.y, axis=0)) / (
                    n_pfr + (c * b_count))

        self.epsilon = epsilon
        self.c = c
        self.n_prf = n_pfr
        self.b_count = b_count

        yt = y_put - (epsilon * self.obj_sense)
        l = [-1 + np.prod(1 + y_put - self.y[i]) if
             cs.compare_solutions(self.y[i], yt, self.obj_sense) == 0
             else 0 for i in range(self.y.shape[0])]
        penalty = (max([0, max(l)]))

        if penalty > 0:
            return -np.array([-penalty])

        # new front
        new_hv = self._compute_hypervolume(np.vstack((self.y, lcb)))
        return -np.array([new_hv - current_hv])

    # def _scalarise_y(self, yp, stdp):
    # #     lower confidence bounds
            # yl = yp - (self.gain * np.multiply(self.obj_sense, stdp))
            # p_inds, d_inds = Pareto_split(self.y, return_indices=True)
            # pen = self._penalty(self.y[p_inds], yl)
            # if pen > 0:
            #     return -np.array([-pen])
            # # new front
            # hv_prev = self._compute_hypervolume()
            # hv_new = self._compute_hypervolume(np.vstack((self.y, yl)))
            # # yn = self._compare_add_solution(self.y[p_inds], yl, self.obj_sense)
            # return -np.array([hv_new-hv_prev])

    def alpha(self, x_put, *args, **kwargs):
            """
            evaluates the predicted efficacy of the putative solution x_put,
            based on predictions from the surrogate and the acquisition
            function.
            :param np.ndarray x_put: putative solution to the objective
            function (1, x_dims)
            :param args: args to acquisition_function
            :param kwargs: kwargs to acquisition_function
            :return float:  assessment of x_put efficacy as a solution to
            the objective function.
            """
            yp, stdp = self.surrogate.predict(x_put)
            ans = self._scalarise_y(yp, stdp)
            return ans


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import wfg
    # set up objective function
    kfactor = 2
    lfactor = 2
    n_objectives = 2
    n_dims = lfactor * 2 + kfactor

    k = kfactor * (n_objectives - 1)
    l = lfactor * 2
    wfg_n = 6
    exec("func = wfg.WFG{}".format(int(wfg_n)))

    def test_function(x):
        if x.ndim == 2:
            assert (x.shape[1] == n_dims)
        else:
            squeezable = np.where([a == 1 for a in x.shape])[0]
            for i in squeezable[::-1]:
                x = x.squeeze(i)

        if x.ndim == 1:
            assert (x.shape[0] == n_dims)
            x = x.reshape(1, -1)
        return np.array([func(xi, k, n_objectives) for xi in x])


    limits = [np.zeros((n_dims)), np.array(range(1, n_dims + 1)) * 2]

    gp_surr_multi = MultiSurrogate(GP, scaled=True)
    opt = SmsEgo(objective_function=test_function, limits=limits, surrogate=gp_surr_multi, n_initial=10, seed=None)

    opt.optimise(n_steps=5)
    print("done")