import lhsmdu
import numpy as np
import testsuite.acquisition_functions as acquisition_functions
import cma
from testsuite.surrogates import *
import inspect
from evoalgos.performance import FonsecaHyperVolume
import os
import pickle
import _csupport as CS


def increment_evaluation_count(f):
    # TODO move this into Optimiser class.
    """decorator to increment the number of evaluations with each function call and log the process"""
    def wrapper(self, *args, **kwargs):
        if self.n_evaluations%self.log_interval==0:
            self.log_optimisation()
        self.n_evaluations += 1
        return f(self, *args, **kwargs)
    return wrapper


class Optimiser:
    def __init__(self, objective_function, limits, n_initial=10, budget=30, seed=None, ref_vector=None, log_dir="./log_data"):
        self.objective_function = objective_function
        self.seed = seed if seed else np.random.randint(0, 10000)
        self.limits = np.array(limits)
        self.n_initial = n_initial
        self.budget = budget
        self.n_evaluations = 0
        self.x_dims = np.shape(limits)[1]
        self.log_interval = budget

        # generate initial samples
        self.x, self.y = self.initial_evaluations(n_initial, self.x_dims, self.limits)
        self.n_objectives = self.y.shape[1]
        # TODO obj_sense currently only allows minimisation of all objecitves.
        self.obj_sense = [-1]*self.n_objectives


        # set up logging
        # if no ref_vector provided, use max of initial evaluations
        self.ref_vector = ref_vector if ref_vector else self.y.max(axis=0)
        self.log_dir = log_dir
        self.log_filename = self._generate_filename()
        if not os.path.exists(log_dir):
            os.makedirs(self.log_dir)
        self.log_data = None
        self.log_optimisation()

    def initial_evaluations(self, n_samples, x_dims, limits):
        """
        Makes the initial evaluations of the parameter space and evaluates the objective function at these locaitons
        :param n_samples: number of initial samples to make
        """
        x = np.array(lhsmdu.sample(x_dims, n_samples, randomSeed=self.seed)).T *\
                 (limits[1]-limits[0])+limits[0]
        y = self.objective_function(x)

        # update evaluation number
        self.n_evaluations += n_samples

        return x, y

    def get_next_x(self, excluded_indices: list):
        """
        Implementation required in child: method to find next parameter to evaluate in optimisation sequence.
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
        """takes one step in the optimisation by getting the next decision vector by calling get_next_x"""

        # ensures unique evaluation
        x_new = self.get_next_x()
        try_count = 0
        aa = self._already_evaluated(x_new)
        while self._already_evaluated(x_new) and try_count<3:
            x_new = self.get_next_x()
            try_count += 1

        try_count = 0
        while self._already_evaluated(x_new):
            # if repeat optimisation of the acquisition function does not produce a unique value for x_new
            excluded_indices = np.argsort(np.linalg.norm((self.x-x_new), axis=1))[:try_count+1]
            x_new = self.get_next_x(excluded_indices)
            try_count += 1
            if try_count>5:
                "This should not happen multiple times. Something is wrong here."
                RuntimeError

        # get objective function evaluation for the new point
        y_new = self.objective_function(x_new)

        # update observations with new observed point
        self.x = np.vstack((self.x, x_new))
        self.y = np.vstack((self.y, y_new))

    def log_optimisation(self):
        """
        updates dictionary of saved optimisation information and saves to disk.
        """
        try:
            #
            self.log_data["x"] = self.x
            self.log_data["y"] = self.y
            self.log_data["hypervolume"].append(self._compute_hypervolume())
        except TypeError:
            # initial information logging
            log_data = {"objective_function": inspect.getsource(self.objective_function),
                        "limits": self.limits,
                        "n_initial": self.n_initial,
                        "seed": self.seed,
                        "hypervolume": [self._compute_hypervolume()],
                        "x": self.x,
                        "y": self.y,
                        "log_dir": self.log_dir,
                        "log_filename": self.log_filename
                        }
            self.log_data = log_data

        # save log_data to file.
        log_filepath = os.path.join(self.log_dir, self.log_filename)
        with open(log_filepath, 'wb') as handle:
            pickle.dump(self.log_data, handle, protocol=2)

    def _compute_hypervolume(self):
        """
        Calcualte the current hypervolume.
        """
        if self.n_objectives > 1:
            y, comp_mat = self._get_dom_matrix(self.y, self.ref_vector)
            front_inds = self._get_front(y, comp_mat)
            hpv = FonsecaHyperVolume(self.ref_vector)
            volume = hpv.assess_non_dom_front(y[front_inds])
            return volume
        else:
            return np.min(self.y)

    def _get_dom_matrix(self, y, r=None):
        """
        Build a dominance comparison matrix between all observed solutions. Cell
        keys for the resulting matrix.
        -1: The same solution, hence identical
         0: Row dominates column.
         1: Column dominates row.
         2: row is equal to column.
         3: row and col mutually non-dominated.

        Parameters.
        -----------
        y (np.array): set of objective vectors.
        r (np.array): reference vector
        """
        if r is not None:
            yr = np.append(y, [r], axis=0)  # append the reference point at the end.
        else:
            yr = y
        n_data, n_obj = yr.shape
        redundancy = np.zeros((n_data, n_data)) - 1  # -1 means its the same solution.
        redundant = np.zeros(n_data)
        for i in range(n_data):
            for j in range(n_data):
                if i != j:
                    redundancy[i, j] = CS.compare_solutions(yr[i], yr[j], \
                                                            self.obj_sense)
        return yr, redundancy

    def _get_front(self, y, comp_mat=None, del_inds=None):
        """
        Get the Pareto front solution indices.

        Parameters.
        -----------
        y (np.array): objective vectors.
        comp_mat (np.array): dominance comparison matrix.
        del_inds (np.array): the indices to ignore.
        """
        if comp_mat is None:
            yr, comp_mat = self._get_dom_matrix(y)
        else:
            yr = y
        dom_inds = np.unique(np.where(comp_mat == 1)[0])
        if del_inds is None:
            ndom_inds = np.delete(np.arange(comp_mat.shape[0]), dom_inds)
        else:
            ndom_inds = np.delete(np.arange(comp_mat.shape[0]), np.concatenate([dom_inds, del_inds], axis=0))
        return ndom_inds

    def _generate_filename(self, repeat=0):
        """
        generates a filename from optimsation parameters, and ensures uniqueness
        :param repeat:
        :return:
        """
        objective_function = str(self.objective_function.__name__)
        optimser = self.__class__.__name__
        inital_samples= str(self.n_initial)
        n_evaluations = str(self.n_evaluations)
        seed = str(self.seed)

        filename = "of{}_opt{}_init{}_tot{}_seed{}_{:03d}".format(objective_function, optimser, inital_samples,
                                                                  n_evaluations, seed, repeat)
        while os.path.exists(os.path.join(self.log_dir, filename)):
            repeat += 1
            filename = "of{}_opt{}_init{}_tot{}_seed{}_{:03d}".format(objective_function, optimser, inital_samples,
                                                                      n_evaluations, seed, repeat)
        return filename

    def _already_evaluated(self, x_new, thresh=1e-9):
        """
        returns True if x_new is in the existing set of evaluated solutions, "same" defined by difference of thresh
        :param x_new: new parameter to be queried
        """
        # TODO do this better. Some meaningful value for thresh instead of arbitrary. Use unit hypercube
        difference_matrix = (self.x-x_new)**2
        evaluated = np.any(np.all((difference_matrix<thresh), axis=1))
        return evaluated


class BayesianOptimiser(Optimiser):
    def __init__(self, objective_function, limits, surrogate, n_initial=10, seed=None,
                 acquisition_function="saf_mu", cmaes_restarts=0):

        super().__init__(objective_function=objective_function, limits=limits, n_initial=n_initial, seed=seed)
        self.bo_steps = 0
        self.surrogate = surrogate
        # set up alpha as generator
        self.acquisition_function = getattr(acquisition_functions, acquisition_function)
        self.cmaes_restarts = cmaes_restarts

    def get_next_x(self, excluded_indices=[]):
        """
        get the next paramter to be evaluated by the objective function
        :return:
        """
        ## handle excluded observations which result in repeat sampling of similar decision vectors.
        x = self.x[[i for i in range(len(self.x)) if i not in excluded_indices]]
        y = self.y[[i for i in range(len(self.x)) if i not in excluded_indices]]

        self.surrogate.update(x, y)

        # optimise the acquisition function using cmaes algorithm for parameter space >1d, else random search
        alpha_arguments = [self.surrogate, y]
        if self.surrogate.x_dims > 1:
            seed = np.random.uniform(self.limits[0], self.limits[1], self.surrogate.x_dims).reshape(1, -1)
            res = cma.fmin(self.alpha, seed, sigma0=0.25, options={'bounds': self.limits}, restarts=self.cmaes_restarts,
                           args=alpha_arguments)
            x_new = res[0]
        else:
            n_search_points = 1000
            search_points = np.random.uniform(self.limits[0], self.limits[1], n_search_points)
            # TODO fix use of argmin to accomodate maximisation
            res_index = np.argmin([self.alpha(search_point, *alpha_arguments) for search_point in search_points])
            x_new = np.array(search_points[res_index:res_index + 1]).reshape(1, -1)

        return x_new

    def alpha(self, x_put, *args, **kwargs):
        """evaluates the predicted efficiacty of the putative solution x_put, based on predictions from teh surrogate
        and the acquisition function.

        :argument:
            x_put [np.array]: putative solution to be predicted, shape (1, n_dim)

        """
        efficiacy_put = self.acquisition_function(x_put, self.surrogate, self.y)
        return efficiacy_put


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    def test_function(x):
        try:
            return np.array([[np.sum([np.sin(xii) for xii in xi]) for xi in x],
                             [np.sum([np.cos(xii) for xii in xi]) for xi in x],
                             [np.sum([np.cos(xii)**2 for xii in xi]) for xi in x]]).T
        except:
            return np.array([np.sum([np.sin(xii) for xii in x]),
                             np.sum([np.cos(xii) for xii in x]),
                             np.sum([np.cos(xii)**2 for xii in x])])


    def plot_surrogate_vs_true(surrogate, title):
        pred_mean, pred_var = surrogate.predict(x)
        fig0 = plt.figure()
        ax0 = fig0.gca()
        ax0.plot(x, y[:,0], c="C0", alpha=0.4)
        ax0.plot(x, y[:,1], c="C1", alpha=0.4)
        ax0.plot(x, y[:,2], c="C2", alpha=0.4)
        ax0.plot(x, pred_mean[:,0], c="C0", linestyle="--", label="objective 1")
        ax0.plot(x, pred_mean[:,1], c="C1", linestyle="--", label="objective 2")
        ax0.plot(x, pred_mean[:,2], c="C2", linestyle="--", label="objective 3")
        ax0.scatter(xtr, ytr[:,0], c="C0", marker="x")
        ax0.scatter(xtr, ytr[:,1], c="C1", marker="x")
        ax0.scatter(xtr, ytr[:,2], c="C2", marker="x")
        plt.legend()
        ax0.set_title(title)


    x = np.linspace(50, 55, 50).reshape(-1, 1)
    y = test_function(x)

    xtr = np.random.uniform(50, 55, size=(10, 1))
    ytr = test_function(xtr)

    limits = [[-5], [5]]
    gp_surr_multi = Surrogate(xtr, ytr, surrogate_type="GP", multi_surrogate=True)
    opt = BayesianOptimiser(objective_function=test_function, limits=limits, surrogate=gp_surr_multi, n_initial=10, seed=None)

    opt.optimise(n_steps=1)
    pass
    opt.optimise(n_steps=1)
    pass
    # gp_surr_multi = Surrogate(x=None, y=None, surrogate_type="GP", multi_surrogate=True)
    # b = BayesianOptimiser(objective_function=test_function, limits=limits, n_initial=10, seed=10, surrogate=gp_surr_multi)

