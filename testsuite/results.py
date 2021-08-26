import pickle
import rootpath
import os
import numpy as np
import matplotlib.pyplot as plt
from testsuite.analysis_tools import strip_problem_names
from testsuite.utilities import SingeTargetDominatedHypervolume
from pymoo.factory import get_performance_indicator
from testsuite.utilities import Pareto_split


class ResultsContainer:
    def __init__(self, results):
        """

        :param results: [Result] or [str]
            list of either Result objects, or strings which provide
            paths to results.pkl files from which Results objects can be
            formed.
        """
        # handle either list of path strings or Result objects creating
        # a list of Results objects stored as self.results
        if isinstance(results, str):
            self.load(results)
        elif isinstance(results, list):
            if isinstance(results[0], str):
                self.results = [Result(r) for r in results]
            elif isinstance(results[0], Result):
                self.results = results
            self.iter_count = 0
            for key in self.results[0].__dict__.keys():
                setattr(self, key, self._amalgamate(key, self.results))
        else:
            raise TypeError


    def compute_hpv_history(self, reference_point, sample_freq=1):
        for result in self.results:
            result.compute_hpv_history(reference_point=reference_point,
                                       sample_freq=sample_freq)

    def compute_igd_history(self, reference_points, sample_freq=1):
        for result in self.results:
            result.compute_igd_history(reference_points=reference_points,
                                       sample_freq=sample_freq)

    @staticmethod
    def _amalgamate(name, results):
        return [getattr(r, name) for r in results]

    def __getitem__(self, item):
        return self.results[item]

    def __next__(self):
        self.iter_count +=1
        return self[self.iter_count-1]

    def sorted(self, attribute: str, reverse=False):
        """
        calls self.sort, but returns self.
        :param attribute:
        :param reverse:
        :return:
        """
        self.sort(attribute, reverse=reverse)
        return self


    def sort(self, attribute: str, reverse=False):
        """
        replaces self.results with copy of itself sorted  by the
        attribute passed as attribute
        :param attribute: str
            string containing the name of the attribute by which to sort
            self.results
        """
        # check attribute exists
        assert attribute in self.__dict__.keys(), \
            "Attribute does not exist, cannot sort. Please pass an attribute "\
            "from: {}".format(self.__dict__.keys())

        result_atts = [getattr(r, attribute) for r in self.results]
        order = np.argsort(result_atts)
        if not reverse:
            for key in self.results[0].__dict__.keys():
                setattr(self, key, [getattr(self, key)[n] for n in order])
        else:
            for key in self.results[0].__dict__.keys():
                setattr(self, key, [getattr(self, key)[n] for n in order[::-1]])

    def save(self, path):
        with open(path, 'wb') as outfile:
            pickle.dump(self, outfile)
            print(f"Saved ResultsContainer to: ", path)

    def load(self, path):
        with open(path, 'rb') as infile:
            replacement = pickle.load(infile)
        for key, value in replacement.__dict__.items():
            setattr(self, key, value)

    def plot_hpv(self, axis=None, c="C0"):
        # create axes if one is not provided
        if axis is None:
            ax_provided = False
            fig = plt.figure(figsize=[10,5])
            axis = fig.gca()
            axis.set_xlabel("Function evaluations")
            axis.set_ylabel("Dominated Hypervolume")
        else:
            ax_provided = True

        # handles different colours provided
        if not isinstance(c, str):
            try:
                # colour is iterable
                for result, ci in zip(self.results, c):
                    axis = result.plot_hpv(axis, c=ci, label=result.seed,
                                           plot_kwargs={
                                               'alpha': 0.3,
                                               'linestyle': ':'
                                           })
            except TypeError:
                # colour is not iterable
                for result in self.results:
                    axis = result.plot_hpv(axis, c=c, label=result.seed,
                                           plot_kwargs = {
                                               'alpha': 0.3,
                                               'linestyle': ':'
                                           })
        else:
            # colour is string and thus should not be iterated
            for result in self.results:
                axis = result.plot_hpv(axis, c=c, label=result.seed,
                                       plot_kwargs = {
                                           'alpha': 0.3,
                                           'linestyle': ':'
                                       })
                pass
        axis.plot(result.hpv_hist_x,
                  np.median([r.hpv_history for r in self.results], axis=0),
                  c=c,
                  linewidth=2,
                  label="median")
        axis.legend()

        if ax_provided:
            return axis
        else:
            return fig

    def plot_igd(self, axis=None, c="C0"):
        # create axes if one is not provided
        if axis is None:
            ax_provided = False
            fig = plt.figure(figsize=[10,5])
            axis = fig.gca()
            axis.set_xlabel("Function evaluations")
            axis.set_ylabel("Dominated Hypervolume")
        else:
            ax_provided = True

        # handles different colours provided
        if not isinstance(c, str):
            try:
                # colour is iterable
                for result, ci in zip(self.results, c):
                    axis = result.plot_igd(axis, c=ci, label=result.seed,
                                           plot_kwargs = {
                                               'alpha': 0.3,
                                               'linestyle': ':'
                                           })
            except TypeError:
                # colour is not iterable
                for result in self.results:
                    axis = result.plot_igd(axis, c=c, label=result.seed,
                                           plot_kwargs = {
                                               'alpha': 0.3,
                                               'linestyle': ':'
                                           })
        else:
            # colour is string and thus should not be iterated
            for result in self.results:
                axis = result.plot_igd(axis, c=c, label=result.seed,
                                       plot_kwargs = {
                                           'alpha': 0.3,
                                           'linestyle': ':'
                                       })
                pass

        axis.plot(result.igd_hist_x,
                  np.median([r.igd_history for r in self.results], axis=0),
                  c=c,
                  linewidth=2,
                  label="median")
        axis.legend()

        if ax_provided:
            return axis
        else:
            return fig

    def get_intervals(self, measure: str, intervals: list):
        """
        calls get_intervals for all Results in self.results and returns
        the median, and inter-quartile ranges of all results at the
        intervals provided.
        :param measure: str
                string dictating which score to return, should be one of
                'igd' or 'hpv'.
        :param intervals: list(int)
                which steps to return the measured score
        :return: tuple(np.ndarray(float), np.ndarray(float))
                (median of measure over all self.results at intervals,
                IQR of measure over all self.results at intervalss)
        """
        scores = [r.get_intervals(measure, intervals) for r in self.results]
        return np.median(scores, axis=0), np.percentile(scores, [0.25, 0.75], axis=0)


class Result:
    """
    class for containing and handling optimisation results
    """

    def __init__(self, result_path: str):
        """
        loads the result data in the pickle file provided by
        results_path and creates class attributes to access formatted
        aspects of the data
        :param result_path: str
                string containing relative, or absolute path to
                results.pkl file.
        """
        # store relative path from project home directory
        self.raw_path = os.path.abspath(result_path)[len(rootpath.detect())+1:]
        self.n_prob, self.n_obj, self.n_dim = strip_problem_names(
            self.raw_path.split('/')[-4])

        # load raw result data
        self.raw_result = self.load_result_from_path(
            os.path.join(rootpath.detect(), self.raw_path))

        # extract information regarding state of optimisation
        self.seed = self.raw_result['seed']
        self.n_initial = self.raw_result['n_initial']
        self.budget = self.raw_result['budget']
        self.n_evaluations = self.raw_result['n_evaluations']
        self.n_optimisation_steps = self.n_evaluations - self.n_initial
        self.train_time = self.raw_result['train_time']

        # extract optimisation step information and assure consistency
        self.x = self.raw_result['x']
        self.y = self.raw_result['y']
        assert self.x.shape[0] == self.y.shape[0]
        self.p = self.raw_result['p']
        self.d = self.raw_result['d']
        assert (self.p.shape[0] + self.d.shape[0]) == self.x.shape[0]
        assert self.x.shape[0] == self.n_evaluations
        assert self.n_dim == self.x.shape[1]
        assert self.n_obj == self.y.shape[1]

        try:
            self.targets = self.raw_result['targets']
        except KeyError:
            # undirected case
            self.targets = None

        try:
            self.target_history = self.raw_result['target_history']
        except KeyError:
            self.target_history = {self.n_initial: self.targets}

        # initially not computed
        self.igd_history = None     # set by calling self.compute_igd_history()
        self.igd_hist_x = None      # set by calling self.compute_igd_history()
        self.igd_refpoints = None   # set by calling self.compute_igd_history()
        self.hpv_history = None     # set by calling self.compute_hpv_history()
        self.hpv_hist_x = None      # set by calling self.compute_hpv_history()
        self.hpv_refpoint = None    # set by calling self.compute_hpv_history()

    @staticmethod
    def load_result_from_path(path):
        """
        unpickles results.pkl file at the location specified by path
        :param path: str
                path to the desired results.pkl file
        :return: dict
                dictionary containing the result of the optimisation in
                question.
        """
        with open(path, "rb") as infile:
            result = pickle.load(infile)
        return result

    def compute_igd_history(self, reference_points, sample_freq=1):
        """
        computes the history of the igd+ score over the course of the
        optimisation, sampled at the frequency dictated by sample_freq
        :param reference_points: np.ndarray shape(n_points, n_obj)
            set of reference points lying on the true Pareto front in
            the objective space against which to measure the igd+score.
        :param sample_freq: int
                frequency at which to sample the igd+ score. defaults to
                1 to sample for every stage.
        """
        # record reference points used
        assert reference_points.ndim == 2
        self.igd_refpoints = reference_points

        # generate measurement tool once for all stages.
        igd_measure = get_performance_indicator("igd+", reference_points)
        self.igd_history, self.igd_hist_x = self._compute_measure_history(
            measure=igd_measure, sample_freq=sample_freq)

    def compute_hpv_history(self, reference_point, sample_freq=1):
        """
        computes the history of the dominated hypervolume score over the
        course of the optimisation, sampled at the frequency dictated by
        sample_freq
        :param reference_point: np.ndarray shape(1, n_obj)
            reference point for the dominated hypervolume computation
        :param sample_freq: int
                frequency at which to sample the igd+ score. defaults to
                1 to sample for every stage.
        """
        # record reference points used
        assert reference_point.ndim == 2
        self.hpv_refpoint = reference_point

        # generate measurement tool once for all stages.
        # hpv_measure = get_performance_indicator("hv", reference_point)
        hpv_measure = SingeTargetDominatedHypervolume(reference_point)

        self.hpv_history, self.hpv_hist_x = self._compute_measure_history(
            measure=hpv_measure, sample_freq=sample_freq)

    def _compute_measure_history(self, measure, sample_freq: int = 1):
        """
        generic utility function to compute the measurement provided by
        measure, over the course of the optimisation history, at
        intervals of sample_freq
        :param measure: pymoo.performance_indicator
        :param sample_freq:
        :return: tuple (np.ndarray, np.ndarray)
            history of the measurement score over the optimisation, at
            the sample frequency specified, as well as the corresponding
            steps for these measurements.
        """
        # steps at intervals determined by sample_freq, bookended by the
        # initial and final evaluations
        steps = range(self.n_initial+sample_freq, self.n_evaluations,
                      sample_freq)
        history = np.zeros(len(steps)+2)

        # initial state
        history[0] = measure.calc(self.y[:self.n_initial])
        # final state
        history[-1] = measure.calc(self.y)
        # step states
        for i, step in enumerate(steps):
            # check all points are Pareto optimal. May not be required
            yi = Pareto_split(self.y[:step])[0]
            history[i+1] = measure.calc(yi)

        hist_x = np.asarray([self.n_initial]+list(steps)+[self.n_evaluations])
        return history, hist_x

    def plot_hpv(self, axis=None, c=None, label=None, plot_kwargs={}):
        """
        plots the dominated hypervolume history, either on a new figure
        or adds to the axes provided
        :param axis: matplotlib.pyplot.axes
            axis object onto which to add the plot of the hpv.
        :param c: str
            string containing the colour to be plotted. Defaults to "C0"
        :param label: str
            string containing the label to be added to the axis for the
            plotted hypervolume
        :return: matplotlib.pyplot.axes OR matplotlib.pyplot.figure
            returns either the provided axis, or a new figure
        """

        # create axes if one is not provided
        if axis is None:
            ax_provided = False
            fig = plt.figure(figsize=[10,5])
            axis = fig.gca()
            axis.set_xlabel("Function evaluations")
            axis.set_ylabel("Dominated Hypervolume")
        else:
            ax_provided = True

        # plot hpv history on axes
        c = "C0" if c is None else c
        label = "dominated hypervolume" if label is None else label
        try:
            axis.plot(self.hpv_hist_x, self.hpv_history, c=c, label=label,
                      **plot_kwargs)
        except Exception as e:
            pass

        # return axis if it was provided, otherwise return the figure
        if ax_provided:
            return axis
        else:
            return fig

    def plot_igd(self, axis=None, c=None, label=None, plot_kwargs={}):
        """
        plots the igd+ history, either on a new figure or adds to the
        axes provided
        :param axis: matplotlib.pyplot.axes
            axis object onto which to add the plot of the hpv.
        :param c: str
            string containing the colour to be plotted. Defaults to "C0"
        :param label: str
            string containing the label to be added to the axis for the
            plotted hypervolume
        :return: matplotlib.pyplot.axes OR matplotlib.pyplot.figure
            returns either the provided axis, or a new figure
        """

        # create axes if one is not provided
        if axis is None:
            ax_provided = False
            fig = plt.figure(figsize=[10, 5])
            axis = fig.gca()
            axis.set_xlabel("Function evaluations")
            axis.set_ylabel("igd+ score")
        else:
            ax_provided = True

        # plot hpv history on ax
        c = "C0" if c is None else c
        label = "igd+ score" if label is None else label
        axis.plot(self.igd_hist_x, self.igd_history, c=c, label=label,
                  **plot_kwargs)

        # return axis if it was provided, otherwise return the figure
        if ax_provided:
            return axis
        else:
            return fig

    def plot_front(self, axis=None, c=None, label=None, plot_kwargs={}):
        # create axes if one is not provided
        if axis is None:
            ax_provided = False
            fig = plt.figure(figsize=[10, 5])
            axis = fig.gca()
            axis.set_xlabel("Function evaluations")
            axis.set_ylabel("igd+ score")
        else:
            ax_provided = True

        # plot hpv history on ax
        c = "C0" if c is None else c
        label = "igd+ score" if label is None else label
        axis.scatter(*self.d.T, c=c, label=label,
                     **plot_kwargs, alpha=0.2)
        axis.scatter(*self.p.T, c=c, label=label,
                  **plot_kwargs)

        # return axis if it was provided, otherwise return the figure
        if ax_provided:
            return axis
        else:
            return fig

    def get_interval(self, measure: str, interval: int):
        """
        returns the scored defined by measure at the interval step of
        the optimisation process.
        :param measure: str
                string dictating which score to return, should be one of
                'igd' or 'hpv'.
        :param interval: int
                which step to return the measured score
        :return: float
                score defined by measure at the interval step of the
                optimisation
        """
        if getattr(self, f"{measure}_history") is None:
            raise AttributeError(f"{measure}_history not yet calculated. "
                                 f"Run compute {measure}_history first.")

        if measure not in ("igd", "hpv"):
            raise ValueError("invlaid measure supplied. Measure should be one "
                             "of 'igd+' or 'hpv'")
        else:
            measure_history = getattr(self, f"{measure}_history")
            measure_steps = getattr(self, f"{measure}_hist_x")
            index = np.argmin(abs(np.array(measure_steps)-interval))
            return measure_history[index]

    def get_intervals(self, measure: str, intervals: list):
        """
        returns the scored defined by measure at the intervals steps of
        the optimisation process. Calls get_interval for each interval
        in intervals.
        :param measure: str
                string dictating which score to return, should be one of
                'igd' or 'hpv'.
        :param intervals: list(int)
                which steps to return the measured score
        :return: np.ndarray(float)
                score defined by measure at the interval step of the
                optimisation
        """
        return np.array([self.get_interval(measure, i) for i in intervals])


if __name__ == "__main__":
    # import copy
    # results_dir = os.path.join(rootpath.detect(),
    #     'experiments/directed/data/wfg1_2obj_3dim/log_data/',
    #     'OF_objective_function__opt_DirectedSaf__ninit_10__surrogate_MultiSurrogateGP__ei_False__target_1p68_1p09__w_0p5/')
    # assert os.path.isdir(results_dir)
    # result_paths = [os.path.join(results_dir, d) for d in
    #                 os.listdir(results_dir) if d[-11:] == "results.pkl"]
    # result_insts = [Result(path) for path in result_paths]
    #
    # result_inst = result_insts[0]
    #
    #
    # test_refpoint0 = np.linspace(0, 1.5, 50)
    # test_refpoint1 = 1.5-np.linspace(0, 1.5, 50)
    # test_refpoints = np.vstack((test_refpoint0, test_refpoint1)).T
    #
    # result_inst.compute_igd_history(reference_points=test_refpoints,
    #                                 sample_freq=1)
    #
    # test_refpoint = (np.ones(result_inst.n_obj)*3.5).reshape(1, -1)
    # result_inst.compute_hpv_history(reference_point=test_refpoint,
    #                                 sample_freq=1)
    #
    # fig0 = result_inst.plot_hpv()
    # fig0.gca().legend()
    #
    # fig1 = result_inst.plot_igd()
    # fig1.gca().legend()
    # plt.show()
    #
    # container_inst = ResultsContainer(results=result_insts)
    # container_inst.compute_igd_history(reference_points=test_refpoints,
    #                                 sample_freq=1)
    # container_inst.compute_hpv_history(reference_point=test_refpoint,
    #                                 sample_freq=1)
    #
    # prev = copy.deepcopy(container_inst)
    # container_inst.sort("seed")
    #
    # # check attributes are sorted accordingly
    # for i, p in enumerate(prev.results):
    #     for j, c in enumerate(container_inst.results):
    #         if p.seed == c.seed:
    #             np.testing.assert_array_equal(p.y, c.y)
    #             np.testing.assert_array_equal(p.x, c.x)
    #
    # container_inst.save("./test_save")
    #
    # container_loaded = ResultsContainer("./test_save")
    #
    # for method in dir(container_inst):
    #     print(method)
    #
    # fig_cont = container_inst.plot_hpv()
    # fig_cont = container_inst.plot_igd(c="C1")
    # plt.show()
    # pass

    results_dir = os.path.join(rootpath.detect(), 'experiments/directed/data/wfg1_2obj_3dim/log_data/OF_objective_function__opt_DirectedSaf__ninit_10__surrogate_MultiSurrogateGP__ei_False__target_0p35_3p14__w_0p5')
    assert os.path.isdir(results_dir)

    result_paths = [os.path.join(results_dir, path) for path in
                    os.listdir(results_dir) if path[-11:] == "results.pkl"]
    results = ResultsContainer(result_paths)
    results.sort("seed")
    results.save("./test_save")
    results_loaded = ResultsContainer("./test_save")
    pass