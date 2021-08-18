import pickle
import rootpath
import os
import numpy as np
import matplotlib.pyplot as plt
from testsuite.analysis_tools import strip_problem_names
from pymoo.factory import get_performance_indicator
from testsuite.utilities import Pareto_split


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

        self.targets = self.raw_result['targets']
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
        hpv_measure = get_performance_indicator("hv", reference_point)
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

    def plot_hpv(self, axis=None, c=None, label=None):
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
        axis.plot(self.hpv_hist_x, self.hpv_history, c=c, label=label)

        # return axis if it was provided, otherwise return the figure
        if ax_provided:
            return axis
        else:
            return fig

    def plot_igd(self, axis=None, c=None, label=None):
        """
        plots the igd+ history, either on a new figure
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
            axis.set_ylabel("igd+ score")
        else:
            ax_provided = True

        # plot hpv history on axes
        c = "C0" if c is None else c
        label = "igd+ score" if label is None else label
        axis.plot(self.igd_hist_x, self.igd_history, c=c, label=label)

        # return axis if it was provided, otherwise return the figure
        if ax_provided:
            return axis
        else:
            return fig


if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    result_inst = Result(path)

    test_refpoint0 = np.linspace(0, 1.5, 50)
    test_refpoint1 = 1.5-np.linspace(0, 1.5, 50)
    test_refpoints = np.vstack((test_refpoint0, test_refpoint1)).T

    result_inst.compute_igd_history(reference_points=test_refpoints,
                                    sample_freq=1)

    test_refpoint = (np.ones(result_inst.n_obj)*3.5).reshape(1, -1)
    result_inst.compute_hpv_history(reference_point=test_refpoint,
                                    sample_freq=1)

    fig0 = result_inst.plot_hpv()
    fig0.gca().legend()

    fig1 = result_inst.plot_igd()
    fig1.gca().legend()

    fig0.show()
    fig1.show()






