import json
import pickle
import rootpath
import os
import numpy as np
import matplotlib.pyplot as plt
from testsuite.analysis_tools import strip_problem_names
from testsuite.analysis_tools import get_target_igd_refpoints
from pymoo.factory import get_performance_indicator
from testsuite.utilities import Pareto_split, DifferenceOfHypervolumes
from copy import deepcopy


class ResultsContainer:
    def __init__(self, results):
        """
        takes a "results" argument and creates a ResultsContainer object
        based on this argument.
        :param results: 1) str - "path/to/directory/"
                        where directory contains results.pkl files
                        2) str - "path/to/saved/ResultsContainer"
                        where results is a path to a previous instance
                        of ResultsContainer saved using save_container
                        3) [str]
                        list of strings to a sequence of results.pkl
                        files
                        4) [Result]
                        list of instances of the Results class to be
                        contained into a ResultsContainer

        :raises
            TypeError: results passed as neither str, [str] or [Results]
        """
        self.reference = None
        if isinstance(results, str):
            if os.path.isdir(results):
                # results provided is a directory path- load all results
                # files from within that dir
                result_paths = [os.path.join(results, path) for path in
                                os.listdir(results) if
                                path[-11:] == "results.pkl"]
                self.results = [Result(r) for r in result_paths]

                self.iter_count = 0
                for key in self.results[0].__dict__.keys():
                    setattr(self, key, self._amalgamate(key, self.results))
            else:
                # results provided is a path to a saved instance of
                # ResultsContainer
                self.load_container(results)

        elif isinstance(results, list):
            if isinstance(results[0], str):
                # results provided is a list of paths to results files
                assert np.all([r[-11:] == "results.pkl" for r in results])
                self.results = [Result(r) for r in results]

            elif isinstance(results[0], Result):
                # results provided is a list of Results objects
                self.results = results

            self.iter_count = 0
            for key in self.results[0].__dict__.keys():
                setattr(self, key, self._amalgamate(key, self.results))
        else:
            print("results provided to ResultsContainer are not recognised.")
            raise TypeError

    def add_reference_data(self, directory_path):
        """
        adds data from an alternative optimisation to use as reference
        comparison.

        :param directory_path: str
            path to the directory containing the corresponding results
            used as a reference for comparison.
        """
        reference_container = ResultsContainer(directory_path)

        ref_results = reference_container.results
        ref_seeds = reference_container.seed

        # check all seeds are present
        try:
            assert set(self.seed) == set(ref_seeds), \
                "reference data does not contain all required random seeds " \
                "and should not be used for reference."
        except AssertionError:
            a = set(self.seed)
            b = set(ref_seeds)
            pass

        # ensure reference results are in the same seed order as self.results
        # (don't question this line of code, it came in a momentary state of
        # enlightenment and it works)
        self.reference = [[ref_results[j] for j in np.argsort(ref_seeds)][i]
                          for i in np.argsort(self.seed).argsort()]

    def get_hpv_refpoint(self, p=None, targetted=False):
        """
        computes an appropriate reference point for the dominated
        hypervolume computation from the provided non-dominated points
        p, or self.p if p is not provided. Reference point is computed
        as the maximum observed in each objective, for the set of
        non-dominated solutions.
        :param p: np.ndarray, (n_points, n_objectives)
            array of non-dominated points
        :return: np.ndarray, (1, n_objectives)
            single reference point form which to calculate dominated
            hypervolume using
        """
        if targetted:
            # use target as reference points for directed optimisation
            assert all([np.array_equal(self.targets[0], t)
                        for t in self.targets])
            rp = self.results[0].targets.reshape(-1)
        else:
            if p is None:
                p = np.vstack(self.p)
            else:
                # check all p are truly non-dominated
                p = Pareto_split(p)

            try:
                # include points from reference_points
                p_ref = np.vstack([r.p for r in self.reference])
                p = np.vstack((p, p_ref))
            except TypeError:
                pass

            rp =  np.max(p, axis=0).reshape(1, -1)

        self.hpv_refpoint = rp
        return rp

    def get_igd_refpoints(self, all_refpoints):
        """
        looks up the reference points for igd+ calculation from a json
        file containing a dictionary of reference points with keys in
        the form wfg{a}_{b}obj_{c}dim where a, b and c are the integers
        stored in self.n_prob, self.n_obj and self.n_dim. If target is
        present then the reference points are downscaled to those
        dominated by the target.
        :param all_refpoints: np.ndarray
            array of evenly distributed reference points on the Pareto
            front of the objective fucntion
        :return: np.ndarray. (n_points, n_obj)
            array of reference points for targeted optimisation
        """
        # ensure problem configurations are identical across all Results
        for attribute in ['n_prob', 'n_obj', 'n_dim', 'targets']:
            assert np.all([getattr(self.results[0], attribute)
                           == getattr(r, attribute) for r in self.results])
            # ensure problem configurations are identical across
            # references Results, no targets in references
            if self.reference is not None and attribute != 'targets':
                try:
                    assert np.all([getattr(self.results[0], attribute) ==
                                   getattr(r, attribute) for r in self.reference])
                except AssertionError:
                    pass

        # ensure correct formatting
        all_refpoints = np.asarray(all_refpoints)

        # down-sample to targeted reference points if target is provided
        if self.targets[0] is not None:
            return get_target_igd_refpoints(self.targets[0], all_refpoints)[0]
        else:
            return all_refpoints

    def compute_hpv_history(self, reference_point=None, sample_freq=1):
        if reference_point is None:
            reference_point = self.get_hpv_refpoint(targetted=True)

        for result in self.results:
            result.compute_hpv_history(reference_point=reference_point,
                                       sample_freq=sample_freq)
        try:
            for result in self.reference:
                result.compute_hpv_history(reference_point=reference_point,
                                           sample_freq=sample_freq)
        except TypeError:
            assert self.reference is None
            # no reference points present

        self.hpv_history = np.vstack([r.hpv_history for r in self.results])
        self.hpv_hist_x = self.results[0].hpv_hist_x

        # TODO: this crashes if no reference data present: fix.
        self.hpvref_history = np.vstack([r.hpv_history
                                         for r in self.reference])
        self.hpvref_hist_x = self.reference[0].hpv_hist_x

    def compute_igd_history(self, reference_points, sample_freq=1):

        reference_points = self.get_igd_refpoints(
            all_refpoints=reference_points)

        for result in self.results:
            result.compute_igd_history(reference_points=reference_points,
                                       sample_freq=sample_freq)
        try:
            for result in self.reference:
                result.compute_igd_history(reference_points=reference_points,
                                           sample_freq=sample_freq)
        except AttributeError:
            assert self.reference is None
            # no reference points present

        self.igd_history = np.vstack([r.igd_history for r in self.results])
        self.igd_hist_x = self.results[0].igd_hist_x

        # TODO: this crashes if no reference data present: fix.
        self.igdref_history = np.vstack([r.igd_history
                                         for r in self.reference])
        self.igdref_hist_x = self.reference[0].igd_hist_x

    def compute_doh_history(self, reference_point=None, sample_freq=1):
        if reference_point is None:
            reference_point = self.get_hpv_refpoint(targetted=True)

        for result in self.results:
            result.compute_doh_history(reference_point=reference_point,
                                       target=self.targets[0].reshape(-1),
                                       sample_freq=sample_freq)
        try:
            for result in self.reference:
                result.compute_doh_history(reference_point=reference_point,
                                           target=self.targets[0].reshape(-1),
                                           sample_freq=sample_freq)
        except TypeError:
            assert self.reference is None
            # no reference points present

        self.doh_history = np.vstack([r.doh_history for r in self.results])
        self.doh_hist_x = self.results[0].doh_hist_x

        # TODO: this crashes if no reference data present: fix.
        self.dohref_history = np.vstack([r.doh_history
                                         for r in self.reference])
        self.dohref_hist_x = self.reference[0].doh_hist_x

    @staticmethod
    def _amalgamate(name, results):
        """
        collects named attributes from all results, returning them as a
        list.
        :param name:
        :param results:
        :return:
        """
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
        duplicate_instance = deepcopy(self)
        duplicate_instance.sort(attribute, reverse=reverse)
        return duplicate_instance

    def sort(self, attribute: str, reverse=False):
        """
        replaces self.results with copy of itself sorted  by the
        attribute passed as attribute
        :param reverse: bool
            if True then reverse sorting otherwise sort normally
        :param attribute: str
            string containing the name of the attribute by which to sort
            self.results
        """
        # check attribute exists
        assert attribute in self.__dict__.keys(), \
            "Attribute does not exist, cannot sort. Please pass an attribute "\
            "from: {}".format(self.__dict__.keys())

        result_atts = [getattr(r, attribute) for r in self.results]

        if reverse is False:
            order = np.argsort(result_atts)
        elif reverse is True:
            order = np.argsort(result_atts)[::-1]
        else:
            raise TypeError

        for key in self.results[0].__dict__.keys():
            setattr(self, key, [getattr(self, key)[n] for n in order])

        # sort self.results
        self.results = [self.results[i] for i in order]
        try:
            self.reference = [self.reference[i] for i in order]
        except TypeError:
            # self.reference not yet set, ignore
            assert self.reference is None
            pass

    def save(self, path):
        with open(path, 'wb') as outfile:
            pickle.dump(self, outfile)
            print(f"Saved ResultsContainer to: ", path)

    def load_container(self, path):
        with open(path, 'rb') as infile:
            replacement = pickle.load(infile)
        for key, value in replacement.__dict__.items():
            setattr(self, key, value)

    def plot_igd(self, axis=None, c="C0", reference=False,
                 show_individuals=False):
        if reference:
            attribute = self.igdref_history
            attribute_x = self.igdref_hist_x
        else:
            attribute = self.igd_history
            attribute_x = self.igd_hist_x

        return self._utility_plotter(axis=axis,
                                     attribute=attribute,
                                     attribute_x=attribute_x,
                                     show_individuals=show_individuals,
                                     ylabel='igd+',
                                     c=c)

    def plot_doh(self, axis=None, c="C0", reference=False,
                 show_individuals=False):
        if reference:
            attribute = self.dohref_history
            attribute_x = self.dohref_hist_x
        else:
            attribute = self.doh_history
            attribute_x = self.doh_hist_x

        return self._utility_plotter(axis=axis,
                                     attribute=attribute,
                                     attribute_x=attribute_x,
                                     show_individuals=show_individuals,
                                     ylabel='doh',
                                     c=c)

    def plot_hpv(self, axis=None, c="C0", reference=False,
                 show_individuals=False):
        if reference:
            attribute = self.hpvref_history
            attribute_x = self.hpvref_hist_x
        else:
            attribute = self.hpv_history
            attribute_x = self.hpv_hist_x

        return self._utility_plotter(axis=axis,
                                     attribute=attribute,
                                     attribute_x=attribute_x,
                                     show_individuals=show_individuals,
                                     ylabel='Dominated Hypervolume',
                                     c=c)

    def _utility_plotter(self, attribute, attribute_x, axis, show_individuals,
                         c, ylabel):
        """
        shared plot utility for plotting perofmance measures for both
        dominated hypervolume and igd+.
        :param attribute:
        :param attribute_x:
        :param axis:
        :param show_individuals:
        :param c:
        :return:
        """

        # create axes if one is not provided
        if axis is None:
            ax_provided = False
            fig = plt.figure(figsize=[10, 5])
            axis = fig.gca()
        else:
            ax_provided = True

        axis.set_xlabel("Function evaluations")
        axis.set_ylabel(ylabel)

        # handles different colours provided
        if show_individuals:
            if not isinstance(c, str):
                try:
                    # colour is iterable
                    for result, ci in zip(attribute, c):
                        axis.plot(attribute_x, result, c=ci, alpha=0.3,
                                  linestyle=':')

                except TypeError:
                    # colour is not iterable
                    for result in attribute:
                        axis.plot(attribute_x, result, c=c, alpha=0.3,
                                  linestyle=':')
            else:
                # colour is string and thus should not be iterated
                for result in attribute:
                    axis.plot(attribute_x, result, c=c, alpha=0.3,
                              linestyle=':')

        axis.plot(attribute_x, np.median(attribute, axis=0),
                  c=c,
                  linewidth=2,
                  label="median")

        q3, q1 = np.percentile(attribute, [75, 25], axis=0)
        axis.fill_between(attribute_x,
                          q1,
                          q3,
                          color=c,
                          alpha=0.25,
                          linewidth=2,
                          label="iqr")
        axis.legend()

        if ax_provided:
            return axis
        else:
            return fig


    def get_intervals(self, measure: str, intervals: list, reference: bool=False):
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
                IQR of measure over all self.results at intervals)
        """
        if not reference:
            scores = [r.get_intervals(measure, intervals)
                      for r in self.results]
        else:
            scores = [r.get_intervals(measure, intervals)
                      for r in self.reference]

        return np.median(scores, axis=0), np.percentile(scores,
                                                        [0.25, 0.75],
                                                        axis=0)


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
        self.seed = int(self.raw_result['seed'])
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

        # initially not computed history of assessment measures over
        # course of optimisation
        # inverted generational distance plus
        self.igd_history = None     # set by calling self.compute_igd_history()
        self.igd_hist_x = None      # set by calling self.compute_igd_history()
        self.igd_refpoints = None   # set by calling self.compute_igd_history()
        # dominated hypervolume
        self.hpv_history = None     # set by calling self.compute_hpv_history()
        self.hpv_hist_x = None      # set by calling self.compute_hpv_history()
        self.hpv_refpoint = None    # set by calling self.compute_hpv_history()
        # difference of hypervolumes
        self.doh_hist = None      # set by calling self.compute_doh_history()
        self.doh_hist_x = None      # set by calling self.compute_doh_history()
        self.doh_refpoint = None    # set by calling self.compute_doh_history()

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
        # configure and save reference points
        if reference_point.ndim == 2:
            reference_point = reference_point.reshape(-1)
        assert reference_point.shape[0] == self.n_obj
        self.hpv_refpoints = reference_point

        # generate measurement tool once for all stages.
        hpv_measure = get_performance_indicator("hv",
                                                ref_point=reference_point)
        self.hpv_history, self.hpv_hist_x = self._compute_measure_history(
            measure=hpv_measure, sample_freq=sample_freq, invert=True)

    def compute_doh_history(self, reference_point, target, sample_freq=1):
        """
        computes the history of the difference of hypervolume score over the
        course of the optimisation, sampled at the frequency dictated by
        sample_freq
        :param reference_point: np.ndarray shape(1, n_obj)
            reference point for the dominated hypervolume computation
        :param sample_freq: int
                frequency at which to sample the igd+ score. defaults to
                1 to sample for every stage.
        """
        # configure and save reference points
        if reference_point.ndim == 2:
            reference_point = reference_point.reshape(-1)
        assert reference_point.shape[0] == self.n_obj
        self.doh_refpoints = reference_point

        # generate measurement tool once for all stages.
        doh_measure = DifferenceOfHypervolumes(target, reference_point)
        self.doh_history, self.doh_hist_x = self._compute_measure_history(
            measure=doh_measure, sample_freq=sample_freq, invert=True)

    def _compute_measure_history(self, measure, sample_freq: int = 1,
                                 invert: bool = False):
        """
        generic utility function to compute the measurement provided by
        measure, over the course of the optimisation history, at
        intervals of sample_freq
        :param measure: pymoo.performance_indicator
        :param sample_freq: int
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

        if invert:
            y = self.y*-1
        else:
            y = self.y

        # initial state
        history[0] = measure.calc(y[:self.n_initial])
        # final state
        history[-1] = measure.calc(y)
        # step states
        for i, step in enumerate(steps):
            # check all points are Pareto optimal. May not be required
            yi = Pareto_split(y[:step])[0]
            history[i+1] = measure.calc(yi)

        hist_x = np.asarray([self.n_initial]+list(steps)+[self.n_evaluations])
        return history, hist_x

    def plot_doh(self, axis=None, c=None, label=None, plot_kwargs={}):
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
            fig = plt.figure(figsize=[10, 5])
            axis = fig.gca()
            axis.set_xlabel("Function evaluations")
            axis.set_ylabel("Difference of Hypervolumes")
        else:
            ax_provided = True

        # plot hpv history on axes
        c = "C0" if c is None else c
        label = "dominated hypervolume" if label is None else label
        try:
            axis.plot(self.doh_hist_x, self.doh_history, c=c, label=label,
                      **plot_kwargs)
        except Exception as e:
            pass

        # return axis if it was provided, otherwise return the figure
        if ax_provided:
            return axis
        else:
            return fig

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

    def plot_front(self, axis=None, c=None, label=None, **kwargs):
        """
        plots the summary attainment front from points self.p
        :param axis: matplotlib.pyplot.axis
            axis object onto which to plot the front. If None then
            matplotlib.pyplot.figure is created.
        :param c: str
            colour argument to plot points
        :param label: str
            string to label the plot
        :param plot_kwargs: dict
            dictionary of keyword arguments to pas to
            matplotlib.pyplot.scatter
        :return: matplotlib.pyplot.axis, matplotlib.pyplot.figure
            either returns the supplied axis, or the new figure created
            if no axis is supplied.
        """
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
                     **kwargs, alpha=0.2)
        axis.scatter(*self.p.T, c=c, label=label,
                  **kwargs)

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

        if measure not in ("igd", "hpv", "doh"):
            raise ValueError("invlaid measure supplied. Measure should be one "
                             "of 'igd+', 'doh' or 'hpv'")
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
    results_dir = os.path.join(
        rootpath.detect(),
        'experiments/directed/data/wfg1_2obj_3dim/log_data/OF_objective_'
        'function__opt_DirectedSaf__ninit_10__surrogate_MultiSurrogateGP__'
        'ei_False__target_0p35_3p14__w_0p5')
    assert os.path.isdir(results_dir)

    reference_dir = os.path.join(
        rootpath.detect(),
        'experiments/directed/data_undirected_comp/wfg1_2obj_3dim/log_data/'
        'OF_objective_function__opt_Saf__ninit_10__surrogate_'
        'MultiSurrogateGP__ei_False')
    assert os.path.isdir(reference_dir)

    results_container = ResultsContainer(results_dir)
    results_container.add_reference_data(reference_dir)

    rp_path = os.path.join(rootpath.detect(),
                           'experiments/directed/template/targets/reference_points')
    with open(rp_path, "r") as infile:
        rp_D = json.load(infile)
    rp = rp_D['wfg1_2obj_3dim']
    results_container.compute_igd_history(reference_points=rp)
    results_container.compute_hpv_history()
    results_container.compute_doh_history()


    fig = results_container.plot_hpv()
    ax = fig.gca()
    results_container.plot_hpv(axis=ax, reference=True, c="C3")

    fig1 = results_container.plot_igd()
    ax1 = fig1.gca()
    results_container.plot_igd(axis=ax1, reference=True, c="C3")

    fig2 = results_container.plot_doh()
    ax = fig2.gca()
    results_container.plot_doh(axis=ax, reference=True, c="C3")

    plt.show()
    print("hello")
    pass

    # results_container.save("./test_save")
    # rc_restore = ResultsContainer("./test_save")
    # pass

    # target_file = os.path.join(rootpath.detect(),
    #                            "experiments/directed/template/targets/targets")
    # with open(target_file, 'r') as infile:
    #     targets = json.load(infile)
    #
    # igdref_file = os.path.join(rootpath.detect(),
    #                            "experiments/directed/template/targets/reference_points")
    # with open(igdref_file, 'r') as infile:
    #     IGD_REFPOINTS = json.load(infile)
    #
    # functions = sorted(os.listdir(
    #     os.path.join(rootpath.detect(), 'experiments/directed/data/')))
    # function_paths = {prob: [
    #     os.path.join(rootpath.detect(), 'experiments/directed/data/', prob,
    #                  'log_data/'),
    #     os.path.join(rootpath.detect(),
    #                  'experiments/directed/data_undirected_comp/', prob,
    #                  'log_data/', os.listdir(os.path.join(rootpath.detect(),
    #                                                       'experiments/directed/data_undirected_comp/',
    #                                                       prob, 'log_data/'))[
    #                      0])] for prob in functions}
    #
    # problem = "wfg2_2obj_6dim"
    # wfg2_2obj_6dim_r = {}
    # for i, path in enumerate(os.listdir(function_paths[problem][0])):
    #     wfg2_2obj_6dim_r[i] = ResultsContainer(
    #         os.path.join(function_paths[problem][0], path))
    #     wfg2_2obj_6dim_r[i].add_reference_data(function_paths[problem][1])
    #
    #
    # problem = "wfg2_2obj_6dim"
    # wfg2_2obj_6dim_r = {}
    # for i, path in enumerate(os.listdir(function_paths[problem][0])):
    #     wfg2_2obj_6dim_r[i] = ResultsContainer(os.path.join(function_paths[problem][0], path))
    #     wfg2_2obj_6dim_r[i].add_reference_data(function_paths[problem][1])
    #
    # for i, rs in wfg2_2obj_6dim_r.items():
    #     print(i)
    #     rs.compute_hpv_history()
    #     rs.compute_igd_history(
    #         reference_points=np.asarray(IGD_REFPOINTS[problem]))