import json
import unittest
from parameterized import parameterized
import rootpath
import os
from copy import deepcopy
from random import choice
import numpy as np

from testsuite.results import Result, ResultsContainer
from testsuite.utilities import Pareto_split


class TestResultsClass(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        # result file path
        results_dir = os.path.join(
            rootpath.detect(),
            "experiments/directed/data_undirected_comp/wfg1_2obj_3dim"
            "/log_data/OF_objective_function__opt_Saf__ninit_10__surrogate"
            "_MultiSurrogateGP__ei_False/")
        assert os.path.isdir(results_dir)

        file_name = [f for f in os.listdir(results_dir)
                     if f[-11:] == "results.pkl"][0]
        cls.file_path = os.path.join(results_dir, file_name)

        # instantiate result to be tested
        cls.result = Result(cls.file_path)
        cls.n_dim = cls.result.n_dim
        cls.n_obj = cls.result.n_obj
        cls.n_evals = cls.result.n_evaluations

        cls.hpv_refpoint = (np.ones(cls.n_obj)*3).reshape(1, -1)
        cls.igd_refpoints = Pareto_split(np.random.randn(10, cls.n_obj))[0]

    def test_igd(self):
        """
        test igd+ computation
        """
        self.assertIsNone(self.result.igd_history)
        self.result.compute_igd_history(self.igd_refpoints)

        # check history is set
        self.assertIsInstance(self.result.igd_history, np.ndarray)

        # check length of history
        self.assertEqual(self.result.n_evaluations-self.result.n_initial+1,
                         self.result.igd_history.shape[0])

    def test_hpv(self):
        """
        test hypervolume computation
        """
        self.assertIsNone(self.result.hpv_history)
        self.result.compute_hpv_history(self.hpv_refpoint)

        # check history is set
        self.assertIsInstance(self.result.hpv_history, np.ndarray)

        # check length of history
        self.assertEqual(self.result.n_evaluations - self.result.n_initial + 1,
                         self.result.hpv_history.shape[0])


class TestResultsContainerClass(unittest.TestCase):

    maxDiff = None  # shows full breakdown of dictionary differences

    @classmethod
    def setUpClass(cls) -> None:
        cls.results_dir = os.path.join(
            rootpath.detect(),
            'experiments/directed/data/wfg1_2obj_3dim/log_data'
            '/OF_objective_function__opt_DirectedSaf__ninit_10__surrogate_'
            'MultiSurrogateGP__ei_False__target_0p35_3p14__w_0p5')
        assert os.path.isdir(cls.results_dir)

        cls.reference_dir = os.path.join(
            rootpath.detect(),
            'experiments/directed/data_undirected_comp/wfg1_2obj_3dim/log_data'
            '/OF_objective_function__opt_Saf__ninit_10__surrogate_'
            'MultiSurrogateGP__ei_False/')
        assert os.path.isdir(cls.reference_dir)

    def setUp(self) -> None:
        # instantiate ResultsContainer object for testing
        self.container = ResultsContainer(self.results_dir)
        self.container.add_reference_data(self.reference_dir)

    def test_isntantiation_methods(self):
        """
        checks that all methods of instantiating ResultsContainer
        objects produce identical instance
        """
        # path to results directory
        rc0 = ResultsContainer(self.results_dir)
        # list of strings to results.pkl files
        result_paths = [os.path.join(self.results_dir, p)
                        for p in os.listdir(self.results_dir)
                        if p[-11:] == "results.pkl"]

        rc1 = ResultsContainer(result_paths)
        # list of Results objects
        r_objs = [Result(file) for file in result_paths]
        rc2 = ResultsContainer(r_objs)

        # compare results dicts, np.testing used to handle numpy arrays
        for i in range(len(self.container.results)):
            np.testing.assert_equal(self.container.results[i].__dict__,
                                    rc0.results[i].__dict__)
            np.testing.assert_equal(self.container.results[i].__dict__,
                                    rc1.results[i].__dict__)
            np.testing.assert_equal(self.container.results[i].__dict__,
                                    rc2.results[i].__dict__)

        # compare ResultsContainers instantiated via different methods
        # results instances differ, so must be removed before container
        # comparision
        comp_dict = deepcopy(self.container)
        comp_dict.__dict__.pop('results')
        comp_dict.__dict__.pop('reference')
        rc0.__dict__.pop('results')
        rc1.__dict__.pop('results')
        rc2.__dict__.pop('results')
        rc0.__dict__.pop('reference')
        rc1.__dict__.pop('reference')
        rc2.__dict__.pop('reference')

        np.testing.assert_equal(rc0.__dict__, comp_dict.__dict__)
        np.testing.assert_equal(rc1.__dict__, comp_dict.__dict__)
        np.testing.assert_equal(rc2.__dict__, comp_dict.__dict__)

        # finally test instance loaded from saved
        self.container.save("./test_ResultsContainer")
        rc3 = ResultsContainer("./test_ResultsContainer")
        for i in range(len(self.container.results)):
            np.testing.assert_equal(self.container.results[i].__dict__,
                                    rc3.results[i].__dict__)

        rc3.__dict__.pop('results')
        rc3.__dict__.pop('reference')
        np.testing.assert_equal(rc3.__dict__, comp_dict.__dict__)

    @parameterized.expand([
        [False],
        [True]
    ])
    def test_sort(self, reverse):
        """
        test result sorting
        """
        initial_container = deepcopy(self.container)
        self.container.sort('seed', reverse=reverse)

        # test sorting reorders ResultContainer attributes
        sorted_seeds = self.container.seed
        if not reverse:
            self.assertListEqual(sorted(initial_container.seed), sorted_seeds)
        else:
            self.assertListEqual(sorted(initial_container.seed)[::-1],
                                 sorted_seeds)

        # test sorting also reorders Results in ResultsContainer.results
        result_seeds = [r.seed for r in self.container.results]
        self.assertListEqual(sorted_seeds, result_seeds)

        # ensure random Result objects attributes are maintained
        # following sort
        random_result = choice(initial_container.results)
        for result in self.container.results:
            if result.seed == random_result.seed:
                # compare results matched by seed
                np.testing.assert_equal(result.__dict__,
                                        random_result.__dict__)

    def test_sorted(self):
        sorted_inst = self.container.sorted("seed")
        self.container.sort("seed")
        sort_inst = deepcopy(self.container)

        # avoid comparing results lists as instance ids of Result differ
        # due to copy process
        Da_res = sorted_inst.__dict__.pop("results")
        Db_res = sort_inst.__dict__.pop("results")
        Da_ref = sorted_inst.__dict__.pop("reference")
        Db_ref = sort_inst.__dict__.pop("reference")

        np.testing.assert_equal(sorted_inst.__dict__, sort_inst.__dict__)
        for Da_rs, Db_rs, Da_rf, Db_rf in zip(Da_res, Db_res, Da_ref, Db_ref):
            np.testing.assert_equal(Da_rs.__dict__, Db_rs.__dict__)
            np.testing.assert_equal(Da_rf.__dict__, Db_rf.__dict__)

    @parameterized.expand([
        [False],
        [True]
    ])
    def test_sort_reference(self, reverse):
        """
        test result sorting also sorts references and keeps them paired
        """
        initial_container = deepcopy(self.container)
        self.container.sort('seed', reverse=reverse)

        initial_seeds = [r.seed for r in initial_container.reference]
        sorted_seeds = [r.seed for r in self.container.reference]
        if not reverse:
            self.assertListEqual(sorted(initial_seeds), sorted_seeds)
        else:
            self.assertListEqual(sorted(initial_seeds)[::-1], sorted_seeds)

        # ensure reference Result objects attributes are maintained
        # following sort
        random_result = choice(initial_container.reference)
        for result in self.container.reference:
            if result.seed == random_result.seed:
                # compare results matched by seed
                np.testing.assert_equal(result.__dict__,
                                        random_result.__dict__)

    def test_get_hpv_refpoint(self):
        rp = self.container.get_hpv_refpoint()
        self.assertEqual(rp.shape, (1, self.container.n_obj[0]))

    def test_get_igd_refpoints(self):
        ex_points = Pareto_split(np.random.randn(50,self.container.n_obj[0]))[0]
        ex_dict = {"wfg1_2obj_3dim": ex_points.tolist()}
        with open("./example_refpoints", 'w') as outfile:
            json.dump(ex_dict, outfile)

        rp = self.container.get_igd_refpoints(file_path='./example_refpoints')[0]
        self.assertEqual(rp.shape[1], self.container.n_obj[0])


if __name__ == '__main__':
    unittest.main()
