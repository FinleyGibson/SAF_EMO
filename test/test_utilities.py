import unittest
from parameterized import parameterized

import numpy as np

from testsuite.utilities import dominates
class TestDominates(unittest.TestCase):
    # basic dominated
    case_00 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': True,
               'maximize': False,
               'strict': True
               }

    # basic non-dominated
    case_01 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': False,
               'maximize': False,
               'strict': True
               }
    # edge dominated, strict
    case_02 = {'a': np.array([[1., 3.],
                              [3., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': False,
               'maximize': False,
               'strict': True
               }
    # edge dominated, strict
    case_03 = {'a': np.array([[1., 3.],
                              [3., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': True,
               'maximize': False,
               'strict': False
               }
    # edge non-dominated, beyond scope
    case_04 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[0., 5.]]),
               'dominated': False,
               'maximize': False,
               'strict': True
               }

    # edge dominated, beyond scope
    case_05 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[2., 5.]]),
               'dominated': True,
               'maximize': False,
               'strict': True
               }

    # edge dominated, beyond scope, strict
    case_06 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[1., 5.]]),
               'dominated': False,
               'maximize': False,
               'strict': True
               }

    # edge dominated, beyond scope, non-strict
    case_07 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[1., 5.]]),
               'dominated': True,
               'maximize': False,
               'strict': False
               }

    # inverted
    # basic dominated
    case_10 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }

    # basic non-dominated
    case_11 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[3., 3.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }
    # edge dominated, strict
    case_12 = {'a': np.array([[1., 3.],
                              [3., 1.]
                              ]),
               'b': np.array([[1., 1.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }
    # edge dominated, strict
    case_13 = {'a': np.array([[1., 3.],
                              [3., 1.]
                              ]),
               'b': np.array([[1., 1.]]),
               'dominated': True,
               'maximize': True,
               'strict': False
               }
    # edge non-dominated, beyond scope
    case_14 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[0., 5.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }

    # edge dominated, beyond scope
    case_15 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[2., 5.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }

    # edge dominated, beyond scope, strict
    case_16 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[0., 4.]]),
               'dominated': False,
               'maximize': True,
               'strict': True
               }

    # edge dominated, beyond scope, non-strict
    case_17 = {'a': np.array([[1., 4.],
                              [4., 1.]
                              ]),
               'b': np.array([[0., 4.]]),
               'dominated': True,
               'maximize': True,
               'strict': False
               }
    case_20 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 3.], [3., 0.]]),
               'dominated': [True, False],
               'maximize': False,
               'strict': True
               }

    case_21 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 3.], [3., 2.]]),
               'dominated': [True, True],
               'maximize': False,
               'strict': True
               }

    case_22 = {'a': np.array([[1., 2.],
                              [2., 1.]
                              ]),
               'b': np.array([[3., 0.], [1., 1.]]),
               'dominated': [False, False],
               'maximize': False,
               'strict': True
               }

    cases = [
        case_00,
        case_01,
        case_02,
        case_03,
        case_04,
        case_05,
        case_06,
        case_07,
        case_10,
        case_11,
        case_12,
        case_13,
        case_14,
        case_15,
        case_16,
        case_17,
        case_20,
        case_21,
        case_22
    ]

    # cases = [case_13]


    @parameterized.expand([[case] for case in cases])
    def test_dominated_case_status(self, case):
        out = dominates(a=case['a'],
                        b=case['b'],
                        maximize=case['maximize'],
                        strict=case['strict'])
        # try:
        self.assertEqual(case['dominated'], out)
        # assert out == case['dominated']
        # except:
        #     pass

    def test_timing(self):
        import numpy as np
        import time

        a = np.random.randn(10, 4)
        b = np.random.randn(100, 4)

        tic = time.time()
        ans0 = dominates(a, b)
        print(time.time() - tic)
        t1 = time.time()-tic

        tic = time.time()
        ans1 = dominates(b, a)
        t2 = time.time()-tic

        print(a.shape, b.shape, t1)
        print(b.shape, a.shape, t2)
        print(t2/t1)

        print(ans0)
        print(ans1)


#
# from testsuite.utilities import Pareto_split
# class TestParetoSplit(unittest.TestCase):
#
#     @classmethod
#     def setUpClass(cls) -> None:
#         # create set of points where alternating points dominate
#         yi = np.linspace(0, 1, 50)
#         y = np.vstack((yi, 1-yi)).T
#         y[::2] = y[::2]+0.05
#
#         cls.y = y
#         cls.p, cls.d = Pareto_split(cls.y)
#         cls.p_ind, cls.d_ind = Pareto_split(cls.y, return_indices=True)
#
#         cls.pmax, cls.dmax = Pareto_split(cls.y, maximize=True)
#         cls.pmax_ind, cls.dmax_ind = Pareto_split(cls.y,
#                                                   return_indices=True,
#                                                   maximize=True)
#
#     def test_basic_pareto_split(self) -> None:
#         self.assertEqual(self.p.shape[0], self.d.shape[0])
#         np.testing.assert_array_equal(self.d, self.y[::2])
#         np.testing.assert_array_equal(self.p, self.y[1::2])
#
#     def test_indices_pareto_split(self) -> None:
#         self.assertEqual(self.p_ind.shape[0], self.d_ind.shape[0])
#         np.testing.assert_array_equal(self.y[self.d_ind], self.y[::2])
#         np.testing.assert_array_equal(self.y[self.p_ind], self.y[1::2])
#
#     def test_maximise(self) -> None:
#         self.assertEqual(self.pmax.shape[0], self.dmax.shape[0])
#         np.testing.assert_array_equal(self.dmax, self.y[1::2])
#         np.testing.assert_array_equal(self.pmax, self.y[::2])
#
#         np.testing.assert_array_equal(self.y[self.dmax_ind], self.y[1::2])
#         np.testing.assert_array_equal(self.y[self.pmax_ind], self.y[::2])
#
#
# from testsuite.utilities import difference_of_hypervolumes
# from pymoo.factory import get_performance_indicator
# class TestDifferenceOfHypervolumes(unittest.TestCase):
#     # case configurations with known doh values
#     # basic dominated
#     case_00 = {'ref_point': np.array([5., 5.]),
#                'target': np.array([4., 4.]),
#                'p': np.array([[1., 3.],
#                               [2., 2.],
#                               [4., 1.]]),
#                'doh': -5.
#                }
#
#     # basic non-dominated
#     case_01 = {'ref_point': np.array([5., 5.]),
#                'target': np.array([1., 1.]),
#                'p': np.array([[1., 3.],
#                               [2., 2.],
#                               [4., 1.]]),
#                'doh': 4.
#                }
#
#     # edge: dominated, in-line with target
#     case_02 = {'ref_point': np.array([5., 5.]),
#                'target': np.array([1., 5.]),
#                'p': np.array([[1., 3.],
#                               [2., 2.],
#                               [4., 1.]]),
#                'doh': 0.
#                }
#
#     # edge: non-dominated, in-line with target
#     case_03 = {'ref_point': np.array([5., 5.]),
#                'target': np.array([5., 0.]),
#                'p': np.array([[1., 3.],
#                               [2., 2.],
#                               [4., 1.]]),
#                'doh': 0.
#                }
#
#     # edge: dominated,  beyond scope of p
#     case_04 = {'ref_point': np.array([5., 5.]),
#                'target': np.array([0., 4.]),
#                'p': np.array([[1., 3.],
#                               [2., 2.],
#                               [4., 1.]]),
#                'doh': 1.
#                }
#
#     # edge: non-dominated, beyond scope of p
#     case_05 = {'ref_point': np.array([5., 5.]),
#                'target': np.array([2., 4.]),
#                'p': np.array([[1., 3.],
#                               [2., 2.],
#                               [4., 1.]]),
#                'doh': -1.
#                }
#
#     # edge: beyond reference point
#     case_06 = {'ref_point': np.array([5., 5.]),
#                'target': np.array([6., 0.]),
#                'p': np.array([[1., 3.],
#                               [2., 2.],
#                               [4., 1.]]),
#                'doh': AssertionError
#                }
#
#     cases = [case_00,
#              case_01,
#              case_02,
#              case_03,
#              case_04,
#              case_05,
#              case_06
#              ]
#     # cases = [case_00]
#
#     @classmethod
#     def setUpClass(cls) -> None:
#
#         # create set of points which split a 1x1 square in objective
#         # space in half
#         yi = np.linspace(0, 1, 1000)
#         y = np.vstack((yi, 1-yi)).T
#
#         cls.y = y
#         cls.ref = np.ones(2)
#         cls.hpv_measure = get_performance_indicator("hv", ref_point=cls.ref)
#         cls.hpv_measure_case = get_performance_indicator("hv",
#                                                          ref_point=[5., 5.])
#
#     def test_basic_nondominated_target_doh(self):
#         # both target and reference point at [1., 1.]
#         target = self.ref
#         hpv = difference_of_hypervolumes(self.y,
#                                          target=target,
#                                          hpv_measure=self.hpv_measure)
#         # target at [0., 0.]
#         target2 = np.zeros_like(self.ref)
#         hpv2 = difference_of_hypervolumes(self.y,
#                                           target=target2,
#                                           hpv_measure=self.hpv_measure)
#
#     def test_basic_dominated_target_doh(self):
#         # both target and reference point at [1., 1.]
#         target = self.ref
#         hpv = difference_of_hypervolumes(self.y,
#                                          target=target,
#                                          hpv_measure=self.hpv_measure)
#
#         # half of 1x1 square area should be close to 0.5
#         self.assertAlmostEqual(hpv, -0.5, 2)
#
#     @parameterized.expand([[case] for case in cases])
#     def test_edge_cases(self, case):
#         if isinstance(case['doh'], float):
#             hpv = difference_of_hypervolumes(p=case['p'],
#                                              target=case['target'],
#                                              hpv_measure=self.hpv_measure_case)
#             self.assertEqual(hpv, case['doh'])
#         elif isinstance(case['doh'], AssertionError):
#             self.assertRaises(AssertionError,
#                               difference_of_hypervolumes,
#                               {'p': case['p'],
#                                'target': case['target'],
#                                'hpv_measure': self.hpv_measure_case})
#
#
# from testsuite.utilities import targetted_hypervolumes_single_target
# class TestTargettedHypervolumesSingleTarget(unittest.TestCase):
#     # target attained
#     case_00 = {'ref_point': np.array([10., 10.]),
#                'target': np.array([[6., 7.]]),
#                'p': np.array([[1., 7.],
#                               [3., 6.],
#                               [5., 5.],
#                               [7., 4.]]),
#                'doh': (12., 4.)
#                }
#
#     # target unattained
#     case_01 = {'ref_point': np.array([10., 10.]),
#                'target': np.array([[2., 4.]]),
#                'p': np.array([[1., 7.],
#                               [3., 6.],
#                               [5., 5.],
#                               [7., 4.]]),
#                'doh': (39., 0.)
#                }
#
#     # target attained, beyond scope
#     case_02 = {'ref_point': np.array([10., 10.]),
#                'target': np.array([[9., 5.]]),
#                'p': np.array([[1., 7.],
#                               [3., 6.],
#                               [5., 5.],
#                               [7., 4.]]),
#                'doh': (5., 2.)
#                }
#
#     # target unattained, beyond scope
#     case_03 = {'ref_point': np.array([10., 10.]),
#                'target': np.array([[0., 5.]]),
#                'p': np.array([[1., 7.],
#                               [3., 6.],
#                               [5., 5.],
#                               [7., 4.]]),
#                'doh': (39., 0.)
#                }
#
#     # target edge
#     case_04 = {'ref_point': np.array([10., 10.]),
#                'target': np.array([[10., 1.]]),
#                'p': np.array([[1., 7.],
#                               [3., 6.],
#                               [5., 5.],
#                               [7., 4.]]),
#                'doh': (0., 0.)
#                }
#
#     # target edge, target outside ref_point span
#     case_05 = {'ref_point': np.array([10., 10.]),
#                'target': np.array([[11., 5.]]),
#                'p': np.array([[1., 7.],
#                               [3., 6.],
#                               [5., 5.],
#                               [7., 4.]]),
#                'doh': AssertionError
#                }
#
#     cases = [case_00,
#              case_01,
#              case_02,
#              case_03,
#              case_04,
#              case_05]
#
#     @parameterized.expand([[case] for case in cases])
#     def test__cases(self, case):
#         if isinstance(case['doh'], float):
#             value = targetted_hypervolumes_single_target(case['p'],
#                                                          case['target'],
#                                                          case['ref_point'])
#             self.assertEqual(value, case["doh"])
#
#         elif isinstance(case['doh'], AssertionError):
#             # test for errors
#             self.assertRaises(AssertionError,
#                               targetted_hypervolumes_single_target,
#                               {'p': case['p'],
#                                'target': case['target'],
#                                'ref_point': case['ref_point']})

