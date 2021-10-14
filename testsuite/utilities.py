import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pymoo.model.indicator
from pymoo.factory import get_performance_indicator
from scipy.spatial import KDTree

def str_format(a):
    replacemen_pairs = {'.':'p',
                        ' ': '_',
                        '[': '',
                        ']': ''
                        }
    a = str(a)
    for k, v in replacemen_pairs.items():
        a = a.replace(k, v)
    return a

def sigmoid(x, beta=1):
    squashed = 1 / (1 + np.exp(-beta * x))
    return squashed


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

def optional_inversion(f):
    """decorator to invert the value of a function, and turn maximisation
    problem to minimization problem. Invoke by passing a keyword argument
    invert=True to the decorated function"""
    def wrapper(*args, **kwargs):
        try:
            if kwargs.pop("invert") is True:
                return -f(*args, **kwargs)
            else:
                return f(*args, **kwargs)
        except KeyError:
            # if no invert argument is passed.
            return f(*args, **kwargs)
    return wrapper


def dominates(a: np.ndarray, b: np.ndarray,
              maximize: bool = False,
              strict: bool = True):
    """
    returns True if any of a dominate b, else returns False

    :param a: np.ndarray (n_points, point_dims)
        dominating query point
    :param b: np.ndarray
        dominated query points (n_points, point_dims)
    :param maximize: bool
        True for finding domination relation in a
        maximisation problem, False for minimisaiton problem.
    :param strict: bool
        if True then computes strict dominance, otherwise allows equal
        value in a given dimension to count as non-dominated
            - swaps < for <=
            - swaps > for >=

    :return bool: True if a dominates b, else returns False"
    """
    if a.ndim < 2:
        a = a.reshape(1, -1)
    if b.ndim < 2:
        b = b.reshape(1, -1)

    if b.shape[0] > 1:
        return [dominates(a, bi.reshape(1, -1)) for bi in b]
    else:
        if maximize and strict:
            return np.all(a > b, axis=1).any()
        elif maximize and not strict:
            return np.all(a >= b, axis=1).any()
        elif not maximize and strict:
            return np.all(a < b, axis=1).any()
        elif not maximize and not strict:
            return np.all(a <= b, axis=1).any()
        else:
            raise

def dominated(a: np.ndarray, b: np.ndarray, maximize: bool = False,
              strict=False):
    """

    returns True if a is dominated all of b, else returns False
    :param np.ndarray a: dominating query points (n_points, point_dims)
    :param np.ndarray b: dominated query points (n_points, point_dims)
    :param bool maximize: True for finding domination relation in a
    maximisation problem, False for minimisaiton problem/
    :return bool: True if a dominate b, else returns False
    """
    if a.ndim < 2:
        a = a.reshape(1, -1)
    if b.ndim < 2:
        b = b.reshape(1, -1)

    if a.shape[0] > 1:
        return [dominated(ai.reshape(1, -1), b) for ai in a]
    else:
        if maximize and not strict:
            return np.all(a < b, axis=1).any()
        elif maximize and strict:
            return np.all(a <= b, axis=1).any()
        elif not maximize and not strict:
            return np.all(a > b, axis=1).any()
        elif not maximize and strict:
            return np.all(a >= b, axis=1).any()

def Pareto_split(x_in, maximize: bool = False, return_indices=False):
    """
    separates the data points in data into non-dominated and dominated.

    :param np.ndarray x: the input data (n_points, data_dimension)
    :param bool maximize: True for finding non-dominated points in a
    maximisation problem, else for minimisaiton.
    :param bool return_indices: if True returns the indices of the
    non-dominated and dominate points if False returns the point values
    themselves.
    :return tuple: (non-dominated points, dominated points)
    """
    x = x_in.copy()
    if not return_indices:
        x_orig = x.copy()
    assert x.ndim==2

    if maximize:
        x *= -1 

    n_points = x.shape[0]
    is_efficient = np.arange(n_points)
    point_index = 0  # Next index in the is_efficient array to search for
    while point_index<len(x):
        pareto_mask = np.any(x<x[point_index], axis=1)
        pareto_mask[point_index] = True
        is_efficient = is_efficient[pareto_mask]  # Remove dominated points
        x = x[pareto_mask]
        point_index = np.sum(pareto_mask[:point_index])+1
    
    nondominated_mask = np.zeros(n_points, dtype = bool)
    nondominated_mask[is_efficient] = True
    if return_indices:
        return nondominated_mask, np.invert(nondominated_mask)
    else:
        return x_orig[nondominated_mask], x_orig[np.invert(nondominated_mask)]


def difference_of_hypervolumes(p, target, hpv_measure):
    """
    computes the difference of the dominated hypervolume metric, using a pymoo.
    1) First determines whether the target point is dominated by any p
    2) -If target is dominated then compute:
        dominated_hypervolume(P)-dominated_hypervolume(P U T)
       - If target is not dominated then compute:
        dominated_hypervolume_T(P U T) where dominated_hypervolume_T is the
        dominated hypervolume calculated using the target point as the
        reference point.

    :param p: np.ndarray, shape: (n_points, n_obj)
        non-dominated points forming an approximation to the Pareto front
    :param target: np.ndarray, shape: (1, n_obj) OR (n_obj,)
        single target point
    :param hpv_measure: pymoo.performance_indicator.hv.Hypervolume
        hypervolume measure from pymoo, with reference point already set.
    :return: float
        value of difference of dominated hypervolumes
    :raises: AssertionError
        if ref_point does not confirm to the requirements specified.

    """
    # check target is valid for hpv_measure reference point
    assert not np.any(target>hpv_measure.ref_point)

    # get dominated state of target and determine inversion
    target_dominated = np.any([dominates(pi, target) for pi in p])

    # compute difference of hypervolumes
    if not target_dominated:
        doh = hpv_measure.calc(np.vstack((p, target)))-hpv_measure.calc(p)
    else:
        hpv_measure_t = get_performance_indicator("hv", ref_point = target)
        doh = -hpv_measure_t.calc(np.vstack((p, target)))
    return doh


class DifferenceOfHypervolumes:
    """
    class wrapper for difference_of_hypervolumes function to
    be used as with pymoo indicators: using calc function.
    """
    def __init__(self, target, ref_point):
        self.target = target
        self.measure = get_performance_indicator("hv", ref_point = ref_point)

    def calc(self, p):
        return difference_of_hypervolumes(p, self.target, self.measure)


# def monte_carlo_targetted_hypervolumes(p, target, ref_point, n_samples=None):
#
#     if target.ndim ==1:
#         target = target.reshape(1,-1)
#     if ref_point.ndim ==1:
#         ref_point = ref_point.reshape(1,-1)
#
#     assert target.shape[1] == ref_point.shape[1]
#     n_obj = target.shape[1]
#
#     if n_samples is None:
#         n_samples = 3e6*n_obj
#
#     outer_sampes = np.random.uniform(0,)

def targetted_hypervolumes_single_target(p, target, ref_point):
    """

    :param p: np.ndarray, shape: (n_points, n_obj)
        non-dominated points forming an approximation to the Pareto front
    :param target: np.ndarray, shape: (1, n_obj) OR (n_obj,)
        single target point
    :param ref_point: np.ndarray, shape: (1, n_obj) OR (n_obj,)
        single reference point
    :return: tuple, (float, float)

    """
    if target.ndim == 1:
        target = target.reshape(1, -1)
    if ref_point.ndim >1:
        ref_point = ref_point.reshape(-1)

    # only tested for 2 objective problems
    # TODO: test/add functionality for higher dimensions - with tests
    # assert ref_point.shape[0] == 2

    assert target.shape[0] == 1
    assert ref_point.shape[0] == target.shape[1]

    return targetted_hypervolume_a_single_target(p, target, ref_point),\
           targetted_hypervolume_b_single_target(p, target, ref_point)


def targetted_hypervolume_a_single_target(p, target, ref_point):
    """

    :param p: np.ndarray, shape: (n_points, n_obj)
        non-dominated points forming an approximation to the Pareto front
    :param target: np.ndarray, shape: (1, n_obj) OR (n_obj,)
        single target point
    :param ref_point: np.ndarray, shape: (1, n_obj) OR (n_obj,)
        single reference point
    :return: tuple, (float, float)

    """
    p = p[np.argsort(p[:, 0])]

    # add points to p at limit of reference point
    pa = np.array([p[0][0], ref_point[1]]).reshape(1, -1)
    pb = np.array([ref_point[0], p[-1][1]]).reshape(1, -1)
    p = np.vstack((pa, p, pb))

    # modify p to limit at the edges of the bounding box.
    t_max = target.max(axis=0)
    p_ = np.vstack([pi if pi[0] > t_max[0] else [t_max[0], pi[1]] for pi in p])
    p_ = np.vstack(
        [pi if pi[1] > t_max[1] else [pi[0], t_max[1]] for pi in p_])

    t_attained = not np.any([dominates(target, pi) for pi in p])

    if t_attained:
        measure = get_performance_indicator("hv", ref_point=ref_point)
        hpv = measure.calc(target)
    else:
        measure = get_performance_indicator("hv", ref_point=ref_point)
        hpv = measure.calc(p_)
    return hpv


def targetted_hypervolume_b_single_target(p, target, ref_point):
    """

    :param p: np.ndarray, shape: (n_points, n_obj)
        non-dominated points forming an approximation to the Pareto front
    :param target: np.ndarray, shape: (1, n_obj) OR (n_obj,)
        single target point
    :param ref_point: np.ndarray, shape: (1, n_obj) OR (n_obj,)
        single reference point
    :return: tuple, (float, float)
    :raises: AssertionError
        if target is outside span of ref_point

    """
    assert not dominates(ref_point, target)
    p = p[np.argsort(p[:, 0])]

    t_attained = not np.any([dominates(target, pi) for pi in p])
    if t_attained:
        measure = get_performance_indicator("hv", ref_point=target.reshape(-1))
        hpv = measure.calc(p)
    else:
        hpv = 0.
    return hpv


def KDTree_distance(a, b):
    """
    quick method to find the neerest neighbours for all items of array a
    in array b
    :param a: np.array()
    :param b: np.array()
    :return: np.array()
        array of minimum distances of array a from b
    """
    tree = KDTree(a)
    return tree.query(b)[0]

def monte_carlo_sample_margin(lower, upper, n_samples):
    assert lower.ndim == 1
    assert upper.ndim == 1

    frac = 1/((np.product(upper)-np.product(lower))/np.product(lower))
    margin = 1.05
    sufficient_samples = False
    while not sufficient_samples:
        ns = int(n_samples + n_samples * frac * margin)
        samples = np.vstack([np.random.uniform(0., u, ns)
                                   for u in upper]).T

        samples = samples[np.logical_or(
            samples[:,0]>lower[0],
            samples[:,1]>lower[1])]

        if len(samples)>n_samples:
            sufficient_samples = True
        else:
            margin *= 1.1
        print(len(samples))
    return samples[:n_samples]



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time

    ans = monte_carlo_sample_margin(np.ones(2)*1.75, np.ones(2)*2, int(1e5))
    # print(ans)
    print(ans.shape)

    pa = np.linspace(0, 2, 30)
    pb = 2 - pa
    p = np.vstack((pa, pb)).T

    fig0 = plt.figure()
    ax0 = fig0.gca()
    ax0.scatter(*ans.T, s=2)
    ax0.scatter(*p.T, s=10, c="magenta")

    tic = time()
    i_p = np.array([dominates(p, ai) for ai in ans])
    toc = time()
    i_d = np.logical_not(i_p)
    fig1 = plt.figure()
    ax1 = fig1.gca()
    ax1.scatter(*ans[i_d].T, s=2, c="C1")
    ax1.scatter(*ans[i_p].T, s=2, c="C0")
    ax1.scatter(*p.T, s=10, c="magenta")

    print(toc-tic)
    print(sum(i_p)/len(ans))
    plt.show()
    pass



    # import numpy as np
    # import matplotlib.pyplot as plt
    # from pymoo.factory import get_performance_indicator
    # from testsuite.utilities import dominates
    #
    #
    # def image_case(case):
    #     rp = case['ref_point']
    #     t = case['target']
    #     p = case['p']
    #
    #     fig = plt.figure(figsize=[6, 6])
    #     ax = fig.gca()
    #     ax.scatter(*case['p'].T, c="C0", label="p")
    #     ax.scatter(*case['target'].T, c="magenta", label="target")
    #     ax.scatter(*case['ref_point'], c="C2", label="reference")
    #     ax.set_title(
    #         f" a expected: {case['doh'][0]} computed: {targetted_hypervolumes_single_target(p, t, rp)[0]} "
    #         f"\n b expected: {case['doh'][1]} computed: {targetted_hypervolumes_single_target(p, t, rp)[1]}")
    #     ax.grid('on')
    #     ax.axis("scaled")
    #     ax.set_xticks(range(0, 12))
    #     ax.set_yticks(range(0, 12))
    #     ax.legend(loc="lower left")
    #     return fig
    #
    # # target attained
    # case_00 = {'ref_point': np.array([10., 10.]),
    #            'target': np.array([[6., 7.]]),
    #            'p': np.array([[1., 7.],
    #                           [3., 6.],
    #                           [5., 5.],
    #                           [7., 4.]]),
    #            'doh': (12., 4.)
    #            }
    #
    # # target unattained
    # case_01 = {'ref_point': np.array([10., 10.]),
    #            'target': np.array([[2., 3.]]),
    #            'p': np.array([[1., 7.],
    #                           [3., 6.],
    #                           [5., 5.],
    #                           [7., 4.]]),
    #            'doh': (39., 0.)
    #            }
    #
    #
    # case = case_00
    # fig = image_case(case)
    # fig.show()
    # pass