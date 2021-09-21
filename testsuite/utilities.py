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
              strict: bool =True):
    """
    returns True if a dominates b, else returns False

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

    :return bool: True if a dominate b, else returns False"
    """
    if len(a) < 2:
        if maximize:
            return np.all(a > b)
        else:
            return np.all(a < b)
    if a.ndim > 1:
        return np.any([dominates(ai, b, maximize, strict) for ai in a])
    else:
        if maximize and strict:
            return np.any(np.all(a > b))
        elif not maximize and strict:
            return np.any(np.all(a < b))
        elif maximize and not strict:
            return np.any(np.all(a >= b))
        elif not maximize and not strict:
            return np.any(np.all(a <= b))
        else:
            raise



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
    def __init__(self, target, reference):
        self.target = target
        self.reference = reference

    def calc(self, p):
        return difference_of_hypervolumes(p, self.target, self.reference)


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(0)
    # test_ref = '/home/finley/phd/code/testsuite/experiments/directed/data/wfg1_2obj_3dim/log_data/OF_objective_function__opt_DirectedSaf__ninit_10__surrogate_MultiSurrogateGP__ei_False__target_1p68_1p09__w_0p5'
    # test_ref = '/home/finley/phd/code/testsuite/experiments/directed/data/wfg2_4obj_5dim/'
    # test_ref = '/home/finley/phd/code/testsuite/experiments/directed/data/'
    # ans = get_filenames_of_incompletes_within_tree(test_ref)
    #
    # print(ans)
    x = np.random.uniform(0.5, 1, (10, 2))
    p, d = Pareto_split(x)
    print(x.shape)
    rp = np.ones(2)*1.2
    t = np.ones(2)*0.4


    # def difference_of_hypervolumes(p, target, ref_point):
    a = np.vstack((p, t))
    rp = np.vstack((p, t)).max(axis=0)
    hv_measure = get_performance_indicator("hv", ref_point=rp)
    ans = difference_of_hypervolumes(p, t, hv_measure)
    print(ans)
    pass

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(*p.T, c="C3", label="non-dominated evaluations")
    ax.scatter(*d.T, c="C0", label="dominated evaluations")
    ax.scatter(*t, c="magenta", label="target")
    ax.scatter(*rp, c="cyan", label="referencepoint")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1.3])
    ax.set_ylim([0, 1.3])
    fig.show()
    pass

    t2 = np.ones(2)
    rp = np.vstack((p, t2)).max(axis=0)
    hv_measure = get_performance_indicator("hv", ref_point=rp)
    ans2 = difference_of_hypervolumes(p, t2, hv_measure)
    print(ans2)

    fig = plt.figure()
    ax = fig.gca()
    ax.scatter(*p.T, c="C3", label="non-dominated evaluations")
    ax.scatter(*d.T, c="C0", label="dominated evaluations")
    ax.scatter(*t2, c="magenta", label="target")
    ax.scatter(*rp, c="cyan", label="referencepoint")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1.3])
    ax.set_ylim([0, 1.3])
    fig.show()
    pass
