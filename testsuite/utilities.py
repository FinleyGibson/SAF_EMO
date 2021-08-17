import os
import pickle
import numpy as np

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


def dominates(a: np.ndarray, b: np.ndarray, maximize: bool = False):
    """
    returns True if a dominates b, else returns False

    :param np.ndarray a: dominating query point
    :param np.ndarray b: dominated query points (n_points, point_dims)
    :param bool maximize: True for finding domination relation in a
    maximisation problem, False for minimisaiton problem.
    :return bool: True if a dominate b, else returns False"
    """
    if len(a) < 2:
        if maximize:
            return np.all(a > b)
        else:
            return np.all(a < b)
    else:
        # allows
        if maximize:
            return np.any(np.all(a > b))
        else:
            return np.any(np.all(a < b))


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



if __name__ == "__main__":
    # test_ref = '/home/finley/phd/code/testsuite/experiments/directed/data/wfg1_2obj_3dim/log_data/OF_objective_function__opt_DirectedSaf__ninit_10__surrogate_MultiSurrogateGP__ei_False__target_1p68_1p09__w_0p5'
    # test_ref = '/home/finley/phd/code/testsuite/experiments/directed/data/wfg2_4obj_5dim/'
    test_ref = '/home/finley/phd/code/testsuite/experiments/directed/data/'
    ans = get_filenames_of_incompletes_within_tree(test_ref)

    print(ans)


