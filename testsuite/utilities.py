import numpy as np


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


def get_target_igd_refpoints(target, ref_points):
    """
    gets dominated reference points for use by IGD+ measurement for directed
    optimisation
    """
    # case where target dominates points
    dominated_by_t = [dominates(target, rp, maximize=False) for rp in ref_points]
    if sum(dominated_by_t) == 0:
        # case where points dominate target
        dominated_by_t = [dominates(target, rp, maximize=True) for rp in ref_points]
        if sum(dominated_by_t) == 0:
            # target on pareto front
            return np.array([]), ref_points
    return ref_points[dominated_by_t], ref_points[np.logical_not(dominated_by_t)]
