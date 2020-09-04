import numpy as np


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


def Pareto_split(data, maximize: bool = False, return_indices=False):
    """
    separates the data points in data into non-dominated and dominated.

    :param np.ndarray data: the input data (n_points, data_dimension)
    :param bool maximize: True for finding non-dominated points in a
    maximisation problem, else for minimisaiton.
    :param bool return_indices: if True returns the indices of the
    non-dominated and dominate points if False returns the point values
    themselves.
    :return tuple: (non-dominated points, dominated points)
    """

    dom_mat = np.zeros((data.shape[0], data.shape[0]))
    for i, A in enumerate(data):
        for j, B in enumerate(data):
            dom_mat[i, j] = dominates(A, B, maximize=maximize)

    # calculate non-dominated and dominated data point indices
    p_ind = np.argwhere(np.all(dom_mat == 0, axis=0)).reshape(-1)
    d_ind = np.argwhere(np.any(dom_mat != 0, axis=0)).reshape(-1)

    if return_indices:
        # return indices
        return p_ind, d_ind
    else:
        # return data points
        return data[p_ind], data[d_ind]


