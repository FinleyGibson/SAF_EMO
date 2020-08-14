import numpy as np


def saf(P, X):
    """
    Calculates summary attainment front distances
    Calculates the distance of n, m-dimensional points X from the summary attainment front defined by points P

    :param P [np.array]: points in the pareto front, shape[?,m]
    :param X [np.array]: points for which the distance to the summary attainment front is to be calculated, shape[n,m]
    :param beta [float]: if not None, the saf distance is passed through sigmoid function with beta=squashed
    :param normalized [Bool]: if not None, the saf distance for points in X is normalized to a range from 0-1

    :return [np.array]: numpy array of saf distances between points in X and saf defined by P, shape[X.shape]
    """
    assert P.shape[1] == X.shape[1]  # check dimensionality of P is the same as that for X

    D = np.zeros((X.shape[0], P.shape[0]))

    for i, p in enumerate(P):
        D[:, i] = np.max(p - X, axis=1).reshape(-1)
    Dq = np.min(D, axis=1)
    return Dq


def dominates(A, B, maximize=False):
    "does A dominate B"
    if maximize:
        return np.all(A>B)
    else:
        return np.all(A<B)


def Pareto_split(X, maximize=False):
    """function to determine the pareto set of a set of values x in X

    args:
        X[np.array] set of 2D points (n,2)

    returns:
        p_set[np.array] set of 2D points (n,2) making up the Pareto set from X
        d_set[np.array] set of 2D points (n,2) making up the dominated points from X

    """

    dom_mat = np.zeros((X.shape[0], X.shape[0]))
    for i, A in enumerate(X):
        for j, B in enumerate(X):
            dom_mat[i, j] = dominates(A, B, maximize=maximize)

    p = X[np.argwhere(np.all(dom_mat == 0, axis=0)).reshape(-1)]
    d = X[np.argwhere(np.any(dom_mat != 0, axis=0)).reshape(-1)]
    return p, d

