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


def save_fig(fig, filename):
    d = "./figures/"
    path = os.path.join(d, filename)
    fig.savefig(path+".png", dpi=300, facecolor=None, edgecolor=None, orientation='portrait', 
                pad_inches=0.12)
