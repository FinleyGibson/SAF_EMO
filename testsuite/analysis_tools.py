import numpy as np
from testsuite.utilities import dominates
import pickle
import os
from itertools import product
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
import lhsmdu
import wfg
from tqdm import tqdm
import matplotlib.pyplot as plt


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
            return target.reshape(1,-1), ref_points
    return ref_points[dominated_by_t], ref_points[np.logical_not(dominated_by_t)]


def get_result_dirs_from_tree(parent_dir):
    """
    Find all directories within a directory tree containing results.pkl
    files. Results are returned as paths extended from the parent_dir
    supplied. If parent_dir is an absolute reference, then the returned
    strings will be also.
    :param parent_dir: str
                       top directory in the tree
    :return: list(str)
             list of directory paths for those which contain at least
             one file ending in results.pkl
    """
    leaf_dirs = []
    for (root, dirs, files) in os.walk(parent_dir, topdown=True):
        leaf = np.any([file[-11:] == 'results.pkl' for file in files])
        if leaf and (root not in leaf_dirs):
            leaf_dirs.append(root)
    return leaf_dirs


def check_state_of_result(dir_path, file_name):
    """
    Loads a result.pkl file from the raw_path supplied by result_path and
    checks the state of it; how many of the intended evaluations have
    been made.
    :param dir_path: str
                string containing the raw_path to the directory containing
                results.pkl files, either an absolute or relative raw_path.
    :param file_name: str
                string containing the name of the file within dir_path
                to be checked
    :return: tuple: (int, bool, (int, int), str)

    :raises: AssertionError: supplied raw_path invalid; not raw_path to results.pkl
                             file
    """
    assert file_name[-11:] == 'results.pkl', "raw_path supplied to " \
                                             "check_state_of_result not a " \
                                             "raw_path to results.pkl file."
    # load result
    with open(os.path.join(dir_path, file_name), 'rb') as infile:
        result = pickle.load(infile)

    # check completion status
    comp = result['n_evaluations'] == result['budget']

    # format and return tuple
    return result['seed'], comp, (result['n_evaluations'], result['budget']), \
           file_name


def check_results_within_directory(dir):
    """
    Checks the state of the results files within the supplied directory.

    :param dir: str
                string raw_path to the directory in which to check the results.
    :return: list [(int, bool, (int, int), str)]
                list of states of the results.pkl files within dir. The results
                take the format:
                (seed: int, complete: bool, state (n_evaluations:int,
                expected_evaluations:int, file_name: str)
    """
    # get paths to all the results.pkl files in dir
    file_names = [f for f in sorted(os.listdir(path=dir))
                  if f[-11:] == 'results.pkl']

    # check the state of each result, and return
    return [check_state_of_result(dir, fn) for fn in file_names]


def check_results_within_tree(top_dir):
    """
    Check the state of the results within all directories in the directory tree
    beneath top_dir.

    :param top_dir: str
                raw_path to the directory which forms the topmost directory in the
                directory tree being considered.
    :return: dict {str, [(int, bool, (int, int), str]}
                dictionary with paths to dirs containing results.pkl files as
                keys, and lists of state tupes as values, as returned by
                check_results_within_directory
    """
    # find all 'leaf' directories; those directories which contain result.pkl
    # files with in the directory tree headed by top_dir
    leaf_dirs = []
    for (root, dirs, files) in os.walk(top_dir, topdown=True):
        leaf = np.any([file[-11:] == 'results.pkl' for file in files])
        if leaf:
            leaf_dirs.append(root)

    return {leaf_dir: check_results_within_directory(leaf_dir) for leaf_dir in
            leaf_dirs}


def get_filenames_of_incompletes_within_tree(top_dir):

    incomplete_paths = []
    for dir, results in check_results_within_tree(top_dir).items():
        for result in results:
            if result[1] is False:
                # if result is incomplete
                incomplete_paths.append(os.path.join(dir, result[-1]))
    return incomplete_paths


def get_filenames_of_all_results_within_tree(top_dir):
    paths = []
    for dir, results in check_results_within_tree(top_dir).items():
        for result in results:
            paths.append(os.path.join(dir, result[-1]))
    return paths


def get_factors(M, n_dim):
    """
    get the k and l factors which produce an n_dim problem for wfg with
    M objectives
    :param M: int
              number of objectives in the problem
    :param n_dim: int
              dimensionality of the parameter space desired
    :return: int, int
             kfactor, lfactor
    """
    best_score = n_dim

    possible_combinations = np.vstack(list(product(range(1, n_dim+1), repeat=2)))
    for kfactor, lfactor in possible_combinations:
        k = kfactor * (M - 1)  # position related params
        l = lfactor * 2  # distance related params
        if k+l == n_dim:
            score = abs(kfactor-lfactor)
            if score<best_score:
                best_score = score
                best_kf, best_lf = kfactor, lfactor
    return best_kf, best_lf


def strip_problem_names(folder):
    """
    get the problem number, number of objectives and number of parameters
    from the directory name
    :param folder:
    :return:
    """
    (prob, obj, dim) = folder.split('_')
    prob = int(prob.lower().strip("wfg"))
    obj = int(obj.lower().strip("obj"))
    dim = int(dim.lower().strip("dim"))
    return prob, obj, dim


def draw_samples(func, n_obj, n_dim, n_samples, random=False):
    """

    :param func:
    :param n_obj:
    :param n_dim:
    :param n_samples:
    :param random:
    :return:
    """
    kfactor, lfactor = get_factors(n_obj, n_dim)
    k = kfactor * (n_obj - 1)  # position related params
    l = lfactor * 2  # distance related params
    if random:
        x = np.array(lhsmdu.sample(numDimensions=n_dim, numSamples=n_samples)).T
        y = np.array([func(xi, k, n_obj) for xi in x])
    else:
        x = np.zeros((n_samples, n_dim))
        y = np.zeros((n_samples, n_obj))
        for n in range(n_samples):
            z = wfg.random_soln(k, l, func.__name__)
            x[n, :] = z
            y[n, :] = func(z, k, n_obj)

    return x, y


def gen_polar_point(theta, gamma, r):
    """
    generates a 3D point from polar coordinates
    """
    x = r * np.sin(theta) * np.cos(gamma)
    y = r * np.sin(theta) * np.sin(gamma)
    z = r * np.cos(theta)
    return np.array([x, y, z]).reshape(1, -1)


def down_sample(y, out_size):
    """
    Down-samples point pool y to size out_size, keeping the
    most sparse population possible.

    params:
        y [np.ndarray]: initial poolof points to be downsampled
        dimensions = [n_points, point_dim]
        out_size [int]: number of points in downsampled population
        muse be smaller than y.shape[0].
    """
    assert out_size < y.shape[0]
    pool = y.copy()
    in_pool = pool[:out_size]
    out_pool = pool[out_size:]
    M = distance_matrix(in_pool, in_pool)
    np.fill_diagonal(M, np.nan)
    for p in out_pool:
        arg_p = np.nanargmin(M)
        i = arg_p // M.shape[0]
        j = arg_p % M.shape[0]
        min_M = M[i, j]

        p_dist = cdist(p[np.newaxis, :], in_pool)[0]
        if p_dist.min() < min_M:
            # query point no improvement
            pass
        else:
            M[i] = p_dist
            M[:, i] = p_dist.T
            M[i, i] = np.nan
            in_pool[i] = p
    return in_pool


def find_neighbours(pool, p, thresh, show_dist=False):
    D = distance_matrix(pool, p)
    pool_nn = np.min(D, axis=1)
    assert pool_nn.shape[0] == pool.shape[0]

    if show_dist:
        plt.hist(pool_nn, int(pool.shape[0] / 2));
        plt.title('Attainment front->Pareto front nn distances');
        plt.axvline(thresh, c="C3", linestyle='--')

    api = pool_nn < thresh
    return api


def weak_dominates(Y, x):
    """
    Test whether rows of Y weakly dominate x

    Parameters
    ----------
    Y : array_like
        Array of points to be tested.

    x : array_like
        Vector to be tested

    Returns
    -------
    c : ndarray (Bool)
        1d-array.  The ith element is True if Y[i] weakly dominates x
    """
    return (Y <= x).sum(axis=1) == Y.shape[1]


def attainment_sample(Y, n_samples=1000):
    """
    Return samples from the attainment surface defined by the mutually non-dominating set Y

    Parameters
    ---------
    Y : array_like
        The surface to be sampled. Each row of Y is vector, that is mutually
        with all the other rows of Y
    n_samples : int
        Number of samples

    Returns
    -------
    S : ndarray
        Array of samples from the attainment surface.
        Shape; Nsamples by Y.shape[1]

    Notes
    -----
    See "Dominance-based multi-objective simulated annealing"
    Kevin Smith, Richard Everson, Jonathan Fieldsend,
    Chris Murphy, Rashmi Misra.
    IEEE Transactions on Evolutionary Computing.
    Volume: 12, Issue: 3, June 2008.
    https://ieeexplore.ieee.org/abstract/document/4358782
    """
    N, D = Y.shape
    Ymin = Y.min(axis=0)
    r = Y.max(axis=0) - Ymin
    S = np.zeros((n_samples, D))

    # Set up arrays of the points sorted according to each coordinate.
    Ys = np.zeros((N, D))
    for d in range(D):
        Ys[:, d] = np.sort(Y[:, d])

    for n in tqdm(range(n_samples)):
        v = np.random.rand(D) * r + Ymin
        m = np.random.randint(D)

        # Bisection search to find the smallest v[m]
        # so that v is weakly dominated by an element of Y
        lo, hi = 0, N
        while lo < hi:
            mid = (lo + hi) // 2
            v[m] = Ys[mid, m]
            if not any(weak_dominates(Y, v)):
                lo = mid + 1
            else:
                hi = mid
        if lo == N: lo -= 1
        v[m] = Ys[lo, m]
        assert lo == N - 1 or any(weak_dominates(Y, v))
        S[n, :] = v[:]
    return S