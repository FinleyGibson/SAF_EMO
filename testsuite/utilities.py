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
    Loads a result.pkl file from the path supplied by result_path and
    checks the state of it; how many of the intended evaluations have
    been made.
    :param dir_path: str
                string containing the path to the directory containing
                results.pkl files, either an absolute or relative path.
    :param file_name: str
                string containing the name of the file within dir_path
                to be checked
    :return: tuple: (int, bool, (int, int), str)

    :raises: AssertionError: supplied path invalid; not path to results.pkl
                             file
    """
    assert file_name[-11:] == 'results.pkl', "path supplied to " \
                                             "check_state_of_result not a " \
                                             "path to results.pkl file."
    # load result
    with open(os.path.join(dir_path, file_name), 'rb') as infile:
        result = pickle.load(infile)

    # check completion status
    comp = result['n_evaluations'] == result['budget']

    # format and return tuple
    return result['seed'], comp, (result['n_evaluations'], result['budget']),\
           file_name


def check_results_within_directory(dir):
    """
    Checks the state of the results files within the supplied directory.

    :param dir: str
                string path to the directory in which to check the results.
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
                path to the directory which forms the topmost directory in the
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


if __name__ == "__main__":
    # test_ref = '/home/finley/phd/code/testsuite/experiments/directed/data/wfg1_2obj_3dim/log_data/OF_objective_function__opt_DirectedSaf__ninit_10__surrogate_MultiSurrogateGP__ei_False__target_1p68_1p09__w_0p5'
    # test_ref = '/home/finley/phd/code/testsuite/experiments/directed/data/wfg2_4obj_5dim/'
    test_ref = '/home/finley/phd/code/testsuite/experiments/directed/data/'
    ans = get_filenames_of_incompletes_within_tree(test_ref)

    print(ans)


