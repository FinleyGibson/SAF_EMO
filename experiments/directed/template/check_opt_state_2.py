"""

- take dir as argument
- count .results files
-

"""

import sys
import os
import numpy as np
import pickle


def get_result_dirs(parent_dir):
    """
    find all directories within a directory tree which contain results.pkl
    files
    :param parent_dir: str
                       top directory in the tree
    :return: list(str)
             list of directory paths for those which contain at least one file
             which ends in results.pkl
    """
    leaf_dirs = []
    for (root, dirs, files) in os.walk(parent_dir, topdown=True):
        leaf = np.any([file[-11:] == 'results.pkl' for file in files])
        if leaf:
            leaf_dirs.append(root)
    return leaf_dirs


def get_result_paths_from_dirs(dir_list):
    return {dir: [f for f in sorted(os.listdir(dir)) if
                  f[-11:] == "results.pkl"] for dir in dir_list}


def check_result_complete(file_path):
    with open(file_path, 'rb') as infile:
        result = pickle.load(infile)
    seed = result["seed"]
    state = (result['budget'] == result['n_evaluations'],
            (result['n_evaluations'], result['budget']))
    return seed, state


def check_results_within_tree(dir):
    results_dirs = get_result_dirs(dir)
    paths_dict = get_result_paths_from_dirs(results_dirs)

    D = {p: [] for p in paths_dict.keys()}
    for dir, paths in paths_dict.items():
        complete = []
        seeds = []
        states = []
        for path in paths:
            seed, state = check_result_complete(os.path.join(dir, path))
            seeds.append(seed)
            complete.append(state[0])
            states.append((seed, state))
        seeds_present = sorted(list(set(seeds))) == list(range(np.max(seeds)+1))
        if seeds_present and np.all(complete):
            D[dir] = (True, np.max(seeds))
        else:
            D[dir] = (False, states)
    return D


if __name__ == "__main__":
    enq_dir = sys.argv[1]
    assert os.path.isdir(enq_dir), "supplied directory does not exist."
    ans = check_results_within_tree(enq_dir)

    for k, v in ans.items():
        print(k)
        if v[0]:
            print(f"Complete up to seed {v[1]}")
        else:
            print(f"Incomplete. Current state:")
            for vi in v:
                print(vi)
        print()
