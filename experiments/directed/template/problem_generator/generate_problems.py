import os
import sys
from itertools import product
import numpy as np
import rootpath


def get_factors(M, n_dim):
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


def strip_folder_names(folder):
    (prob, obj, dim) = folder.split('_')
    prob = int(prob.strip("wfg"))
    obj = int(obj.strip("obj"))
    dim = int(dim.strip("dim"))
    return prob, obj, dim

target_dir = os.path.join(rootpath.detect(), "experiments/directed/data/")
# target_dir = "./test_dir"
assert os.path.isdir(target_dir)

folder_list = [
    'wfg1_2obj_3dim',
    'wfg1_3obj_4dim',
    'wfg1_4obj_5dim',
    'wfg2_2obj_6dim',
    'wfg2_3obj_6dim',
    'wfg2_4obj_10dim',
    'wfg3_2obj_6dim',
    'wfg3_3obj_10dim',
    'wfg3_4obj_10dim',
    'wfg4_2obj_6dim',
    'wfg4_3obj_8dim',
    'wfg4_4obj_8dim',
    'wfg5_2obj_6dim',
    'wfg5_3obj_8dim',
    'wfg5_4obj_10dim',
    'wfg6_2obj_6dim',
    'wfg6_3obj_8dim',
    'wfg6_4obj_10dim']

for folder in folder_list:
    prob, obj, dim = strip_folder_names(folder)
    kf, lf = get_factors(obj, dim)

    with open("./problem_setup_template") as infile:
        contents = infile.readlines()

    contents.insert(8, "M = {}".format(obj))
    contents.insert(9+1, "n_dim = {}".format(dim))
    contents.insert(10+2, "kfactor, lfactor = {}, {}".format(kf, lf))
    contents.insert(16+3, "func = getattr(wfg, 'WFG{}')".format(prob))

    os.makedirs(os.path.join(target_dir, folder))
    with open(os.path.join(target_dir, folder, "problem_setup.py"), "w") as f:
        contents = "".join(contents)
        f.write(contents)