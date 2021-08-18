import os
import rootpath
from testsuite.analysis_tools import strip_problem_names, get_factors


# target_dir = os.raw_path.join(rootpath.detect(), "experiments/directed/data/")
target_dir = os.path.join(rootpath.detect(), "experiments/directed/data_undirected_comp/")
# target_dir = "./test_dir"
assert os.path.isdir(target_dir)

problem_list = [
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

if __name__ == "__main__":
    for folder in problem_list:
        prob, obj, dim = strip_problem_names(folder)
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