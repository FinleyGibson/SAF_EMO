import rootpath
import os
import sys
import json
import numpy as np
from tqdm import tqdm

from testsuite.results import ResultsContainer

def get_result_leaves_from_tree_parent(parent_path):
    p = []
    for parent, dirs, files in os.walk(parent_path):
        if any([file[-11:] == "results.pkl" for file in os.listdir(parent)]):
            p.append(parent)
    return p


igdref_file = os.path.join(
    rootpath.detect(),
    "experiments/directed/template/targets/reference_points")
with open(igdref_file, 'r') as infile:
    IGD_REFPOINTS = json.load(infile)

target_file = os.path.join(
    rootpath.detect(),
    "experiments/directed/template/targets/targets")
with open(target_file, 'r') as infile:
    TARGETS = json.load(infile)


# configure paths from arguments
PROCESSED_RESULTS_DIR = \
    os.path.join(rootpath.detect(),
                 'experiments/directed/analysis/processed_results/')

RESULTS_DIR = sys.argv[1]
REFERENCE_DIR = sys.argv[2]

# acquire all result dirs within provided tree
t_result_paths = sorted(get_result_leaves_from_tree_parent(RESULTS_DIR))
ref_path = get_result_leaves_from_tree_parent(REFERENCE_DIR)[0]

# generate output file names from objective and target
filename_dict = {t: t.split('/')[-3]+"_"+t.split('/')[-1].split("target")[-1]
                    +'_processed.pkl'
                 for t in t_result_paths}
for result_path, file_name in tqdm(filename_dict.items(), leave=False):
    file_path = os.path.join(PROCESSED_RESULTS_DIR, file_name)
    problem_name = result_path.split("/")[-3]
    ref_points = np.asarray(IGD_REFPOINTS[problem_name])

    rc = ResultsContainer(result_path)
    rc.add_reference_data(ref_path)

    # rc.compute_hpv_history(sample_freq=10)
    # rc.compute_doh_history(sample_freq=10)
    rc.compute_igd_history(reference_points=ref_points, sample_freq=10)
    rc.compute_dual_hpv_history(sample_freq=10)

    rc.save(path=file_path)

if __name__ == "__main__":
    os.path.isdir(RESULTS_DIR)
    os.path.isdir(REFERENCE_DIR)
    os.path.isdir(PROCESSED_RESULTS_DIR)

    ans = []
    for file_name in os.listdir(PROCESSED_RESULTS_DIR):
        file_path = os.path.join(PROCESSED_RESULTS_DIR, file_name)
        ans.append(ResultsContainer(file_path))

    pass
