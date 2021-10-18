"""

- take dir as argument
- count .results files
-

"""
import sys
import os
import rootpath
sys.path.append(rootpath.detect())
import numpy as np
from testsuite.analysis_tools import get_result_dirs_from_tree
from testsuite.analysis_tools import check_results_within_directory


enq_dir = sys.argv[1]
assert os.path.isdir(enq_dir)
print(enq_dir)
try:
    n_expected = int(sys.argv[2])
except IndexError:
    n_expected = 31

results_dirs = get_result_dirs_from_tree(enq_dir)
n_complete = 0
n_incomplete = 0
for rd in results_dirs:

    ans = check_results_within_directory(rd)
    seeds = [state[0] for state in ans]
    completed = sorted([state[1] for state in ans])
    expected_seeds = list(range(n_expected))
    if np.all(completed):
        if expected_seeds == seeds:
            n_complete += 1
            print(f"All complete up to seed: {max(seeds)}, \t Directory: {rd}")
        else:
            n_incomplete += 1
            missing_seeds = set(expected_seeds)-set(seeds)
            print(f"INCOMPLETE missing seeds: {missing_seeds}, \t Directory: {rd}")
    else:
        print(f"Directory: {rd}")
        for seed, complete, (n_evaluated, budget), filename in ans:
            if not complete:
                print(f"\t seed {seed} incomplete ({n_evaluated}/{budget})")
            else:
                print(f"\t seed {seed} complete ({n_evaluated}/{budget})")
