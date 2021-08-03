"""

- take dir as argument
- count .results files
-

"""
import sys
import rootpath
sys.path.append(rootpath.detect())
import numpy as np
from testsuite.utilities import get_result_dirs_from_tree
from testsuite.utilities import check_results_within_directory


enq_dir = sys.argv[1]
try:
    n_expected = int(sys.argv[2])
except IndexError:
    n_expected = 0

results_dirs = get_result_dirs_from_tree(enq_dir)
n_complete = 0
n_incomplete = 0
for rd in results_dirs:
    # print(rd)
    ans = check_results_within_directory(rd)
    seeds = [state[0] for state in ans]
    pass
    completed = [state[1] for state in ans]
    if np.all(completed):
        if max(seeds)>=n_expected:
            n_complete += 1
            print(f"All complete up to seed: {max(seeds)}, \t Directory: {rd}")
        else:
            n_incomplete += 1
            print(f"MISSING! Complete up to seed: {max(seeds)}, expected {n_expected} \t Directory: {rd}")
    else:
        print(f"Directory: {rd}")
        for seed, complete, (n_evaluated, budget), filename in ans:
            if not complete:
                print(f"\t seed {seed} incomplete ({n_evaluated}/{budget})")
            else:
                print(f"\t seed {seed} complete ({n_evaluated}/{budget})")
