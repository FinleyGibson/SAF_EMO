import rootpath
import sys
sys.path.append(rootpath.detect())
import os
import pickle
import numpy as np
from itertools import product
from testsuite.surrogates import GP, MultiSurrogate
from testsuite.optimisers import Saf
from testsuite.analysis_tools import get_filenames_of_all_results_within_tree
sys.path.append(sys.argv[1])
from problem_setup import func, objective_function, limits, n_dim, n_obj
from json import load
from persistqueue import SQLiteAckQueue
from filelock import FileLock
print(rootpath.detect())


def get_seed_from_string(string):
    with open(string, 'rb') as infile:
        result = pickle.load(infile)
    return result['seed']


# raw_path to optimisier
optimiser_path = str(sys.argv[1])

# lock raw_path
lock_path = os.path.abspath(os.path.join(optimiser_path, "lock"))
queue_path = os.path.abspath(os.path.join(optimiser_path, "queue"))
log_path = os.path.abspath(os.path.join(optimiser_path, "log_data"))

# strip out function number
func_n = int(func.__name__.strip('WFG'))

# get targets
if n_obj>4:
    raise Exception

if func_n < 4:
    target_name = "WFG{}_{}obj_{}dim".format(func_n, n_obj, n_dim)
else:
    target_name = "ELLIPSOID_{}obj".format(n_obj)
with open("../targets/targets", "r") as infile:
    targets = load(infile)
targets = targets[target_name]
targets =[np.array(t).reshape(1,-1) for t in targets]

# set optimiser parameters
budget = 150
log_dir = os.path.join(optimiser_path, "log_data/")
cmaes_restarts = 1
surrogate = MultiSurrogate(GP, scaled=True)

# set up lock file to manage queue access
lock = FileLock(lock_path)
with lock:
    q = SQLiteAckQueue(queue_path, multithreading=True)


opt_opts = {'saf': "Saf(objective_function=objective_function, "
                   "ei=False, limits=limits, surrogate=surrogate,"
                   "n_initial=10, budget=budget, log_dir=log_path, seed=seed)"}

# do initial optimisations
seeds = list(range(6, 7))

# find which exist already
existing_result_paths = get_filenames_of_all_results_within_tree(log_dir)
existing_configs = [get_seed_from_string(path) for path in
                    existing_result_paths]


required_configs = np.array(list(seeds))

# finds which configs in required_configs are not in existing_configs
remaining_configs = [i for i in required_configs if i not in existing_configs]


# add outstanding optimsations to queue
optimisers = []
for seed in remaining_configs:
    exec('optimisers += [{}]'.format(opt_opts['saf']))
n_opt = len(optimisers)

if __name__ == "__main__":
    import shutil
    if q.size>0:
        print("{} items already in queue".format(q.size))
        reset = input("Would you like to delete the existing queue? Y/N:\t").lower()
        if reset == "y":
            reset = True
        elif reset == "n":
            reset = False
        else:
            print("Input not recognised")
    else:
        reset = True

    if reset == True:
        shutil.rmtree(queue_path, ignore_errors=True)
        print("removed existing queue.")
        with lock:
            q = SQLiteAckQueue(queue_path, multithreading=True)

    else:
        pass

    # add to queue

    with lock:
        for optimiser in optimisers:
            q.put(optimiser)

    print("Added {}  optimisers to ./opt_queue, queue length now {}.".format(n_opt, q.size))
