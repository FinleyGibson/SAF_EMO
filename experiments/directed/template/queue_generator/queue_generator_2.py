import rootpath
import sys
sys.path.append(rootpath.detect())
import os
from testsuite.surrogates import GP, MultiSurrogate
from testsuite.directed_optimisers import DirectedSaf
sys.path.append(sys.argv[1])
from problem_setup import func, objective_function, limits, n_dim, n_obj
from json import load
from persistqueue import SQLiteAckQueue
from filelock import FileLock
print(rootpath.detect())

# path to optimisier
optimiser_path = str(sys.argv[1])

# lock path
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

# set optimiser parameters
budget = 150
log_dir = os.path.join(optimiser_path, "log_data/")
cmaes_restarts = 1
surrogate = MultiSurrogate(GP, scaled=True)

# set up lock file to manage queue access
lock = FileLock(lock_path)
with lock:
    q = SQLiteAckQueue(queue_path, multithreading=True)


opt_opts = {'dsaf': "DirectedSaf(objective_function=objective_function, "
                    "ei=False,  targets=t, w=0.5, limits=limits, "
                    "surrogate=surrogate, n_initial=10, budget=budget, "
                    "log_dir=log_path, seed=seed)"}

if len(sys.argv) > 2:
    pass
    # TODO: add this
    # # if remaining optimisations are provided
    #
    # # read remaining opts
    # with open(sys.argv[2], 'rb') as infile:
    #     lst = pickle.load(infile)

else:
    # do initial optimisations
    seeds = range(0, 6)

    # add optimsers to queue
    optimisers = []
    for seed in seeds:
        for t in targets:
            exec('optimisers += [{}]'.format(opt_opts['dsaf']))
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
