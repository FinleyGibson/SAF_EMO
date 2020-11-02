import rootpath 
import sys
sys.path.append(rootpath.detect())

import numpy as np
import persistqueue
from testsuite.surrogates import GP, MultiSurrogate
from testsuite.optimisers import *
from problem_setup import func, objective_function, limits
exec("objective_function.__name__ = '{}'".format(func.__name__))

## set up optimisers
mono_surrogate = GP(scaled=True)
multi_surrogate = MultiSurrogate(GP, scaled=True)

budget = 250
log_dir = "./log_data"
cmaes_restarts=0

seeds = range(11)

## set up queue
q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)

# add optimsers to queue
optimisers = []
for seed in seeds:
    #create optimisers
    optimisers += [SmsEgo(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed, ei=True, log_dir=log_dir, cmaes_restarts=cmaes_restarts, log_interval=5),
                  SmsEgo(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed, ei=False, log_dir=log_dir, cmaes_restarts=cmaes_restarts, log_interval=5),
                  Saf(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed, ei=True, log_dir=log_dir, cmaes_restarts=cmaes_restarts, log_interval=5),
                  Saf(objective_function=objective_function, limits=limits, surrogate=multi_surrogate,  n_initial=10, budget=budget, seed=seed, ei=False, log_dir=log_dir, cmaes_restarts=cmaes_restarts, log_interval=5),
                  ParEgo(objective_function=objective_function, limits=limits, surrogate=mono_surrogate, n_initial=10, seed=seed, budget=budget, cmaes_restarts=cmaes_restarts ,log_interval=5),
                  Mpoi(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, seed=seed, budget=budget, cmaes_restarts=cmaes_restarts ,log_interval=5),
                  Lhs(objective_function = objective_function, limits=limits, n_initial=10, budget=budget, seed=seed)]
 
n_opt = len(optimisers)

if __name__ == "__main__":
    import shutil
    if not q.empty():
        print("{} items already in queue".format(q.size))
        reset = input("Would you like to delete the existing queue? Y/N:\t").lower()
        if reset == "y":
            reset = True
        elif reset == "n":
            reset = False
        else:
            print("Input not recognised")
            raise InputError
    else:
        reset = False

    if reset == True:
        shutil.rmtree('./opt_queue')
        print("removed existing queue.")
        ## set up queue
        q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)
    else: 
        pass

    # add to queue
    for optimiser in optimisers:
        q.put(optimiser)

    print("Added {}  optimisers to ./opt_queue, queue length now {}.".format(n_opt, q.size))
