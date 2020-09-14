import os 
import numpy as np
import multiprocessing as mp
import rootpath
import sys
sys.path.append(rootpath.detect())
from testsuite.optimisers import SmsEgo, Saf
from testsuite.surrogates import GP, MultiSurrogate
from push_problems import push8
from numpy import pi

# define push8 test problem
p = push8()
x_limits = [p.lb, p.ub]
def test_function(x):
    return p(x)


# establish optimisers and surrogates.
surrogate = MultiSurrogate(GP, scaled=True)
optimisers = []
for n in range(30):
    optimisers += [Saf(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=0, log_interval=10),
                  Saf(test_function, x_limits, surrogate,  n_initial=10, budget=100, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=0, log_interval=10),
                  SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=0, log_interval=10),
                  SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=0, log_interval=10)]

def objective_function(opt):
    opt.optimise()


## establish parallel processing pool
n_proc = mp.cpu_count()
print("{} processors found".format(n_proc))
n_proc_cap = 12
pool = mp.Pool(min(n_proc, n_proc_cap))

pool.map(objective_function, optimisers)

pool.close()
print("finished")

