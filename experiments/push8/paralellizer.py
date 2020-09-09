import os 
import numpy as np
import multiprocessing as mp
import rootpath
import sys
sys.path.append(rootpath.detect())
from testsuite.optimisers import SmsEgo, Saf
from testsuite.surrogates import GP, MultiSurrogate
from push_world import push_8D
from numpy import pi

# define push8 test problem
x_limits = [[-5, -5, 10, 0]*2, [5, 5, 300, 2*pi]*2]
o1 = [4, 4]
o2 = [0, -4]
t1 = [-3, 0]
t2 = [3, 0]


def test_function(x):
    return push_8D(x=x, t1=t1, t2=t2, o1=o1, o2=o2, draw=False)


# establish optimisers and surrogates.
surrogate = MultiSurrogate(GP, scaled=True)
optimisers = []
# for n in range(10, 30):
#     optimisers += [Saf(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=2),
#                   Saf(test_function, x_limits, surrogate,  n_initial=10, budget=100, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=2),
#                   SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=2),
#                   SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=2)]

optimisers = [Saf(test_function, x_limits, surrogate, ei=False, n_initial=10, budget=15, seed=0, log_dir="./log_data", cmaes_restarts=0, log_models=False)]

# def objective_function(optimiser):
optimisers[0].optimise()

# ## establish parallel processing pool
# n_proc = mp.cpu_count()
# print("{} processors found".format(n_proc))
# n_proc_cap = 12
# pool = mp.Pool(min(n_proc, n_proc_cap))
#
# pool.map(objective_function, optimisers)
#
# pool.close()
print("finished")

