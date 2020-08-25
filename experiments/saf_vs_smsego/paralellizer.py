import os 
import numpy as np
import multiprocessing as mp
import rootpath
import sys
sys.path.append(rootpath.detect())
import wfg
from testsuite.optimisers import SmsEgo, Saf
from testsuite.surrogates import GP, MultiSurrogate

## establish objective function
kfactor = 1
lfactor = 2 
M = 5 # number of "underlying positional parameters" +1 
k = kfactor*(M-1) # position related parameers (must be devisible by M-1)
l = lfactor*2 # distance-related parameters, muist be even for WFG2 & WFG3
l = 3

n_obj = 2 # must be from 1:M 
n_dim = l+k

x_limits = np.zeros((2, n_dim))
x_limits[1] = np.array(range(1,n_dim+1))*2

# fun = BM.wfg
fun =wfg.WFG6
args = [k, n_obj] # number of objectives as argument

ans = fun(np.ones_like(x_limits[0]), k, n_obj)
surrogate = MultiSurrogate(GP, scaled=True)

optimisers = []
for n in range(10):
    optimisers += [Saf(fun, x_limits, surrogate, of_args=args, n_initial=10, budget=30, seed=n, ei=True, log_dir="./log_data", log_interval=10, cmaes_restarts=2),
                  Saf(fun, x_limits, surrogate,  of_args=args, n_initial=10, budget=30, seed=n, ei=False, log_dir="./log_data", log_interval=10, cmaes_restarts=2),
                  SmsEgo(fun, x_limits, surrogate, of_args=args, n_initial=10, budget=30, seed=n, ei=False, log_dir="./log_data", log_interval=10, cmaes_restarts=2),
                  SmsEgo(fun, x_limits, surrogate, of_args=args, n_initial=10, budget=30, seed=n, ei=False, log_dir="./log_data", log_interval=10, cmaes_restarts=2)]


def objective_function(optimiser):
    optimiser.optimise(n_steps=100)

## establish parallel processing pool
n_proc = mp.cpu_count()
print("{} processors found".format(n_proc))
n_proc_cap = 15
pool = mp.Pool(min(n_proc, n_proc_cap))

pool.map(objective_function, optimisers)

pool.close()
print("finished")

