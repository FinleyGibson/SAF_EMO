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


def test_function(x):
    x = x.reshape(1, -1)
    return np.array([fun(xi, k, n_obj) for xi in x])


ans = fun(np.ones_like(x_limits[0]), k, n_obj)
surrogate = MultiSurrogate(GP, scaled=True)

optimisers = []
log_dir = "./log_test_data"
# for n in range(10, 30):

def objective_function(optimiser):
    optimiser.optimise(n_steps=2)

if __name__ == '__main__':

    for n in range(1):
        optimisers += [Saf(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir=log_dir, cmaes_restarts=2),
                      Saf(test_function, x_limits, surrogate,  n_initial=10, budget=100, seed=n, ei=False, log_dir=log_dir, cmaes_restarts=2),
                      SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=False, log_dir=log_dir, cmaes_restarts=2),
                      SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir=log_dir, cmaes_restarts=2)]

    ## establish parallel processing pool
    n_proc = mp.cpu_count()
    print("{} processors found".format(n_proc))
    n_proc_cap = 4
    
    with mp.Pool(4) as p:
        p.map(objective_function, optimisers)
    # for opt in optimisers:
    #     opt.optimise()
