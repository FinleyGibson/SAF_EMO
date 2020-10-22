import os  
import numpy as np 
import multiprocessing as mp 
import rootpath 
import sys 
sys.path.append(rootpath.detect()) 
import wfg 
from testsuite.optimisers import *
from testsuite.surrogates import GP, MultiSurrogate 

n_obj = 3                                   # Number of objectives
kfactor = 2
lfactor = 1

k = kfactor*(n_obj-1)   # position related params
l = lfactor*2           # distance related params
n_dim = k+l

func = wfg.WFG2


x_limits = np.zeros((2, n_dim))
x_limits[1] = np.array(range(1,n_dim+1))*2


ref = np.ones(n_obj)*1.2
surrogate = MultiSurrogate(GP, scaled=True)

args = [k, n_obj] # number of objectives as argument

def test_function(x):
    if x.ndim<2:
        x = x.reshape(1, -1)
    return np.array([func(xi, k, n_obj) for xi in x])

budget = 250

if __name__ == "__main__":
    optimisers = []
    for n in range(8, 9):
        optimisers += [SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=0),
                      SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=0),
                      Saf(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=0), 
                      Saf(test_function, x_limits, surrogate,  n_initial=10, budget=budget, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=0),
                      ParEgo(objective_function=test_function, limits=x_limits, surrogate=GP(), seed=n, n_initial=10, s=5, rho=0.5, budget=budget, log_dir="./log_data", cmaes_restarts=0),
                      Mpoi(objective_function=test_function, limits=x_limits, surrogate=surrogate, n_initial=10, seed=n, budget=budget, cmaes_restarts=0),
                      Lhs(objective_function = test_function, limits=x_limits, n_initial=10, budget=budget, seed=n)]
    optimisers = [SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=0)] 
    # optimisers = [Saf(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=0, ei=True, log_dir="./log_data", cmaes_restarts=0)]
    # optimisers = [SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=0)]
    # optimisers = [SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=0)]
    # optimisers = [Saf(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=0)]
    # optimisers = [ParEgo(objective_function=test_function, limits=x_limits, surrogate=GP(), n_initial=10, s=5, rho=0.5, budget=budget, log_dir="./log_data", cmaes_restarts=0)]
    # optimisers = [Mpoi(objective_function=test_function, limits=x_limits, surrogate=surrogate, n_initial=10, seed=n, budget=budget, cmaes_restarts=0)]
    #optimisers = [Lhs(objective_function = test_function, limits=x_limits, n_initial=10, budget=budget, seed=n)]
    # optimisers = [SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=1, ei=True, log_dir="./log_data", cmaes_restarts=0)]
    
    
    def objective_function(optimiser):
        optimiser.optimise()
    
    ## establish parallel processing pool
    n_proc = mp.cpu_count()
    print("{} processors found".format(n_proc))
    n_proc_cap = 8 
    pool = mp.Pool(min(n_proc, n_proc_cap))
    
    pool.map(objective_function, optimisers)
    
    pool.close()
    print("finished")
