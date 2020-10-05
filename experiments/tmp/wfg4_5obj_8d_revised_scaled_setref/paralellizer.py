import os  
import numpy as np 
import multiprocessing as mp 
import rootpath 
import sys 
sys.path.append(rootpath.detect()) 
import wfg 
from testsuite.optimisers import SmsEgo, Saf, Saf_Saf, Sms_Saf, Saf_Sms 
from testsuite.surrogates import GP, MultiSurrogate 


n_obj = 5                                   # Number of objectives
kfactor = 1
lfactor = 2

k = kfactor*(n_obj-1)   # position related params
l = lfactor*2           # distance related params
n_dim = k+l

func = wfg.WFG4

x_limits = np.zeros((2, n_dim))
x_limits[1] = np.array(range(1,n_dim+1))*2


ref = np.ones(n_obj)*1.2
surrogate = MultiSurrogate(GP, scaled=True)

args = [k, n_obj] # number of objectives as argument

def test_function(x):
    if x.ndim<2:
        x = x.reshape(1, -1)
    return np.array([func(xi, k, n_obj) for xi in x])/(np.array(range(1, n_obj+1))*2)

optimisers = []
for n in range(1, 10):
    optimisers += [SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir="./log_data", ref_vector=ref, cmaes_restarts=0),
                   SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=False, log_dir="./log_data", ref_vector=ref, cmaes_restarts=0),
                   Saf(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir="./log_data", ref_vector=ref, cmaes_restarts=0), 
                   Saf(test_function, x_limits, surrogate,  n_initial=10, budget=100, seed=n, ei=False, log_dir="./log_data", ref_vector=ref, cmaes_restarts=0)]
 

    # optimisers += [Sms_Saf(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=False, log_dir="./log_data", ref_vector=ref, cmaes_restarts=2), 
    #               Saf_Sms(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=False, log_dir="./log_data",  ref_vector=ref, cmaes_restarts=2), 
    #               Saf_Saf(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir="./log_data", ref_vector=ref, cmaes_restarts=2), 
    #               Saf(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir="./log_data", ref_vector=ref, cmaes_restarts=2), 
    #               Saf(test_function, x_limits, surrogate,  n_initial=10, budget=100, seed=n, ei=False, log_dir="./log_data", ref_vector=ref, cmaes_restarts=2), 
    #               SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=100, seed=n, ei=True, log_dir="./log_data", ref_vector=ref, cmaes_restarts=2)] 

def objective_function(optimiser):
    optimiser.optimise()

## establish parallel processing pool
n_proc = mp.cpu_count()
print("{} processors found".format(n_proc))
n_proc_cap = 4
pool = mp.Pool(min(n_proc, n_proc_cap))

pool.map(objective_function, optimisers)

pool.close()
print("finished")

