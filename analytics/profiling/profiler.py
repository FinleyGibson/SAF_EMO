import rootpath 
import sys 
sys.path.append(rootpath.detect()) 
import cProfile

import numpy as np
import wfg 
from testsuite.surrogates import GP, MultiSurrogate 

opt_name = str(sys.argv[1])
exec('from testsuite.optimisers import {} as optimiser'.format(opt_name))

print('Loaded optimiser', optimiser, 'from testsuite.optimisers')


# establish optimisation problem
n_obj = 5                                   # Number of objectives
kfactor = 2
lfactor = 2

k = kfactor*(n_obj-1)   # position related params
l = lfactor*2           # distance related params
n_dim = k+l

func = wfg.WFG6

args = [k, n_obj] # number of objectives as argument

def test_function(x):
    if x.ndim<2:
        x = x.reshape(1, -1)
    return np.array([func(xi, k, n_obj) for xi in x])
print('Establishe', func, ' as test problem with {} objectives and {} dimensional parameter space.'.format(n_obj, n_dim))


x_limits = np.zeros((2, n_dim))
x_limits[1] = np.array(range(1,n_dim+1))*2
surrogate = MultiSurrogate(GP, scaled=True)


cProfile.run('opt_instance = optimiser(test_function, x_limits, surrogate, n_initial=10, budget=11, seed=0, ei=True, log_dir="./log_data", cmaes_restarts=0)',  '{}_init.profile'.format(opt_name))
print('optimiser instantiated')

cProfile.run('opt_instance.optimise()', '{}_step.profile'.format(opt_name))
print('optimiser finished')

