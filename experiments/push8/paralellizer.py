import os  
import numpy as np 
import multiprocessing as mp 
import rootpath 
import sys 
sys.path.append(rootpath.detect()) 
import wfg 
from testsuite.optimisers import *
from testsuite.surrogates import GP, MultiSurrogate 
from push_world import push_8D

class push8:
    """Robot pushing simulation: Two robots pushing an object each.
    The objects' initial locations are at [-3, 0] and [3, 0] respectively,
    with the robot 1 pushing the first target and robot 2 pushing the second
    target. See paper for full problem details.
    Parameters
    ----------
    tx_1 : float
        x-axis location for the target of robot 1, should reside in [-5, 5].
    ty_1 : float
        y-axis location for the target of robot 1, should reside in [-5, 5].
    tx_2 : float
        x-axis location for the target of robot 2, should reside in [-5, 5].
    ty_2 : float
        y-axis location for the target of robot 2, should reside in [-5, 5].
    Examples
    --------
    >> f_class = push8
    >> # initial positions (tx_1, ty_1) and (tx_2, ty_2) for both robots
    >> tx_1 = 3.5; ty_1 = 4
    >> tx_2 = -2; ty_2 = 1.5
    >> # instantiate the test problem
    >> f = f_class(tx_1, ty_1, tx_2, ty_2)
    >> # evaluate some solution x in [0, 1]^8
    >> x = numpy.array([0.5, 0.7, 0.2, 0.3, 0.3, 0.1, 0.5, 0.6])
    >> f(x)
    array([24.15719287])
    """
    def __init__(self, t1_x=-5, t1_y=-5, t2_x=5, t2_y=5):
        self.dim = 8
        self.lb = np.array([-5, -5,   1, -5*np.pi, -5, -5,   1, -5*np.pi])
        self.ub = np.array([ 5,  5, 300, 5*np.pi,  5,  5, 300, 5*np.pi])

        # object target locations
        self.t1_x = t1_x
        self.t1_y = t1_y
        self.t2_x = t2_x
        self.t2_y = t2_y

        # initial object locations (-3, 0) and (3, 0)
        self.o1_x = -3
        self.o1_y = 0
        self.o2_x = 3
        self.o2_y = 0

        # optimum location unknown as defined by inputs
        self.yopt = np.array([0.])
        self.xopt = None

        self.cf = None

    def __call__(self, x):
        x = np.atleast_2d(x)

        val = np.zeros((x.shape[0], 2))
        for i in range(x.shape[0]):
            val[i, :] = push_8D(x[i, :],
                                self.t1_x, self.t1_y,
                                self.t2_x, self.t2_y,
                                self.o1_x, self.o1_y,
                                self.o2_x, self.o2_y,
                                draw=False)

        return val.ravel()

## establish objective function
func = push8()
x_limits = [func.lb, func.ub]
def test_function(x):
    if x.ndim ==1:
        x = x.reshape(1,-1)
    return np.array([func(xi) for xi in x])

surrogate = MultiSurrogate(GP, scaled=True)


budget = 250 

if __name__ == "__main__":
    optimisers = []
    for n in range(0, 11):
        optimisers += [SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=0),
                      SmsEgo(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=0),
                      Saf(test_function, x_limits, surrogate, n_initial=10, budget=budget, seed=n, ei=True, log_dir="./log_data", cmaes_restarts=0), 
                      Saf(test_function, x_limits, surrogate,  n_initial=10, budget=budget, seed=n, ei=False, log_dir="./log_data", cmaes_restarts=0),
                      ParEgo(objective_function=test_function, limits=x_limits, surrogate=GP(), seed=n, n_initial=10, s=5, rho=0.5, budget=budget, log_dir="./log_data", cmaes_restarts=0),
                      Mpoi(objective_function=test_function, limits=x_limits, surrogate=surrogate, n_initial=10, seed=n, budget=budget, cmaes_restarts=0),
                      Lhs(objective_function = test_function, limits=x_limits, n_initial=10, budget=budget, seed=n)]
    
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
    n_proc_cap = 11 
    pool = mp.Pool(min(n_proc, n_proc_cap))
    
    pool.map(objective_function, optimisers)
    
    pool.close()
    print("finished")
