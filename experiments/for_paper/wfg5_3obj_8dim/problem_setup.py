
#! /bin/env python

import numpy as np
import wfg

## establish the objective function
M = 3                                   # Number of objectives
kfactor = 2
lfactor = 2
k = kfactor*(M-1)   # position related params
l = lfactor*2           # distance related params

n_dim = k+l
n_obj = M
func = wfg.WFG5


limits = np.zeros((2, n_dim))
limits[1] = np.array(range(1,n_dim+1))*2

def objective_function(x):
    if x.ndim<2:
        x = x.reshape(1,-1)
    return np.array([func(xi, k, M) for xi in x])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    N = 500 
    y = np.zeros((N, M))
    for n in range(N):
        z = wfg.random_soln(k, l, func.__name__)
        y[n,:] = func(z, k, M)
    print("objectives: {}".format(M))
    print("parameters: {}".format(n_dim))
    print("position params: {}". format(k))
    print("distance params: {}". format(l))
    assert(z.shape[0]==n_dim)
    
    
    fig = plt.figure()

    if n_obj > 4:
        pass
    else:
        if n_obj == 2:
            ax = fig.gca() 
        elif n_obj == 3:
            ax = fig.gca(projection="3d") 
        ax.scatter(*y.T)
        plt.suptitle(func.__name__)
        
        plt.show(block=True)
    print("Done!")
