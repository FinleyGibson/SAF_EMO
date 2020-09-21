
#! /bin/env python
#
#  Plot solutions from a Walking Fish Group front with 3 objectives
#

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import wfg

N = 500 
M = 4                                   # Number of objectives
n_obj = M
kfactor = 2
lfactor = 3

k = kfactor*(M-1)
l = lfactor*2
n_dim = k+l

func = wfg.WFG4

y = np.zeros((N, M))
for n in range(N):
        z = wfg.random_soln(k, l, func.__name__)
        y[n,:] = func(z, k, M)


y = np.zeros((N, n_obj))
for n in range(N):
    z = wfg.random_soln(k, l, func.__name__)
    y[n,:] = func(z, k, n_obj)
print("objectives: {}".format(n_obj))
print("parameters: {}".format(n_dim))
print("position params: {}". format(k))
print("distance params: {}". format(l))
assert(z.shape[0]==n_dim)


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(y[:,0], y[:,1], y[:,2])
plt.suptitle(func.__name__)

plt.show(block=True)
