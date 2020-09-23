
#! /bin/env python
#
#  Plot solutions from a Walking Fish Group front with 3 objectives
#

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import wfg

n_obj = 3                                   # Number of objectives
kfactor = 5
lfactor = 2

k = kfactor*(n_obj-1)   # position related params
l = lfactor*2           # distance related params
n_dim = k+l

func = wfg.WFG6


N = 500 
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