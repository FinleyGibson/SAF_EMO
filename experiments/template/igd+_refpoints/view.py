import numpy as np
import matplotlib.pyplot as plt
import sys

def plot_2d(y):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.gca()
    ax.scatter(*y.T, s=5)
    ax.set_aspect('equal')
    return fig

def plot_3d(y):
    fig = plt.figure(figsize=[10, 10])
    ax = fig.gca(projection="3d")
    ax.scatter(*y.T, s=5)
    return fig

def plot_4d(y):
    fig = plt.figure(figsize=[18, 10])
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)

    ax0.scatter(*y[:, [0, 1]].T, s=5)
    ax1.scatter(*y[:, [0, 2]].T, s=5)
    ax2.scatter(*y[:, [0, 3]].T, s=5)

    fig2 = plt.figure(figsize=[18, 10])
    ax20 = fig2.add_subplot(131, projection="3d")
    ax21 = fig2.add_subplot(132, projection="3d")
    ax22 = fig2.add_subplot(133, projection="3d")

    ax20.scatter(*y[:, [0, 1, 2]].T, s=5)
    ax21.scatter(*y[:, [0, 2, 3]].T, s=5)
    ax22.scatter(*y[:, [1, 2, 3]].T, s=5)
    return fig, fig2

y = np.load(sys.argv[1])
n_obj = y.shape[1]
print("Problem details:")
print(n_obj, " objectives")
print(y.shape[0], "ref points")

scales = np.round(y.max(axis=0), 0)

if n_obj == 2:
    fig = plot_2d(y)
elif n_obj == 3:
    fig = plot_3d(y)
elif n_obj ==4:
    fig, fig2 = plot_4d(y)
else:
    print("ERROR {} OBJECTIVES DETECTED".format(n_obj))
plt.show()

