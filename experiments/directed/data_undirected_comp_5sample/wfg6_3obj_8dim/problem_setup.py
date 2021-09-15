import numpy as np
import wfg
import lhsmdu
# establish the objective function




M = 3
n_dim = 8
kfactor, lfactor = 2, 2
k = kfactor*(M-1)   # position related params
l = lfactor*2           # distance related params

n_obj = M

func = getattr(wfg, 'WFG6')
def draw_samples(N, random=False):
    if random:
        x = np.array(lhsmdu.sample(numDimensions=n_dim, numSamples=N)).T
        y = np.array([func(xi, k, M) for xi in x])
    else:
        x = np.zeros((N, n_dim))
        y = np.zeros((N, n_obj))
        for n in range(N):
            z = wfg.random_soln(k, l, func.__name__)
            x[n,:] = z
            y[n,:] = func(z, k, M)

    return x, y

x, y = draw_samples(500)


print(func.__name__)

limits = np.zeros((2, n_dim))
limits[1] = np.array(range(1,n_dim+1))*2

def objective_function(x):
    if x.ndim<2:
        x = x.reshape(1,-1)
    return np.array([func(xi, k, M) for xi in x])

weighting = np.round(y.max(axis=0)*20, -1)/20
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    print("objectives: {}".format(M))
    print("factors k:{}\t l:\t{}".format(kfactor, lfactor))
    print("parameters: {}".format(n_dim))
    print("position params: {}". format(k))
    print("distance params: {}". format(l))
    print("weighting: {}".format(weighting))
    assert(x.shape[1]==n_dim)
    assert y.shape[1] == n_obj
    print("*************************")
    print((y/weighting).max(axis=0))
    np.testing.assert_array_almost_equal((y/weighting).max(axis=0), np.ones_like(y.max(axis=0)), 1)

    if n_obj > 3:
        pass
    else:
        samples = draw_samples(100, random=True)[1]
        fig = plt.figure(figsize=[10, 4])
        if n_obj == 2:
            ax1=fig.add_subplot(1,2,1)
            ax2=fig.add_subplot(1,2,2)
        elif n_obj == 3:
            ax1=fig.add_subplot(1,2,1, projection="3d")
            ax2=fig.add_subplot(1,2,2, projection="3d")
        ax1.scatter(*y.T, s=5)
        ax1.scatter(*samples.T, s=5, c="C3")
        ax2.scatter(*(y/weighting).T, c="C1", s=5)
        plt.suptitle(func.__name__)
        ax1.set_title("unscaled")
        ax2.set_title("scaled with axis maxima: {}".format((y/weighting).max(axis=0).round(3)))

        plt.show(block=True)
    print("Done!")
