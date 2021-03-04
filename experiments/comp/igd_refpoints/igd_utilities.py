import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
REPO_FIGDIR="/home/finley/phd/papers/Pareto_sampling/figures/"

def gen_polar_point(theta, gamma, r):
    """
    generates a 3D point from polar coordinates
    """
    x = r*np.sin(theta)*np.cos(gamma)
    y = r*np.sin(theta)*np.sin(gamma)
    z = r*np.cos(theta)
    return np.array([x, y, z]).reshape(1, -1)

def down_sample(y, out_size):
    """
    Down-samples point pool y to size out_size, keeping the 
    most sparse population possible.
    
    params:
        y [np.ndarray]: initial poolof points to be downsampled
        dimensions = [n_points, point_dim]
        out_size [int]: number of points in downsampled population
        muse be smaller than y.shape[0].
    """
    assert out_size<y.shape[0]
    pool = y.copy()
    in_pool = pool[:out_size] 
    out_pool = pool[out_size:] 
    M = distance_matrix(in_pool, in_pool)
    np.fill_diagonal(M, np.nan)
    for p in out_pool:
        arg_p = np.nanargmin(M)
        i = arg_p//M.shape[0]
        j = arg_p%M.shape[0]
        min_M = M[i,j]
        
        p_dist = cdist(p[np.newaxis,:], in_pool)[0]
        if p_dist.min()<min_M:
            # query point no improvement
            pass
        else:
            M[i] = p_dist 
            M[:, i] = p_dist.T
            M[i, i] = np.nan
            in_pool[i] = p
    return in_pool

def find_neighbours(pool, p, thresh, show_dist=False):
    D = distance_matrix(pool, p)
    pool_nn = np.min(D, axis=1)
    assert pool_nn.shape[0] == pool.shape[0]
    
    if show_dist:
        plt.hist(pool_nn, int(pool.shape[0]/2));
        plt.title('Attainment front->Pareto front nn distances');
        plt.axvline(thresh, c="C3", linestyle='--')
        
    api = pool_nn<thresh
    return api
    
def weak_dominates(Y, x):
    """
    Test whether rows of Y weakly dominate x
    
    Parameters
    ----------
    Y : array_like
        Array of points to be tested. 
        
    x : array_like
        Vector to be tested
        
    Returns
    -------
    c : ndarray (Bool)
        1d-array.  The ith element is True if Y[i] weakly dominates x
    """
    return (Y <= x ).sum(axis=1) == Y.shape[1]


def attainment_sample(Y, Nsamples=1000):
    """
    Return samples from the attainment surface defined by the mutually non-dominating set Y

    Parameters
    ---------
    Y : array_like
        The surface to be sampled. Each row of Y is vector, that is mutually
        with all the other rows of Y
    Nsamples : int
        Number of samples

    Returns
    -------
    S : ndarray
        Array of samples from the attainment surface.
        Shape; Nsamples by Y.shape[1] 
    
    Notes
    -----
    See "Dominance-based multi-objective simulated annealing"
    Kevin Smith, Richard Everson, Jonathan Fieldsend, 
    Chris Murphy, Rashmi Misra.
    IEEE Transactions on Evolutionary Computing. 
    Volume: 12, Issue: 3, June 2008.
    https://ieeexplore.ieee.org/abstract/document/4358782
    """
    N, D = Y.shape
    Ymin = Y.min(axis=0)
    r = Y.max(axis=0) - Ymin
    S = np.zeros((Nsamples, D))
    
    # Set up arrays of the points sorted according to each coordinate.
    Ys = np.zeros((N, D))
    for d in range(D):
        Ys[:,d] = np.sort(Y[:,d])

    for n in tqdm(range(Nsamples)):
        v = np.random.rand(D)*r + Ymin
        m = np.random.randint(D)

        # Bisection search to find the smallest v[m] 
        # so that v is weakly dominated by an element of Y
        lo, hi = 0, N
        while lo < hi:
            mid = (lo+hi)//2
            v[m] = Ys[mid,m]
            if not any(weak_dominates(Y, v)):
                lo = mid+1
            else:
                hi = mid
        if lo == N: lo -= 1
        v[m] = Ys[lo, m]      
        assert lo == N-1 or any(weak_dominates(Y, v))
        S[n,:] = v[:]
    return S

def ax_format(ax, axes, vp=None):
    ax.set_xlabel(r"$f_1\mathbf{x}$")
    ax.set_ylabel(r"$f_2\mathbf{x}$")
    if ax.name =="3d":
        print(axes)
        print("3d")
        ax.set_zlabel(r"$f_3\mathbf{x}$")
        ax.set_box_aspect(list(axes))
        if vp:
            ax.view_init(elev=vp[0], azim=vp[1])
    else:
        ax.set_aspect('equal')
    ax.legend()
    
def save_fig(fig, title):
    file_name= title.lower().replace(" ", "_").replace(".", "").replace(",", "").replace("\\", "").replace("$", "")
    for dr in [REPO_FIGDIR, "figures/"]:
        try: 
            fig.savefig(dr+file_name+".png", dpi=200, format="png", transparent=True)
            print("saved to ", dr+file_name+".png")
        except:
            print("failed to save ", dr+file_name+".png")
        try: 
            fig.savefig(dr+file_name+".pdf", dpi=200, format="pdf", transparent=True)
            print("saved to ", dr+file_name+".pdf")
        except:
            print("failed to save ", dr+file_name+".pdf")
    
    
def simple_3d_plot(x, title, axes=[1., 1., 1.], save=False, viewpoint=[35., 65.]):
    axes = np.asarray(axes).flatten()
    assert x.shape[1] == 3
    assert axes.shape[0] == 3
    
    fig = plt.figure(figsize=[8, 8])
    ax = fig.gca(projection="3d")
    ax.scatter(*x.T, s=8, c="C0", alpha=0.4, label="distributed points")
    ax.scatter(*np.zeros(3).T, s=35, c="C3", alpha=1., label="origin")
    ax_format(ax, axes, viewpoint)
    if save:
        save_fig(fig, title)
    return fig

def simple_2d_plot(x, title, axes=[1., 1.], save=False):
    axes = np.asarray(axes).flatten()
    assert x.shape[1] == 2
    assert axes.shape[0] == 2
    file_name= title.lower().replace(" ", "_").replace(".", "").replace(",", "").replace("\\", "").replace("$", "")
    fig = plt.figure(figsize=[10, 10])
    ax = fig.gca()
    
    x_norm = normalise_to_axes(x, axes)
    ax.scatter(*x.T, c="C0", alpha=0.3, s=5, label="{} original points".format(len(x)))
    ax.scatter(*x_norm.T, c="C1", alpha=0.1, s=8, label = "points projected to elipse")
    ax.scatter([0],[0], c="C3", label="origin")
    ax_format(ax, axes)
    if save:
        save_fig(fig, title)
    return fig

def normalise_length(points):
    magnitudes = np.sqrt(np.diag(np.dot(points, points.T)))
    return points/magnitudes.reshape(-1,1)

def normalise_to_axes(x, axes=None):
    r = 1
    assert x.ndim == 2
    axes = np.array(axes) if axes is not None else np.ones(x.shape[1])
    
    x_norm = np.zeros_like(x)
    for i, xi in enumerate(x):
        lmbda = r**2/np.sum([xi[j]**2/axes[j]**2 for j in range(x.shape[1])])
        x_norm[i] = xi*np.sqrt(lmbda)
    return x_norm

def gen_polar_point(theta, gamma, r):
    x = r*np.sin(theta)*np.cos(gamma)
    y = r*np.sin(theta)*np.sin(gamma)
    z = r*np.cos(theta)
    return np.array([x, y, z]).reshape(1, -1)