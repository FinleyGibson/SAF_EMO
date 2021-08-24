import json
from testsuite.analysis_tools import strip_problem_names, draw_samples, \
    attainment_sample, get_target_igd_refpoints
from testsuite.utilities import Pareto_split
from scipy.spatial import distance_matrix, KDTree
import matplotlib.pyplot as plt
import wfg
import numpy as np

def KDTree_distance(a, b):
    """
    quick method to find the neerest neighbours for all items of array a
    in array b
    :param a: np.array()
    :param b: np.array()
    :return: np.array()
        array of minimum distances of array a from b
    """
    tree = KDTree(a)
    return tree.query(b)[0]

n_samples = 100000
n_attainment = 10000
D = {}

with open("./targets", "rb") as infile:
    target_dict = json.load(infile)

count = 0
for name, targets in target_dict.items():
    count += 1
    # if count != 2:
    #     continue
    # get problem
    try:
        prob, obj, dim = strip_problem_names(name)
    except ValueError:
        prob = 4
        obj = int(name.split("_")[-1].strip("obj"))
        dim = obj+1
    func = getattr(wfg, f"WFG{prob}")

    # draw pareto front samples
    x, y = draw_samples(func=func, n_obj=obj, n_dim=dim, n_samples=n_samples)
    assert y.shape[0] == n_samples
    assert x.shape[1] == dim
    assert y.shape[1] == obj
    # check for pareto dominance
    p_ind, d_ind = Pareto_split(y, return_indices=True)
    x = x[p_ind]
    y = y[p_ind]

    # attainment sample
    ya = attainment_sample(y, n_attainment)

    # prune attainment samples based on proximity to samples
    min_dist = KDTree_distance(y, ya)
    threshold = KDTree_distance(ya, y).max()
    keep_indices = min_dist < threshold

    ya = ya[keep_indices]

    # find reference points for each target
    out_dict = {}
    for target in targets:
        target = np.array(target).reshape(1, -1)
        igd_points, other_points = get_target_igd_refpoints(target, ya)
        # add to target_dict
        out_dict[str(target)] = igd_points

    if obj == 2:
        fig, axes = plt.subplots(2, 3, figsize=[10, 10])
    elif obj == 3:
        fig = plt.figure(figsize=[10,10])
        ax0 = fig.add_subplot(231, projection="3d")
        ax1 = fig.add_subplot(232, projection="3d")
        ax2 = fig.add_subplot(233, projection="3d")
        ax3 = fig.add_subplot(234, projection="3d")
        ax4 = fig.add_subplot(235, projection="3d")
        ax5 = fig.add_subplot(236, projection="3d")
        axes = np.asarray(fig.axes)
    else:
        continue
    for ax, target in zip(axes.flatten(), targets):
        target = np.array(target).reshape(1, -1)
        ax.scatter(*ya[::10].T, s=5, c="C0", alpha=0.2)
        ax.scatter(*target.T, s=5, c="magenta", alpha=1.)
        c = out_dict[str(target)]
        ax.scatter(*c.T, s=5, c="C3", alpha=1.)
    D[name] = target_dict

with open("./reference_points", "w") as outfile:
    json.dump(D, outfile)
# print("Done")
plt.show()
# print("Done")
