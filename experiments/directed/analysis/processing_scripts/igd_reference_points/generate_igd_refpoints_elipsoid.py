import json
import numpy as np
from testsuite.analysis_tools import strip_problem_names, attainment_sample
from testsuite.utilities import KDTree_distance


def normalise_to_axes(x, axes=None):
    r = 1
    assert x.ndim == 2
    axes = np.array(axes) if axes is not None else np.ones(x.shape[1])

    x_norm = np.zeros_like(x)
    for i, xi in enumerate(x):
        lmbda = r ** 2 / np.sum(
            [xi[j] ** 2 / axes[j] ** 2 for j in range(x.shape[1])])
        x_norm[i] = xi * np.sqrt(lmbda)
    return x_norm

# path to save file
dict_path = 'reference_points'

# number of desired points for n dimensions
N_POINTS = {
    2: 2000,
    3: 4000,
    4: 8000
}

PROBLEMS = [
    "wfg4_2obj_6dim",
    "wfg4_3obj_8dim",
    "wfg4_4obj_8dim",
    "wfg5_2obj_6dim",
    "wfg5_3obj_8dim",
    "wfg5_4obj_10dim",
    "wfg6_2obj_6dim",
    "wfg6_3obj_8dim",
    "wfg6_4obj_10dim"]

try:
    with open(dict_path, 'r') as infile:
        D = json.load(infile)
except FileNotFoundError:
    D = {problem: None for problem in PROBLEMS}

for problem in PROBLEMS:
    D.__delitem__(problem)

axis_scales = np.array([2, 4, 6, 8])

if __name__ == "__main__":
    for name in PROBLEMS:
        if name not in D.keys() or D[name] is None:
            # generate problem from name
            prob, obj, dim = strip_problem_names(name)
            samples_pareto = N_POINTS[obj]
            samples_attainment = N_POINTS[obj]
            samples_approved = False

            # draw random samples in multivariate norm
            y = abs(np.random.multivariate_normal(
                np.zeros(obj),
                np.diag(axis_scales[:obj]**2),
                size=samples_pareto))

            while samples_approved is False:
                # project to elipsoide surface (non-uniform distribution)
                y = abs(normalise_to_axes(y, axis_scales[:obj]))

                # sample uniformly
                ya = attainment_sample(y, samples_attainment)

                # prune attainment samples based on proximity to samples
                min_dist = KDTree_distance(y, ya)
                threshold = KDTree_distance(ya, y).max()
                keep_indices = min_dist < threshold
                ya = ya[keep_indices]

                if ya.shape[0] > N_POINTS[obj]:
                    ya = ya[np.random.choice(ya.shape[0], N_POINTS[obj],
                                             replace=False)]
                    samples_approved = True

                    for n in PROBLEMS:
                        p, o, d = strip_problem_names(n)
                        if o == obj:
                            D[n] = ya.tolist()
                    print("saving")
                    with open(dict_path, "w") as outfile:
                        json.dump(D, outfile)
                else:
                    samples_pareto *= 2
                    samples_attainment *= 2

    print("Done")
