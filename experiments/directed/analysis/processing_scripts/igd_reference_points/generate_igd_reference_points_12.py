import json
from testsuite.analysis_tools import strip_problem_names, draw_samples, \
    attainment_sample, get_target_igd_refpoints
from testsuite.utilities import Pareto_split, KDTree_distance
import matplotlib.pyplot as plt
import wfg
import numpy as np

# path to save file
dict_path = 'reference_points'

# number of desired points for n dimensions
N_POINTS = {
    2: 2000,
    3: 4000,
    4: 8000
}

PROBLEMS = [
    "wfg1_2obj_3dim",
    "wfg1_3obj_4dim",
    "wfg1_4obj_5dim",
    "wfg2_2obj_6dim",
    "wfg2_3obj_6dim",
    "wfg2_4obj_10dim"]

try:
    with open(dict_path, 'r') as infile:
        D = json.load(infile)
except FileNotFoundError:
    D = {problem: None for problem in PROBLEMS}

if __name__ == "__main__":
    for name in PROBLEMS:
        if name in D.keys() or D[name] is None:
            # generate problem from name
            prob, obj, dim = strip_problem_names(name)
            func = getattr(wfg, f"WFG{prob}")

            samples_approved = False
            samples_pareto = N_POINTS[obj]*2
            samples_attainment = N_POINTS[obj]
            while samples_approved is False:
                # draw pareto front samples
                x, y = draw_samples(func=func, n_obj=obj, n_dim=dim,
                                    n_samples=samples_pareto)
                assert y.shape[0] == samples_pareto
                assert x.shape[1] == dim
                assert y.shape[1] == obj

                # ensure pareto dominance
                p_ind, d_ind = Pareto_split(y, return_indices=True)
                x = x[p_ind]
                y = y[p_ind]

                # attainment sample
                ya = attainment_sample(y, samples_attainment)

                # prune attainment samples based on proximity to samples
                min_dist = KDTree_distance(y, ya)
                threshold = KDTree_distance(ya, y).max()
                keep_indices = min_dist < threshold
                ya = ya[keep_indices]

                n_points = ya.shape[0]
                if n_points > N_POINTS[obj]:
                    ya = ya[np.random.choice(ya.shape[0], N_POINTS[obj],
                                             replace=False)]
                    samples_approved = True
                    D[name] = ya.tolist()

                    fig = plt.figure(figsize=[8, 8])
                    if obj == 2:
                        ax = fig.gca()
                        ax.scatter(*y.T, c="C0", alpha=0.2, s=10)
                        ax.scatter(*ya.T, c="C3", s=3)
                    elif obj == 3:
                        ax = fig.gca(projection="3d")
                        ax.scatter(*y.T, c="C0", alpha=0.2, s=10)
                        ax.scatter(*ya.T, c="C3", s=3)
                    # plt.show()

                    print("saving")
                    with open(dict_path, "w") as outfile:
                        json.dump(D, outfile)
                else:
                    samples_pareto *= 2
                    samples_attainment *= 2

    print("Done")
    plt.show()
    print("Done")
