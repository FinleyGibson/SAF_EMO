import json
import os
import sys

import numpy as np
from multiprocessing import Pool
from testsuite.results import ResultsContainer
from testsuite.utilities import dominates
import time
import fasteners
from tqdm import tqdm

def trim_var(points, P, targets, ref_point):
    va_r = np.asarray(
        [dominates(targets, point) and dominates(point, ref_point)
         for point in points])
    vb_r = np.asarray([np.any(
        [dominates(pareto, point) for pareto in P]) and np.any(
        [dominates(point, t) for t in targets]) for point in points])

    try:
        return points[va_r], points[vb_r]
    except:
        pass


def sample_var_vbr(n, P, T, ref_point, ref_ideal):
    ref_point = ref_point.flatten()
    ref_ideal = ref_ideal.flatten()

    monte_points = np.vstack([np.random.uniform(l, u, n)
                              for l, u in zip(ref_ideal, ref_point)]).T
    assert monte_points.shape == (n, P.shape[1])

    # initial samples
    v_ar_points, v_br_points = trim_var(monte_points, P, T, ref_point)

    try:
        fraction_in_sample = n / (v_ar_points.shape[0] + v_br_points.shape[0])
    except ZeroDivisionError:
        fraction_in_sample = 10

    while v_ar_points.shape[0] + v_br_points.shape[0] < n:
        # sample an estimate of the number required times sampling fraction +.1
        n_to_sample = int((n-v_ar_points.shape[0] - v_br_points.shape[0])
                          *fraction_in_sample*1.2)

        monte_points = np.vstack([np.random.uniform(l, u, n_to_sample)
                                  for l, u in zip(ref_ideal, ref_point)]).T
        v_ar_new, v_br_new = trim_var(monte_points, P, T, ref_point)
        v_ar_points = np.append(v_ar_points, v_ar_new, axis=0)
        v_br_points = np.append(v_br_points, v_br_new, axis=0)
        pass

    # randomly down-sample to required number
    n_points = v_ar_points.shape[0] + v_br_points.shape[0]
    keep_inds = np.random.choice(n_points, n, False)
    keep_inds.sort()

    ar_inds = keep_inds[keep_inds<v_ar_points.shape[0]]
    br_inds = keep_inds[keep_inds>=v_ar_points.shape[0]]-v_ar_points.shape[0]

    return v_ar_points[ar_inds], v_br_points[br_inds]

problem = str(sys.argv[1])

n_samples_desired = int(3e2)
step_size = min(n_samples_desired, 500)

result_path = '../../../data/directed/'
PROBLEMS = sorted(list(os.listdir(result_path)))
assert problem in PROBLEMS, f"Problem '{problem}' not found. Specify one" \
                            f"of: \n" +"\n".join(PROBLEMS)

# compute reference point
rp_path = "../igd_reference_points/reference_points"
with open(rp_path, "r") as infile:
    igd_rps = json.load(infile)


problem_path = os.path.join(result_path, problem, "log_data")

D_path = f"./points/{problem}.json"
lock = fasteners.InterProcessLock(f'./locks/{problem}_lock.file')


# for target_dir in os.listdir(problem_path):
def worker_f(target_dir):
    tic = time.time()
    time.sleep(np.random.uniform(0,15))
    problem_target_path = os.path.join(problem_path, target_dir)
    reference_path = os.path.join("../../../data/undirected_comp", problem, "log_data")
    reference_path = os.path.join(reference_path, os.listdir(reference_path)[0])
    assert os.path.isdir(problem_target_path)
    assert os.path.isdir(reference_path)

    # load igd reference points
    igd_rp = np.asarray(igd_rps[problem])

    results = ResultsContainer(problem_target_path)
    results.add_reference_data(reference_path)
    results_p = np.vstack(results.p)
    reference_p = np.vstack([r.p for r in results.reference])
    targets = np.vstack(results.targets)

    ref_point = np.vstack((igd_rp, results_p, reference_p)).max(axis=0)
    ref_ideal = np.vstack((igd_rp, results_p, reference_p)).min(axis=0)

    n_obj = results_p.shape[1]

    key_string = problem + "__" + str(targets[0].round(3)).replace("[", "") \
        .replace("]", "").replace(" ", "_").replace(".", "p")

    try:
        with lock:
            with open(D_path, "r") as infile:
                D = json.load(infile)
    except :
        D = {}

    if key_string in D.keys():
        n_completed = len(D[key_string][0]) + len(D[key_string][1])
        n_samples_to_add = n_samples_desired - n_completed
    else:
        n_samples_to_add = n_samples_desired

    if key_string in D.keys():
        ar, br = D[key_string]
        ar, br = np.asarray(ar), np.asarray(br)

        if len(ar) == 0:
            ar = ar.reshape(0, n_obj)
        if len(br) == 0:
            br = br.reshape(0, n_obj)
    else:
        ar = np.zeros((0, n_obj))
        br = np.zeros((0, n_obj))

    if n_samples_to_add > 0:

        steps_set = [step_size]*int(n_samples_to_add//step_size)
        if n_samples_to_add%step_size>0:
            steps_set+= [int(n_samples_to_add%step_size)]

        for steps in tqdm(steps_set):
            v_ar_points, v_br_points = sample_var_vbr(n=steps,
                                                      P=igd_rp,
                                                      T=targets,
                                                      ref_point=ref_point,
                                                      ref_ideal=ref_ideal
                                                      )

            ar = np.vstack((ar, v_ar_points))
            br = np.vstack((br, v_br_points))

            with lock:
                try:
                    with open(D_path, "r") as infile:
                        D_new = json.load(infile)
                except:
                    D_new = {}
                D_new[key_string] = (ar.tolist(), br.tolist())
                with open(D_path, "w") as outfile:
                    json.dump(D_new, outfile)

        print("For "+key_string+" added:", sum(steps_set), "points")
        print("total points now: " , np.shape(ar)[0] + np.shape(br)[0])


if __name__ == "__main__":
    with Pool(6) as p:
        p.map(worker_f, os.listdir(problem_path))

    # worker_f(os.listdir(problem_path)[0])
