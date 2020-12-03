import os
import sys
import rootpath
sys.path.append(rootpath.detect())

import pickle
from testsuite.utilities import Pareto_split
import numpy as np
from generate_queue import func, n_obj, n_dim, n_obj
from pymoo.factory import get_performance_indicator
from tqdm import tqdm

def load_all(directory):
    paths = [file for file in os.listdir(directory) if file[-4:] == "results.pkl"]
    combined_resuts = {}
    for log_path in paths:
        result = pickle.load(open(os.path.join(directory, log_path), "rb"))
        for key, value in result.items():
            try:
                combined_resuts[key] += [value]
            except KeyError:
                combined_resuts[key] = [value]

    return combined_resuts


def extract_performance(z, indicator):
    ANS  = []
    for i in range(9, len(z)):
        p = Pareto_split(z[:i])[0]
        ans = indicator.calc(p)
        ANS.append(ans)

    return ANS

def scatter_nsphere(n_points, n_dims, weighting=None):
    """scatter n_points onto unit n-spere with n_dims dimensions"""
    if weighting is None:
        weighting = np.ones(n_dims)
    else:
        weighting = np.array(weighting)
    points = np.random.randn(n_points,n_dims)*weighting
    d = (points**2).sum(axis=1)**0.5
    norm_points = (points.T/d.T).T
    return np.abs(norm_points)

# set directories and paths
pkl_dir = "./pkl_data/"
results_dir = "./log_data/"
result_dirs = sorted(os.listdir("./log_data/"))
filename = pkl_dir+'results.pkl'
if not os.path.isdir(pkl_dir):
    os.makedirs(pkl_dir)

# load data
results = []
for i, result_dir in enumerate(result_dirs):
    result = load_all(os.path.join(results_dir, result_dir))
    name = result_dir.split("_")[2]
    if "ei" in  result_dir:
        name+=" ei"
    elif "mean" in result_dir:
        name+=" $\mu$"
    else:
        pass
    result["name"] = name
    results.append(result)

print("results loaded from ", results_dir)
print("loaded:")
for result in results:
    print(result["name"])

weighting = np.arange(1,n_obj+1)*2
y = scatter_nsphere(500, n_obj, weighting)

print()
print("weighting:\t", weighting)
print("n_obj:\t\t", n_obj)
print("n_dim:\t\t", n_dim)

print()

igdp = get_performance_indicator("igd+", y)
ref = np.ones(n_obj)*1.2
hv = get_performance_indicator("hv", ref_point=ref)

print("computing hvs and igds")
for result in tqdm(results):
    ys_adjusted = [y/weighting for y in result["y"] if type(y) is np.ndarray]
    result["igd"] = [extract_performance(y, igdp) for y in ys_adjusted]
    result["hpv"] = [extract_performance(y, hv) for y in ys_adjusted]

lhs_ind = int(np.where([np.shape(result["hpv"]) == (0,) for result in results])[0])
print("lhs index: \t", lhs_ind)

results[lhs_ind]["hpv"] = [[hv.calc(Pareto_split(y/weighting)[0]) for y in ys] for ys in results[lhs_ind]["y"]]
results[lhs_ind]["igd"] = [[igdp.calc(Pareto_split(y/weighting)[0]) for y in ys] for ys in results[lhs_ind]["y"]]

outfile = open(filename,'wb')
pickle.dump(results, outfile)
outfile.close()

print("DONE!")
