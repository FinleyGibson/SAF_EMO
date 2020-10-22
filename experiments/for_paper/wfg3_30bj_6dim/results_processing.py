import os
import sys
import rootpath
sys.path.append(rootpath.detect())

import wfg
import pickle
from testsuite.utilities import Pareto_split
from testsuite.analysis import load_all
import numpy as np
from paralellizer import func, n_obj, x_limits, n_dim, n_obj, k, l
from pymoo.factory import get_performance_indicator
from tqdm import tqdm

def extract_performance(z, indicator):
    ANS  = []
    for i in range(9, len(z)):
        p = Pareto_split(z[:i])[0]
        ans = indicator.calc(p)
        ANS.append(ans)

    return ANS


weighting = np.array([1.5, 3, 6])

N = 500
y = np.zeros((N, n_obj))
for n in range(N):
    z = wfg.random_soln(k, l, func.__name__)
    y[n,:] = func(z, k, n_obj)/weighting


# set directories and paths
pkl_dir = "./pkl_data/"
results_dir = "./log_data/"
result_dirs = sorted(os.listdir("./log_data/"))
filename = pkl_dir+'results.pkl'
if not os.path.isdir(pkl_dir):
    os.makedirs(pkl_dir)

    
    
if __name__ == "__main__":
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
