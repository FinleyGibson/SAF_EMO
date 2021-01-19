import os
import sys
import rootpath
sys.path.append(rootpath.detect())

import numpy as np
import pickle
import inspect
from tqdm import tqdm
import wfg
from pymoo.factory import get_performance_indicator
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist 

from testsuite import optimisers
from testsuite.utilities import Pareto_split
from testsuite.analysis import load_all
from problem_setup import weighting, func, k, l, n_obj

OPTIMISER_NAMES = [cls[0].lower() for cls in inspect.getmembers(optimisers, inspect.isclass)  
        if cls[1].__module__ == 'testsuite.optimisers']

def get_name_from_dir(dir_string):
    bits = dir_string.split('_')
    name = [bit for bit in bits if bit.lower() in OPTIMISER_NAMES]

    if type(name) is list:
        name = name[0]
    if 'ei' in bits:
        name+='_ei'
    elif 'mean' in bits:
        name+='_$\mu$'
    return name

def load_result(directory):
    result = load_all(directory, trailing_text = "_results.pkl")
    name = get_name_from_dir(directory)
    result['name'] = name
    return result

def extract_performance(y, indicator):
    performance = []
    for i in range(9, len(y)):
        p = Pareto_split(y[:i])[0]
        ans = indicator.calc(p)
        performance.append(ans)
    return performance

def generate_wfg_pareto_samples(n_samples):
    y = np.zeros((n_samples, n_obj))
    for n in range(n_samples):
        z = wfg.random_soln(k, l, func.__name__)
        y[n,:] = func(z, k, n_obj)
    p = Pareto_split(y)[0]
    return p

def down_sample(y, out_size):
    """
    Down samples point pool y to size out_size, keeping the 
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

# load all results from directory tree
result_dirs = sorted(os.listdir("./log_data/"))
results= []
for path in result_dirs:
    result = load_result(os.path.join('./log_data/', path))
    results.append(result)

# set up ypervolume measure
hv_measure = get_performance_indicator("hv", ref_point=np.ones(n_obj)*1.2)

# set up igdi+ measure
# initial_samples = 10000
# final_n = 200
# y = generate_wfg_pareto_samples(initial_samples)
# y = down_sample(y, final_n)

y = np.load(sys.argv[1])
igdp_measure = get_performance_indicator("igd+", y)

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(*y.T)
plt.show()

# get all results aside from lhs as lhs is in list(np.array) format
for result in tqdm(results):
    ys_adjusted = [y/weighting for y in result["y"] if type(y) is np.ndarray]
#     result["hpv"] = [extract_performance(y, hv_measure) for y in ys_adjusted]
    result["igd+"] = [extract_performance(y, igdp_measure) for y in ys_adjusted]

# get results for lhsi by doing computation for every item in lhs['y'] list
lhs_ind = int(np.where([np.shape(result["hpv"]) == (0,) for result in results])[0]) # find which result is lhs
# results[lhs_ind]["hpv"] = [[hv_measure.calc(Pareto_split(y/weighting)[0]) for y in ys] for ys in results[lhs_ind]["y"]]
results[lhs_ind]["igd+"] = [[igdp_measure.calc(Pareto_split(y/weighting)[0]) for y in ys] for ys in results[lhs_ind]["y"]]

# pickle results
pkl_dir = "./pkl_data/"                 # set pickle dirs and paths
if not os.path.isdir(pkl_dir):
    os.makedirs(pkl_dir)
pkl_filename = pkl_dir+'results.pkl'

with open(pkl_filename,'wb') as outfile:    # pickle data
    pickle.dump(results, outfile)
print("DONE!")

