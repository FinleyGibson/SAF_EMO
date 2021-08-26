import os
import sys
import rootpath
sys.path.append(rootpath.detect())

import inspect
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pymoo.factory import get_performance_indicator
from testsuite.analysis import load_all
from testsuite import optimisers


def load_result(directory):
    result = load_all(directory, trailing_text = "_results.pkl")
    name = get_name_from_dir(directory)
    result['name'] = name
    return result


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

OPTIMISER_NAMES = [cls[0].lower() for cls in inspect.getmembers(optimisers, inspect.isclass)
        if cls[1].__module__ == 'testsuite.optimisers']

ref_path = sys.argv[1]
problem_path = sys.argv[2]


# load all results from directory tree
result_dirs = sorted(os.listdir(os.path.join(problem_path, "log_data/")))
results= []
for path in result_dirs:
    print(os.path.join(problem_path, path))
    result = load_result(os.path.join(problem_path, 'log_data/',  path))
    results.append(result)
    
n_obj = np.shape(results[0]['y'])[-1]
print("n_obj", n_obj)
print("loading ref points from , ", ref_path)
print("saving processed results to ",  os.path.join(problem_path, "log_data/"))

# get refpoints
p = np.load(sys.argv[1])
y_maxs = np.concatenate([r['y'] for r in results if r['name'] != "lhs"], axis=0).reshape(-1, n_obj)
ref_point =  y_maxs.max(axis=0)

# setup measurement systems
hv_measure = get_performance_indicator("hv", ref_point=ref_point)
igdp_measure = get_performance_indicator("igd+", p)

# process results, storing in D
D = {}
for result in tqdm(results):
    print(result['name'])
    y = np.array(result['y'])

    if result['name'] == 'lhs':
        hvs = np.zeros((y.shape[0], y.shape[1]+10))
        igdps = np.zeros((y.shape[0], y.shape[1]+10))
        for i, yi in tqdm(enumerate(y)):
            for j, yii in enumerate(yi):
                hvs[i, j+10] = hv_measure.calc(yii)
                igdps[i, j+10] = igdp_measure.calc(yii)
    else:
        hvs = np.zeros((y.shape[0], y.shape[1]))
        igdps = np.zeros((y.shape[0], y.shape[1]))
        for i, yi in tqdm(enumerate(y)):
            for j in range(1, y.shape[1]+1):
                hvs[i, j-1] = hv_measure.calc(yi[:j])
                igdps[i, j-1] = igdp_measure.calc(yi[:j])

    D[result['name']] = {'name':result['name'], 'hypervolume': hvs, 'igd+':igdps, 'y': result['y'], 'hv_ref': ref_point, 'igd_ref': p, 'x': result['x'], 'seed': resuts['seed']}

# save processed results
with open(os.path.join(problem_path, 'pkl_data/results__newsms.pkl'), 'wb') as outfile:
    pkl.dump(D, outfile)
