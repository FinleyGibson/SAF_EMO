import sys
import os
import pickle
import numpy as np
import pickle
import rootpath
sys.path.append(rootpath.detect())

def get_missing(a, total): 
    b = np.arange(total) 
    duplicates = a.copy() 
    missing = [] 
    for i in b: 
        try: 
            duplicates.remove(i) 
        except: 
            missing.append(i) 
    return duplicates, missing 

def get_seed_from_str(string): 
    ind = string.find('seed')+4 
    numstr = [i for i in string[ind:ind+4] if i in list('0987654321')] 
    return int(''.join(numstr))

# set up parent dir
try:
    parent_dir = sys.argv[1]
    if parent_dir[-1] == '/':
        # robust to dirs which end in "/"
        parent_dir = parent_dir[:-1]
except IndexError:
    parent_dir = '.'

sys.path.insert(1, parent_dir)
from problem_setup import n_obj

## logged data
log_dir = parent_dir+"/log_data/"
# get result paths from provided parent dir, else use ./log_data/
directory_paths = sorted([os.path.join(log_dir, dir_name) for dir_name in os.listdir(log_dir)])
result_paths = [[os.path.join(d, f) for f in os.listdir(d) if f[-11:]=="results.pkl"] for d in directory_paths]
print("{} directories found:".format(len(directory_paths)))

print("\n Logged data:")
# load each result fom result_paths
D = []
for parent, file_paths in zip(directory_paths, result_paths):
    print("Directory", parent, " "*(50-len(parent)),  " contains {} files: ".format(len(file_paths)), end="    ")

    seeds = []
    evals = []
    for f in file_paths:
        seed = get_seed_from_str(f)
        with open(f, 'rb') as infile:
            result = pickle.load(infile)
        seeds.append(seed)
        evals.append(result['n_evaluations'])

    expected = range(31)
    if sorted(seeds) == sorted(expected):
        print("All files accounted for", end=", ")
        if all([i == 250 for i in evals]):
            print("All otimisations complete!")
        else:
            for s, i in zip(seeds, evals):     
                if i != 250:
                    print(s, "\t",  i)
        duplicates = []
        missing = []
    else:
        duplicates, missing = get_missing(seeds, 31)
        if len(duplicates)>0:
            print("Directory contains duplicates: ", *duplicates)
        if len(missing)>0:
            print("Directory contains missing opts: ", *missing)
        for s, i in zip(seeds, evals):     
            if i != 250:
                print(s, "\t", i)

    # store remiaining opts
    D.append({'directory': parent, 'duplicates': duplicates, 'missing': missing, 'seeds':seeds})

# write remaining optimisations to disk
with open('./remaining_opts', 'wb') as outfile:
    pickle.dump(D, outfile)

# check queue
with open('opt_queue', 'rb') as infile:
    q = pickle.load(infile)
if q != None:
    print("current queue length: ", len(q), end = "\n\n")
else:
    print("no queue found")


## Pickled data
        
print("Pickle data:")
pkl_dir = parent_dir +"/pkl_data/"
try:
    with open(pkl_dir+'results.pkl', 'rb') as infile:
        results = pickle.load(infile)
    for result in results:
        try:
            assert np.shape(result['y']) == (31, 250,  n_obj)
            print(result['name'], ":"+' '*(20-len(result['name'])), len(result['log_dir']), "/31", "\tComplete!")
        except:
            if result['name'].lower() == 'lhs':
                print("???") 
            else:
                print(result['log_dir'][0], ":", len(result['log_dir']), "/31", "\tIncomplete!")
                for seed, y in zip(result['seed'], result["y"]):
                    shape = np.shape(y)
                    if shape[0] != 250:
                        print(seed, "\t", shape)
except FileNotFoundError:
    print("No pickle data found")
