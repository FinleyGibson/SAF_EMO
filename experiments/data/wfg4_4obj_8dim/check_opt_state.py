import sys
import os
import pickle
import numpy as np
import persistqueue
import pickle
from filelock import FileLock
log_dir = "./log_data"

lock = FileLock('./lock')

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

results_dirs = sorted([os.path.join(log_dir, result_dir) for result_dir in os.listdir(log_dir)])

# result_files = [[os.path.join(resuts_dir, file_dir) for file_dir in os.listdir(results_dir)] for results_dir in results_dirs]
result_files = [[os.path.join(results_dir, file_dir) for file_dir in os.listdir(results_dir) if file_dir[-11:]=="results.pkl"] for results_dir in results_dirs]


print("{} directories found containing {} results files.".format(len(results_dirs), sum([len(l) for l in result_files])))
print()

D = []
for directory in results_dirs:
    files = [f for f in os.listdir(directory) if f[-10:] != '_model.pkl']
    print("Directory", directory, "contains {} files: ".format(len(files)))
    seeds = []
    for f in files:
        seed = get_seed_from_str(f)
        seeds.append(seed)

    expected = range(31)
    if sorted(seeds) == sorted(expected):
        print("All accounted for")
        duplicates = []
        missing = []
    else:
        duplicates, missing = get_missing(seeds, 31)
        if len(duplicates)>0:
            print("Directory contains duplicates: ", *duplicates)
        if len(missing)>0:
            print("Directory contains missing opts: ", *missing)

    print('\n')
    D.append({'directory': directory, 'duplicates': duplicates, 'missing': missing, 'seeds':seeds})

        
with open('./remaining_opts', 'wb') as outfile:
    pickle.dump(D, outfile)
# results = []
# n_results = 0
# for filei in result_files:
#     for f in filei:
#         n_results += 1
#         with open(f, "rb") as infile:
#             result = pickle.load(infile)
#         results.append(result)
# 
# print(n_results)
# completed = []
# print(len(completed))
# for i, result in enumerate(results):
#     if result["budget"]==result["n_evaluations"]:
#         completed.append(True)
#     else:
#         completed.append(False)
# 
# print()
# print("{} completed out of {} started.".format(sum(completed), len(completed)))
# 
# for result in results:
#     print(result["budget"], result["n_evaluations"])
# 
with lock:
    q = persistqueue.SQLiteAckQueue('./opt_queue')
print("current queue length: ", q.size)
# 
