import sys
import os
import pickle
import numpy as np
import persistqueue
log_dir = "./log_data"

results_dirs = [os.path.join(log_dir, result_dir) for result_dir in os.listdir(log_dir)]

# result_files = [[os.path.join(resuts_dir, file_dir) for file_dir in os.listdir(results_dir)] for results_dir in results_dirs]
result_files = [[os.path.join(results_dir, file_dir) for file_dir in os.listdir(results_dir) if file_dir[-11:]=="results.pkl"] for results_dir in results_dirs]


print("{} directories found containing {} results files.".format(len(results_dirs), sum([len(l) for l in result_files])))
print()
for directory, files in zip(results_dirs, result_files):
    print("Directory", directory, "contains {} files: ".format(len(files)))
    for filei in files:
        print(filei)

results = []
n_results = 0
for filei in result_files:
    for f in filei:
        n_results += 1
        with open(f, "rb") as infile:
            result = pickle.load(infile)
            result["path"] = f
        results.append(result)

print(n_results)
completed = []
print(len(completed))
for i, result in enumerate(results):
    if result["budget"]==result["n_evaluations"]:
        completed.append(True)
    else:
        completed.append(False)

print()
print("{} completed out of {}.".format(sum(completed), len(completed)))

BUDGET = 0
EVALS = 0
for result in results:
    budget = result["budget"]
    evals = result["n_evaluations"]
    print(result["n_evaluations"], "/",result["budget"], "\t", result["path"].split("/")[3][:-49])
    BUDGET += budget
    EVALS+=evals

q = persistqueue.SQLiteAckQueue('./opt_queue')
print("{:.2f}%".format((EVALS/BUDGET)*100))
print("current queue length: ", q.size)

