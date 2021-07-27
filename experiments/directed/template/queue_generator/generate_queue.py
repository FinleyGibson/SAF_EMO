"""
generates queue of optimisation tasks for problem_setup.py in directory
provided as first argument
"""
import rootpath
import sys
sys.path.append(rootpath.detect())

from filelock import FileLock

# import pickle
# import persistqueue
# from testsuite.surrogates import GP, MultiSurrogate
# from testsuite.directed_optimisers import DirectedSaf
# from problem_setup import func, objective_function, limits
# exec("objective_function.__name__ = '{}'".format(func.__name__))
# import numpy as np

lock = FileLock("../q_lock")

surrogate = MultiSurrogate(GP, scaled=True)

budget = 150
log_dir = "./log_data"
cmaes_restarts = 1

## TODO: change the targets
# targets = np.array([[1.79, 1.79],[1.79*0.8, 1.79*0.8],[1.79*1.1, 1.79*1.1], [.985, 3.48],[0.985*0.9, 3.48*0.9], [0.985*1.1, 3.48*1.1]])
# set up queue
# with lock:
#     q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)
#
# if len(sys.argv)>1:
#     opt_opts = {
#                 'dsaf': "DirectedSaf(objective_function=objective_function, ei=False,  targets=t, w=0.5, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed)"}
#
#     with open(sys.argv[1], 'rb') as infile:
#         lst = pickle.load(infile)
#
#     optimisers = []
#     for D in lst:
#         path = D['directory']
#         print(path)
#         path_split = [i.strip('_').lower() for i in path.split('_')]
#         optimiser = list(opt_opts.keys())[np.nonzero([opt in path_split for opt in opt_opts.keys()])[0][0]]
#         ei = 'ei' in  path_split
#         for seed in D["missing"]:
#             if seed <31:
#                 opt = opt_opts[optimiser].format(seed, ei)
#                 exec('optimisers.append('+opt+')')
# else:
#     seeds = range(0, 11)
#
#     # add optimsers to queue
#     optimisers = []
#     for seed in seeds:
#         #create optimisers
#         for t in targets:
#             optimisers += [DirectedSaf(objective_function=objective_function, ei=False,  targets=t, w=0.5, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed)]
# n_opt = len(optimisers)
#
# if __name__ == "__main__":
#     import shutil
#     if q.size>0:
#         print("{} items already in queue".format(q.size))
#         reset = input("Would you like to delete the existing queue? Y/N:\t").lower()
#         if reset == "y":
#             reset = True
#         elif reset == "n":
#             reset = False
#         else:
#             print("Input not recognised")
#     else:
#         reset = True
#
#     if reset == True:
#         shutil.rmtree('./opt_queue', ignore_errors=True)
#         print("removed existing queue.")
#         with lock:
#             q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)
#
#     else:
#         pass
#
#     # add to queue
#
#     with lock:
#         for optimiser in optimisers:
#             q.put(optimiser)
#
#     print("Added {}  optimisers to ./opt_queue, queue length now {}.".format(n_opt, q.size))
