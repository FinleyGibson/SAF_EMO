import rootpath 
import sys
sys.path.append(rootpath.detect())
import pickle
import persistqueue
from testsuite.surrogates import GP, MultiSurrogate
from testsuite.directed_optimisers import DmVector 
from problem_setup import func, objective_function, limits
exec("objective_function.__name__ = '{}'".format(func.__name__))
from filelock import FileLock
import numpy as np

lock = FileLock("./lock")

multi_surrogate = MultiSurrogate(GP, scaled=True)

budget = 150
log_dir = "./log_data"
cmaes_restarts=0
dmvs = np.array([[1.5, 1.], [1., 2.5]])

# set up queue
with lock: 
    q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)

if len(sys.argv)>1:
    opt_opts = {
                'dm_saf': "DmVector(objective_function=objective_function, ei=False,  dmv=dmv, w=0.5, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed)"}

    with open(sys.argv[1], 'rb') as infile:
        lst = pickle.load(infile)

    optimisers = []
    for D in lst:
        path = D['directory']
        print(path)
        path_split = [i.strip('_').lower() for i in path.split('_')]
        optimiser = list(opt_opts.keys())[np.nonzero([opt in path_split for opt in opt_opts.keys()])[0][0]]
        ei = 'ei' in  path_split
        for seed in D["missing"]:
            if seed <31:
                opt = opt_opts[optimiser].format(seed, ei)
                exec('optimisers.append('+opt+')')
else:
    seeds = range(0, 11)
    
    # add optimsers to queue
    optimisers = []
    for seed in seeds:
        #create optimisers
        for dmv in dmvs:
            optimisers += [DmVector(objective_function=objective_function, ei=False,  dmv=dmv, w=0.5, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed)]
n_opt = len(optimisers)

if __name__ == "__main__":
    import shutil
    if q.size>0:
        print("{} items already in queue".format(q.size))
        reset = input("Would you like to delete the existing queue? Y/N:\t").lower()
        if reset == "y":
            reset = True
        elif reset == "n":
            reset = False
        else:
            print("Input not recognised")
    else:
        reset = True 

    if reset == True:
        shutil.rmtree('./opt_queue', ignore_errors=True)
        print("removed existing queue.")
        with lock:
            q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True) 

    else: 
        pass

    # add to queue

    with lock:
        for optimiser in optimisers:
            q.put(optimiser)

    print("Added {}  optimisers to ./opt_queue, queue length now {}.".format(n_opt, q.size))