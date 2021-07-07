import rootpath 
import sys
sys.path.append(rootpath.detect())
import pickle
import persistqueue
from testsuite.surrogates import GP, MultiSurrogate
from testsuite.directed_optimisers import DirectedSaf
from problem_setup import func, objective_function, limits, n_obj
exec("objective_function.__name__ = '{}'".format(func.__name__))
from filelock import FileLock
import numpy as np

lock = FileLock("./lock")

multi_surrogate = MultiSurrogate(GP, scaled=True)

budget = 150
log_dir = "./log_data"
cmaes_restarts=0

if n_obj == 2:
    # 2d targets
    targets = np.array([[1.79, 1.79],[1.79*0.8, 1.79*0.8],[1.79*1.1, 1.79*1.1], [.985, 3.48],[0.985*0.9, 3.48*0.9], [0.985*1.1, 3.48*1.1]])
elif n_obj ==3:
    # 3d targets
    t1 = np.array([1.73, 1.63, 1.72])
    t2= np.array([0.51, 3.67, 1.83])
    targets = np.array([t1, t1*0.9, t1*1.1, t2, t2*0.9, t2*1.1])
## 

# set up queue
with lock: 
    q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)

if len(sys.argv)>3:
    opt_opts = {
                'directedsaf': "DirectedSaf(objective_function=objective_function, ei=False,  targets=t, w=0.5, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed)"}

    with open(sys.argv[1], 'rb') as infile:
        lst = pickle.load(infile)

    optimisers = []
    for D in lst:
        path = D['directory']
        print(path)
        path_split = [i.strip('_').lower() for i in path.split('_')]
        print(1, path_split)
        print(opt_opts.keys())
        print(2, [opt in path_split for opt in opt_opts.keys()])
        optimiser = list(opt_opts.keys())[np.nonzero([opt in path_split for opt in opt_opts.keys()])[0][0]]
        ei = 'ei' in  path_split
        t = D["target"] 
        for seed in D["missing"]:
            if seed <11:
                opt = opt_opts[optimiser].format(seed, ei)
                print(t, seed)
                print(opt)
                exec('optimisers.append('+opt+')')
else:
    seeds = range(0, 11)
    
    # add optimsers to queue
    optimisers = []
    for seed in seeds:
        #create optimisers
        for t in targets:
            optimisers += [DirectedSaf(objective_function=objective_function, ei=False,  targets=t, w=0.5, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed)]
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
