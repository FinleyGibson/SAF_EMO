import rootpath 
import sys
sys.path.append(rootpath.detect())
import pickle
import persistqueue
from testsuite.surrogates import GP, MultiSurrogate
from testsuite.optimisers import *
from problem_setup import func, objective_function, limits
exec("objective_function.__name__ = '{}'".format(func.__name__))
from filelock import FileLock

lock = FileLock("./lock")

## set up optimisers
mono_surrogate = GP(scaled=True)
multi_surrogate = MultiSurrogate(GP, scaled=True)

budget = 250
log_dir = "./log_data"
cmaes_restarts=0


# set up queue
with lock: 
    q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)

if len(sys.argv)>1:
    opt_opts = {
                'smsego': "SmsEgo(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed={}, ei={}, log_dir=log_dir, cmaes_restarts=cmaes_restarts)", 
                'saf':"Saf(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed={}, ei={}, log_dir=log_dir, cmaes_restarts=cmaes_restarts)",
                'mpoi':"Mpoi(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, seed={}, budget=budget, cmaes_restarts=cmaes_restarts)",
                'lhs': "Lhs(objective_function = objective_function, limits=limits, n_initial=10, budget=budget, seed={})",
                'parego': "ParEgo(objective_function=objective_function, limits=limits, surrogate=mono_surrogate, n_initial=10, seed={}, budget=budget, cmaes_restarts=cmaes_restarts)"
                }

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
            opt = opt_opts[optimiser].format(seed, ei)
            exec('optimisers.append('+opt+')')
else:
    seeds = range(28, 31)
    
    # add optimsers to queue
    optimisers = []
    for seed in seeds:
        #create optimisers
        optimisers += [SmsEgo(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed, ei=True, log_dir=log_dir, cmaes_restarts=cmaes_restarts),
                      SmsEgo(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed, ei=False, log_dir=log_dir, cmaes_restarts=cmaes_restarts),
                      Saf(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, budget=budget, seed=seed, ei=True, log_dir=log_dir, cmaes_restarts=cmaes_restarts),
                      Saf(objective_function=objective_function, limits=limits, surrogate=multi_surrogate,  n_initial=10, budget=budget, seed=seed, ei=False, log_dir=log_dir, cmaes_restarts=cmaes_restarts),
                      ParEgo(objective_function=objective_function, limits=limits, surrogate=mono_surrogate, n_initial=10, seed=seed, budget=budget, cmaes_restarts=cmaes_restarts),
                      Mpoi(objective_function=objective_function, limits=limits, surrogate=multi_surrogate, n_initial=10, seed=seed, budget=budget, cmaes_restarts=cmaes_restarts),
                      Lhs(objective_function = objective_function, limits=limits, n_initial=10, budget=budget, seed=seed)]
    
             
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
