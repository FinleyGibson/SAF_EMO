from multiprocessing import cpu_count, Pool
# from generate_queue import n_opt
import persistqueue

import pickle
from filelock import FileLock
import logging
import rootpath
import copy
import sys
sys.path.append(rootpath.detect())
import testsuite

def get_item():
    with lock:
        with open('opt_queue' , 'rb') as infile:
            q = pickle.load(infile)
        if  q == None or len(q)<1:
            return None
        task = q[0]
        with open('opt_queue' , 'wb') as outfile:
            pickle.dump(q[1:], outfile)
    return task

def add_item(item):
    with lock:
        with open('opt_queue' , 'rb') as infile:
            q = pickle.load(infile)
        q.append(item)
        with open('opt_queue' , 'wb') as outfile:
            pickle.dump(q, outfile)


# get processor count and cap cpu usage 
proc_count = cpu_count()
try:
    m_proc = int(sys.argv[1])
except IndexError:
    m_proc = proc_count
n_proc = min(proc_count, m_proc)

# define lock
lock = FileLock("./lock")

def worker(i):
    cont = True
    while cont:
        optimiser = get_item()
        optimiser_cp = copy.deepcopy(optimiser)
        if optimiser is None:
            cont = False
            continue
        else:
            try:
                optimiser.optimise()
                # print("optimised: \t",  optimiser, optimiser.seed)
            except  Exception as e:
                logging.error('Exception met in {}, seed {}, at step {}.'.
                        format(optimiser.__class__, optimiser.log_data["seed"], optimiser.n_evaluations))
                add_item(optimiser_cp)

print("{} processors found, limited to access {} processors.".format(proc_count, n_proc))


with lock:
    with open('opt_queue' , 'rb') as infile:
        q = pickle.load(infile)
if q != None:
    ql = len(q)
else:
    ql = None 
print("{} optimsations found in queue.".format(ql))


go = input("Press Enter to begin, optimisation, input N to cancel:\t").lower()

logging.basicConfig(filename='error.log',level=logging.ERROR)

if go != "n":
    with Pool(n_proc) as pool:
        pool.map(worker, range(n_proc))


