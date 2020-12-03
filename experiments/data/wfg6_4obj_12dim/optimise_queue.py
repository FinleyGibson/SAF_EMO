from multiprocessing import cpu_count, Pool
# from generate_queue import n_opt
import persistqueue

from filelock import FileLock
import logging
import rootpath
import copy
import sys
sys.path.append(rootpath.detect())
import testsuite

# get processor count
proc_count = cpu_count()
# cap processor usage
try:
    m_proc = int(sys.argv[1])
except IndexError:
    m_proc = proc_count

lock = FileLock("./lock")
n_proc = min(proc_count, m_proc)

def worker(i):
    with lock:
        q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)
        cont = not q.empty()
    while cont:
        with lock:
            q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)
            optimiser = q.get()
            q.ack(optimiser)
        optimiser_cp = copy.deepcopy(optimiser)
        try:
            optimiser.optimise()
        except Exception as e:
            logging.error('Exception met in {}, seed {}, at step {}.'.
                    format(optimiser.__class__, optimiser.log_data["seed"], optimiser.n_evaluations))
            with lock:
                q.put(optimiser_cp)
        with lock:
            q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)
            cont = not q.empty()


with lock:
    q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)
print("{} processors found, limited to access {} processors.".format(proc_count, n_proc))
print("{} optimsations found in queue.".format(q.size))
go = input("Press Enter to begin, optimisation, input N to cancel:\t").lower()

logging.basicConfig(filename='error.log',level=logging.ERROR)
if go != "n":
    with Pool(n_proc) as pool:
        pool.map(worker, range(n_proc))


