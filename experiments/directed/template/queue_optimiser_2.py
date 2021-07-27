import rootpath
import sys
sys.path.append(rootpath.detect())
import os
from multiprocessing import cpu_count, Pool
from persistqueue import SQLiteAckQueue
from filelock import FileLock

# path to optimisier
optimiser_path = str(sys.argv[1])

# lock path
lock_path = os.path.join(os.path.dirname(__file__), optimiser_path, "lock")
queue_path = os.path.join(os.path.dirname(__file__), optimiser_path, "queue")

def worker(i):
    with lock:
        q = persistqueue.SQLiteAckQueue(queue_path, multithreading=True)
        cont = not q.empty()
    while cont:
        with lock:
            q = persistqueue.SQLiteAckQueue(queue_path, multithreading=True)
            optimiser = q.get()
            q.ack(optimiser)
        optimiser_cp = copy.deepcopy(optimiser)
        try:
            optimiser.optimise()
        except  Exception as e:
            logging.error('Exception met in {}, seed {}, at step {}.'.
                          format(optimiser.__class__, optimiser.log_data["seed"], optimiser.n_evaluations))
            with lock:
                q.put(optimiser_cp)
        with lock:
            q = SQLiteAckQueue(queue_path, multithreading=True)
            cont = not q.empty()


# decide how many processors
proc_count = cpu_count()
try:
    # cap processor usage
    n_proc = min(int(sys.argv[2]), proc_count)
except IndexError:
    # use all available
    n_proc = proc_count

lock = FileLock(lock_path)

with lock:
    q = SQLiteAckQueue(queue_path, multithreading=True)
print("{} processors found, limited access to {} ".format(proc_count, n_proc))
print("{} optimsations found in queue.".format(q.size))
go = input("Press Enter to begin, optimisation, input N to cancel:\t").lower()

logging.basicConfig(filename='error.log', level=logging.ERROR)
if go != "n":
    with Pool(n_proc) as pool:
        pool.map(worker, range(n_proc))

