import sys
from multiprocessing import cpu_count, Pool
from filelock import FileLock
import os
from persistqueue import SQLiteQueue
import copy

optimiser_path = sys.argv[1]
lock_path = os.path.join(os.path.dirname(__file__), optimiser_path, "lock")
queue_path = os.path.join(os.path.dirname(__file__), optimiser_path, "queue")

lock = FileLock(lock_path)


def worker(i):
        with lock:
                q = SQLiteQueue(queue_path, multithreading=True)
                optimiser = q.get()
        optimiser_cp = copy.deepcopy(optimiser)
        try:
                print(ping)
                optimiser.optimise()
        except Exception as e:
                logging.error('Exception met in {}, seed {}, at step {}.'.
                              format(optimiser.__class__, optimiser.log_data["seed"], optimiser.n_evaluations))
                with lock:
                        q.put(optimiser_cp)
        with lock:
                q = SQLiteQueue(queue_path, multithreading=True)
                cont = not q.empty()


try:
        n_cpu = cpu_count()
        cap = int(sys.argv[2])
        n_processes = min([n_cpu, cap])
except IndexError:
        n_processes = cpu_count()

with lock:
        q = SQLiteQueue(queue_path, multithreading=True)
        fin = q.empty()
        n_renamining = q.qsize()

with Pool(n_processes) as p:
        p.map(worker, range(n_renamining))
