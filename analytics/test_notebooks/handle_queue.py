import persistqueue
from generate_queue import sleeper
from multiprocessing import cpu_count, Pool
import portalocker
import time
import copy
from filelock import FileLock 

lock = FileLock("./lock")
q = persistqueue.SQLiteAckQueue("./sleep_queue", multithreading=True) 

def worker(i, q):

#     q = persistqueue.SQLiteAckQueue("./sleep_queue", multithreading=True) 
    if q.size:
        cont=True
    else:
        cont = False

    while not q.empty():
        with lock:
            task = q.get()
            q.ack(task)
        task_c = copy.deepcopy(task)
        try:
            task()
            print("task completed!")
        except:
            print("task failed!")
            with lock:
                q.put(task_c)

        with lock:
            q = persistqueue.SQLiteAckQueue("./sleep_queue", multithreading=True) 
        if q.size<1:
            cont=False


n_cpus = cpu_count()
print(n_cpus, " detected.")

with Pool(n_cpus) as pool:
    pool.map(worker, range(n_cpus), q)

