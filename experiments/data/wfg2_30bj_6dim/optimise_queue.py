# from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count, Pool 
import persistqueue

import copy
import time
import logging
import rootpath
import sys
sys.path.append(rootpath.detect())
import testsuite


def worker(i):
    time.sleep(i*0.1)
    while True:
         q = persistqueue.SQLiteAckQueue('./opt_queue', multithreading=True)
         if q.size>0:
             task = q.get()
             q.ack(task)
             task_c = copy.deepcopy(task)
             try:
                 task.optimise()
                 print("task completed!")
             except Exception as e:
                 print("task failed!")
                 logging.error("an error occured: "+str(e))
                 q.put(task_c)
         else:
             # portalocker.unlock(fl)
             break

# get processor count
proc_count = cpu_count()
# cap processor usage
try:
    m_proc = int(sys.argv[1])
except IndexError:
    m_proc = proc_count

# set number of available processors
n_proc = min(proc_count, m_proc)


print("{} processors found, limited to access {} processors.".format(proc_count, n_proc))
cont = input("Press Enter to begin, optimisation, input N to cancel:\t").lower()

logging.basicConfig(filename='error.log',level=logging.INFO)

with Pool(n_proc) as pool:
    pool.map(worker, range(n_proc))


