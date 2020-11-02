import time
import random
import persistqueue
import shutil
import os

class sleeper:
    def __init__(self, i):
        self.i = i

    def __call__(self):
        t = random.uniform(0, 5)
        code = "{:03d}".format(self.i)
        print("executing {}...".format(code))
        time.sleep(t)
        if t<1:
            raise Exception("Crashed")
        message = "finished {} in time {:.2f}s".format(code, t)
        print(message)
        with open('sleep_out.txt', 'a') as f:
            print(message, file=f)

if __name__ == "__main__":
    import sys

    try:
        n_tasks = int(sys.argv[1])
    except:
        n_tasks = 10 
    print("adding ", n_tasks, " tasks.")


    try: 
        os.remove("./sleep_out.txt")
        print("Exsting log deleted")
    except:
        print("No log to delete")
    
    sleeper_list = [sleeper(i) for i in range(n_tasks)]
    
    try:
        shutil.rmtree("./sleep_queue")
        print("Exsting results deleted")
    except:
        print("No results to delete.")

    q = persistqueue.SQLiteAckQueue("./sleep_queue", multithreading=True)
    for sl in sleeper_list:
        q.put(sl)
    
    print(q.size," tasks in queue")
