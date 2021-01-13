import numpy as np
import persistqueue
q = persistqueue.SQLiteAckQueue("./sleep_queue", multithreading=True)


f = open("./sleep_out.txt", "r")


lines = f.readlines()

ints = []
for i, line in enumerate(lines):
    print(line, end="")
    num = int(line.split(" ")[1])
    ints.append(num)
   
print("\n\n")
print("Queue size: ", q.size)
# print(list(range(len(ints))))
# print(ints)

try:
    assert sorted(list(range(len(ints)))) == sorted(ints)
except AssertionError as e:
    print("Failed!")
    print(e)
    print("missing: ")
    for i in range(max(ints)):
        if i not in ints:
            print(i) 

    print("Duplicates")
    for i in set(ints):
        occur = [i==ii for ii in ints]
        if np.sum(occur)>1:
            print(i)
