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
    print("All remaining tasks accounted for")
    print("No duplicates")
except Exception as e:
    print("failed: ", e)
