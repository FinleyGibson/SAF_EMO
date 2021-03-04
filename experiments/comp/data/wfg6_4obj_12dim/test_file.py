import time
from numpy.random import randint

while True:
    print("This should appear in terminal.")
    time.sleep(0.1)
    
    d = randint(0, 30)
    if d==1:
        # raise error
        assert 1==2
print("done")
