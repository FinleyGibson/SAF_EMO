import os
import sys

try: 
    d = sys.argv[1]
except:
       d = './'

dirs =  os.listdir(d)
dirs = [d for d in dirs if d[-3:] != '.py']

for dirr in dirs:
    dirr = os.path.join(d, dirr)
    files = os.listdir(dirr)
    files = [f for f in files if f[-3:]!=".py"]
    
    ints = '0987654321'

    for f in files:
        ind = f.find('seed')+5    
        if f[ind] in ints and f[ind+1] not in ints:
            new_f = f[:ind]+"{:02d}".format(int(f[ind]))+f[ind+1:]
            old = os.path.join(dirr,f)
            new = os.path.join(dirr,new_f)
            print()
            print(old)
            print(new)
            print()
            os.rename(old, new)
