import os
import sys


log_dir = sys.argv[1]
dirs = os.listdir(log_dir)

for d in dirs:
    files = [f for f in os.listdir(log_dir+d) if f[-10:]!='_model.pkl' and f[-12:]!='_results.pkl']
    new_files = [f[:-4]+'_results.pkl' for f in files]
    for f, nf in zip(files, new_files):
        a = os.path.join(log_dir, d, f)
        b = os.path.join(log_dir, d, nf)
        os.rename(a, b)
        print(f, "\t--->\t", nf)

