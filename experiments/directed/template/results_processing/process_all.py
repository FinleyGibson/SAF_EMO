import os
import sys

results_dir = sys.argv[1]
for i in os.listdir(results_dir):
    print(os.path.join(results_dir, i))
