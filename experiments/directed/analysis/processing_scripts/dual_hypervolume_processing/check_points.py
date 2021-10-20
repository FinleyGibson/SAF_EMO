import json
import sys
import numpy as np
import os


for i, f in enumerate(os.listdir("./points")):
    print(f"{i}: {f}")
    path = os.path.join("./points", f)

    with open(path, "r") as infile:
        D = json.load(infile)
    
    for i, (k, (va, vb)) in enumerate(D.items()):
        print(f"{i}: {k}: {len(va)} va points, {len(vb)} vb points, total: {len(va)+len(vb)}")

    print()
