import json
import sys
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "./points.json"

with open(path, "r") as infile:
    D = json.load(infile)

for i, (k, (va, vb)) in enumerate(D.items()):
    print(f"{i}: {k}: {len(va)} va points, {len(vb)} vb points, total: {len(va)+len(vb)}")