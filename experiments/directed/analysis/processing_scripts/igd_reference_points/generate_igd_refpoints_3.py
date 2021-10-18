import json
from testsuite.analysis_tools import strip_problem_names
import numpy as np

# path to save file
dict_path = 'reference_points'

# number of desired points for n dimensions
N_POINTS = {
    2: 2000,
    3: 4000,
    4: 8000
}

PROBLEMS = [
    "wfg3_2obj_6dim",
    "wfg3_3obj_10dim",
    "wfg3_4obj_10dim"]

try:
    with open(dict_path, 'r') as infile:
        D = json.load(infile)
except FileNotFoundError:
    D = {problem: None for problem in PROBLEMS}

if __name__ == "__main__":
    for name in PROBLEMS:
        if name not in D.keys() or D[name] is None:
            prob, obj, dim = strip_problem_names(name)
            if obj == 2:
                x = np.linspace(0, 2, N_POINTS[obj])
                y = -2 * x + 4
                yf = np.vstack((x, y)).T
            elif obj == 3:
                x = np.linspace(0, 1, N_POINTS[obj])
                y = 2 * x
                z = -6 * x + 6
                yf = np.vstack((x, y, z)).T
            elif obj == 4:
                x = np.linspace(0, .5, N_POINTS[obj])
                y = 2 * x
                z = 6 * x
                zz = -16 * x + 8
                yf = np.vstack((x, y, z, zz)).T

            D[name] = yf.tolist()
            print("saving")
            with open(dict_path, "w") as outfile:
                json.dump(D, outfile)
