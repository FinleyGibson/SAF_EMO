from tqdm import tqdm
import os

for name in os.listdir("../../../data"):
    os.system("python process_results.py {} {}".format(
        os.path.join("../../../data", name),
        os.path.join("../../data_undirected_comp", name)
    ))
