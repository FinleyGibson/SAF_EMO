import os

for name in os.listdir("../../data"):
    print(name)
    if name.split("_")[0] == "wfg4":
        os.system("python process_results.py {} {}".format(
            os.path.join("../../data", name),
            os.path.join("../../data_undirected_comp", name)
        ))
