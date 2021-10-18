import sys
import os

from testsuite.analysis_tools import get_filenames_of_incompletes_within_tree

top_dir = sys.argv[1]
print("Cleaning incomplete files from ", top_dir)
incomplete_files = get_filenames_of_incompletes_within_tree(top_dir)

if incomplete_files == []:
    print("log_dir clean: no incomplete files to remove.")
for file in incomplete_files:
    try:
        os.remove(file) # remove results
        print("Removed file: ", file)
        model = file[:-11]+"model.pkl"
        os.remove(model)   # also remove model
        print("Removed file: ", model)
    except:
        print(f"Removeing {file} FAILED!")
