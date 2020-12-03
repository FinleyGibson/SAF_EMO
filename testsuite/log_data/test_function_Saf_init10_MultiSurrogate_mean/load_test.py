"""
simple script to test the loading of logged data
arg1: name of .plk file locaed in ./
"""
import pickle
import sys

path = "./"+str(sys.argv[1])

try:
    result = pickle.load(open(path, "rb"))
    print("Load success!!:")
    print()
    print()
    for k, v in result.items():
        print(k, "\t", type(v))
except:
    print("Load failed!!")



