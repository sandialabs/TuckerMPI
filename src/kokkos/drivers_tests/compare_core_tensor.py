import numpy as np
import sys
from argparse import ArgumentParser

def extract_sthosvd_path(paramfile):
    with open(paramfile) as fh:
        for line in fh:
            if line.startswith("STHOSVD directory = "):
                print(line.split("="))
                return line.split("=")[1].strip()

def extract_sthosvd_prefix(paramfile):
    with open(paramfile) as fh:
        for line in fh:
            if line.startswith("STHOSVD file prefix = "):
                print(line.split("="))
                return line.split("=")[1].strip()

if __name__== "__main__":

    parser = ArgumentParser()
    parser.add_argument("--rtol", dest="relTol", type=float)
    parser.add_argument("--atol", dest="absTol", type=float)
    args = parser.parse_args()

    # figure out where sthosvd values are written to
    sthosvdPath     = extract_sthosvd_path("./paramfile.txt")
    sthosvdPrefix   = extract_sthosvd_prefix("./paramfile.txt")
    print("STHOSVD directory = {}".format(sthosvdPath))
    print("STHOSVD file prefix = {}".format(sthosvdPrefix))

    # convert real data into text file
    binary = np.fromfile(sthosvdPath + "/" + sthosvdPrefix + "_core.mpi")
    np.savetxt("computedCoreTensor.txt", binary)

    # use data from text files
    gold = np.loadtxt("goldCoreTensor.txt")
    computed = np.loadtxt("computedCoreTensor.txt")

    # compare
    assert(len(gold) == len(computed)), \
        "Failing due to mismatching extents!"
    assert(np.allclose(gold, computed, rtol=args.relTol, atol=args.absTol)), \
        "Failing due to numerical differences!"
