import numpy as np
import sys

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

    # figure out where sthosvd values are written to
    sthosvdPath     = extract_sthosvd_path("./paramfile.txt")
    sthosvdPrefix   = extract_sthosvd_prefix("./paramfile.txt")

    # convert real data into text file
    binary = np.fromfile(sthosvdPath + "/" + sthosvdPrefix + "_core.mpi")
    np.savetxt("computedCoreTensor.txt", binary)

    # use data from text files
    gold = np.loadtxt("goldCoreTensor.txt")
    computed = np.loadtxt("computedCoreTensor.txt")

    # compare
    assert(len(gold) == len(computed)), \
        "Failing due to mismatching extents!"
    assert(np.allclose(gold, computed, rtol=1e-9, atol=1e-11)), \
        "Failing due to numerical differences!"
