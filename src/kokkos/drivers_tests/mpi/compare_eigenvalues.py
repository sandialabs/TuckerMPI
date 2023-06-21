
import numpy as np
import sys

def extract_datatensor_rank(paramfile):
    with open(paramfile) as fh:
        for line in fh:
            if line.startswith("Global dims = "):
                ss = line.split("=")[1].strip()
                arr = np.fromstring(ss, sep=" ")
                return len(arr)

def extract_singvals_path(paramfile):
    with open(paramfile) as fh:
        for line in fh:
            if line.startswith("SV directory = "):
                print(line.split("="))
                return line.split("=")[1].strip()

def extract_singvals_prefix(paramfile):
    with open(paramfile) as fh:
        for line in fh:
            if line.startswith("SV file prefix = "):
                print(line.split("="))
                return line.split("=")[1].strip()

if __name__== "__main__":

    # we have as many modes to read as the ranks of the tensor
    modes = extract_datatensor_rank("./paramfile.txt")
    print("Rank = {}".format(modes))

    # figure out where sing values are written to
    svPath   = extract_singvals_path("./paramfile.txt")
    svPrefix = extract_singvals_prefix("./paramfile.txt")
    print("SV directory = {}".format(svPath))
    print("SV prefix = {}".format(svPrefix))

    # compare
    for i in range(0, modes-1):
        print("Comparing eigenvalues for mode = {}".format(i))

        gold = np.loadtxt("gold"+str(i)+".txt")
        computed = np.loadtxt(svPath + "/" + svPrefix + "_mode_"+str(i)+".txt")

        assert (len(gold) == len(computed)), \
            "Failing due to mismatching extents!"
        assert (np.allclose(gold, computed, rtol=1e-9, atol=1e-11)), \
            "Failing due to numerical differences!"
