
import numpy as np
import sys, os
import matplotlib.pyplot as plt
from matplotlib import cm

if __name__== "__main__":

    # read gold.txt
    goldtxt_path = "gold.txt"
    goldtxt_lines = []
    with open(goldtxt_path) as goldtxt_data:
        goldtxt_lines = [line.rstrip() for line in goldtxt_data]

    # read sv_mode_*
    svmodetxt_lines = []
    for i in range(0, 4):
        svmodetxt_path = "sv_mode_" + str(i) + ".txt"
        with open(svmodetxt_path) as svmodetxt_data:
            svmodetxt_lines += [line.rstrip() for line in svmodetxt_data]

    # assert lines size
    assert(len(goldtxt_lines) == len(svmodetxt_lines))

    # assert line



    for a in goldtxt_lines:
        print(a)

    print("====")

    for b in svmodetxt_lines:
        print(b)





    if(1 == 2):

        # SUCCES
        sys.exit(0)
    else:

        # ERROR
        sys.exit(1)

    # compare gold.txt with sv_mode_*
