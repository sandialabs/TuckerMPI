
import numpy as np
import sys

if __name__== "__main__":

    # read gold.txt
    goldtxt_path = "gold.txt"
    goldtxt_lines = []
    with open(goldtxt_path) as goldtxt_data:
        goldtxt_lines = [float(line.rstrip()) for line in goldtxt_data]

    # read sv_mode_*
    svmodetxt_lines = []
    for i in range(0, 4):
        svmodetxt_path = "sv_mode_" + str(i) + ".txt"
        with open(svmodetxt_path) as svmodetxt_data:
            svmodetxt_lines += [float(line.rstrip()) for line in svmodetxt_data]

    # assert lines size
    assert(len(goldtxt_lines) == len(svmodetxt_lines))

    # assert lines data
    for i in range(0, len(goldtxt_lines)):
        assert(np.allclose(goldtxt_lines[i], svmodetxt_lines[i],rtol=1e-9, atol=1e-11))

    # SUCCESS
    sys.exit(0)