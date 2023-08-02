
import numpy as np
import sys
from argparse import ArgumentParser

if __name__== "__main__":

    parser = ArgumentParser()
    parser.add_argument("-i", dest="inFile",  type=str)
    parser.add_argument("-o", dest="outFile", type=str)
    parser.add_argument("--skip", dest="rowsToSkipWhenReading", type=int)
    args = parser.parse_args()

    data = np.loadtxt(args.inFile, skiprows=args.rowsToSkipWhenReading)
    data.tofile(args.outFile, format='%20.15f')
