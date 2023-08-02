import numpy as np
from argparse import ArgumentParser

def get_comparison_arguments():
    parser = ArgumentParser()
    parser.add_argument("--rtol", dest="relTol", type=float)
    parser.add_argument("--atol", dest="absTol", type=float)
    return parser.parse_args()

def extract_datatensor_rank(paramfile):
    with open(paramfile) as fh:
        for line in fh:
            if line.startswith("Global dims = "):
                ss = line.split("=")[1].strip()
                arr = np.fromstring(ss, sep=" ")
                return len(arr)

def extract_single_line_value_from_paramfile(line_in_paramfile, paramfile):
    with open(paramfile) as fh:
        for line in fh:
            if line.startswith(line_in_paramfile):
                print(line.split("="))
                return line.split("=")[1].strip()
