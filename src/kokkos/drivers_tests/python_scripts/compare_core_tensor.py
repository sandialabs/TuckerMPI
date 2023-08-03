import numpy as np
import compare_utils

if __name__== "__main__":

    args = compare_utils.get_comparison_arguments()

    # figure out where sthosvd values are written to
    sthosvdPath     = compare_utils.extract_single_line_value_from_paramfile("STHOSVD directory = ", "./paramfile.txt")
    sthosvdPrefix   = compare_utils.extract_single_line_value_from_paramfile("STHOSVD file prefix = ", "./paramfile.txt")

    # convert real data into text file
    binary = np.fromfile(sthosvdPath + "/" + sthosvdPrefix + "_core.mpi")
    np.savetxt("computedCoreTensor.txt", binary)

    # use data from text files
    gold = np.loadtxt("goldCoreTensor.txt")
    computed = np.loadtxt("computedCoreTensor.txt")

    # compare
    assert(len(gold) == len(computed)), \
        "Failing due to mismatching extents!"
    assert(np.allclose(np.abs(gold), np.abs(computed), rtol=args.relTol, atol=args.absTol)), \
        "Failing due to numerical differences!"
