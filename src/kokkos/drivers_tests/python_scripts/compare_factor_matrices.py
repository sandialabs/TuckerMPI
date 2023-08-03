import numpy as np
import compare_utils

if __name__== "__main__":

    args = compare_utils.get_comparison_arguments()

    # we have as many modes to read as the ranks of the tensor
    modes = compare_utils.extract_datatensor_rank("./paramfile.txt")

    # figure out where sthosvd values are written to
    sthosvdPath     = compare_utils.extract_single_line_value_from_paramfile("STHOSVD directory = ", "./paramfile.txt")
    sthosvdPrefix   = compare_utils.extract_single_line_value_from_paramfile("STHOSVD file prefix = ", "./paramfile.txt")

    # convert
    for i in range(0, modes):
        binary = np.fromfile(sthosvdPath + "/" + sthosvdPrefix + "_mat_" + str(i) + ".mpi")
        np.savetxt("factorMatrix_" + str(i) + ".txt", binary)

    # compare
    for i in range(0, modes):
        gold = np.loadtxt("goldFactorMatrix" + str(i) + ".txt")
        computed = np.loadtxt("factorMatrix_" + str(i) + ".txt")

        assert (len(gold) == len(computed)), \
            "Failing due to mismatching extents!"
        assert (np.allclose(np.abs(gold), np.abs(computed), rtol=args.relTol, atol=args.absTol)), \
            "Failing due to numerical differences!"