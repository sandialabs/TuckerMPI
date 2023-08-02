import numpy as np
import compare_utils

if __name__== "__main__":

    args = compare_utils.get_comparison_arguments()

    # we have as many modes to read as the ranks of the tensor
    modes = compare_utils.extract_datatensor_rank("./paramfile.txt")

    # figure out where sing values are written to
    svPath   = compare_utils.extract_single_line_value_from_paramfile("SV directory = ", "./paramfile.txt")
    svPrefix = compare_utils.extract_single_line_value_from_paramfile("SV file prefix = ", "./paramfile.txt")

    # compare
    for i in range(0, modes):
        print("Comparing eigenvalues for mode = {}".format(i))

        gold = np.loadtxt("gold"+str(i)+".txt")
        computed = np.loadtxt(svPath + "/" + svPrefix + "_mode_"+str(i)+".txt")

        assert (len(gold) == len(computed)), \
            "Failing due to mismatching extents!"
        assert (np.allclose(gold, computed, rtol=args.relTol, atol=args.absTol)), \
            "Failing due to numerical differences!"
