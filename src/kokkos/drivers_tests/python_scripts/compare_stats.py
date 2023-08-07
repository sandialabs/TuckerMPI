import numpy as np
import compare_utils

if __name__== "__main__":

    args = compare_utils.get_comparison_arguments()

    gold = np.loadtxt("gold_stats.txt", skiprows=3)
    computed = np.loadtxt("stats.txt", skiprows=3)

    assert(len(gold) == len(computed)), \
        "Failing due to mismatching extents!"
    assert(np.allclose(np.abs(gold), np.abs(computed), rtol=args.relTol, atol=args.absTol)), \
        "Failing due to numerical differences!"
