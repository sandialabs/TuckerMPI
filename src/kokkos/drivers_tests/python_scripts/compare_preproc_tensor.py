import numpy as np
import compare_utils

if __name__== "__main__":

    args = compare_utils.get_comparison_arguments()

    gold = np.fromfile("gold_normalized_tensor.bin")
    computed = np.fromfile("normalized_tensor.bin")

    # compare
    assert(len(gold) == len(computed)), \
        "Failing due to mismatching extents!"
    assert(np.allclose(np.abs(gold), np.abs(computed), rtol=args.relTol, atol=args.absTol)), \
        "Failing due to numerical differences!"
