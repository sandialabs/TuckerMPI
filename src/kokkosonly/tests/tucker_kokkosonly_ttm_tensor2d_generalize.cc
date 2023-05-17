#include <gtest/gtest.h>

#include "Tucker_Tensor.hpp"
#include "ttm.hpp"

// At the moment the creation of the tensor and the matrix
// are in the same file but this should be separated.
#include "InitTensorAndMatrixForTesting.hpp"

/** Idea of generalised testing:
 * We want to be able to choose the size of the tensor and the matrix(es) to be tested.
 * 
 * We proceed with the following steps:
 * 1. Thanks to Test Fixture we import:
 *      - a tensor with the size and the data we want
 *      - a matrix with the size and the data we want
 * 2. Get results of a TTM call with these previously defined values
 * 3. Calculation of good results/true data with our own matrix calculation method
 * 4. Loop and compare results
 */

// RESULT HARDCODE
TEST_F(InitTensorAndMatrixForTesting, Tensor2d5x7_Matrix1x5_Mode0) {
    tensor my_tensor = create_tensor_for_testing(2, {5, 7});
    int nRows = 1;
    int nCols = my_tensor.size(0);
    // Get a matrix 
    matrix mat = create_matrix_for_testing(nRows, nCols);
    // Compute TTM result (A = mat; B = X)
    TuckerKokkos::Tensor<scalar_t, memory_space> result =
        TuckerKokkos::ttm(&my_tensor, 0, mat, false);
    scalar_t* ttm_data = result.data().data();
    // Find true data
    std::vector<scalar_t> trueData = compute_true_data(my_tensor, mat, 0);
    // Compare results
    for(int i=0; i<2; i++) { // TODO
        ASSERT_EQ(ttm_data[i], trueData[i]);
    }
}

TEST_F(InitTensorAndMatrixForTesting, Tensor2d5x7_AllMatrixUntil6x5_Mode0) {

    // Test for matrix 1x5, 2x5, 3x5, 4x5, 5x5 and 6x5
    int nRowsMatrix = 7;  // num rows max test matrix
    // Get a 2d tensor with 5 values x 7 values
    tensor my_tensor = create_tensor_for_testing(2, {5, 7});
    int mode = 0;

    int nCols = my_tensor.size(0); // = dims.at(0) = X.size(0);
    for (int nRows=1; nRows<nRowsMatrix; ++nRows) {
        std::cout << "Test Matrix " << nRows << "x" << nCols << std::endl;
        // Get a matrix 
        matrix mat = create_matrix_for_testing(nRows, nCols);
        // Compute TTM result (A = mat; B = X)
        TuckerKokkos::Tensor<scalar_t, memory_space> result =
            TuckerKokkos::ttm(&my_tensor, 0, mat, false);
        scalar_t* ttm_data = result.data().data();
        // Find true data
        std::vector<scalar_t> trueData = compute_true_data(my_tensor, mat, mode);
        // Compare results
        for(int i=0; i<result.getNumElements(); i++) {
            ASSERT_EQ(ttm_data[i], trueData[i]);
        }
    }
}

/*
TEST_F(InitTensorAndMatrixForTesting, Tensor3d_4x2x3_...) {

    // PARAMS
    int nRowsMatrix = 7;
    tensor my_tensor = create_tensor_for_testing(3, {4, 2, 3});

    // ...
    // TODO
    ASSERT_EQ(2, 3);
}
*/