#include <gtest/gtest.h>

#include "Tucker_Tensor.hpp"
#include "ttm.hpp"

// At the moment the creation of the tensor and the matrix
// are in the same file but this should be separated.
#include "Tensor_2d_XxY_order.hpp"

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

TEST_F(Tensor_2d_XxY_order, Tensor_5x7_Matrix_4x5) {
    // Get a tensor
    // TODO
    // my_tensor = create_tensor_for_testing(2, {5, 7});
    for (int nRows=1; nRows<nRowsMatrix; ++nRows) { 
        int nCols = X.size(0); // = dims.at(0) = X.size(0);
        //for (int nCols=1; nCols<nColsMatrix; ++nCols) {
            // Get a matrix 
            matrix mat = create_matrix_for_testing(nRows, nCols);
            // Compute TTM result (A = mat; B = X)
            TuckerKokkos::Tensor<scalar_t, memory_space> result =
                TuckerKokkos::ttm(&X, 0, mat, false);
            scalar_t* ttm_data = result.data().data();
            // Find true data
            // TODO
            std::vector<scalar_t> trueData = compute_true_data(X, mat);
            // Compare results
            // result.getNumElements() = 21
            // trueData.size() = ? => TODO
            for(int i=0; i<result.getNumElements(); i++) {
                // TODO
                // ASSERT_EQ(ttm_data[i], trueData[i]);
                ASSERT_EQ(2, 3);
            }
        //}
    }
}

//TEST_F(Tensor_2d_XxY_order, Tensor_AxB_Matrix_CxD) {}