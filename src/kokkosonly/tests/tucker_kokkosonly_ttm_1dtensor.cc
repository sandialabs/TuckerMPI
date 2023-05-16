#include <gtest/gtest.h>

#include "Tucker_SizeArray.hpp"
#include "Tucker_Tensor.hpp"
#include "ttm.hpp"

using scalar_t = double;
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
using matrix = Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space>;

// ================================================== GENERALIZE ==================================================

/**
 * 1. create tensor X with [var_x] dims and size [var_y x var_z x etc ]
 * 2. fill this X tensor witht data
 * 3. create a matrix with random size / size that we choose
 * 4. fill this matrix with data
 * 5. call ttm and put inside a TuckerKokkos::Tensor result
 * 6. get data from ttm with result.data();
 * 7. compute true data in a std::array<scalar_t, NUMBERVALUES> trueData;  
 * 8. loop check values with ASSERT_EQ
 */

class TestTensor2D : public ::testing::Test {
    protected:

    /**
     * @brief Set the Up object. Create a tensor and its values
     * 
     */
    void SetUp() override {
        size = TuckerKokkos::SizeArray(n);
        for(int i=0; i<n; i++) {
            size[i] = dims.at(i);
        }
        // Step 1
        X = TuckerKokkos::Tensor<scalar_t, memory_space>(size);
        auto view1d_d = X.data();
        auto view1d_h = Kokkos::create_mirror(view1d_d);
        // Step 2
        int values = 0;
        for(int i=0; i<n; i++) {
            for(int j=0; j<dims.at(i); j++) {
                // Values will be 0, 1, 2, etc
                view1d_h(values) = values;
                values = values + 1;
            }
        }
        Kokkos::deep_copy(view1d_d, view1d_h);
    }

    /**
     * @brief Create a matrix for testing object
     * 
     * @param number_rows 
     * @param number_cols 
     * @return matrix 
     */
    matrix create_matrix_for_testing(const int number_rows, const int number_cols) {
        // Step 3
        matrix my_mat("my_mat", number_rows, number_cols);
        // Step 4
        auto view2d_h = Kokkos::create_mirror(my_mat);
        int values = 0;
        // Values will be 0, 1, 2, etc sort by column-major
        for(int c=0; c<number_cols; c++){
            for(int r=0; r<number_rows; r++){
                view2d_h(r, c) = values;
                values = values + 1;
            }
        }
        Kokkos::deep_copy(my_mat, view2d_h);
        return my_mat;
    }

    /**
     * @brief Our own loop to do the matrix-matrix multiply 
     * 
     * @param tensor 
     * @param matrice 
     * @return std::vector<scalar_t> 
     */
    std::vector<scalar_t> compute_true_data(TuckerKokkos::Tensor<scalar_t, memory_space> tensor, matrix matrice) {
        // TODO


        return {10, 11, 12};
    }
    
    // PARAMS 
    int n = 2;              // n-d tensor
    std::vector<int> dims = // with X values for each n dimensions 
        { 10, 15 }; 
    int nRowsMatrix = 4;    // num rows max test matrix
    int nColsMatrix = 5;    // num cols max test matrix
    // PARAMS

    TuckerKokkos::SizeArray size;
    TuckerKokkos::Tensor<scalar_t, memory_space> X;
    scalar_t* data;
};

// TENSOR 2D
TEST_F(TestTensor2D, Matrice_TODO) {
    // Step 1 & 2: Done with Test Fixtures
    for (int nRows=1; nRows<nRowsMatrix; ++nRows) { 
        for (int nCols=1; nCols<nColsMatrix; ++nCols) {
            // Step 3 & 4
            matrix mat = create_matrix_for_testing(nRows, nCols);
            // 5. TODO
            std::cout << "X.N(): " << X.N() << std::endl;
            std::cout << "?: " << X.size().prod(1,X.N()-1) << std::endl;
            //TuckerKokkos::Tensor<scalar_t, memory_space> result =
                //TuckerKokkos::ttm(&X, 0, mat, false);
            // 6. TODO
            //ttm_data = result.data().data();
            // 7. TODO
            std::vector<scalar_t> trueData = compute_true_data(X, mat);
            // Final step 
            /*for(int i=0; i<trueData.size(); i++) {
                ASSERT_EQ(ttm_data[i], trueData[i]);
            }*/
            ASSERT_EQ(2, 2);
        }
    }
    ASSERT_EQ(4, 3);
}
