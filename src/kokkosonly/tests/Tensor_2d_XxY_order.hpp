#include <gtest/gtest.h>

#include "Tucker_SizeArray.hpp"
#include "Tucker_Tensor.hpp"

using scalar_t = double;
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
using matrix = Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space>;

class Tensor_2d_XxY_order : public ::testing::Test {
    protected:

    /**
     * @brief Set the Up object. Create a tensor and its values
     * 
     */
    void SetUp() override {
        size = TuckerKokkos::SizeArray(n);
        for(int i=0; i < n ; i++) {
            size[i] = dims.at(i);
        }
        // Create the tensor
        X = TuckerKokkos::Tensor<scalar_t, memory_space>(size);
        auto view1d_d = X.data();
        auto view1d_h = Kokkos::create_mirror(view1d_d);
        // Fill the tensor with data (0, 1, 2, etc)
        int values = 0;
        for(int i=0; i<X.getNumElements(); i++) {
            view1d_h(values) = values;
            values = values + 1;
        }
        Kokkos::deep_copy(view1d_d, view1d_h);
    }

    // function create tensor who return a tensor with the size that I want !

    /**
     * @brief Create a matrix for testing object
     * 
     * @param number_rows 
     * @param number_cols 
     * @return matrix 
     */
    matrix create_matrix_for_testing(const int number_rows, const int number_cols) {
        matrix my_mat("my_mat", number_rows, number_cols);
        // Fill the matrix with data (0, 1, 2, etc)
        auto view2d_h = Kokkos::create_mirror(my_mat);
        int values = 0;
        // Values are order and sort by column-major
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
        { 5, 7 }; 
    int nRowsMatrix = 4;  // num rows max test matrix
    //int nColsMatrix = 5;    // num cols max test matrix
    // PARAMS

    TuckerKokkos::SizeArray size;
    TuckerKokkos::Tensor<scalar_t, memory_space> X;
};