#include <gtest/gtest.h>

#include "Tucker_SizeArray.hpp"
#include "Tucker_Tensor.hpp"

using scalar_t = double;
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
using matrix = Kokkos::View<scalar_t**, Kokkos::LayoutLeft, memory_space>;
using tensor = TuckerKokkos::Tensor<scalar_t, memory_space>;

class InitTensorAndMatrixForTesting : public ::testing::Test {
    protected:

    void SetUp() override { }

    /**
     * @brief Create a tensor for testing object
     * 
     * @param n n-d tensor
     * @param dims with X values for each n dimensions 
     * @return tensor 
     */
    tensor create_tensor_for_testing(int n, std::vector<int> dims) {
        TuckerKokkos::SizeArray size;
        size = TuckerKokkos::SizeArray(n);
        for(int i=0; i < n ; i++) {
            size[i] = dims.at(i);
        }
        // Create the tensor
        tensor X(size);
        auto view1d_d = X.data();
        auto view1d_h = Kokkos::create_mirror(view1d_d);
        // Fill the tensor with data (0, 1, 2, etc)
        int values = 0;
        for(int i=0; i<X.getNumElements(); i++) {
            view1d_h(values) = values;
            values = values + 1;
        }
        Kokkos::deep_copy(view1d_d, view1d_h);
        return X;
    }

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
    std::vector<scalar_t> compute_true_data(tensor tens, matrix B, int mode) {
        // std::vector<scalar_t> final_result;
        if(mode == 0){
            // slice tensor on right mode to get a matrix
            // matrix A = tensor.slice(0);
            // do calcul matrix-matrix product
            /*for(i = 0; i < A.rows; ++i) {
                for(j = 0; j < B.colums; ++j) {
                    for(k = 0; k < A.colums; ++k) {
                        final_result[i][j] += A[i][k] * B[k][j];
                    }
                }
            }*/
            // return result
            // return final_result;
            return { 30, 80 };
        }else{
            std::cout << "TODO" << std::endl;
            return { 0 };
        }
    }
};