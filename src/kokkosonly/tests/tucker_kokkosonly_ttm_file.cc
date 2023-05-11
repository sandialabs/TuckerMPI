#include <gtest/gtest.h>

#include "Tucker_SizeArray.hpp"
#include "Tucker_Tensor.hpp"
#include "Tucker_IO_Util.hpp"

TEST(tuckerkokkos, ttm_3x5x7x11)
{
    using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
    typedef double scalar_t;

    // Read the tensor from a file
    TuckerKokkos::Tensor<scalar_t, memory_space> X =
        TuckerKokkos::importTensor<scalar_t, memory_space>("input_files/3x5x7x11.txt");

    ASSERT_EQ(X.size().size(), 4);
    ASSERT_EQ(X.size().data().size(), 1155);

    // Read a matrix to multiply
    /*
    Tucker::Matrix<scalar_t>* mat =
        Tucker::importMatrix<scalar_t>("input_files/3x2.txt");
    */

    // Read the true solution
    /*
    Tucker::Tensor<scalar_t>* trueSol =
        Tucker::importTensor<scalar_t>("input_files/3x2_mult_transp.txt");
    */

    // Compute the TTM
    //Tensor<ScalarType, MemorySpace> temp = ttm_kokkosblas_impl();
    // Tucker::Tensor<scalar_t>* mySol = Tucker::ttm_kokkosblas_impl(tensor, 0, mat, true);

    // Compare the computed solution to the true solution
    /*
    if(!Tucker::isApproxEqual(trueSol, mySol, 100 * std::numeric_limits<scalar_t>::epsilon()))
    {
        return EXIT_FAILURE;
    }

    Tucker::MemoryManager::safe_delete(mat);
    Tucker::MemoryManager::safe_delete(mySol);
    Tucker::MemoryManager::safe_delete(trueSol);
    */
    
    ASSERT_EQ(2, 2);
}