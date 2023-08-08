#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

using namespace TuckerMpi;

using scalar_t = double;
using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
using result_t = impl::Matrix<scalar_t>;

const MPI_Comm comm = MPI_COMM_WORLD;

void setup_and_run(int mode){

  // Tensor T
  std::vector<int> dataTensorDim = {3, 4, 5, 6};
  std::vector<int> proc_grid_dims = {2, 1, 2, 2};
  Tensor<scalar_t, memory_space> T(dataTensorDim, proc_grid_dims);
  read_tensor_binary(T, "./tensor_data_files/3x5x7x11.bin");

  // Map M
  const int ndims = T.rank();
  const auto & sz = T.localDimensionsOnHost();
  const int nrows = T.globalExtent(mode);
  size_t ncols = impl::prod(sz, 0, mode-1, 1) * impl::prod(sz, mode+1, ndims-1, 1);
  ASSERT_TRUE(ncols <= std::numeric_limits<int>::max());
  result_t recvY(nrows, (int)ncols, comm, false);
  const Map* M = recvY.getMap();

  const std::vector<scalar_t> gold = impl::pack_for_gram_fallback_copy_host(T, mode, M);
  const std::vector<scalar_t> computed = impl::pack_for_gram(T, mode, M);
  for (int i=0; i<gold.size(); ++i){
    ASSERT_EQ(gold[i], computed[i]);
  }
}

// TEST(tuckermpi_pack_for_gram, test_along_mode_0){
//   setup_and_run(0);
// }

// TEST(tuckermpi_pack_for_gram, test_along_mode_1){
//   setup_and_run(1);
// }

// // TEST(tuckermpi_pack_for_gram, test_along_mode_2){
// //   setup_and_run(2);
// // }

TEST(tuckermpi_pack_for_gram, test_along_mode_3){
   setup_and_run(3);
}
