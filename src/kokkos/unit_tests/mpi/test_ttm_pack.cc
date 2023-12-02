/**
 * Originally import tests here:
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/mpi/tests/TuckerMPI_pack_data.cpp
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/mpi/tests/TuckerMPI_pack_data2.cpp
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/mpi/tests/TuckerMPI_pack_data3.cpp
 */

#include <gtest/gtest.h>
#include "mpi.h"
#include "TuckerMpi.hpp"

TEST(tuckermpi, ttm_pack_data){
  using scalar_t = double;

  // Create a 4x2x2 tensor
  std::vector<int> dims = {4, 2, 2};
  TuckerOnNode::Tensor<scalar_t> T(dims);

  auto T_h = Tucker::create_mirror(T);
  auto T_h_view = T_h.data();

  // Fill it with the entries 0:15
  for(int i=0; i<16; i++){ T_h_view(i) = i; }
  Tucker::deep_copy(T, T_h);

  // Create a map describing the distribution
  TuckerMpi::Map map(4, MPI_COMM_WORLD);

  // Pack the tensor
  TuckerMpi::impl::packForTTM(T, 0, &map);
  Tucker::deep_copy(T_h, T);

  // Check the result
  std::vector<scalar_t> trueResult = {0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15};
  for(int i=0; i<16; i++) {
    ASSERT_TRUE(T_h_view(i) == trueResult[i]);
  }
}

TEST(tuckermpi, ttm_pack_data2){
  using scalar_t = double;

  // Create a 3x4x3 tensor
  std::vector<int> dims = {3, 4, 3};
  TuckerOnNode::Tensor<scalar_t> T(dims);
  auto T_h = Tucker::create_mirror(T);
  auto T_h_view = T_h.data();

  // Fill it with the entries 0:35
  for(int i=0; i<36; i++){ T_h_view(i) = i; }
  Tucker::deep_copy(T, T_h);

  // Create a map describing the distribution
  TuckerMpi::Map map(4, MPI_COMM_WORLD);

  // Pack the tensor
  TuckerMpi::impl::packForTTM(T, 1, &map);
  Tucker::deep_copy(T_h, T);

  // Check the result
  std::vector<scalar_t> trueResult = {0,1,2,3,4,5,12,13,14,15,16,17,24,25,26,27,28,29,6,7,8,9,10,11,18,19,20,21,22,23,30,31,32,33,34,35};
  for(int i=0; i<36; i++) {
    ASSERT_TRUE(T_h_view(i) == trueResult[i]);
  }
}

TEST(tuckermpi, ttm_pack_data3){
  using scalar_t = double;

  // Create a 3x4x3 tensor
  std::vector<int> dims = {3, 4, 3};
  TuckerOnNode::Tensor<scalar_t> T(dims);
  auto T_h = Tucker::create_mirror(T);
  auto T_h_view = T_h.data();

  // Fill it with the entries 0:35
  for(int i=0; i<36; i++){ T_h_view(i) = i; }
  Tucker::deep_copy(T, T_h);

  // Create a map describing the distribution
  TuckerMpi::Map map(4, MPI_COMM_WORLD);

  // Pack the tensor
  TuckerMpi::impl::packForTTM(T, 2, &map);
  Tucker::deep_copy(T_h, T);

  // Fill true result
  std::vector<scalar_t> trueResult(36);
  std::iota(trueResult.begin(), trueResult.end(), 0);

  // Check the result
  for(int i=0; i<36; i++) {
    ASSERT_TRUE(T_h_view(i) == trueResult[i]);
  }
}
