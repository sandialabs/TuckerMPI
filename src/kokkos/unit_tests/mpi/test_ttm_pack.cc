/**
 * Originally import tests here:
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/mpi/tests/TuckerMPI_pack_data.cpp
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/mpi/tests/TuckerMPI_pack_data2.cpp
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/mpi/tests/TuckerMPI_pack_data3.cpp
 */

#include <gtest/gtest.h>
#include "mpi.h"
#include "TuckerMpi.hpp"

TEST(tuckermpi, pack_data){
  using scalar_t = double;

  // Create a 4x2x2 tensor
  std::vector<int> dims = {4, 2, 2};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);

  // Fill it with the entries 0:15
  auto tensor_d = tensor.data();
  for(int i=0; i<16; i++){
    tensor_d(i) = i;
  }

  // Create a map describing the distribution
  TuckerMpi::Map map(4, MPI_COMM_WORLD);

  // Pack the tensor
  TuckerMpi::impl::packForTTM(tensor, 0, &map);

  // Check the result
  scalar_t trueResult[16] = {0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15};
  for(int i=0; i<16; i++) {
    ASSERT_TRUE(tensor_d(i) == trueResult[i]);
  }
}

TEST(tuckermpi, pack_data2){
  using scalar_t = double;

  // Create a 3x4x3 tensor
  std::vector<int> dims = {3, 4, 3};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);

  // Fill it with the entries 0:35
  auto tensor_d = tensor.data();
  for(int i=0; i<36; i++){
    tensor_d(i) = i;
  }

  // Create a map describing the distribution
  TuckerMpi::Map map(4, MPI_COMM_WORLD);

  // Pack the tensor
  TuckerMpi::impl::packForTTM(tensor, 1, &map);

  // Check the result
  scalar_t trueResult[36] = {0,1,2,3,4,5,12,13,14,15,16,17,24,25,26,27,28,29,6,7,8,9,10,11,18,19,20,21,22,23,30,31,32,33,34,35};
  for(int i=0; i<36; i++) {
    ASSERT_TRUE(tensor_d(i) == trueResult[i]);
  }
}

TEST(tuckermpi, pack_data3){
  using scalar_t = double;

  // Create a 3x4x3 tensor
  std::vector<int> dims = {3, 4, 3};
  TuckerOnNode::Tensor<scalar_t> tensor(dims);

  // Fill it with the entries 0:35
  auto tensor_d = tensor.data();
  for(int i=0; i<36; i++){
    tensor_d(i) = i;
  }

  // Create a map describing the distribution
  TuckerMpi::Map map(4, MPI_COMM_WORLD);

  // Pack the tensor
  TuckerMpi::impl::packForTTM(tensor, 2, &map);

  // Fill true result
  scalar_t trueResult[36];
  for(int i=0; i<36; i++) trueResult[i] = i;

  // Check the result
  for(int i=0; i<36; i++) {
    ASSERT_TRUE(tensor_d(i) == trueResult[i]);
  }
}
