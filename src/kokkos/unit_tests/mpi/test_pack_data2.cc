/**
 * This test comes from:
 * TuckerMPI/src/mpi/tests/TuckerMPI_pack_data2.cpp
 */

#include <gtest/gtest.h>
#include "mpi.h"
#include "TuckerMpi.hpp"

TEST(tuckermpi_pack_data2, main){
  // Prepare
  typedef double scalar_t;

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