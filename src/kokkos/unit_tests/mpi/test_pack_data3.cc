/**
 * This test comes from:
 * TuckerMPI/src/mpi/tests/TuckerMPI_pack_data3.cpp
 */

#include <gtest/gtest.h>
#include "mpi.h"
#include "TuckerMpi.hpp"

TEST(tuckermpi_pack_data3, main){
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
  TuckerMpi::impl::packForTTM(tensor, 2, &map);

  // Fill true result
  scalar_t trueResult[36];
  for(int i=0; i<36; i++) trueResult[i] = i;

  // Check the result
  for(int i=0; i<36; i++) {
    ASSERT_TRUE(tensor_d(i) == trueResult[i]);
  }
}