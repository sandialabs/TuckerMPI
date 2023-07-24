/**
 * This test comes from:
 * TuckerMPI/src/mpi/tests/TuckerMPI_pack_data.cpp
 */

#include <gtest/gtest.h>
#include "mpi.h"
#include "TuckerMpi.hpp"
#include "TuckerMpi_Map.hpp"

TEST(tuckermpi_pack_data, main){
  // Prepare
  typedef double scalar_t;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

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
    ASSERT_TRUE(tensor_d[i] == trueResult[i]);
  }
}
