/**
 * This test comes from:
 * TuckerMPI/src/mpi/tests/TuckerMPI_pack_data.cpp
 */

#include <gtest/gtest.h>
#include "mpi.h"
#include "TuckerMpi.hpp"
#include "TuckerMpi_Map.hpp"

const MPI_Comm comm = MPI_COMM_WORLD;

TEST(tuckermpi_pack_data, main){
  // Prepare
  typedef double scalar_t;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  // Create a 4x2x2 tensor
  auto data_tensor_dim = {4, 2, 2};
  auto proc_grid_dims = {2, 2, 2};
  TuckerMpi::Tensor<scalar_t, memory_space> tensor(data_tensor_dim, proc_grid_dims);

  // Fill it with the entries 0:15
  auto local_tensor_view_d = tensor.localTensor().data();
  auto local_tensor_view_h = Kokkos::create_mirror(local_tensor_view_d);
  for(int i=0; i<16; i++){
    // ERROR HERE
    local_tensor_view_h(i) = i;
    // Who can I modify data of a TuckerMpi::Tensor?
  }
  Kokkos::deep_copy(local_tensor_view_d, local_tensor_view_h);

  // Create a map describing the distribution
  TuckerMpi::Map m(4, comm);

  // Pack the tensor
  TuckerMpi::impl::packForTTM(tensor.localTensor(), 0, &m);

  // Check the result
  scalar_t trueResult[16] = {0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15};
  for(int i=0; i<16; i++) {
    ASSERT_TRUE(local_tensor_view_d[i] == trueResult[i]);
  }
}
