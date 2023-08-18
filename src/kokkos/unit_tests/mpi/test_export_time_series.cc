#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

using namespace TuckerMpi;

const MPI_Comm comm = MPI_COMM_WORLD;

int mpi_rank(){
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

TEST(tuckermpi_tensor_io, export_time_series_A)
{
  using scalar_t = double;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;
  using tensor_type = TuckerMpi::Tensor<scalar_t, memory_space>;

  const std::vector<int> tensorDims = {4,6,3};
  const std::vector<int> procGrid = {2,2,1};
  TuckerMpi::Distribution dist(tensorDims, procGrid);
  tensor_type T(dist);
  auto T_l = T.localTensor();
  auto Tl_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), T_l);
  auto T_h_view = Tl_h.data();
  for(size_t i=0; i<T_h_view.extent(0); i++) {
    T_h_view(i) = i;
  }

  const std::vector<std::string> filenames = {"f1.bin", "f2.bin", "f3.bin"};
  TuckerMpi::write_tensor_binary_multifile(mpi_rank(), T, filenames);
}
