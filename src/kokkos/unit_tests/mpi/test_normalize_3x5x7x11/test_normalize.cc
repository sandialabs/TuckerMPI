/**
 * Originally import test here:
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/mpi/tests/TuckerMPI_normalize_test.cpp
 */

#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

using namespace TuckerMpi;
using scalar_t = double;

const MPI_Comm comm = MPI_COMM_WORLD;
const scalar_t stdThresh = 1e-9;
const scalar_t tol = 100 * std::numeric_limits<scalar_t>::epsilon();

int mpi_size(){
  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  return nprocs;
}

int mpi_rank(){
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

template<class scalar_t, class ...Props1, class ...Props2>
bool checks(const Tensor<scalar_t, Props1...> t1,
            const Tensor<scalar_t, Props2...> t2,
            scalar_t tol)
{
  auto t1_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), t1);
  auto t2_h = Tucker::create_mirror_and_copy(Kokkos::HostSpace(), t2);

  // GLOBAL - 1) Same size
  if(t1_h.globalSize() != t2_h.globalSize()) {
    return false;
  }

  // LOCAL - 1) Owns any data
  if(t1_h.localTensor().size() == 0 && t2_h.localTensor().size() == 0){
    return true;
  }

  // LOCAL - 2) Same size
  if(t1_h.localTensor().size() != t2_h.localTensor().size()) {
    return false;
  }

  // LOCAL - 3) Values
  int numElements = t1_h.localTensor().size();
  scalar_t errNorm2 = 0;
  for(int i=0; i<numElements; i++) {
    scalar_t err = std::abs(t1_h.localTensor().data()[i] - t2_h.localTensor().data()[i]);
    if(std::isnan(err)){
      return false;
    }
    errNorm2 += (err*err);
  }

  // LOCAL - 4) Tol
  scalar_t origNorm2 = t1_h.localTensor().frobeniusNormSquared();
  scalar_t relErr = std::sqrt(errNorm2/origNorm2);
  if(relErr > tol){
    return false;
  }

  // Return
  return true;
}

bool runSim(std::initializer_list<int> procs)
{
  std::vector<int> dims = {3, 5, 7, 11};
  Tensor<scalar_t> computed_T(dims, procs);
  Tensor<scalar_t> GOLD_TENSOR(dims, procs);
  
  // Normalize Tensor with MinMax for mode 0, 1, 2 and 3
  {
    for(int scaleMode = 0; scaleMode < 4; scaleMode++){
      read_tensor_binary(mpi_rank(), computed_T, "../tensor_data_files/3x5x7x11.bin");
      read_tensor_binary(mpi_rank(), GOLD_TENSOR, "./gold_3x5x7x11_mm"+std::to_string(scaleMode)+".bin");
      auto metricsData = TuckerMpi::compute_slice_metrics(mpi_rank(), computed_T, scaleMode, Tucker::defaultMetrics);   
      [[maybe_unused]] auto [r1,r2] = TuckerMpi::normalize_tensor(mpi_rank(), computed_T, metricsData, "MinMax", scaleMode, stdThresh);    
      if (!checks(computed_T, GOLD_TENSOR, tol)){
        return false;
      }
    }
  }
  
  // // Normalize Tensor with StandardCentering for mode 0, 1, 2 and 3
  // {
  //   for(int scaleMode = 0; scaleMode < 4; scaleMode++){
  //     read_tensor_binary(computed_T, "../tensor_data_files/3x5x7x11.bin");
  //     read_tensor_binary(GOLD_TENSOR, "./gold_3x5x7x11_sc"+std::to_string(scaleMode)+".bin");
  //     auto metricsData = TuckerMpi::compute_slice_metrics(mpi_rank(), computed_T, scaleMode, Tucker::defaultMetrics);
  //     TuckerMpi::normalize_tensor(mpi_rank(), computed_T, metricsData, "StandardCentering", scaleMode, stdThresh);
  //     if (!checks(computed_T, GOLD_TENSOR, tol)){
  //       return false;
  //     }
  //   }
  // }

  return true;
}

TEST(tuckermpi, normalize_nprocs1)
{
  if(mpi_size() == 1) {
    ASSERT_TRUE(runSim({1,1,1,1}));
  }
}

TEST(tuckermpi, normalize_nprocs2)
{
  if(mpi_size() == 2) {
    ASSERT_TRUE(runSim({2,1,1,1}));
    ASSERT_TRUE(runSim({1,2,1,1}));
    ASSERT_TRUE(runSim({1,1,2,1}));
    ASSERT_TRUE(runSim({1,1,1,2}));
  }
}

TEST(tuckermpi, normalize_nprocs3)
{
  if(mpi_size() == 3) {
    ASSERT_TRUE(runSim({3,1,1,1}));
    ASSERT_TRUE(runSim({1,3,1,1}));
    ASSERT_TRUE(runSim({1,1,3,1}));
    ASSERT_TRUE(runSim({1,1,1,3}));
  }
}

TEST(tuckermpi, normalize_nprocs4)
{
  if(mpi_size() == 4) {
    // ASSERT_TRUE(runSim({4,1,1,1}));
    ASSERT_TRUE(runSim({1,4,1,1}));
    ASSERT_TRUE(runSim({1,1,4,1}));
    ASSERT_TRUE(runSim({1,1,1,4}));
    ASSERT_TRUE(runSim({2,2,1,1}));
    ASSERT_TRUE(runSim({2,1,2,1}));
    ASSERT_TRUE(runSim({2,1,1,2}));
    ASSERT_TRUE(runSim({1,2,2,1}));
    ASSERT_TRUE(runSim({1,2,1,2}));
    ASSERT_TRUE(runSim({1,1,2,2}));
  }
}

TEST(tuckermpi, normalize_nprocs5)
{
  if(mpi_size() == 5) {
    ASSERT_TRUE(runSim({5,1,1,1}));
    ASSERT_TRUE(runSim({1,5,1,1}));
    ASSERT_TRUE(runSim({1,1,5,1}));
    ASSERT_TRUE(runSim({1,1,1,5}));
  }
}

TEST(tuckermpi, normalize_nprocs6)
{
  if(mpi_size() == 6) {
    ASSERT_TRUE(runSim({6,1,1,1}));
    ASSERT_TRUE(runSim({1,6,1,1}));
    ASSERT_TRUE(runSim({1,1,6,1}));
    ASSERT_TRUE(runSim({1,1,1,6}));
    ASSERT_TRUE(runSim({2,3,1,1}));
    ASSERT_TRUE(runSim({2,1,3,1}));
    ASSERT_TRUE(runSim({2,1,1,3}));
    ASSERT_TRUE(runSim({1,2,3,1}));
    ASSERT_TRUE(runSim({1,2,1,3}));
    ASSERT_TRUE(runSim({1,1,2,3}));
    ASSERT_TRUE(runSim({3,2,1,1}));
    ASSERT_TRUE(runSim({3,1,2,1}));
    ASSERT_TRUE(runSim({3,1,1,2}));
    ASSERT_TRUE(runSim({1,3,2,1}));
    ASSERT_TRUE(runSim({1,3,1,2}));
    ASSERT_TRUE(runSim({1,1,3,2}));
  }
}

TEST(tuckermpi, normalize_nprocs7)
{
  if(mpi_size() == 7) {
    ASSERT_TRUE(runSim({7,1,1,1}));
    ASSERT_TRUE(runSim({1,7,1,1}));
    ASSERT_TRUE(runSim({1,1,7,1}));
    ASSERT_TRUE(runSim({1,1,1,7}));
  }
}

TEST(tuckermpi, normalize_nprocs8)
{
  if(mpi_size() == 8) {
    ASSERT_TRUE(runSim({8,1,1,1}));
    ASSERT_TRUE(runSim({1,8,1,1}));
    ASSERT_TRUE(runSim({1,1,8,1}));
    ASSERT_TRUE(runSim({1,1,1,8}));
    ASSERT_TRUE(runSim({4,2,1,1}));
    ASSERT_TRUE(runSim({4,1,2,1}));
    ASSERT_TRUE(runSim({4,1,1,2}));
    ASSERT_TRUE(runSim({1,4,2,1}));
    ASSERT_TRUE(runSim({1,4,1,2}));
    ASSERT_TRUE(runSim({1,1,4,2}));
    ASSERT_TRUE(runSim({2,4,1,1}));
    ASSERT_TRUE(runSim({2,1,4,1}));
    ASSERT_TRUE(runSim({2,1,1,4}));
    ASSERT_TRUE(runSim({1,2,4,1}));
    ASSERT_TRUE(runSim({1,2,1,4}));
    ASSERT_TRUE(runSim({1,1,2,4}));
    ASSERT_TRUE(runSim({2,2,2,1}));
    ASSERT_TRUE(runSim({2,2,1,2}));
    ASSERT_TRUE(runSim({2,1,2,2}));
    ASSERT_TRUE(runSim({1,2,2,2}));
  }
}
