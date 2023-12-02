/**
 * Originally import test here:
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/mpi/tests/TuckerMPI_norm_test.cpp
 */

#include <gtest/gtest.h>
#include "TuckerMpi.hpp"

using namespace TuckerMpi;
using scalar_t = double;

const MPI_Comm comm = MPI_COMM_WORLD;

int mpi_rank(){
  int rank;
  MPI_Comm_rank(comm, &rank);
  return rank;
}

int mpi_size(){
  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  return nprocs;
}

bool runSim(std::initializer_list<int> procs)
{
  std::vector<int> dims = {3, 5, 7, 11};
  Tensor<scalar_t> T(dims, procs);
  read_tensor_binary(mpi_rank(), T, "./tensor_data_files/3x5x7x11.bin");

  scalar_t computed_norm = sqrt(T.frobeniusNormSquared());

  const scalar_t TRUE_SOLUTION = 9.690249359274157;

  scalar_t diff = std::abs(computed_norm - TRUE_SOLUTION);
  if(diff < 100 * std::numeric_limits<scalar_t>::epsilon()) {
    return true;
  }

  return false;
}

TEST(tuckermpi, norm_nprocs2)
{
  if(mpi_size() == 2) {
    ASSERT_TRUE(runSim({2,1,1,1}));
    ASSERT_TRUE(runSim({1,2,1,1}));
    ASSERT_TRUE(runSim({1,1,2,1}));
    ASSERT_TRUE(runSim({1,1,1,2}));
  }
}

TEST(tuckermpi, norm_nprocs3)
{
  if(mpi_size() == 3) {
    ASSERT_TRUE(runSim({3,1,1,1}));
    ASSERT_TRUE(runSim({1,3,1,1}));
    ASSERT_TRUE(runSim({1,1,3,1}));
    ASSERT_TRUE(runSim({1,1,1,3}));
  }
}

TEST(tuckermpi, norm_nprocs4)
{
  if(mpi_size() == 4) {
    ASSERT_TRUE(runSim({4,1,1,1}));
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

TEST(tuckermpi, norm_nprocs5)
{
  if(mpi_size() == 5) {
    ASSERT_TRUE(runSim({5,1,1,1}));
    ASSERT_TRUE(runSim({1,5,1,1}));
    ASSERT_TRUE(runSim({1,1,5,1}));
    ASSERT_TRUE(runSim({1,1,1,5}));
  }
}

TEST(tuckermpi, norm_nprocs6)
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

TEST(tuckermpi, norm_nprocs7)
{
  if(mpi_size() == 7) {
    ASSERT_TRUE(runSim({7,1,1,1}));
    ASSERT_TRUE(runSim({1,7,1,1}));
    ASSERT_TRUE(runSim({1,1,7,1}));
    ASSERT_TRUE(runSim({1,1,1,7}));
  }
}

TEST(tuckermpi, norm_nprocs8)
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
