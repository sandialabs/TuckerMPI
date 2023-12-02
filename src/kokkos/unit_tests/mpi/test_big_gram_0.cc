/**
 * Originally import test here:
 * https://gitlab.com/tensors/TuckerMPI/-/blob/master/src/mpi/tests/TuckerMPI_bigGram0.cpp
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

template <class scalar_t>
bool checkUTEqual(const scalar_t* arr1, const scalar_t* arr2, int numRows)
{
  for(int r=0; r<numRows; r++) {
    for(int c=r; c<numRows; c++) {
      int ind = r+c*numRows;
      if(arr1[ind] != arr2[ind]) {
        std::cerr << "arr[" << ind << "] (" << r << "," << c
          << ") are not equal(" << arr1[ind] << " "
          << arr2[ind] << ")\n";
        return false;
      }
    }
  }
  return true;
}

TEST(tuckermpi, big_gram_0)
{
  std::vector<int> dims = {4, 4, 4};
  std::vector<int> procs = {2, 2, 2};
  Tensor<scalar_t> T(dims, procs);
  read_tensor_binary(mpi_rank(), T, "./tensor_data_files/4x4x4.bin");
 
  auto matrix = TuckerMpi::compute_gram(T, 0);
  auto m_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrix);

  const scalar_t TRUE_SOLUTION[36] =
    {19840, 20320, 20800, 21280,
     20320, 20816, 21312, 21808,
     20800, 21312, 21824, 22336,
     21280, 21808, 22336, 22864};

  bool matchesTrueSol = checkUTEqual(m_h.data(), TRUE_SOLUTION, 4);
  ASSERT_TRUE(matchesTrueSol);
}