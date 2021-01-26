/*
 * matrix_read_test.cpp
 *
 *  Created on: Jul 12, 2016
 *      Author: amklinv
 */

#include<cstdlib>
#include "TuckerMPI.hpp"

template <class scalar_t>
bool checkUTEqual(const scalar_t* arr1, const scalar_t* arr2, int numRows);

int main(int argc, char* argv[])
{

// specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t; 
  std::string filename = "input_files/tensor24_single.mpi";
#else
  typedef double scalar_t;
  std::string filename = "input_files/tensor24.mpi";
#endif

  // Initialize MPI
  MPI_Init(&argc,&argv);

  // Get rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // Create a distribution object
  int ndims = 3;
  Tucker::SizeArray* sz =
      Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*sz)[0] = 2; (*sz)[1] = 2; (*sz)[2] = 6;
  Tucker::SizeArray* nprocsPerDim =
      Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*nprocsPerDim)[0] = 1; (*nprocsPerDim)[1] = 1; (*nprocsPerDim)[2] = 3;
  TuckerMPI::Distribution* dist =
        Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*sz,*nprocsPerDim);
  Tucker::MemoryManager::safe_delete(sz);
  Tucker::MemoryManager::safe_delete(nprocsPerDim);

  // Create a tensor
  TuckerMPI::Tensor<scalar_t>* tensor =
      Tucker::MemoryManager::safe_new<TuckerMPI::Tensor<scalar_t>>(dist);

  // Read the entries from a file
  TuckerMPI::importTensorBinary(filename.c_str(),tensor);

  // Compute the gram matrix in dimension 2
  const Tucker::Matrix<scalar_t>* mat = TuckerMPI::newGram(tensor,2);

  scalar_t trueData[36] = {14, 38, 62, 86, 110, 134,
                         38, 126, 214, 302, 390, 478,
                         62, 214, 366, 518, 670, 822,
                         86, 302, 518, 734, 950, 1166,
                         110, 390, 670, 950, 1230, 1510,
                         134, 478, 822, 1166, 1510, 1854};

  bool equal = checkUTEqual(mat->data(), trueData, 6);
  if(!equal) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);
  Tucker::MemoryManager::safe_delete(mat);

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  // Finalize MPI
  MPI_Finalize();

  return EXIT_SUCCESS;
}

template <class scalar_t>
bool checkUTEqual(const scalar_t* arr1, const scalar_t* arr2, int numRows)
{
  for(int r=0; r<numRows; r++) {
    for(int c=r; c<numRows; c++) {
      int ind = r+c*numRows;
      if(arr1[ind] != arr2[ind]) {
        return false;
      }
    }
  }
  return true;
}
