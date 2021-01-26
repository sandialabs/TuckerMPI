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
  std::string filename = "input_files/tensor64_single.mpi"; 
#else
  typedef double scalar_t;
  std::string filename = "input_files/tensor64.mpi";
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
  (*sz)[0] = 4; (*sz)[1] = 4; (*sz)[2] = 4;
  Tucker::SizeArray* nprocsPerDim =
      Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*nprocsPerDim)[0] = 2; (*nprocsPerDim)[1] = 2; (*nprocsPerDim)[2] = 2;
  TuckerMPI::Distribution* dist =
        Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*sz,*nprocsPerDim);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(sz);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(nprocsPerDim);

  // Create a tensor
  TuckerMPI::Tensor<scalar_t>* tensor =
      Tucker::MemoryManager::safe_new<TuckerMPI::Tensor<scalar_t>>(dist);

  // Read the entries from a file
  TuckerMPI::importTensorBinary(filename.c_str(),tensor);

  // Compute the gram matrix in dimension 0
  const Tucker::Matrix<scalar_t>* mat = TuckerMPI::newGram(tensor,0);

//  mat->print();

  scalar_t trueData[36] = {19840, 20320, 20800, 21280,
                         20320, 20816, 21312, 21808,
                         20800, 21312, 21824, 22336,
                         21280, 21808, 22336, 22864};

  bool equal = checkUTEqual(mat->data(), trueData, 4);
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
//        std::cerr << "arr[" << ind << "] (" << r << "," << c
//            << ") are not equal(" << arr1[ind] << " "
//            << arr2[ind] << ")\n";
        return false;
      }
    }
  }
  return true;
}
