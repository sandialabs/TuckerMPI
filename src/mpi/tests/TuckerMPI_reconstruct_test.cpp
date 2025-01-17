/*
 * reconstruct_test.cpp
 *
 *  Created on: Jul 25, 2016
 *      Author: amklinv
 */

#include<cstdlib>
#include "TuckerMPI.hpp"

int main(int argc, char* argv[])
{

// specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t; 
  const char* filename = "input_files/tensor24_single.mpi";
#else
  typedef double scalar_t;
  const char* filename = "input_files/tensor24.mpi";
#endif

  // Initialize MPI
  MPI_Init(&argc,&argv);

  // Create the SizeArray
  Tucker::SizeArray* size =
      Tucker::MemoryManager::safe_new<Tucker::SizeArray>(3);
  (*size)[0] = 2;
  (*size)[1] = 3;
  (*size)[2] = 5;

  // Create the MPI processor grid
  int ndims = 3;
  Tucker::SizeArray* nprocsPerDim =
      Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*nprocsPerDim)[0] = 1; (*nprocsPerDim)[1] = 2; (*nprocsPerDim)[2] = 3;

  // Create the distribution object
  TuckerMPI::Distribution* dist =
        Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*size,*nprocsPerDim);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(size);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(nprocsPerDim);

  // Create the tensor
  TuckerMPI::Tensor<scalar_t>* tensor =
      Tucker::MemoryManager::safe_new<TuckerMPI::Tensor<scalar_t>>(dist);

  // Read the entries from a file
  TuckerMPI::importTensorBinary(filename,tensor);

  // Compute approximation to 1e-3 (loose enough for single precision)
  const struct TuckerMPI::TuckerTensor<scalar_t>* factorization =
      TuckerMPI::STHOSVD(tensor, (scalar_t) 1e-3);

  // Reconstruct the original tensor
  TuckerMPI::Tensor<scalar_t>* temp = TuckerMPI::ttm(factorization->G,0,
      factorization->U[0]);
  TuckerMPI::Tensor<scalar_t>* temp2 = TuckerMPI::ttm(temp,1,
        factorization->U[1]);
  TuckerMPI::Tensor<scalar_t>* temp3 = TuckerMPI::ttm(temp2,2,
        factorization->U[2]);

  // Set elementwise tolerance an order of magnitude larger than normwise error
  bool eq = isApproxEqual(temp3, tensor, (scalar_t) 1e-2);
  if(!eq) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  // Free some memory
  Tucker::MemoryManager::safe_delete(factorization);
  Tucker::MemoryManager::safe_delete(temp);
  Tucker::MemoryManager::safe_delete(temp2);
  Tucker::MemoryManager::safe_delete(temp3);
  Tucker::MemoryManager::safe_delete(tensor);

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  MPI_Finalize();

  return EXIT_SUCCESS;
}



