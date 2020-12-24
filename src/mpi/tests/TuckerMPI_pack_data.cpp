/*
 * pack_data.cpp
 *
 * This tests the mode-0 packData function
 *
 *  Created on: Jul 12, 2016
 *      Author: amklinv
 */

#include <cstdlib>
#include "mpi.h"
#include "TuckerMPI.hpp"

template <class scalar_t>
bool checkArrayEqual(scalar_t* arr1, scalar_t* arr2, int numEl);

int main(int argc, char* argv[])
{
  typedef double scalar_t; // specify precision

  // Initialize MPI
  MPI_Init(&argc,&argv);

  // Create a 4x2x2 tensor
  Tucker::SizeArray* sa =
      Tucker::MemoryManager::safe_new<Tucker::SizeArray>(3);
  (*sa)[0] = 4; (*sa)[1] = 2; (*sa)[2] = 2;
  Tucker::Tensor<scalar_t>* tensor =
      Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*sa);
  Tucker::MemoryManager::safe_delete(sa);

  // Fill it with the entries 0:15
  scalar_t* data = tensor->data();
  for(int i=0; i<16; i++)
    data[i] = i;

  // Create a map describing the distribution
  TuckerMPI::Map* map =
      Tucker::MemoryManager::safe_new<TuckerMPI::Map>(4,MPI_COMM_WORLD);

  // Pack the tensor
  packForTTM(tensor, 0, map);
  Tucker::MemoryManager::safe_delete(map);

  // Check the result
  scalar_t trueResult[16] = {0,1,4,5,8,9,12,13,2,3,6,7,10,11,14,15};
  bool equal = checkArrayEqual(tensor->data(),trueResult,16);
  if(!equal) {
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::safe_delete(tensor);

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  // Call MPI_Finalize
  MPI_Finalize();
  return EXIT_SUCCESS;
}

template <class scalar_t>
bool checkArrayEqual(scalar_t* arr1, scalar_t* arr2, int numEl)
{
  for(int i=0; i<numEl; i++) {
    if(arr1[i] != arr2[i]) {
      return false;
    }
  }
  return true;
}
