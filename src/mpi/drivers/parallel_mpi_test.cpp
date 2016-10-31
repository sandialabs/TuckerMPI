/*
 * mpi_test.cpp
 *
 *  Created on: Jun 27, 2016
 *      Author: amklinv
 */

#include<mpi.h>
#include<memory>
#include "TuckerMPI.hpp"

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc,&argv);
  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD,&globalRank);

  // Create a processor grid
  int ndims = 2;
  Tucker::SizeArray nprocsPerDim(ndims);
  nprocsPerDim[0] = 2;
  nprocsPerDim[1] = 2;

  // Set the dimensions
  Tucker::SizeArray dims(ndims);
  dims[0] = 6;
  dims[1] = 6;

  // Create a distribution
  TuckerMPI::Distribution dist(dims,nprocsPerDim);

  // Create a tensor
  TuckerMPI::Tensor tensor(&dist);

  // Read the tensor from a file
  TuckerMPI::importTensorBinary("../../../../input_data/boringFile.mpi",&tensor);

  // Write the tensor to a file
  TuckerMPI::exportTensorBinary("../../../../output_data/boringFile.mpi",&tensor);

  // Terminate MPI
  MPI_Finalize();
  return 0;
}


