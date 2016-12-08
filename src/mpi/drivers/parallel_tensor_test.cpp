/*
 * parallel_tensor_test.cpp
 *
 *  Created on: Jul 8, 2016
 *      Author: amklinv
 */

#include <fstream>
#include <sstream>
#include "mpi.h"
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
//  nprocsPerDim[1] = 3;
//  nprocsPerDim[2] = 4;
  nprocsPerDim[1] = 2;

  // Set the dimensions
  Tucker::SizeArray dims(ndims);
//  dims[0] = 5;
//  dims[1] = 7;
//  dims[2] = 11;
  dims[0] = 3;
  dims[1] = 3;

  // Create a distribution
  TuckerMPI::Distribution dist(dims,nprocsPerDim);
  const TuckerMPI::ProcessorGrid* pg = dist.getProcessorGrid();

  // Create filename
  std::stringstream ss;
  ss << "pg_" << globalRank << ".txt";
  std::ofstream ofs;
  ofs.open(ss.str().c_str());

  // Get information about the processor grid
  for(int i=0; i<ndims; i++) {
    const MPI_Comm& comm = pg->getColComm(i,false);

    int localRank, localNumProcs;
    MPI_Comm_rank(comm,&localRank);
    MPI_Comm_size(comm,&localNumProcs);
    ofs << "Dimension " << i << ": I am rank " << localRank
        << " of " << localNumProcs << std::endl;
  }

//  const Tucker::SizeArray& localSizes = dist->getLocalDims();
//  for(int i=0; i<localSizes.size(); i++) {
//    ofs << "localSizes[" << i << "] = " << localSizes[i] << std::endl;
//  }
//
//  const Tucker::SizeArray& globalSizes = dist->getGlobalDims();
//  for(int i=0; i<globalSizes.size(); i++) {
//    ofs << "globalSizes[" << i << "] = " << globalSizes[i] << std::endl;
//  }

  // What is my location in the grid?
  int* coords = Tucker::safe_new_array<int>(ndims);
  pg->getCoordinates(coords);
  ofs << "Coordinates: ";
  for(int i=0; i<ndims; i++) {
    ofs << coords[i] << " ";
  }
  ofs << std::endl;

  // Who else do I interact with?
  for(int d=0; d<ndims; d++)
  {
    ofs << "In dimension " << d << ", I interact with processes ";
    int origVal = coords[d];
    for(int i=0; i<nprocsPerDim[d]; i++)
    {
      coords[d] = i;
      int neighborRank = pg->getRank(coords);
      ofs << neighborRank << " ";
    }
    ofs << std::endl;
    coords[d] = origVal;

//    ofs << "This is the size of each subtensor I interact with\n";
//    const int* ttmSizes = dist.getTTMSizes(d);
//    for(int i=0; i<nprocsPerDim[d]; i++) {
//      ofs << ttmSizes[i] << " ";
//    }
//    ofs << std::endl;
  }

  delete[] coords;

  // Create a tensor
  TuckerMPI::Tensor tensor(&dist);
  double* tensorData = tensor.getLocalTensor()->data();
  for(size_t i=0; i<tensor.getLocalNumEntries(); i++) {
    tensorData[i] = (double)i+1;
  }

  // Create a matrix
  Tucker::Matrix matrix(3,2);
  double* matrixData = matrix.data();
  for(size_t i=0; i<matrix.getNumElements(); i++) {
    matrixData[i] = (double)i+1;
  }

  // Perform a TTM
  TuckerMPI::Tensor* result = TuckerMPI::ttm(&tensor,0,&matrix,true);

  std::cout << "result:\n";
  result->print();

  delete result;

  ofs.close();

  MPI_Finalize();
  return 0;
}
