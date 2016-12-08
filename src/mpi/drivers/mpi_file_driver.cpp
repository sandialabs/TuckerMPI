/*
 * driver.cpp
 *
 *  Created on: Jun 3, 2016
 *      Author: Alicia Klinvex (amklinv@sandia.gov)
 */

#include "TuckerMPI.hpp"
#include "Tucker.hpp"
#include <iostream>

int main(int argc, char* argv[])
{
  //
  // Initialize MPI
  //
  MPI_Init(&argc, &argv);

  //
  // Get the rank of this MPI process
  // Only rank 0 will print to stdout
  //
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  //
  // Set the initial size of the tensor
  // In this case, our tensor has four dimensions,
  // and it is of order 672 x 672 x 33 x 10.
  // The first two dimensions represent a 672 x 672
  // spatial grid, the third dimension represents 33
  // variables, and the fourth dimension represents 10
  // timesteps.
  //
  int ndims = 4;
  Tucker::SizeArray size(ndims);
  size[0] = 672;
  size[1] = 672;
  size[2] = 33;
  size[3] = 10;

  //
  // Create the MPI processor grid
  // In this case, we expect to have 10 MPI processes;
  // the first dimension is divided into 5 segments,
  // and the third is divided into 2 segments.
  //
  if(rank == 0) {
    std::cout << "Creating process grid\n";
  }
  Tucker::SizeArray numProcsPerDim(ndims);
  numProcsPerDim[0] = 5;
  numProcsPerDim[1] = 1;
  numProcsPerDim[2] = 2;
  numProcsPerDim[3] = 1;

  //
  // Create the distribution object
  // A distribution is defined by the tensor size and the
  // MPI process grid
  //
  TuckerMPI::Distribution dist(size,numProcsPerDim);

  //
  // Create the tensor, using the distribution object
  // This only allocates the memory; it does not fill in any values
  //
  TuckerMPI::Tensor t(&dist);

  //
  // Read the tensor values from a set of files
  // The string passed to the importTimeSeries function is the
  // name of a text file.  The number of lines in the text file
  // must be the same as the number of timesteps (in this case 10).
  // Each line is the name of a MPI_IO binary
  // file from which to read the data associated with a particular
  // timestep.
  //
  std::string filename = "filenames.txt";
  TuckerMPI::importTimeSeries(filename.c_str(), &t);

  //
  // Normalize the tensor
  // Dimension 2 represents the variables, so we normalize along that
  // dimension.
  //
  if(rank == 0) {
    std::cout << "Preprocessing data\n";
  }
  TuckerMPI::normalizeTensorStandardCentering(&t, 2);

  //
  // Compute the factorization using STHOSVD
  // We are using a tolerance of 1e-3, and we flip the sign
  // of the eigenvectors for consistency with the MATLAB
  // tensor toolbox.
  //
  const TuckerMPI::TuckerTensor* factorization =
      TuckerMPI::STHOSVD(&t, 1e-3, true);

  //
  // Output the size of the core tensor, and the eigenvalues
  // computed during STHOSVD
  //
  if(rank == 0) {
    std::cout << "Core size: " << factorization->G->getGlobalSize() << std::endl;

    const Tucker::SizeArray& sizes = t.getGlobalSize();
    for(int i=0; i<sizes.size(); i++) {
      for(int j=0; j<sizes[i]; j++) {
        std::cout << "eigenvalues[" << i << "," << j << "] = "
            << factorization->eigenvalues[i][j] << std::endl;
      }
    }
  }
//
//  for(int i=0; i<sizes.size(); i++) {
//    factorization->U[i]->print();
//  }
//
//  // Send the core to a file
//  filename = "../../../output_data/hcci_all.mpi";
//  TuckerMPI::exportTensorBinary(filename.c_str(), factorization->G);
//
//  // Free some memory
//  Tucker::safe_delete<TuckerMPI::TuckerTensor>(factorization);

  // Finalize MPI
  MPI_Finalize();
}
