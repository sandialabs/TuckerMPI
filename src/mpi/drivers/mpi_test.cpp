/*
 * mpi_test.cpp
 *
 *  Created on: Jun 27, 2016
 *      Author: amklinv
 */

#include<mpi.h>
#include "Tucker.hpp"
#include "TuckerMPI.hpp"

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc,&argv);

  // Create the tensor of appropriate size
  Tucker::SizeArray sz(4);
  sz[0] = 500;
  sz[1] = 500;
  sz[2] = 500;
  sz[3] = 11;

  Tucker::Tensor t(sz);

  // Read the tensor from a file
  TuckerMPI::importTensorBinary("../input_data/stat_planar.1.5000E-03.field.mpi",&t);

  // Normalize each slice (each variable) based on min and max entries
  int mode = 1;
  Tucker::normalizeTensorMinMax(&t,mode);

  // Create the reduced size
  Tucker::SizeArray newSize(4);
  newSize[0] = 72;
  newSize[1] = 113;
  newSize[2] = 117;
  newSize[3] = 3;

  // Compute the Tucker decomposition
  std::cout << "Computing Tucker decomposition\n";
  const Tucker::TuckerTensor* ttensor = Tucker::STHOSVD(&t,&newSize,true);
  std::cout << "Computed Tucker decomposition\n";

  // Send core to binary file
  TuckerMPI::exportTensorBinary("../output_data/stat_planar.mpi",ttensor->G);

//  // Print the eigenvalues
//  for(int i=0; i<ttensor->N; i++) {
//    for(int j=0; j<t.size(i); j++)
//      std::cout << "mode " << i << " eigenvalue " << j << " = "
//      << ttensor->eigenvalues[i][j] << std::endl;
//  }

  // Free some memory
  delete ttensor;

  // Terminate MPI
  MPI_Finalize();
  return 0;
}


