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
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Set the initial size
  Tucker::SizeArray size(3);
  size[0] = 672;
  size[1] = 672;
  size[2] = 33;

  // Read tensor from file
  std::string filename = "../../../input_data/hcci.0.0000E+00.field.mpi";
  Tucker::Tensor t(size);
  TuckerMPI::importTensorBinary(filename.c_str(), &t);

  // Normalize the tensor
  Tucker::normalizeTensorMinMax(&t, 2);

  // Compute the factorization
  const struct Tucker::TuckerTensor* factorization = Tucker::STHOSVD(&t, 1e-3, true);

  const Tucker::SizeArray& sizes = t.size();
  for(int i=0; i<sizes.size(); i++) {
    for(int j=0; j<sizes[i]; j++) {
      std::cout << "eigenvalues[" << i << "," << j << "] = " << factorization->eigenvalues[i][j] << std::endl;
    }
  }

//  for(int i=0; i<sizes.size(); i++) {
//    factorization->U[i]->print();
//  }

  // Send the core to a file
  filename = "../../../output_data/hcci.mpi";
  TuckerMPI::exportTensorBinary(filename.c_str(), factorization->G);

  // Free some memory
  Tucker::MemoryManager::safe_delete<const Tucker::TuckerTensor>(factorization);

  // Finalize MPI
  MPI_Finalize();
}
