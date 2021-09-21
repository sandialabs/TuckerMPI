/*
 * TuckerMPI_converter.cpp
 *
 *  Created on: Sep 1, 2016
 *      Author: amklinv
 */

#include "TuckerMPI.hpp"
#include "Tucker.hpp"

int main(int argc, char* argv[])
{

// specify precision
#ifdef TEST_SINGLE
  typedef float scalar_t; 
#else
  typedef double scalar_t;
#endif
  
  MPI_Init(&argc, &argv);

  Tucker::Tensor<scalar_t>* t = Tucker::importTensor<scalar_t>(argv[1]);
  TuckerMPI::exportTensorBinary(argv[2],t);

  MPI_Finalize();
}


