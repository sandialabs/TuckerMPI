#include "TuckerMPI.hpp"
#include "Tucker.hpp"
#include "Tucker_IO_Util.hpp"
#include "TuckerMPI_IO_Util.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "assert.h"

int main(int argc, char* argv[])
{
  #ifdef DRIVER_SINGLE
    using scalar_t = float;
  #else
    using scalar_t = double;
  #endif  // specify precision

  //
  // Initialize MPI
  //
  MPI_Init(&argc, &argv);

  //
  // Get the rank of this MPI process
  // Only rank 0 will print to stdout
  //
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  //
  // Get the name of the input file
  //
  std::string paramfn = Tucker::parseString(argc, (const char**)argv,
      "--parameter-file", "paramfile.txt");

  std::vector<std::string> fileAsString = Tucker::getFileAsStrings(paramfn);
  std::string in_fns_file               = Tucker::stringParse<std::string>(fileAsString, "Input file list", "raw.txt");
  Tucker::SizeArray* I_dims             = Tucker::stringParseSizeArray(fileAsString, "Global dims");
  Tucker::SizeArray* proc_grid_dims     = Tucker::stringParseSizeArray(fileAsString, "Grid dims");

    TuckerMPI::Distribution* dist =
      Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*I_dims, *proc_grid_dims);

  ///////////////////////////
  // Read full tensor data //
  ///////////////////////////
  Tucker::Timer readTimer;
  readTimer.start();
  TuckerMPI::Tensor<scalar_t> X(dist);
  std::cout << in_fns_file << std::endl;
  TuckerMPI::readTensorBinary(in_fns_file,X);
  std::cout << "tensor read" << std::endl;
  Tucker::Tensor<scalar_t>* t = X.getLocalTensor();
  std::cout << "local tensor got" << std::endl;
  // t->print()
  // for(size_t i=0; i<50; i++) {
  //   std::cout << "data[" << i << "] = " << t->data()[i] << std::endl;
  // }
  std::cout << t->norm2();
  MPI_Finalize();
}