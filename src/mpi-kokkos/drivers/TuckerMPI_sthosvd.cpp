// #include "TuckerMPI.hpp"
// #include "Tucker.hpp"
// #include "TuckerMPI_IO_Util.hpp"
// #include <cmath>
// #include <iostream>
// #include <iomanip>
// #include <fstream>
// #include "assert.h"
#include "Tucker_IO_Util.hpp"
#include "TuckerMPI_Distribution.hpp"

#include "init_args.hpp"
#include <mpi.h>
#include <Kokkos_Core.hpp>

int main(int argc, char* argv[])
{
  #ifdef DRIVER_SINGLE
    using scalar_t = float;
  #else
    using scalar_t = double;
  #endif  // specify precision
  
  // Initialize MPI
  MPI_Init(&argc, &argv);

  // Initialize Kokkos
  Kokkos::initialize();
  {
    // Get the rank of this MPI process
    // Only rank 0 will print to stdout
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    // Get the name of the input file
    const std::string paramfn = Tucker::parseString(argc, (const char**)argv, "--parameter-file", "paramfile.txt");

    // Parse parameter file
    const std::vector<std::string> fileAsString = Tucker::getFileAsStrings(paramfn);
    InputArgs args = parse_input_file<scalar_t>(fileAsString);

    // Check options
    int checkArgs = check_args(args);
    if (rank == 0) { std::cout << "Argument checking: passed" << std::endl; }

    // Print options
    if (rank == 0) { print_args(args); }

    assert(args.boolAuto || args.R_dims->size() == args.nd);

    // Check array sizes
    chech_array_sizes(args, rank, nprocs);
    if (rank == 0) { std::cout << "Array sizes checking: passed" << std::endl; }

    // Set up processor grid
    if (rank == 0) { std::cout << "Creating process grid" << std::endl; }

    // Set up distribution object
    TuckerMPI::Distribution* dist =
      Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*args.I_dims, *args.proc_grid_dims);
    
    // Read full tensor data

    // Tucker::Timer readTimer;
    // readTimer.start();
    // TuckerMPI::Tensor<scalar_t> X(dist);
    // TuckerMPI::readTensorBinary(in_fns_file,X);
    // readTimer.stop();
    // ...
    // [[code]]

    // Compute statistics
    // [[code]]

    // Perform preprocessing
    // [[code]]

    // Perform STHOSVD
    if(args.boolSTHOSVD) {

      // [[code]]
    
    } // end if(boolSTHOSVD)

    // Free memory
    Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(args.I_dims);
    if(args.R_dims) Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(args.R_dims);
    Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(args.proc_grid_dims);
    
    if(rank == 0) {
      Tucker::MemoryManager::printMaxMemUsage();
    }
    
  }
  
  // Finalize Kokkos
  Kokkos::finalize();

  // Finalize MPI
  MPI_Finalize();
  return 0;
}


#if 0
  

  
  ///////////////////////////
  // Read full tensor data //
  ///////////////////////////
  Tucker::Timer readTimer;
  readTimer.start();
  TuckerMPI::Tensor<scalar_t> X(dist);
  TuckerMPI::readTensorBinary(in_fns_file,X);
  readTimer.stop();

  double localReadTime = readTimer.duration();
  double globalReadTime;

  MPI_Reduce(&localReadTime,&globalReadTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  if(rank == 0) {
    std::cout << "Time to read tensor: " << globalReadTime << " s\n";

    size_t local_nnz = X.getLocalNumEntries();
    size_t global_nnz = X.getGlobalNumEntries();
    std::cout << "Local input tensor size: " << X.getLocalSize() << ", or ";
    Tucker::printBytes(local_nnz*sizeof(scalar_t));
    std::cout << "Global input tensor size: " << X.getGlobalSize() << ", or ";
    Tucker::printBytes(global_nnz*sizeof(scalar_t));
  }

  // bunch of stuff missing

}
#endif
