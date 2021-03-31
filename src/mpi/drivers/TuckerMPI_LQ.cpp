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
  typedef double scalar_t;
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

  //
  // Parse parameter file
  // Put's each line as a string into a vector ignoring empty lines
  // and comments
  //
  std::vector<std::string> fileAsString = Tucker::getFileAsStrings(paramfn);
  bool boolPrintOptions                 = Tucker::stringParse<bool>(fileAsString, "Print options", false);
  bool boolUseButterflyTSQR             = Tucker::stringParse<bool>(fileAsString, "Use butterfly TSQR", false);
  int mode                              = Tucker::stringParse<int>(fileAsString, "Mode", 0);
  std::string timing_file               = Tucker::stringParse<std::string>(fileAsString, "Timing file", "runtime.csv");
  Tucker::SizeArray* I_dims             = Tucker::stringParseSizeArray(fileAsString, "Global dims");
  Tucker::SizeArray* proc_grid_dims     = Tucker::stringParseSizeArray(fileAsString, "Grid dims");

  int ndims = I_dims->size();;
  //
  // Print options
  //
  if (rank == 0 && boolPrintOptions) {
    std::cout << "The global dimensions of the tensor to be scaled or compressed\n";
    std::cout << "- Global dims = " << *I_dims << std::endl << std::endl;

    std::cout << "The global dimensions of the processor grid\n";
    std::cout << "- Grid dims = " << *proc_grid_dims << std::endl << std::endl;

    std::cout << "Mode for Gram computation\n";
    std::cout << "- Mode = " << mode << std::endl << std::endl;

    std::cout << "Name of the CSV file holding the timing results\n";
    std::cout << "- Timing file = " << timing_file << std::endl << std::endl;

    std::cout << "- Use butterfly TSQR = " << (boolUseButterflyTSQR ? "true" : "false") << std::endl << std::endl;

    std::cout << "If true, print the parameters\n";
    std::cout << "- Print options = " << (boolPrintOptions ? "true" : "false") << std::endl << std::endl;

    std::cout << std::endl;
  }

  ///////////////////////
  // Check array sizes //
  ///////////////////////

  // Does |grid| == nprocs?
  if ((int)proc_grid_dims->prod() != nprocs){
    if (rank==0) {
      std::cerr << "Processor grid dimensions do not multiply to nprocs" << std::endl;
      std::cout << "Processor grid dimensions: " << *proc_grid_dims << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (ndims != proc_grid_dims->size()) {
    if (rank == 0) {
      std::cerr << "Error: The size of global dimension array (" << ndims;
      std::cerr << ") must be equal to the size of the processor grid ("
          << proc_grid_dims->size() << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  TuckerMPI::Distribution* dist =
    Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*I_dims,*proc_grid_dims);

  TuckerMPI::Tensor<scalar_t> X(dist);
  X.rand();

  if(rank == 0) {
    size_t local_nnz = X.getLocalNumEntries();
    size_t global_nnz = X.getGlobalNumEntries();
    std::cout << "Local input tensor size: " << X.getLocalSize() << ", or ";
    Tucker::printBytes(local_nnz*sizeof(scalar_t));
    std::cout << "Global input tensor size: " << X.getGlobalSize() << ", or ";
    Tucker::printBytes(global_nnz*sizeof(scalar_t));
  }

  Tucker::Timer total_timer;
  Tucker::Timer redistribute_timer;
  Tucker::Timer local_qr_timer;
  Tucker::Timer tsqr_timer;
  Tucker::Timer bcast_timer;

  MPI_Barrier(MPI_COMM_WORLD);
  total_timer.start();
  Tucker::Matrix<scalar_t>* L0 = TuckerMPI::LQ<scalar_t>(&X, 0, boolUseButterflyTSQR, &tsqr_timer, &local_qr_timer, &redistribute_timer, &bcast_timer);
  total_timer.stop();
  if(rank == 0){
    std::cout << "total time used: " << total_timer.duration()
      << " \n redistribute time: " << redistribute_timer.duration()
      << "\n local qr time: " << local_qr_timer.duration() 
      << "\n tsqr time: " << tsqr_timer.duration()
      << "\n bcast time: " << bcast_timer.duration() << std::endl;
  }

  ////////////////////////////////////////
  // Determine the maximum memory usage //
  ////////////////////////////////////////
  size_t max_mem = Tucker::MemoryManager::maxMemUsage;

  if(rank == 0) {
    Tucker::MemoryManager::printMaxMemUsage();
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Gather the resource usage information to process 0 and write it to a file //
  ///////////////////////////////////////////////////////////////////////////////
  double total_time = total_timer.duration();
  double redistribute_time = redistribute_timer.duration();
  double local_qr_time = local_qr_timer.duration();
  double tsqr_time = tsqr_timer.duration();
  double bcast_time = bcast_timer.duration();


  double *total_times, *redistribute_times, *local_qr_times, *tsqr_times, *bcast_times;
  size_t* max_mems;

  if(rank == 0) {
    total_times = Tucker::MemoryManager::safe_new_array<double>(nprocs);
    redistribute_times = Tucker::MemoryManager::safe_new_array<double>(nprocs);
    local_qr_times = Tucker::MemoryManager::safe_new_array<double>(nprocs);
    tsqr_times = Tucker::MemoryManager::safe_new_array<double>(nprocs);
    bcast_times = Tucker::MemoryManager::safe_new_array<double>(nprocs);
    max_mems = Tucker::MemoryManager::safe_new_array<size_t>(nprocs);
  }

  MPI_Gather(&total_time, 1, MPI_DOUBLE, total_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&redistribute_time, 1, MPI_DOUBLE, redistribute_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&local_qr_time, 1, MPI_DOUBLE, local_qr_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&tsqr_time, 1, MPI_DOUBLE, tsqr_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&bcast_time, 1, MPI_DOUBLE, bcast_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gather(&max_mem, sizeof(size_t), MPI_CHAR, max_mems, sizeof(size_t), MPI_CHAR, 0, MPI_COMM_WORLD);

if(rank == 0) {
    // Send the data to a file
    std::ofstream os(timing_file);

    // Create the header row
    os << "redistribution,localQR,TSQR,bcast,total,mem\n";

    // For each MPI process
    for(int r=0; r<nprocs; r++) {
      os << redistribute_times[r] << "," << local_qr_times[r] << "," << tsqr_times[r] << "," << bcast_times[r] << ","
      << total_times[r] << "," << max_mems[r] << std::endl;
    }

    os.close();

    Tucker::MemoryManager::safe_delete_array<double>(redistribute_times,nprocs);
    Tucker::MemoryManager::safe_delete_array<double>(local_qr_times,nprocs);
    Tucker::MemoryManager::safe_delete_array<double>(tsqr_times,nprocs);
    Tucker::MemoryManager::safe_delete_array<double>(bcast_times,nprocs);
    Tucker::MemoryManager::safe_delete_array<double>(total_times,nprocs);
    Tucker::MemoryManager::safe_delete_array<size_t>(max_mems,nprocs);
  }

  /////////////////
  // Free memory //
  /////////////////
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(I_dims);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(proc_grid_dims);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
