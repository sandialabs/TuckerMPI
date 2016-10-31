/*
 * gram_driver.cpp
 *
 *  Created on: Oct 31, 2016
 *      Author: Alicia Klinvex (amklinv@sandia.gov)
 *
 *  Used to generate results for Tammy Kolda's invited
 *  talk at SC 2016.  Times a single Gram matrix computation
 *  and writes the runtimes to a file, which can be read
 *  into MATLAB and plotted.
 */

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
  ////////////////////
  // Initialize MPI //
  ////////////////////
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  ////////////////////////////////////
  // Get the name of the input file //
  ////////////////////////////////////
  std::string paramfn = Tucker::parseString(argc,
      (const char**)argv, "--parameter-file", "paramfile.txt");

  //////////////////////////
  // Parse parameter file //
  //////////////////////////
  std::vector<std::string> fileAsString = Tucker::getFileAsStrings(paramfn);
  bool boolPrintOptions                 = Tucker::stringParse<bool>(fileAsString, "Print options", false);
  Tucker::SizeArray* I_dims             = Tucker::stringParseSizeArray(fileAsString, "Global dims");
  Tucker::SizeArray* proc_grid_dims     = Tucker::stringParseSizeArray(fileAsString, "Grid dims");

  int nd = I_dims->size();
  bool boolUseOldGram                   = Tucker::stringParse<bool>(fileAsString, "Use old Gram", true);

  ///////////////////
  // Print options //
  ///////////////////
  if (rank == 0 && boolPrintOptions) {
    std::cout << "Global dims = " << *I_dims << std::endl;
    std::cout << "Grid dims = " << *proc_grid_dims << std::endl;
    std::cout << "Use old Gram = " << (boolUseOldGram ? "true" : "false") << std::endl;
    std::cout << std::endl;
  }

  /////////////////////////
  // Check for bad input //
  /////////////////////////
  if (proc_grid_dims->prod() != nprocs){
    if (rank==0) {
      std::cerr << "Processor grid dimensions do not multiply to nprocs" << std::endl;
      std::cout << "Processor grid dimensions: " << *proc_grid_dims << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (nd != proc_grid_dims->size()) {
    if (rank == 0) {
      std::cerr << "Error: The size of global dimension array (" << nd;
      std::cerr << ") must be equal to the size of the processor grid ("
          << proc_grid_dims->size() << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ////////////////////////////////
  // Set up distribution object //
  ////////////////////////////////
  TuckerMPI::Distribution dist(*I_dims, *proc_grid_dims);

  //////////////////////////
  // Create random tensor //
  //////////////////////////
  TuckerMPI::Tensor X(&dist);
  X.rand();

  /////////////////////////////
  // Compute the Gram matrix //
  /////////////////////////////
  Tucker::Timer gram_timer[nd];
  Tucker::Timer mult_timer[nd];
  Tucker::Timer shift_timer[nd];
  Tucker::Timer allreduce_timer[nd];
  Tucker::Timer allgather_timer[nd];
  Tucker::Timer pack_timer[nd];
  Tucker::Timer alltoall_timer[nd];
  Tucker::Timer unpack_timer[nd];
  for(int mode = 0; mode < nd; mode++) {
    MPI_Barrier(MPI_COMM_WORLD);
    gram_timer[mode].start();
    if(boolUseOldGram) {
      Tucker::Matrix* gram = TuckerMPI::oldGram(&X, mode,
          &mult_timer[mode], &shift_timer[mode],
          &allreduce_timer[mode], &allgather_timer[mode]);
      delete gram;
    }
    else {
      Tucker::Matrix* gram = TuckerMPI::newGram(&X, mode,
          &mult_timer[mode], &pack_timer[mode],
          &alltoall_timer[mode], &unpack_timer[mode],
          &allreduce_timer[mode]);
      delete gram;
    }
    gram_timer[mode].stop();
  }

  ///////////////////////////
  // Pack the runtime data //
  ///////////////////////////
  const int NTIMERS = 8;
  double timeArray[nd*NTIMERS];
  for(int mode=0; mode<nd; mode++) {
    timeArray[mode*NTIMERS]   = gram_timer[mode].duration();
    timeArray[mode*NTIMERS+1] = mult_timer[mode].duration();
    timeArray[mode*NTIMERS+2] = shift_timer[mode].duration();
    timeArray[mode*NTIMERS+3] = allreduce_timer[mode].duration();
    timeArray[mode*NTIMERS+4] = allgather_timer[mode].duration();
    timeArray[mode*NTIMERS+5] = pack_timer[mode].duration();
    timeArray[mode*NTIMERS+6] = alltoall_timer[mode].duration();
    timeArray[mode*NTIMERS+7] = unpack_timer[mode].duration();
  }

  //////////////////////////////////////
  // Gather the runtimes to process 0 //
  //////////////////////////////////////
  double* gathered_data;
  if(rank == 0) {
    gathered_data = Tucker::safe_new<double>(nd*NTIMERS*nprocs);
  }
  MPI_Gather(timeArray, nd*NTIMERS, MPI_DOUBLE, gathered_data,
      nd*NTIMERS, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  /////////////////////////////
  // Send the data to a file //
  /////////////////////////////
  std::ofstream os("gram_runtimes.csv");

  for(int mode=0; mode<nd; mode++) {
    os << "Gram(" << mode << "),Gram local multiply(" << mode << "),Gram shift("
        << mode << "),Gram all-reduce(" << mode << "),Gram all-gather("
        << mode << "),Gram packing(" << mode << "),Gram all-to-all(" << mode
        << "),Gram unpacking(" << mode << ")";
    if(mode < nd-1) os << ",";
  }

  // For each MPI process
  for(int r=0; r<nprocs; r++) {
    // For each timer belonging to that process
    for(int t=0; t<NTIMERS*nd; t++) {
      os << gathered_data[r*(NTIMERS*nd+1)+t];
      if(t < NTIMERS*nd-1) os << ",";
    }
    os << std::endl;
  }

  os.close();

  /////////////////
  // Free memory //
  /////////////////
  if(rank == 0) delete[] gathered_data;
  delete I_dims;
  delete proc_grid_dims;

  //////////////////
  // Finalize MPI //
  //////////////////
  MPI_Finalize();
}
