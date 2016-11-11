/*
 * driver.cpp
 *
 *  Created on: Jun 3, 2016
 *      Author: Alicia Klinvex (amklinv@sandia.gov)
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

  //
  // Parse parameter file
  // Put's each line as a string into a vector ignoring empty lines
  // and comments
  //
  std::vector<std::string> fileAsString = Tucker::getFileAsStrings(paramfn);
  bool boolPrintOptions                 = Tucker::stringParse<bool>(fileAsString, "Print options", false);

  Tucker::SizeArray* proc_grid_dims     = Tucker::stringParseSizeArray(fileAsString, "Grid dims");
  Tucker::SizeArray* I_dims             = Tucker::stringParseSizeArray(fileAsString, "Global dims");

  std::string sthosvd_dir               = Tucker::stringParse<std::string>(fileAsString, "STHOSVD directory", "compressed");
  std::string sthosvd_fn                = Tucker::stringParse<std::string>(fileAsString, "STHOSVD file prefix", "sthosvd");
  std::string in_fns_file               = Tucker::stringParse<std::string>(fileAsString, "Input file list", "raw.txt");

  //
  // Print options
  //
  if (rank == 0 && boolPrintOptions) {
    std::cout << "STHOSVD directory = " << sthosvd_dir << std::endl;
    std::cout << "STHOSVD file prefix = " << sthosvd_fn << std::endl;
    std::cout << "Input file list = " << in_fns_file << std::endl;
    std::cout << "Global dims = " << *I_dims << std::endl;
    std::cout << "Grid dims = " << *proc_grid_dims << std::endl;
    std::cout << std::endl;
  }

  ///////////////////////
  // Check array sizes //
  ///////////////////////
  int nd = I_dims->size();

  // Does |grid| == nprocs?
  if ((int)proc_grid_dims->prod() != nprocs){
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

  if((*proc_grid_dims)[nd-1] != 1) {
    if(rank == 0) {
      std::cerr << "Error: The number of processes in the time dimension must be 1.\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ////////////////////////////////////
  // Read the core size from a file //
  ////////////////////////////////////
  Tucker::SizeArray coreSize(nd);
  if(rank == 0)
  {
    std::string dimFilename = sthosvd_dir + "/" + sthosvd_fn +
        "_ranks.txt";
    std::ifstream ifs(dimFilename);

    for(int mode=0; mode<nd; mode++) {
      ifs >> coreSize[mode];
    }
    ifs.close();
  }
  MPI_Bcast(coreSize.data(),nd,MPI_INT,0,MPI_COMM_WORLD);

  /////////////////////////////////
  // Set up factorization object //
  /////////////////////////////////
  TuckerMPI::TuckerTensor fact(nd);

  /////////////////////////////////////////////
  // Set up distribution object for the core //
  /////////////////////////////////////////////
  TuckerMPI::Distribution dist(coreSize, *proc_grid_dims);

  ///////////////////////////
  // Read core tensor data //
  ///////////////////////////
  std::string coreFilename = sthosvd_dir + "/" + sthosvd_fn +
            "_core.mpi";
  fact.G = new TuckerMPI::Tensor(&dist);
  TuckerMPI::importTensorBinary(coreFilename.c_str(),fact.G);

  //////////////////////////
  // Read factor matrices //
  //////////////////////////
  for(int mode=0; mode<nd; mode++)
  {
    std::ostringstream ss;
    ss << sthosvd_dir << "/" << sthosvd_fn << "_mat_" << mode << ".mpi";

    fact.U[mode] = new Tucker::Matrix((*I_dims)[mode],coreSize[mode]);
    TuckerMPI::importTensorBinary(ss.str().c_str(), fact.U[mode]);
  }

  //////////////////////////
  // For each timestep... //
  //////////////////////////
  int nsteps = (*I_dims)[nd-1];
  (*I_dims)[nd-1] = 1;
  TuckerMPI::Distribution ssdist(*I_dims, *proc_grid_dims);
  TuckerMPI::Tensor ssTensor(&ssdist);
  std::ifstream ifs(in_fns_file);
  double localError = 0;
  double localMaxError = 0;
  double localNorm = 0;
  for(int timestep=0; timestep < nsteps; timestep++)
  {
    std::string stepFilename;
    ifs >> stepFilename;

    // Read a timestep from a file
    TuckerMPI::importTensorBinary(stepFilename.c_str(), &ssTensor);

    // Reconstruct that timestep
    const TuckerMPI::Tensor* recTen = reconstructSingleSlice(&fact,
        nd-1, timestep);

    // Compute the error in this slice
    size_t nnz = recTen->getLocalNumEntries();
    assert(ssTensor.getLocalNumEntries() == nnz);
    const double* data1 = ssTensor.getLocalTensor()->data();
    const double* data2 = recTen->getLocalTensor()->data();
    for(int i=0; i<nnz; i++)
    {
      double err = std::abs(data1[i] - data2[i]);
      if(err > localMaxError)
        localMaxError = err;
      localError += (err*err);
      localNorm += (data1[i]*data1[i]);
    }

    delete recTen;
  }

  double maxError, totalError, totalNorm;
  MPI_Reduce(&localMaxError,&maxError,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(&localError,&totalError,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&localNorm,&totalNorm,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);

  if(rank == 0)
  {
    std::cout << "Norm of original data: " << std::sqrt(totalNorm) << std::endl;
    std::cout << "Absolute error: " << std::sqrt(totalError) << std::endl;
    std::cout << "Largest difference in elements: " << maxError << std::endl;
  }

  //
  // Free memory
  //
  delete I_dims;
  delete proc_grid_dims;

  // Finalize MPI
  MPI_Finalize();
}
