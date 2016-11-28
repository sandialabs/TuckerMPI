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
  Tucker::SizeArray* subs_begin         = Tucker::stringParseSizeArray(fileAsString, "Beginning subscripts");
  Tucker::SizeArray* subs_end           = Tucker::stringParseSizeArray(fileAsString, "Ending subscripts");

  std::string sthosvd_dir               = Tucker::stringParse<std::string>(fileAsString, "STHOSVD directory", "compressed");
  std::string sthosvd_fn                = Tucker::stringParse<std::string>(fileAsString, "STHOSVD file prefix", "sthosvd");
  std::string out_fns_file              = Tucker::stringParse<std::string>(fileAsString, "Output file list", "rec.txt");

  //
  // Print options
  //
  if (rank == 0 && boolPrintOptions) {
    std::cout << "STHOSVD directory = " << sthosvd_dir << std::endl;
    std::cout << "STHOSVD file prefix = " << sthosvd_fn << std::endl;
    std::cout << "Output file list = " << out_fns_file << std::endl;
    std::cout << "Global dims = " << *I_dims << std::endl;
    std::cout << "Grid dims = " << *proc_grid_dims << std::endl;
    std::cout << "Beginning subscripts = " << *subs_begin << std::endl;
    std::cout << "Ending subscripts = " << *subs_end << std::endl;
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

  ////////////////////////////////////////////////////
  // Reconstruct the requested pieces of the tensor //
  ////////////////////////////////////////////////////
  TuckerMPI::Tensor* result = fact.G;
  for(int mode=0; mode<nd; mode++)
  {
    // Grab the requested rows of the factor matrix
    int start_subs = (*subs_begin)[mode];
    int end_subs = (*subs_end)[mode];
    Tucker::Matrix* factMat =
        fact.U[mode]->getSubmatrix(start_subs, end_subs);

    // Perform the TTM
    TuckerMPI::Tensor* temp = TuckerMPI::ttm(result,mode,factMat);

    delete factMat;
    if(mode > 0)
      delete result;
    result = temp;
  }

  ////////////////////////////////////////////
  // Write the reconstructed tensor to disk //
  ////////////////////////////////////////////
  result->print();
  TuckerMPI::writeTensorBinary(out_fns_file, *result);

  /////////////////
  // Free memory //
  /////////////////
  delete I_dims;
  delete proc_grid_dims;
  delete subs_begin;
  delete subs_end;
  delete result;

  //////////////////
  // Finalize MPI //
  //////////////////
  MPI_Finalize();
}
