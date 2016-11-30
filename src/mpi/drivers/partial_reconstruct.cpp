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
  Tucker::SizeArray* rec_order          = Tucker::stringParseSizeArray(fileAsString, "Reconstruction order");

  std::string sthosvd_dir               = Tucker::stringParse<std::string>(fileAsString, "STHOSVD directory", "compressed");
  std::string sthosvd_fn                = Tucker::stringParse<std::string>(fileAsString, "STHOSVD file prefix", "sthosvd");
  std::string out_fns_file              = Tucker::stringParse<std::string>(fileAsString, "Output file list", "rec.txt");

  /////////////////////////////////////////////////
  // Assert that none of the SizeArrays are null //
  /////////////////////////////////////////////////
  if(proc_grid_dims == NULL) {
    if(rank == 0)
      std::cerr << "Error: Grid dims is a required parameter\n";
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  if(I_dims == NULL) {
    if(rank == 0)
      std::cerr << "Error: Global dims is a required parameter\n";
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  if(subs_begin == NULL) {
    if(rank == 0)
      std::cerr << "Error: Beginning subscripts is a required parameter\n";
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  if(subs_end == NULL) {
    if(rank == 0)
      std::cerr << "Error: Ending subscripts is a required parameter\n";
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD,1);
  }

  ///////////////////
  // Print options //
  ///////////////////
  if (rank == 0 && boolPrintOptions) {
    std::cout << "STHOSVD directory = " << sthosvd_dir << std::endl;
    std::cout << "STHOSVD file prefix = " << sthosvd_fn << std::endl;
    std::cout << "Output file list = " << out_fns_file << std::endl;
    std::cout << "Global dims = " << *I_dims << std::endl;
    std::cout << "Grid dims = " << *proc_grid_dims << std::endl;
    std::cout << "Beginning subscripts = " << *subs_begin << std::endl;
    std::cout << "Ending subscripts = " << *subs_end << std::endl;
    if(rec_order != NULL) std::cout << "Reconstruction order = " << *rec_order << std::endl;
    std::cout << std::endl;
  }

  ///////////////////////
  // Check array sizes //
  ///////////////////////
  int nd = proc_grid_dims->size();

  // Does |grid| == nprocs?
  if ((int)proc_grid_dims->prod() != nprocs){
    if (rank==0) {
      std::cerr << "Processor grid dimensions do not multiply to nprocs" << std::endl;
      std::cout << "Processor grid dimensions: " << *proc_grid_dims << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (nd != I_dims->size()) {
    if (rank == 0) {
      std::cerr << "Error: The size of global dimension array (" << I_dims->size();
      std::cerr << ") must be equal to the size of the processor grid ("
          << nd << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (nd != subs_begin->size()) {
    if (rank == 0) {
      std::cerr << "Error: The size of the subs_begin array (" << subs_begin->size();
      std::cerr << ") must be equal to the size of the processor grid ("
          << nd << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (nd != subs_end->size()) {
    if (rank == 0) {
      std::cerr << "Error: The size of the subs_end array (" << subs_end->size();
      std::cerr << ") must be equal to the size of the processor grid ("
          << nd << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (rec_order != NULL && nd != rec_order->size()) {
    if (rank == 0) {
      std::cerr << "Error: The size of the rec_order array (" << rec_order->size();
      std::cerr << ") must be equal to the size of the processor grid ("
          << nd << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ////////////////////////////////////////////////////////
  // Make sure the subs begin and end arrays make sense //
  ////////////////////////////////////////////////////////
  for(int i=0; i<nd; i++) {
    if((*subs_begin)[i] < 0) {
      if(rank == 0) {
        std::cerr << "Error: subs_begin[" << i << "] = "
            << (*subs_begin)[i] << " < 0\n";
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if((*subs_begin)[i] > (*subs_end)[i]) {
      if(rank == 0) {
        std::cerr << "Error: subs_begin[" << i << "] = "
            << (*subs_begin)[i] << " > subs_end[" << i << "] = "
            << (*subs_end)[i] << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if((*subs_end)[i] >= (*I_dims)[i]) {
      if(rank == 0) {
        std::cerr << "Error: subs_end[" << i << "] = "
            << (*subs_end)[i] << " >= I_dims[" << i << "] = "
            << (*I_dims)[i] << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
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

    if(!ifs.is_open()) {
      if(rank == 0) {
        std::cerr << "Failed to open core size file: " << dimFilename
            << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for(int mode=0; mode<nd; mode++) {
      ifs >> coreSize[mode];
    }
    ifs.close();
  }
  MPI_Bcast(coreSize.data(),nd,MPI_INT,0,MPI_COMM_WORLD);

  //////////////////////////////////////////////
  // Make sure the core size data makes sense //
  //////////////////////////////////////////////
  for(int i=0; i<nd; i++) {
    if(coreSize[i] <= 0) {
      if(rank == 0) {
        std::cerr << "coreSize[" << i << "] = " << coreSize[i]
                  << " <= 0\n";
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD,1);
    }

    if(coreSize[i] > (*I_dims)[i]) {
      if(rank == 0) {
        std::cerr << "coreSize[" << i << "] = " << coreSize[i]
                  << " > I_dims[" << (*I_dims)[i] << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD,1);
    }
  }

  ////////////////////////////////////////////////////////////
  // Create the optimal reconstruction order if unspecified //
  ////////////////////////////////////////////////////////////
  if(rec_order == NULL) {
    // Create the SizeArray
    rec_order = new Tucker::SizeArray(nd);
    for(int i=0; i<nd; i++) {
      (*rec_order)[i] = i;
    }

    // Compute the ratios of reconstructed size to core size
    int* rec_size = Tucker::safe_new<int>(nd);
    double* ratios = Tucker::safe_new<double>(nd);
    for(int i=0; i<nd; i++) {
      rec_size[i] = 1 + (*subs_end)[i] - (*subs_begin)[i];
      ratios[i] = (double)rec_size[i] / coreSize[i];
    }

    // Sort the ratios
    for(int i=1; i<nd; i++) {
      for(int j=0; j<nd-i; j++) {
        if(ratios[j] > ratios[j+1]) {
          std::swap(ratios[j],ratios[j+1]);
          std::swap((*rec_order)[j],(*rec_order)[j+1]);
        }
      }
    }
    if(rank == 0) std::cout << "Reconstruction order: " << *rec_order << std::endl;

    // Free the memory
    delete[] rec_size;
    delete[] ratios;
  }

  //////////////////////////////////////////////////////////
  // Make sure the reconstruction order array makes sense //
  //////////////////////////////////////////////////////////
  for(int i=0; i<nd; i++) {
    if((*rec_order)[i] < 0) {
      if(rank == 0) {
        std::cerr << "Error: rec_order[" << i << "] = "
            << (*rec_order)[i] << " < 0\n";
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if((*rec_order)[i] >= nd) {
      if(rank == 0) {
        std::cerr << "Error: rec_order[" << i << "] = "
            << (*rec_order)[i] << " >= nd = " << nd << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for(int j=i+1; j<nd; j++) {
      if((*rec_order)[i] == (*rec_order)[j]) {
        if(rank == 0) {
          std::cerr << "Error: rec_order[" << i << "] == rec_order["
              << j << "] = " << (*rec_order)[i] << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
  }

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
  for(int i=0; i<nd; i++)
  {
    int mode = (*rec_order)[i];
    // Grab the requested rows of the factor matrix
    int start_subs = (*subs_begin)[mode];
    int end_subs = (*subs_end)[mode];
    Tucker::Matrix* factMat =
        fact.U[mode]->getSubmatrix(start_subs, end_subs);

    // Perform the TTM
    TuckerMPI::Tensor* temp = TuckerMPI::ttm(result,mode,factMat);

    delete factMat;
    if(result != fact.G)
      delete result;
    result = temp;
  }

  ////////////////////////////////////////////
  // Write the reconstructed tensor to disk //
  ////////////////////////////////////////////
  TuckerMPI::writeTensorBinary(out_fns_file, *result);

  /////////////////
  // Free memory //
  /////////////////
  delete I_dims;
  delete proc_grid_dims;
  delete subs_begin;
  delete subs_end;
  delete rec_order;
  delete result;

  //////////////////
  // Finalize MPI //
  //////////////////
  MPI_Finalize();
}
