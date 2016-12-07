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
  bool boolAuto                         = Tucker::stringParse<bool>(fileAsString, "Automatic rank determination", false);
  bool boolSTHOSVD                      = Tucker::stringParse<bool>(fileAsString, "Perform STHOSVD", false);
  bool boolWriteSTHOSVD                 = Tucker::stringParse<bool>(fileAsString, "Write STHOSVD result", false);
  bool boolPrintOptions                 = Tucker::stringParse<bool>(fileAsString, "Print options", false);
  bool boolWritePreprocessed            = Tucker::stringParse<bool>(fileAsString, "Write preprocessed data", false);
  bool boolUseOldGram                   = Tucker::stringParse<bool>(fileAsString, "Use old Gram", false);
  bool boolReconstruct                  = Tucker::stringParse<bool>(fileAsString, "Reconstruct tensor", false);

  double tol                            = Tucker::stringParse<double>(fileAsString, "SV Threshold", 1e-6);
  double stdThresh                      = Tucker::stringParse<double>(fileAsString, "STD Threshold", 1e-9);

  Tucker::SizeArray* I_dims             = Tucker::stringParseSizeArray(fileAsString, "Global dims");
  Tucker::SizeArray* R_dims = 0;
  if(!boolAuto)  R_dims                 = Tucker::stringParseSizeArray(fileAsString, "Ranks");
  Tucker::SizeArray* proc_grid_dims     = Tucker::stringParseSizeArray(fileAsString, "Grid dims");

  std::string scaling_type              = Tucker::stringParse<std::string>(fileAsString, "Scaling type", "None");
  std::string sthosvd_dir               = Tucker::stringParse<std::string>(fileAsString, "STHOSVD directory", "compressed");
  std::string sthosvd_fn                = Tucker::stringParse<std::string>(fileAsString, "STHOSVD file prefix", "sthosvd");
  std::string sv_dir                    = Tucker::stringParse<std::string>(fileAsString, "SV directory", ".");
  std::string sv_fn                     = Tucker::stringParse<std::string>(fileAsString, "SV file prefix", "sv");
  std::string in_fns_file               = Tucker::stringParse<std::string>(fileAsString, "Input file list", "raw.txt");
  std::string pre_fns_file              = Tucker::stringParse<std::string>(fileAsString, "Preprocessed output file list", "pre.txt");
  std::string stats_file                = Tucker::stringParse<std::string>(fileAsString, "Stats file", "stats.txt");
  std::string timing_file               = Tucker::stringParse<std::string>(fileAsString, "Timing file", "runtime.csv");

  int nd = I_dims->size();
  int scale_mode                        = Tucker::stringParse<int>(fileAsString, "Scale mode", nd-1);

  //
  // Print options
  //
  if (rank == 0 && boolPrintOptions) {
    std::cout << "Automatic rank determination = " << boolAuto << std::endl;
    std::cout << "Perform STHOSVD = " << boolSTHOSVD << std::endl;
    std::cout << "Write STHOSVD result = " << boolWriteSTHOSVD << std::endl;
    std::cout << "Stopping tolerance = " << tol << std::endl;
    std::cout << "STHOSVD directory = " << sthosvd_dir << std::endl;
    std::cout << "STHOSVD file prefix = " << sthosvd_fn << std::endl;
    std::cout << "SV directory = " << sv_dir << std::endl;
    std::cout << "SV file prefix = " << sv_fn << std::endl;
    std::cout << "Input file list = " << in_fns_file << std::endl;
    std::cout << "Scale mode = " << scale_mode << std::endl;

    std::cout << "Global dims = " << *I_dims << std::endl;
    if(!boolAuto) std::cout << "Ranks = " << *R_dims << std::endl;
    std::cout << "Grid dims = " << *proc_grid_dims << std::endl;
    std::cout << std::endl;
  }

  assert(boolAuto || R_dims->size() == nd);

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

  if (nd != proc_grid_dims->size()) {
    if (rank == 0) {
      std::cerr << "Error: The size of global dimension array (" << nd;
      std::cerr << ") must be equal to the size of the processor grid ("
          << proc_grid_dims->size() << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (!boolAuto && R_dims->size() != 0 && R_dims->size() != nd) {
    if (rank == 0) {
      std::cerr << "Error: The size of the ranks array (" << R_dims->size();
      std::cerr << ") must be 0 or equal to the size of the processor grid (" << nd << ")" << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  ///////////////////////////
  // Set up processor grid //
  ///////////////////////////
  if (rank == 0) {
    std::cout << "Creating process grid" << std::endl;
  }
  ////////////////////////////////
  // Set up distribution object //
  ////////////////////////////////
  TuckerMPI::Distribution dist(*I_dims, *proc_grid_dims);

  ///////////////////////////
  // Read full tensor data //
  ///////////////////////////
  TuckerMPI::Tensor X(&dist);
  TuckerMPI::readTensorBinary(in_fns_file,X);

  ////////////////////////
  // Compute statistics //
  ////////////////////////
  Tucker::MetricData* metrics = TuckerMPI::computeSliceMetrics(&X,
      scale_mode,
      Tucker::MIN+Tucker::MAX+Tucker::MEAN+Tucker::VARIANCE);

  // Determine whether I need to communicate with rank 0
  int* myCoordinates = Tucker::safe_new<int>(nd);
  int* zeroCoordinates = Tucker::safe_new<int>(nd);
  const TuckerMPI::ProcessorGrid* grid = dist.getProcessorGrid();
  grid->getCoordinates(myCoordinates);
  grid->getCoordinates(zeroCoordinates,0);

  bool needToSendToZero = true;
  for(int i=0; i<nd; i++) {
    if(i == scale_mode) continue;

    if(myCoordinates[i] != zeroCoordinates[i]) {
      needToSendToZero = false;
      break;
    }
  }

  const TuckerMPI::Map* map = dist.getMap(scale_mode,false);
  const MPI_Comm& rowComm = grid->getColComm(scale_mode,false);
  if(needToSendToZero) {
    int numEntries = map->getGlobalNumEntries();
    double* mins = Tucker::safe_new<double>(numEntries);
    double* maxs = Tucker::safe_new<double>(numEntries);
    double* means = Tucker::safe_new<double>(numEntries);
    double* vars = Tucker::safe_new<double>(numEntries);
    MPI_Gatherv (metrics->getMinData(), map->getLocalNumEntries(),
        MPI_DOUBLE, mins, (int*)map->getNumElementsPerProc()->data(),
        (int*)map->getOffsets()->data(), MPI_DOUBLE, 0, rowComm);
    MPI_Gatherv (metrics->getMaxData(), map->getLocalNumEntries(),
        MPI_DOUBLE, maxs, (int*)map->getNumElementsPerProc()->data(),
        (int*)map->getOffsets()->data(), MPI_DOUBLE, 0, rowComm);
    MPI_Gatherv (metrics->getMeanData(), map->getLocalNumEntries(),
        MPI_DOUBLE, means, (int*)map->getNumElementsPerProc()->data(),
        (int*)map->getOffsets()->data(), MPI_DOUBLE, 0, rowComm);
    MPI_Gatherv (metrics->getVarianceData(), map->getLocalNumEntries(),
        MPI_DOUBLE, vars, (int*)map->getNumElementsPerProc()->data(),
        (int*)map->getOffsets()->data(), MPI_DOUBLE, 0, rowComm);

    if(rank == 0) {
      std::cout << "Writing file " << stats_file << std::endl;

      std::ofstream statStream(stats_file);
      statStream << std::setw(5) << "Mode"
          << std::setw(13) << "Mean"
          << std::setw(13) << "Stdev"
          << std::setw(13) << "Min"
          << std::setw(13) << "Max"
          << std::endl;

      for(int i=0; i<numEntries; i++) {
        double stdev = sqrt(vars[i]);

        if(stdev < stdThresh) {
          std::cout << "Slice " << i
              << " is below the cutoff. True value is: "
              << stdev << std::endl;
          stdev = 1;
        }

        statStream << std::setw(5) << i
            << std::setw(13) << means[i]
            << std::setw(13) << stdev
            << std::setw(13) << mins[i]
            << std::setw(13) << maxs[i] << std::endl;
      }

      statStream.close();
    }
  }


  ///////////////////////////
  // Perform preprocessing //
  ///////////////////////////
  if(scaling_type == "Max") {
    normalizeTensorMax(&X, scale_mode);
  }
  if(scaling_type == "MinMax") {
    normalizeTensorMinMax(&X, scale_mode);
  }
  else if(scaling_type == "StandardCentering") {
    normalizeTensorStandardCentering(&X, scale_mode, stdThresh);
  }
  else if(scaling_type == "None") {

  }
  else {
    std::cerr << "Error: invalid scaling type: " << scaling_type << std::endl;
  }

  if(boolWritePreprocessed) {
    TuckerMPI::writeTensorBinary(pre_fns_file,X);
  }

  /////////////////////
  // Perform STHOSVD //
  /////////////////////
  if(boolSTHOSVD) {
    const TuckerMPI::TuckerTensor* solution;

    if(boolAuto) {
      solution = TuckerMPI::STHOSVD(&X, tol, boolUseOldGram);
    }
    else {
      solution = TuckerMPI::STHOSVD(&X, R_dims, boolUseOldGram);
    }

    // Send the timing information to a CSV
    solution->printTimers(timing_file);

    if(boolReconstruct) {
      TuckerMPI::Tensor* t = solution->reconstructTensor();

      TuckerMPI::Tensor* diff = X.subtract(t);
      double nrm = X.norm2();
      double err = diff->norm2();
      double maxEntry = diff->maxEntry();
      double minEntry = diff->minEntry();
      if(rank == 0) {
        std::cout << "Norm of X: " << std::sqrt(nrm) << std::endl;
        std::cout << "Norm of X - Xtilde: "
            << std::sqrt(err) << std::endl;
        std::cout << "Maximum entry of X - Xtilde: "
            << std::max(maxEntry,-minEntry) << std::endl;
      }
    }

    if(rank == 0) {
      // Write the eigenvalues to files
      std::string filePrefix = sv_dir + "/" + sv_fn + "_mode_";
      TuckerMPI::printEigenvalues(solution, filePrefix);

      // Print the core tensor size
      std::cout << "Core tensor size: " <<
          solution->G->getGlobalSize() << std::endl;
    }

    if(boolWriteSTHOSVD) {
      // Write dimension of core tensor
      if(rank == 0) {
        std::string dimFilename = sthosvd_dir + "/" + sthosvd_fn +
            "_ranks.txt";
        std::ofstream of(dimFilename);
        for(int mode=0; mode<nd; mode++) {
          of << solution->G->getGlobalSize(mode) << std::endl;
        }
        of.close();
      }

      // Write core tensor
      std::string coreFilename = sthosvd_dir + "/" + sthosvd_fn +
          "_core.mpi";
      TuckerMPI::exportTensorBinary(coreFilename.c_str(), solution->G);

      // Write each factor
      if(rank == 0) {
        for(int mode=0; mode<nd; mode++) {
          // Create the filename by appending the mode #
          std::ostringstream ss;
          ss << sthosvd_dir << "/" << sthosvd_fn << "_mat_" << mode
              << ".mpi";       // Open the file
          TuckerMPI::exportTensorBinary(ss.str().c_str(), solution->U[mode]);
        }
      }
    }
  }

  //
  // Free memory
  //
  delete I_dims;
  if(R_dims) delete R_dims;
  delete proc_grid_dims;

  // Finalize MPI
  MPI_Finalize();
}
