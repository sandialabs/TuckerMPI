// This file allows you do multiple runs of sthosvd with different parameter files on the same data.
// This file should only be used when experimenting with different varaibles on a large dataset that 
// you don't want to read over and over. Note for now, since there is no way of doing a general 
// redistribution, it is required that the processor grids also stay the same among those runs.

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
  std::string general_input = Tucker::parseString(argc, (const char**)argv, "--parameter-file", "paramfile.txt");
  std::vector<std::string> fileAsString = Tucker::getFileAsStrings(general_input);
  std::string in_fns_file               = Tucker::stringParse<std::string>(fileAsString, "Input file list", "raw.txt");
  Tucker::SizeArray* I_dims             = Tucker::stringParseSizeArray(fileAsString, "Global dims");
  Tucker::SizeArray* proc_grid_dims     = Tucker::stringParseSizeArray(fileAsString, "Grid dims");
  std::string parameter_file_list            = Tucker::stringParse<std::string>(fileAsString, "Parameter files list", "params.txt");
  int nd = I_dims->size();

  ///////////////////////////
  // Read full tensor data //
  ///////////////////////////
  // Check array sizes //
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

  // Set up processor grid //
  if (rank == 0) {
    std::cout << "Creating process grid" << std::endl;
  }
  // Set up distribution object //
  TuckerMPI::Distribution* dist =
      Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*I_dims, *proc_grid_dims);
  
  // Read full tensor data //
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


  //////////////////////////////////
  // Loop through parameter files //
  //////////////////////////////////
  std::ifstream ifs;
  ifs.open(parameter_file_list.c_str());
  std::string parameter_file;
  while(ifs >> parameter_file){
    // Parse parameter file
    // Put's each line as a string into a vector ignoring empty lines
    // and comments
    std::vector<std::string> fileAsString = Tucker::getFileAsStrings(parameter_file);
    bool boolAuto                         = Tucker::stringParse<bool>(fileAsString, "Automatic rank determination", false);
    bool boolSTHOSVD                      = Tucker::stringParse<bool>(fileAsString, "Perform STHOSVD", false);
    bool boolWriteSTHOSVD                 = Tucker::stringParse<bool>(fileAsString, "Write core tensor and factor matrices", false);
    bool boolPrintOptions                 = Tucker::stringParse<bool>(fileAsString, "Print options", false);
    bool boolWritePreprocessed            = Tucker::stringParse<bool>(fileAsString, "Write preprocessed data", false);
    bool boolUseOldGram                   = Tucker::stringParse<bool>(fileAsString, "Use old Gram", false);
    bool boolUseLQ                        = Tucker::stringParse<bool>(fileAsString, "Compute SVD via LQ", false);
    bool boolPrintSV                      = Tucker::stringParse<bool>(fileAsString, "Print factor matrices", false);
    bool boolReconstruct                  = Tucker::stringParse<bool>(fileAsString, "Reconstruct tensor", false);
    bool useButterflyTSQR                 = Tucker::stringParse<bool>(fileAsString, "Use butterfly TSQR", false);

    scalar_t tol                          = Tucker::stringParse<scalar_t>(fileAsString, "SV Threshold", 1e-6);
    scalar_t stdThresh                    = Tucker::stringParse<scalar_t>(fileAsString, "STD Threshold", 1e-9);

    Tucker::SizeArray* R_dims = 0;
    if(!boolAuto)  R_dims                 = Tucker::stringParseSizeArray(fileAsString, "Ranks");
    Tucker::SizeArray* modeOrder          = Tucker::stringParseSizeArray(fileAsString, "Decompose mode order");

    std::string scaling_type              = Tucker::stringParse<std::string>(fileAsString, "Scaling type", "None");
    std::string sthosvd_dir               = Tucker::stringParse<std::string>(fileAsString, "STHOSVD directory", "compressed");
    std::string sthosvd_fn                = Tucker::stringParse<std::string>(fileAsString, "STHOSVD file prefix", "sthosvd");
    std::string sv_dir                    = Tucker::stringParse<std::string>(fileAsString, "SV directory", ".");
    std::string sv_fn                     = Tucker::stringParse<std::string>(fileAsString, "SV file prefix", "sv");
    std::string pre_fns_file              = Tucker::stringParse<std::string>(fileAsString, "Preprocessed output file list", "pre.txt");
    std::string reconstruct_report_file   = Tucker::stringParse<std::string>(fileAsString, "Reconstruction report file", "reconstruction.txt");
    std::string stats_file                = Tucker::stringParse<std::string>(fileAsString, "Stats file", "stats.txt");
    std::string timing_file               = Tucker::stringParse<std::string>(fileAsString, "Timing file", "runtime.csv");
    int scale_mode                        = Tucker::stringParse<int>(fileAsString, "Scale mode", nd-1);

    //
    // Assert that we either have automatic rank determination or the user
    // has supplied their own ranks
    //
    if(!boolAuto && !R_dims) {
      std::cerr << "ERROR: Please either enable Automatic rank determination, "
                << "or provide the desired core tensor size via the Ranks parameter\n";
      return EXIT_FAILURE;
    }

    if (!boolAuto && R_dims->size() != 0 && R_dims->size() != nd) {
      if (rank == 0) {
        std::cerr << "Error: The size of the ranks array (" << R_dims->size();
        std::cerr << ") must be 0 or equal to the size of the processor grid (" << nd << ")" << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    assert(boolAuto || R_dims->size() == nd);

    //
    // Print options
    //
    if (rank == 0 && boolPrintOptions) {
      std::cout << "The global dimensions of the tensor to be scaled or compressed\n";
      std::cout << "- Global dims = " << *I_dims << std::endl << std::endl;

      std::cout << "The global dimensions of the processor grid\n";
      std::cout << "- Grid dims = " << *proc_grid_dims << std::endl << std::endl;

      std::cout << "Mode order for decomposition\n";
      std::cout << "- Decompose mode order " << *modeOrder << std::endl << std::endl;

      std::cout << "If true, automatically determine rank; otherwise, use the user-defined ranks\n";
      std::cout << "- Automatic rank determination = " << (boolAuto ? "true" : "false") << std::endl << std::endl;

      std::cout << "Used for automatic rank determination; the desired error rate\n";
      std::cout << "- SV Threshold = " << tol << std::endl << std::endl;

      if(!boolAuto) {
        std::cout << "Global dimensions of the desired core tensor\n";
        std::cout << "Not used if \"Automatic rank determination\" is enabled\n";
        std::cout << "- Ranks = " << *R_dims << std::endl << std::endl;
      }

      std::cout << "List of filenames of raw data to be read\n";
      std::cout << "- Input file list = " << in_fns_file << std::endl << std::endl;

      std::cout << "How to scale the tensor\n";
      std::cout << "- Scaling type = " << scaling_type << std::endl << std::endl;

      std::cout << "Which mode's hyperslices will be scaled\n";
      std::cout << "- Scale mode = " << scale_mode << std::endl << std::endl;

      std::cout << "Threshold for standard deviation before we simply set it to 1\n";
      std::cout << "Used in StandardCentering scaling\n";
      std::cout << "- STD Threshold = " << stdThresh << std::endl << std::endl;

      std::cout << "If true, perform ST-HOSVD\n";
      std::cout << "- Perform STHOSVD = " << (boolSTHOSVD ? "true" : "false") << std::endl << std::endl;

      std::cout << "If true, use the old Gram algorithm; otherwise use the new one\n";
      std::cout << "- Use old Gram = " << (boolUseOldGram ? "true" : "false") << std::endl << std::endl;

      std::cout << "Location of a report of the reconstruction errors \n";
      std::cout << "- Reconstruction report file = " << reconstruct_report_file << std::endl << std::endl;

      std::cout << "Location of statistics file containing min, max, mean, and std of each hyperslice\n";
      std::cout << "- Stats file = " << stats_file << std::endl << std::endl;

      std::cout << "If true, write the preprocessed data to a file\n";
      std::cout << "- Write preprocessed data = " << (boolWritePreprocessed ? "true" : "false") << std::endl << std::endl;

      std::cout << "File containing a list of filenames to output the scaled data into\n";
      std::cout << "- Preprocessed output file list = " << pre_fns_file << std::endl << std::endl;

      std::cout << "If true, record the result of ST-HOSVD (the core tensor and all factors\n";
      std::cout << "- Write STHOSVD result = " << (boolWriteSTHOSVD ? "true" : "false") << std::endl << std::endl;

      std::cout << "Directory location of ST-HOSVD output files\n";
      if(boolWriteSTHOSVD) std::cout << "NOTE: Please ensure that this directory actually exists!\n";
      std::cout << "- STHOSVD directory = " << sthosvd_dir << std::endl << std::endl;

      std::cout << "Base name of ST-HOSVD output files\n";
      std::cout << "- STHOSVD file prefix = " << sthosvd_fn << std::endl << std::endl;

      std::cout << "Directory to place singular value files into\n";
      if(boolWriteSTHOSVD) std::cout << "NOTE: Please ensure that this directory actually exists!\n";
      std::cout << "- SV directory = " << sv_dir << std::endl << std::endl;

      std::cout << "Base name for writing the singular value files\n";
      std::cout << "- SV file prefix = " << sv_fn << std::endl << std::endl;

      std::cout << "Name of the CSV file holding the timing results\n";
      std::cout << "- Timing file = " << timing_file << std::endl << std::endl;

      std::cout << "If true, reconstruct an approximation of the original tensor after ST-HOSVD\n";
      if(boolReconstruct) std::cout << "WARNING: This may require a great deal of memory\n";
      std::cout << "- Reconstruct tensor = " << (boolReconstruct ? "true" : "false") << std::endl << std::endl;

      std::cout << "If true, print the parameters\n";
      std::cout << "- Print options = " << (boolPrintOptions ? "true" : "false") << std::endl << std::endl;

      std::cout << std::endl;
    }

    ////////////////////////
    // Compute statistics //
    ////////////////////////
    Tucker::MetricData<scalar_t>* metrics = TuckerMPI::computeSliceMetrics(&X,
        scale_mode,
        Tucker::MIN+Tucker::MAX+Tucker::MEAN+Tucker::VARIANCE);

    // Determine whether I need to communicate with rank 0
    int* myCoordinates = Tucker::MemoryManager::safe_new_array<int>(nd);
    int* zeroCoordinates = Tucker::MemoryManager::safe_new_array<int>(nd);
    const TuckerMPI::ProcessorGrid* grid = dist->getProcessorGrid();
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

    const TuckerMPI::Map* map = dist->getMap(scale_mode,false);
    const MPI_Comm& rowComm = grid->getColComm(scale_mode,false);
    if(needToSendToZero) {
      int numEntries = map->getGlobalNumEntries();
      scalar_t* mins = Tucker::MemoryManager::safe_new_array<scalar_t>(numEntries);
      scalar_t* maxs = Tucker::MemoryManager::safe_new_array<scalar_t>(numEntries);
      scalar_t* means = Tucker::MemoryManager::safe_new_array<scalar_t>(numEntries);
      scalar_t* vars = Tucker::MemoryManager::safe_new_array<scalar_t>(numEntries);
      TuckerMPI::MPI_Gatherv_ (metrics->getMinData(), map->getLocalNumEntries(),
          mins, (int*)map->getNumElementsPerProc()->data(),
          (int*)map->getOffsets()->data(), 0, rowComm);
      TuckerMPI::MPI_Gatherv_ (metrics->getMaxData(), map->getLocalNumEntries(),
          maxs, (int*)map->getNumElementsPerProc()->data(),
          (int*)map->getOffsets()->data(), 0, rowComm);
      TuckerMPI::MPI_Gatherv_ (metrics->getMeanData(), map->getLocalNumEntries(),
          means, (int*)map->getNumElementsPerProc()->data(),
          (int*)map->getOffsets()->data(), 0, rowComm);
      TuckerMPI::MPI_Gatherv_ (metrics->getVarianceData(), map->getLocalNumEntries(),
          vars, (int*)map->getNumElementsPerProc()->data(),
          (int*)map->getOffsets()->data(), 0, rowComm);

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

        if(scaling_type != "None") {
          std::string scale_file = sthosvd_dir + "/" + sthosvd_fn +
              "_scale.txt";
          std::ofstream outStream(scale_file);

          outStream << scale_mode << std::endl;

          for(int i=0; i<numEntries; i++)
          {
            double scales, shifts;
            if(scaling_type == "Max") {
              scales = maxs[i] - mins[i];
              shifts = -mins[i];
            }
            else if(scaling_type == "MinMax") {
              scales = std::max(-mins[i], maxs[i]);
              shifts = 0;
            }
            else if(scaling_type == "StandardCentering") {
              scales = sqrt(vars[i]);
              shifts = -means[i];

              if(scales < stdThresh) {
                scales = 1;
              }
            }

            outStream << scales << " " << shifts << std::endl;
          }

          outStream.close();
        }
      }
    }

    ///////////////////////////
    // Perform preprocessing //
    ///////////////////////////
    MPI_Barrier(MPI_COMM_WORLD);
    Tucker::Timer preprocessTimer;
    preprocessTimer.start();
    if(scaling_type == "Max") {
      if(rank == 0) {
        std::cout << "Normalizing the tensor by maximum entry - mode " 
                  << scale_mode << std::endl;
      }
      normalizeTensorMax(&X, scale_mode);
    }
    else if(scaling_type == "MinMax") {
      if(rank == 0) {
        std::cout << "Normalizing the tensor using minmax scaling - mode "
                  << scale_mode << std::endl;
      }
      normalizeTensorMinMax(&X, scale_mode);
    }
    else if(scaling_type == "StandardCentering") {
      if(rank == 0) {
        std::cout << "Normalizing the tensor using standard centering - mode "
                  << scale_mode << std::endl;
      }
      normalizeTensorStandardCentering(&X, scale_mode, stdThresh);
    }
    else if(scaling_type == "None") {
      if(rank == 0) {
        std::cout << "Not scaling the tensor\n";
      }
    }
    else {
      std::cerr << "Error: invalid scaling type: " << scaling_type << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    preprocessTimer.stop();
    if(rank == 0) {
      std::cout << "Time to preprocess: " << preprocessTimer.duration() << std::endl;
    }

    if(boolWritePreprocessed) {
      TuckerMPI::writeTensorBinary(pre_fns_file,X);
    }
    
    /////////////////////
    // Perform STHOSVD //
    /////////////////////
    if(boolSTHOSVD) {
      const TuckerMPI::TuckerTensor<scalar_t>* solution;
      bool flipSign = false; // confirm its default as false
      if(boolAuto) {
        solution = TuckerMPI::STHOSVD(&X, tol, modeOrder->data(), boolUseOldGram, flipSign, boolUseLQ, useButterflyTSQR);
      }
      else {
        solution = TuckerMPI::STHOSVD(&X, R_dims, modeOrder->data(), boolUseOldGram, flipSign, boolUseLQ, useButterflyTSQR);
      }

      // Send the timing information to a CSV
      if(boolUseLQ) solution->printTimersLQ(timing_file);
      else solution->printTimers(timing_file);

      if(rank == 0) {
        // Write the eigenvalues to files
        std::string filePrefix = sv_dir + "/" + sv_fn + "value_mode_";
        TuckerMPI::printSingularValues(solution, filePrefix, boolUseLQ);
        if(boolPrintSV){
          filePrefix = sv_dir + "/" + sv_fn + "vector_mode_";
          TuckerMPI::printEigenvectors(solution, filePrefix);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      scalar_t xnorm2 = X.norm2();
      scalar_t xnorm = std::sqrt(xnorm2);
      scalar_t gnorm = std::sqrt(solution->G->norm2());
      scalar_t errorBound =0;
      
      if(rank == 0) {
        std::cout << "Norm of input tensor: " << xnorm << std::endl;
        std::cout << "Norm of core tensor: " << gnorm << std::endl;
        // Compute the error bound based on the eigenvalues
        
        if(boolUseLQ){
          for(int i=0; i<nd; i++) {
            for(int j=solution->G->getGlobalSize(i); j<X.getGlobalSize(i); j++) {
              errorBound += solution->singularValues[i][j];
            }
          }
        }
        else{
          for(int i=0; i<nd; i++) {
            for(int j=solution->G->getGlobalSize(i); j<X.getGlobalSize(i); j++) {
              errorBound += solution->eigenvalues[i][j];
            }
          }
        }
        std::cout << "Error bound: " << std::sqrt(errorBound)/xnorm << std::endl;
      
        
        // Write dimension of core tensor
        std::string dimFilename = sthosvd_dir + "/" + sthosvd_fn +
            "_ranks.txt";
        std::ofstream of(dimFilename);
        for(int mode=0; mode<nd; mode++) {
          of << solution->G->getGlobalSize(mode) << std::endl;
        }
        of.close();
        of.clear();
        // Write dimension of global tensor
        std::string sizeFilename = sthosvd_dir + "/" + sthosvd_fn +
            "_size.txt";
        of.open(sizeFilename);
        for(int mode=0; mode<nd; mode++) {
          of << (*I_dims)[mode] << std::endl;
        }
        of.close();
      }

      if(boolReconstruct) {
        TuckerMPI::Tensor<scalar_t>* t = solution->reconstructTensor();
        TuckerMPI::Tensor<scalar_t>* diff = X.subtract(t);
        // scalar_t nrm = X.norm2();
        scalar_t err = diff->norm2();
        scalar_t maxEntry = diff->maxEntry();
        scalar_t minEntry = diff->minEntry();
        if(rank == 0) {
          std::ofstream of(reconstruct_report_file);
          of << "Norm of input tensor: " << xnorm2 << std::endl;
          of << "Norm of core tensor: " << gnorm << std::endl;
          of << "Error bound: " << std::sqrt(errorBound)/xnorm << std::endl;

          of << "Norm of inputTensor - reconstruction: "
              << std::sqrt(err) << std::endl;
          of << "relative error: "
              << std::sqrt(err/xnorm2) << std::endl;
          of << "Maximum entry of inputTensor - reconstruction: "
              << std::max(maxEntry,-minEntry) << std::endl;
          of.close();
        }
        Tucker::MemoryManager::safe_delete(diff);
        Tucker::MemoryManager::safe_delete(t);
      }

      if(boolWriteSTHOSVD) {
        Tucker::Timer writeTimer;
        writeTimer.start();

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
        writeTimer.stop();

        double localWriteTime = writeTimer.duration();
        double globalWriteTime;

        MPI_Reduce(&localWriteTime,&globalWriteTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

        if(rank == 0) {
          std::cout << "Time to write factorization: " << globalWriteTime << " s\n";
        }

      } // end if(boolWriteSTHOSVD)
    } // end if(boolSTHOSVD)
    if(R_dims) Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(R_dims);
    if(rank == 0) {
      Tucker::MemoryManager::printMaxMemUsage();
    }
  }

  //
  // Free memory
  //
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(I_dims);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(proc_grid_dims);

  // Finalize MPI
  MPI_Finalize();
}
