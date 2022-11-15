/*
 * driver.cpp
 *
 *  Created on: Nov 8, 2022
 *      Author: Hemanth Kolla (hnkolla@sandia.gov)
 */

#include "Tucker.hpp"
#include "Tucker_IO_Util.hpp"
#include "Tucker_StreamingTuckerTensor.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "assert.h"

int main(int argc, char* argv[])
{
  typedef double scalar_t;  // specify precision
  
  Tucker::Timer totalTimer;
  totalTimer.start();

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
  bool useLQ                            = Tucker::stringParse<bool>(fileAsString, "Compute SVD via LQ", false);

  double tol                            = Tucker::stringParse<double>(fileAsString, "SV Threshold", 1e-6);

  Tucker::SizeArray* I_dims             = Tucker::stringParseSizeArray(fileAsString, "Global dims");
  Tucker::SizeArray* R_dims = 0;
  if(!boolAuto)  R_dims                 = Tucker::stringParseSizeArray(fileAsString, "Ranks");

  std::string scaling_type              = Tucker::stringParse<std::string>(fileAsString, "Scaling type", "None");
  std::string sthosvd_dir               = Tucker::stringParse<std::string>(fileAsString, "STHOSVD directory", "compressed");
  std::string sthosvd_fn                = Tucker::stringParse<std::string>(fileAsString, "STHOSVD file prefix", "sthosvd");
  std::string sv_dir                    = Tucker::stringParse<std::string>(fileAsString, "SV directory", ".");
  std::string sv_fn                     = Tucker::stringParse<std::string>(fileAsString, "SV file prefix", "sv");
  std::string in_fns_file               = Tucker::stringParse<std::string>(fileAsString, "Initial input file list", "raw.txt");
  std::string streaming_fns_file        = Tucker::stringParse<std::string>(fileAsString, "Streaming input file list", "stream_files.txt");

  int nd = I_dims->size();


  //
  // Assert that we either have automatic rank determination or the user
  // has supplied their own ranks
  //
  if(!boolAuto && !R_dims) {
    std::cerr << "ERROR: Please either enable Automatic rank determination, "
              << "or provide the desired core tensor size via the Ranks parameter\n";
    return EXIT_FAILURE;
  }

  //
  // Assert that user is not expecting to perform scaling or preprocessing
  // This driver expects data that is already normalized for the time being
  //
  if(scaling_type != "None") {
    std::cerr << "Error: scaling not enabled in this driver, use sthosvd driver for preprocessing/scaling: " << scaling_type << std::endl;
    return EXIT_FAILURE;
  }

  //
  // Print options
  //
  if (boolPrintOptions) {
    std::cout << "The global dimensions of the initial tensor to be compressed\n";
    std::cout << "- Global dims = " << *I_dims << std::endl << std::endl;

    std::cout << "If true, automatically determine rank; otherwise, use the user-defined ranks\n";
    std::cout << "- Automatic rank determination = " << (boolAuto ? "true" : "false") << std::endl << std::endl;

    std::cout << "Used for automatic rank determination; the desired error rate\n";
    std::cout << "- SV Threshold = " << tol << std::endl << std::endl;

    if(!boolAuto) {
      std::cout << "Global dimensions of the desired core of the initial compressed tensor\n";
      std::cout << "Not used if \"Automatic rank determination\" is enabled\n";
      std::cout << "- Ranks = " << *R_dims << std::endl << std::endl;
    }

    std::cout << "List of filenames of raw data to be read for initial sthosvd\n";
    std::cout << "- Initial input file list = " << in_fns_file << std::endl << std::endl;

    std::cout << "List of filenames of raw data to be read for streaming hosvd\n";
    std::cout << "- Streaming input file list = " << streaming_fns_file << std::endl << std::endl;

    std::cout << "How to scale the tensor\n";
    std::cout << "- Scaling type = " << scaling_type << std::endl << std::endl;

    std::cout << "If true, perform ST-HOSVD\n";
    std::cout << "- Perform STHOSVD = " << (boolSTHOSVD ? "true" : "false") << std::endl << std::endl;

    std::cout << "If true, record the result of ST-HOSVD (the core tensor and all factors)\n";
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

    std::cout << "If true, print the parameters\n";
    std::cout << "- Print options = " << (boolPrintOptions ? "true" : "false") << std::endl << std::endl;

    std::cout << std::endl;
  }

  assert(boolAuto || R_dims->size() == nd);

  ///////////////////////
  // Check array sizes //
  ///////////////////////
  if (!boolAuto && R_dims->size() != 0 && R_dims->size() != nd) {
    std::cerr << "Error: The size of the ranks array (" << R_dims->size();
    std::cerr << ") must be 0 or equal to the size of the global dimensions (" << nd << ")" << std::endl;

    return EXIT_FAILURE;
  }

  ///////////////////////////
  // Read initial tensor data //
  ///////////////////////////
  Tucker::Timer readTimer;
  readTimer.start();
  Tucker::Tensor<scalar_t>* X = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*I_dims);
  Tucker::readTensorBinary(X,in_fns_file.c_str());
  readTimer.stop();

  size_t nnz = X->getNumElements();
  std::cout << "Initial input tensor size: " << X->size() << ", or ";
  Tucker::printBytes(nnz*sizeof(double));

  ///////////////////////////
  // Perform preprocessing //
  ///////////////////////////

  /////////////////////////////
  // Perform Initial STHOSVD //
  /////////////////////////////
  Tucker::Timer sthosvdTimer, writeTimer;
  if(boolSTHOSVD) {
    const Tucker::TuckerTensor<scalar_t>* initial_solution;

    sthosvdTimer.start();
    if(boolAuto) {
      initial_solution = Tucker::STHOSVD(X, tol, useLQ);
    }
    else {
      initial_solution = Tucker::STHOSVD(X, R_dims, useLQ);
    }
    sthosvdTimer.stop();

    // Write the eigenvalues to files
    std::string filePrefix = sv_dir + "/" + sv_fn + "_mode_";
    Tucker::printEigenvalues(initial_solution, filePrefix, useLQ);

    double xnorm = std::sqrt(X->norm2());
    double gnorm = std::sqrt(initial_solution->G->norm2());
    std::cout << "Norm of input tensor: " << xnorm << std::endl;
    std::cout << "Norm of core tensor: " << gnorm << std::endl;

    // Compute the error bound based on the eigenvalues
    double eb =0;
    if(useLQ){
      for(int i=0; i<nd; i++) {
        for(int j=initial_solution->G->size(i); j<X->size(i); j++) {
          eb += std::pow(initial_solution->singularValues[i][j],2);
        }
      }
    }
    else{
      for(int i=0; i<nd; i++) {
        for(int j=initial_solution->G->size(i); j<X->size(i); j++) {
          eb += initial_solution->eigenvalues[i][j];
        }
      }
    }
    std::cout << "Error bound: " << eb << ", " << std::sqrt(eb) << " / " << xnorm << std::endl;


    /////////////////////////////
    // Perform Streaming HOSVD //
    /////////////////////////////

    const Tucker::StreamingTuckerTensor<scalar_t>* solution;

    solution = Tucker::StreamingHOSVD(X, initial_solution, streaming_fns_file.c_str(), tol, useLQ);

    /*
    writeTimer.start();
    if(boolWriteSTHOSVD) {
      // Write dimension of core tensor
      std::string dimFilename = sthosvd_dir + "/" + sthosvd_fn +
          "_ranks.txt";
      std::cout << "Writing core tensor dimensions to " << dimFilename << std::endl;
      std::ofstream of(dimFilename);
      assert(of.is_open());
      for(int mode=0; mode<nd; mode++) {
        of << solution->G->size(mode) << std::endl;
      }
      of.close();

      // Write dimension of global tensor
      std::string sizeFilename = sthosvd_dir + "/" + sthosvd_fn +
          "_size.txt";
      std::cout << "Writing global tensor dimensions to " << sizeFilename << std::endl;
      of.open(sizeFilename);
      assert(of.is_open());
      for(int mode=0; mode<nd; mode++) {
        of << (*I_dims)[mode] << std::endl;
      }
      of.close();

      // Write core tensor
      std::string coreFilename = sthosvd_dir + "/" + sthosvd_fn +
          "_core.mpi";
      std::cout << "Writing core tensor to " << coreFilename << std::endl;
      Tucker::exportTensorBinary(solution->G, coreFilename.c_str());

      // Write each factor
      for(int mode=0; mode<nd; mode++) {
        // Create the filename by appending the mode #
        std::ostringstream ss;
        ss << sthosvd_dir << "/" << sthosvd_fn << "_mat_" << mode
            << ".mpi";       // Open the file
        std::cout << "Writing factor " << mode << " to " << ss.str() << std::endl;
        Tucker::exportTensorBinary(solution->U[mode], ss.str().c_str());
      }
    }
    writeTimer.stop();
    */
  }

  //
  // Free memory
  //
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(I_dims);
  if(R_dims) Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(R_dims);

  Tucker::MemoryManager::printMaxMemUsage();

  totalTimer.stop();
  std::cout << "Read time: " << readTimer.duration() << std::endl;
  std::cout << "STHOSVD time: " << sthosvdTimer.duration() << std::endl;
  std::cout << "Write time: " << writeTimer.duration() << std::endl;
  std::cout << "Total time: " << totalTimer.duration() << std::endl;

  return EXIT_SUCCESS;
}
