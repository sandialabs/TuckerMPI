/*
 * driver.cpp
 *
 *  Created on: Jun 3, 2016
 *      Author: Alicia Klinvex (amklinv@sandia.gov)
 */

#include "Tucker.hpp"
#include "Tucker_IO_Util.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "assert.h"

int main(int argc, char* argv[])
{
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
  if(I_dims == NULL) {
    std::cerr << "Error: Global dims is a required parameter\n";
    return EXIT_FAILURE;
  }

  if(subs_begin == NULL) {
    std::cerr << "Error: Beginning subscripts is a required parameter\n";
    return EXIT_FAILURE;
  }

  if(subs_end == NULL) {
    std::cerr << "Error: Ending subscripts is a required parameter\n";
    return EXIT_FAILURE;
  }

  ///////////////////
  // Print options //
  ///////////////////
  if (boolPrintOptions) {
    std::cout << "Global dimensions of the original tensor\n";
    std::cout << "- Global dims = " << *I_dims << std::endl << std::endl;

    std::cout << "Start of subscripts to be recomputed\n";
    std::cout << "- Beginning subscripts = " << *subs_begin << std::endl << std::endl;

    std::cout << "End of subscripts to be recomputed\n";
    std::cout << "- Ending subscripts = " << *subs_end << std::endl << std::endl;

    std::cout << "Directory location of ST-HOSVD output files\n";
    std::cout << "- STHOSVD directory = " << sthosvd_dir << std::endl << std::endl;

    std::cout << "Base name of the ST-HOSVD output files\n";
    std::cout << "- STHOSVD file prefix = " << sthosvd_fn << std::endl << std::endl;

    std::cout << "File containing a list of filenames to output the reconstructed data into\n";
    std::cout << "- Output file list = " << out_fns_file << std::endl << std::endl;

    if(rec_order != NULL) {
      std::cout << "Mode order for reconstruction\n";
      std::cout << "NOTE: if left unspecified, the memory-optimal one will be automatically selected\n";
      std::cout << "- Reconstruction order = " << *rec_order << std::endl << std::endl;
    }

    std::cout << "If true, print the parameters\n";
    std::cout << "- Print options = " << (boolPrintOptions ? "true" : "false") << std::endl << std::endl;

    std::cout << std::endl;
  }

  ///////////////////////
  // Check array sizes //
  ///////////////////////
  int nd = I_dims->size();

  if (nd != subs_begin->size()) {
    std::cerr << "Error: The size of the subs_begin array (" << subs_begin->size();
    std::cerr << ") must be equal to the size of the global dimension array ("
        << nd << ")" << std::endl;
    return EXIT_FAILURE;
  }

  if (nd != subs_end->size()) {
    std::cerr << "Error: The size of the subs_end array (" << subs_end->size();
    std::cerr << ") must be equal to the size of the global dimension array ("
        << nd << ")" << std::endl;

    return EXIT_FAILURE;
  }

  if (rec_order != NULL && nd != rec_order->size()) {
    std::cerr << "Error: The size of the rec_order array (" << rec_order->size();
    std::cerr << ") must be equal to the size of the processor grid ("
        << nd << ")" << std::endl;

    return EXIT_FAILURE;
  }

  ////////////////////////////////////////////////////////
  // Make sure the subs begin and end arrays make sense //
  ////////////////////////////////////////////////////////
  for(int i=0; i<nd; i++) {
    if((*subs_begin)[i] < 0) {
      std::cerr << "Error: subs_begin[" << i << "] = "
          << (*subs_begin)[i] << " < 0\n";

      return EXIT_FAILURE;
    }

    if((*subs_begin)[i] > (*subs_end)[i]) {
      std::cerr << "Error: subs_begin[" << i << "] = "
          << (*subs_begin)[i] << " > subs_end[" << i << "] = "
          << (*subs_end)[i] << std::endl;

      return EXIT_FAILURE;
    }

    if((*subs_end)[i] >= (*I_dims)[i]) {
      std::cerr << "Error: subs_end[" << i << "] = "
          << (*subs_end)[i] << " >= I_dims[" << i << "] = "
          << (*I_dims)[i] << std::endl;

      return EXIT_FAILURE;
    }
  }

  ////////////////////////////////////
  // Read the core size from a file //
  ////////////////////////////////////
  Tucker::SizeArray* coreSize = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(nd);
  std::string dimFilename = sthosvd_dir + "/" + sthosvd_fn +
      "_ranks.txt";
  std::ifstream ifs(dimFilename);

  if(!ifs.is_open()) {
    std::cerr << "Failed to open core size file: " << dimFilename
              << std::endl;

    return EXIT_FAILURE;
  }

  for(int mode=0; mode<nd; mode++) {
    ifs >> (*coreSize)[mode];
  }
  ifs.close();

  //////////////////////////////////////////////
  // Make sure the core size data makes sense //
  //////////////////////////////////////////////
  for(int i=0; i<nd; i++) {
    if((*coreSize)[i] <= 0) {
      std::cerr << "coreSize[" << i << "] = " << (*coreSize)[i]
                << " <= 0\n";

      return EXIT_FAILURE;
    }

    if((*coreSize)[i] > (*I_dims)[i]) {
      std::cerr << "coreSize[" << i << "] = " << (*coreSize)[i]
                << " > I_dims[" << (*I_dims)[i] << std::endl;

      return EXIT_FAILURE;
    }
  }

  ////////////////////////////////////////////////////////////
  // Create the optimal reconstruction order if unspecified //
  ////////////////////////////////////////////////////////////
  if(rec_order == NULL) {
    // Create the SizeArray
    rec_order = Tucker::MemoryManager::safe_new<Tucker::SizeArray>(nd);
    for(int i=0; i<nd; i++) {
      (*rec_order)[i] = i;
    }

    // Compute the ratios of reconstructed size to core size
    int* rec_size = Tucker::MemoryManager::safe_new_array<int>(nd);
    double* ratios = Tucker::MemoryManager::safe_new_array<double>(nd);
    for(int i=0; i<nd; i++) {
      rec_size[i] = 1 + (*subs_end)[i] - (*subs_begin)[i];
      ratios[i] = (double)rec_size[i] / (*coreSize)[i];
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

    std::cout << "Reconstruction order: " << *rec_order << std::endl;

    // Free the memory
    Tucker::MemoryManager::safe_delete_array<int>(rec_size,nd);
    Tucker::MemoryManager::safe_delete_array<double>(ratios,nd);
  }

  //////////////////////////////////////////////////////////
  // Make sure the reconstruction order array makes sense //
  //////////////////////////////////////////////////////////
  for(int i=0; i<nd; i++) {
    if((*rec_order)[i] < 0) {
      std::cerr << "Error: rec_order[" << i << "] = "
          << (*rec_order)[i] << " < 0\n";

      return EXIT_FAILURE;
    }

    if((*rec_order)[i] >= nd) {
        std::cerr << "Error: rec_order[" << i << "] = "
            << (*rec_order)[i] << " >= nd = " << nd << std::endl;

      return EXIT_FAILURE;
    }

    for(int j=i+1; j<nd; j++) {
      if((*rec_order)[i] == (*rec_order)[j]) {
        std::cerr << "Error: rec_order[" << i << "] == rec_order["
            << j << "] = " << (*rec_order)[i] << std::endl;

        return EXIT_FAILURE;
      }
    }
  }

  /////////////////////////////////
  // Set up factorization object //
  /////////////////////////////////
  Tucker::TuckerTensor* fact =
      Tucker::MemoryManager::safe_new<Tucker::TuckerTensor>(nd);

  ///////////////////////////
  // Read core tensor data //
  ///////////////////////////
  Tucker::Timer* readTimer = Tucker::MemoryManager::safe_new<Tucker::Timer>();
  readTimer->start();
  std::string coreFilename = sthosvd_dir + "/" + sthosvd_fn +
            "_core.mpi";
  fact->G = Tucker::MemoryManager::safe_new<Tucker::Tensor>(*coreSize);
  Tucker::importTensorBinary(coreFilename.c_str(),fact->G);

  size_t nnz = fact->G->getNumElements();
  std::cout << "Core tensor size: " << fact->G->size() << ", or ";
  Tucker::printBytes(nnz*sizeof(double));

  //////////////////////////
  // Read factor matrices //
  //////////////////////////
  for(int mode=0; mode<nd; mode++)
  {
    std::ostringstream ss;
    ss << sthosvd_dir << "/" << sthosvd_fn << "_mat_" << mode << ".mpi";

    fact->U[mode] = Tucker::MemoryManager::safe_new<Tucker::Matrix>((*I_dims)[mode],(*coreSize)[mode]);
    Tucker::importTensorBinary(ss.str().c_str(), fact->U[mode]);
  }
  readTimer->stop();
  std::cout << "Time spent reading: " << readTimer->duration() << "s\n";
  Tucker::MemoryManager::safe_delete<Tucker::Timer>(readTimer);

  ////////////////////////////////////////////////////
  // Reconstruct the requested pieces of the tensor //
  ////////////////////////////////////////////////////
  Tucker::Timer* reconstructTimer = Tucker::MemoryManager::safe_new<Tucker::Timer>();
  reconstructTimer->start();
  Tucker::Tensor* result = fact->G;
  for(int i=0; i<nd; i++)
  {
    int mode = (*rec_order)[i];
    // Grab the requested rows of the factor matrix
    int start_subs = (*subs_begin)[mode];
    int end_subs = (*subs_end)[mode];
    Tucker::Matrix* factMat =
        fact->U[mode]->getSubmatrix(start_subs, end_subs);

    // Perform the TTM
    Tucker::Tensor* temp = Tucker::ttm(result,mode,factMat);

    Tucker::MemoryManager::safe_delete<Tucker::Matrix>(factMat);
    if(result != fact->G) {
      Tucker::MemoryManager::safe_delete<Tucker::Tensor>(result);
    }
    result = temp;

    size_t nnz = result->getNumElements();
    std::cout << "Tensor size after reconstruction iteration "
        << i << ": " << result->size() << ", or ";
    Tucker::printBytes(nnz*sizeof(double));
  }
  reconstructTimer->stop();
  std::cout << "Time spent reconstructing: " << readTimer->duration() << "s\n";
  Tucker::MemoryManager::safe_delete<Tucker::Timer>(reconstructTimer);

  ////////////////////////////////////////////
  // Write the reconstructed tensor to disk //
  ////////////////////////////////////////////
  Tucker::Timer* writeTimer = Tucker::MemoryManager::safe_new<Tucker::Timer>();
  writeTimer->start();
  Tucker::writeTensorBinary(result, out_fns_file.c_str());
  writeTimer->stop();
  std::cout << "Time spent writing: " << writeTimer->duration() << "s\n";
  Tucker::MemoryManager::safe_delete<Tucker::Timer>(writeTimer);

  /////////////////
  // Free memory //
  /////////////////
  Tucker::MemoryManager::safe_delete<Tucker::TuckerTensor>(fact);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(coreSize);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(I_dims);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(subs_begin);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(subs_end);
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(rec_order);
  Tucker::MemoryManager::safe_delete<Tucker::Tensor>(result);

  if(Tucker::MemoryManager::curMemUsage > 0) {
    Tucker::MemoryManager::printCurrentMemUsage();
    return EXIT_FAILURE;
  }

  Tucker::MemoryManager::printMaxMemUsage();

  return EXIT_SUCCESS;
}
