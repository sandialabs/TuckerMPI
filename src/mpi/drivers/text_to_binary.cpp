/*
 * text_to_binary.cpp
 *
 *  Created on: Jun 28, 2016
 *      Author: amklinv
 */

#include<mpi.h>
#include<stdlib.h>
#include "Tucker.hpp"

int main(int argc, char* argv[])
{
  // Initialize MPI
  MPI_Init(&argc,&argv);

  // Ensure that the number of arguments is correct
  if(argc < 3) {
    std::cerr << "ERROR: Must provide at least two arguments for input and output filenames\n";
    exit(EXIT_FAILURE);
  }

  // Read the tensor from a text file
  Tucker::Tensor* t = Tucker::importTensor(argv[1]);

  // Open the file for writing
  int ret;
  MPI_File fh;
  ret = MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not open file " << argv[2] << std::endl;
  }

  // Write the tensor to a binary file
  MPI_Status status;
  size_t nentries = t->size().prod();
  assert(nentries <= std::numeric_limits<int>::max());
  double* entries = t->data();
  ret = MPI_File_write(fh, entries, (int)nentries, MPI_DOUBLE, &status);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not write file " << argv[2] << std::endl;
  }

  // Close the file
  MPI_File_close(&fh);

  // Free memory
  Tucker::MemoryManager::safe_delete<Tucker::Tensor>(t);

  // Finalize MPI
  MPI_Finalize();

  return EXIT_SUCCESS;
}
