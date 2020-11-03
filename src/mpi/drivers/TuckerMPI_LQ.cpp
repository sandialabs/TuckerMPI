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
  MPI_Init(&argc, &argv);

  //
  // Get the rank of this MPI process
  // Only rank 0 will print to stdout
  //
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

  int ndims = 4;
  Tucker::SizeArray* sz =
    Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*sz)[0] = 100; (*sz)[1] = 100; (*sz)[2] = 100; (*sz)[3] = 100;

  Tucker::SizeArray* nprocsPerDim =
    Tucker::MemoryManager::safe_new<Tucker::SizeArray>(ndims);
  (*nprocsPerDim)[0] = 1; (*nprocsPerDim)[1] = 1; (*nprocsPerDim)[2] = 1; (*nprocsPerDim)[3] = 1;
  TuckerMPI::Distribution* dist =
    Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*sz,*nprocsPerDim);

  TuckerMPI::Tensor X(dist);
  X.rand();

  if(rank == 0) {
    size_t local_nnz = X.getLocalNumEntries();
    size_t global_nnz = X.getGlobalNumEntries();
    std::cout << "Local input tensor size: " << X.getLocalSize() << ", or ";
    Tucker::printBytes(local_nnz*sizeof(double));
    std::cout << "Global input tensor size: " << X.getGlobalSize() << ", or ";
    Tucker::printBytes(global_nnz*sizeof(double));
  }

  Tucker::Timer local_qr_timer;
  Tucker::Timer tsqr_timer;
  Tucker::Timer total_timer;
  Tucker::Timer redistribute_timer;
  Tucker::Timer localqr_dcopy_timer;
  Tucker::Timer localqr_decompose_timer;
  Tucker::Timer localqr_transpose_timer;
  if(rank == 0) total_timer.start();
  Tucker::Matrix* L0 = TuckerMPI::LQ(&X, 0, &tsqr_timer, &local_qr_timer, &redistribute_timer,
    &localqr_dcopy_timer, &localqr_decompose_timer, &localqr_transpose_timer);
  if(rank == 0){
    total_timer.stop();
    std::cout << "total time used: " << total_timer.duration()
      << " \n redistribute time: " << redistribute_timer.duration()
      << "\n local qr time: " << local_qr_timer.duration() 
      << "\n local qr dcopy time: " << localqr_dcopy_timer.duration()
      << "\n local qr decompose time: " << localqr_decompose_timer.duration()
      << "\n local qr transpose time: " << localqr_transpose_timer.duration()
      << "\n tsqr time: " << tsqr_timer.duration() << std::endl;
  }

  MPI_Finalize();
  return EXIT_SUCCESS;
}
