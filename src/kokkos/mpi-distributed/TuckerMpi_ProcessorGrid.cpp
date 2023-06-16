#include "TuckerMpi_ProcessorGrid.hpp"
#include <iostream>

namespace TuckerMpi {

ProcessorGrid::ProcessorGrid(const std::vector<int>& sz,
    const MPI_Comm& comm) :
        size_(sz.size()),
        squeezed_(false),
        cartComm_squeezed_(MPI_COMM_NULL),
        rowcomms_squeezed_(),
        colcomms_squeezed_()

{
  int ndims = sz.size();

  for(int i=0; i<ndims; i++) {
    size_[i] = sz[i];
  }

  // Get the number of MPI processes
  int nprocs;
  MPI_Comm_size(comm, &nprocs);

  // Assert that the number of processes is consistent
  int nprocsRequested = 1;
  for(int i=0; i<ndims; i++) {
    nprocsRequested *= sz[i];
  }
  if(nprocsRequested != nprocs) {
    std::cerr << "ERROR in ProcessorGrid constructor: the processor grid "
        << "supplied is inconsistent with the total number of processes\n";
  }

  // Create a virtual topology MPI communicator
  std::vector<int> periods(ndims);
  for(int i=0; i<ndims; i++) periods[i] = 1;
  int reorder = 0;
  MPI_Cart_create(comm, ndims, (int*)sz.data(), periods.data(),
      reorder, &cartComm_);

  // Allocate memory for subcommunicators
  rowcomms_ = std::vector<MPI_Comm>(ndims);
  colcomms_ = std::vector<MPI_Comm>(ndims);

  // Get the subcommunicators
  std::vector<int> remainDims(ndims);
  for(int i=0; i<ndims; i++) remainDims[i] = 0;
  for(int i=0; i<ndims; i++)
  {
    remainDims[i] = 1;
    MPI_Cart_sub(cartComm_, remainDims.data(), &(colcomms_[i]));
    remainDims[i] = 0;
  }

  for(int i=0; i<ndims; i++) remainDims[i] = 1;
  for(int i=0; i<ndims; i++)
  {
    remainDims[i] = 0;
    MPI_Cart_sub(cartComm_, remainDims.data(), &(rowcomms_[i]));
    remainDims[i] = 1;
  }
}


void ProcessorGrid::squeeze(const std::vector<int>& sz, const MPI_Comm& comm)
{
  squeezed_ = true;
  int ndims = size_.size();

  // Get the number of MPI processes
  int nprocs;
  MPI_Comm_size(comm, &nprocs);

  // Assert that the number of processes is consistent
  int nprocsRequested = 1;
  for(int i=0; i<ndims; i++) {
    nprocsRequested *= sz[i];
  }
  if(nprocsRequested != nprocs) {
    std::cerr << "ERROR in ProcessorGrid::squeeze: the processor grid "
        << "supplied is inconsistent with the total number of processes\n";
  }

  // Create a virtual topology MPI communicator
  std::vector<int> periods(ndims);
  for(int i=0; i<ndims; i++) periods[i] = 1;
  int reorder = 0;
  MPI_Cart_create(comm, ndims, (int*)sz.data(), periods.data(),
      reorder, &cartComm_squeezed_);

  // Allocate memory for subcommunicators
  std::vector<MPI_Comm> rowcomms_squeezed_(ndims);
  std::vector<MPI_Comm> colcomms_squeezed_(ndims);

  // Get the subcommunicators
  std::vector<int> remainDims(ndims);
  for(int i=0; i<ndims; i++) remainDims[i] = 0;
  for(int i=0; i<ndims; i++)
  {
    remainDims[i] = 1;
    MPI_Cart_sub(cartComm_squeezed_, remainDims.data(), &(colcomms_squeezed_[i]));
    remainDims[i] = 0;
  }

  for(int i=0; i<ndims; i++) remainDims[i] = 1;
  for(int i=0; i<ndims; i++)
  {
    remainDims[i] = 0;
    MPI_Cart_sub(cartComm_squeezed_, remainDims.data(), &(rowcomms_squeezed_[i]));
    remainDims[i] = 1;
  }
}

} /* namespace Tucker */
