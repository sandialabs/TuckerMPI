#include "TuckerMpi_ProcessorGrid.hpp"
#include <iostream>

namespace TuckerMpi {

bool operator==(const ProcessorGrid& a, const ProcessorGrid& b)
{
  if (a.size_ != b.size_){ return false; }
  if (!impl::comms_equal(a.cartComm_,b.cartComm_)){ return false; }
  if (!impl::stdvectors_of_comms_equal(a.rowcomms_, b.rowcomms_)){ return false; }
  if (!impl::stdvectors_of_comms_equal(a.colcomms_, b.colcomms_)){ return false; }

  return true;
}

bool operator!=(const ProcessorGrid& a, const ProcessorGrid& b){
  return !(a==b);
}

ProcessorGrid::ProcessorGrid(const std::vector<int>& sz,
                             const MPI_Comm& comm)
  : size_(sz.size()),
    cartComm_(new MPI_Comm)
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
  std::fill(periods.begin(), periods.end(), 1);
  int reorder = 0;
  MPI_Comm cc;
  MPI_Cart_create(comm, ndims, (int*)sz.data(), periods.data(), reorder, &cc);
  *cartComm_ = cc;

  // Allocate memory for subcommunicators
  rowcomms_ = std::vector<MPI_Comm>(ndims);
  colcomms_ = std::vector<MPI_Comm>(ndims);

  // Get the subcommunicators
  std::vector<int> remainDims(ndims);
  std::fill(remainDims.begin(), remainDims.end(), 0);
  for(int i=0; i<ndims; i++){
    remainDims[i] = 1;
    MPI_Cart_sub(*cartComm_, remainDims.data(), &(colcomms_[i]));
    remainDims[i] = 0;
  }

  std::fill(remainDims.begin(), remainDims.end(), 1);
  for(int i=0; i<ndims; i++){
    remainDims[i] = 0;
    MPI_Cart_sub(*cartComm_, remainDims.data(), &(rowcomms_[i]));
    remainDims[i] = 1;
  }
}

}
