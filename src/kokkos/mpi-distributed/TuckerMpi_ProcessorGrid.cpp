#include "TuckerMpi_ProcessorGrid.hpp"
#include "Tucker_Util.hpp"

namespace TuckerMpiDistributed {

ProcessorGrid::ProcessorGrid(const Tucker::SizeArray& sz,
    const MPI_Comm& comm) :
        size_(sz.size()),
        squeezed_(false),
        cartComm_squeezed_(MPI_COMM_NULL),
        rowcomms_squeezed_(0),
        colcomms_squeezed_(0)

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
  int* periods = std::vector<int>(ndims);
  for(int i=0; i<ndims; i++) periods[i] = 1;
  int reorder = 0;
  MPI_Cart_create(comm, ndims, (int*)sz.data(), periods,
      reorder, &cartComm_);

  // Allocate memory for subcommunicators
  rowcomms_ = std::vector<MPI_Comm>(ndims);
  colcomms_ = std::vector<MPI_Comm>(ndims);

  // Get the subcommunicators
  int* remainDims = std::vector<int>(ndims);
  for(int i=0; i<ndims; i++) remainDims[i] = 0;
  for(int i=0; i<ndims; i++)
  {
    remainDims[i] = 1;
    MPI_Cart_sub(cartComm_, remainDims, &(colcomms_[i]));
    remainDims[i] = 0;
  }

  for(int i=0; i<ndims; i++) remainDims[i] = 1;
  for(int i=0; i<ndims; i++)
  {
    remainDims[i] = 0;
    MPI_Cart_sub(cartComm_, remainDims, &(rowcomms_[i]));
    remainDims[i] = 1;
  }
}

const MPI_Comm& ProcessorGrid::getComm(bool squeezed) const
{
  if(squeezed && squeezed_) {
    return cartComm_squeezed_;
  }
  return cartComm_;
}

void ProcessorGrid::getCoordinates(int* coords) const
{
  int globalRank;
  MPI_Comm_rank(cartComm_, &globalRank);
  getCoordinates(coords, globalRank);
}

void ProcessorGrid::getCoordinates(int* coords, int globalRank) const
{
  int ndims = size_.size();
  MPI_Cart_coords(cartComm_, globalRank, ndims, coords);
}

const MPI_Comm& ProcessorGrid::getRowComm(const int d, bool squeezed) const
{
  if(squeezed && squeezed_) {
    return rowcomms_squeezed_[d];
  }
  return rowcomms_[d];
}

const MPI_Comm& ProcessorGrid::getColComm(const int d, bool squeezed) const
{
  if(squeezed && squeezed_) {
    return colcomms_squeezed_[d];
  }
  return colcomms_[d];
}

int ProcessorGrid::getRank(const int* coords) const
{
  int rank;
  MPI_Cart_rank(cartComm_,(int*)coords,&rank);
  return rank;
}

int ProcessorGrid::getNumProcs(int d, bool squeezed) const
{
  int nprocs;
  if(squeezed && squeezed_) {
    MPI_Comm_size(colcomms_squeezed_[d],&nprocs);
  }
  else {
    MPI_Comm_size(colcomms_[d],&nprocs);
  }
  return nprocs;
}

void ProcessorGrid::squeeze(const Tucker::SizeArray& sz, const MPI_Comm& comm)
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
  int* periods = std::vector<int>(ndims);
  for(int i=0; i<ndims; i++) periods[i] = 1;
  int reorder = 0;
  MPI_Cart_create(comm, ndims, (int*)sz.data(), periods,
      reorder, &cartComm_squeezed_);

  // Allocate memory for subcommunicators
  rowcomms_squeezed_ = std::vector<MPI_Comm>(ndims);
  colcomms_squeezed_ = std::vector<MPI_Comm>(ndims);

  // Get the subcommunicators
  int* remainDims = std::vector<int>(ndims);
  for(int i=0; i<ndims; i++) remainDims[i] = 0;
  for(int i=0; i<ndims; i++)
  {
    remainDims[i] = 1;
    MPI_Cart_sub(cartComm_squeezed_, remainDims, &(colcomms_squeezed_[i]));
    remainDims[i] = 0;
  }

  for(int i=0; i<ndims; i++) remainDims[i] = 1;
  for(int i=0; i<ndims; i++)
  {
    remainDims[i] = 0;
    MPI_Cart_sub(cartComm_squeezed_, remainDims, &(rowcomms_squeezed_[i]));
    remainDims[i] = 1;
  }
}

const Tucker::SizeArray getSizeArray() const
{
  return size_;
}

} /* namespace Tucker */
