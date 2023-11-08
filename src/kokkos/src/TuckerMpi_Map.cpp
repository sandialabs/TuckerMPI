#include "TuckerMpi_Map.hpp"
#include <cassert>
#include <limits>

namespace TuckerMpi {

bool operator==(const Map& a, const Map& b)
{
  // comparison operators for shared_ptr simply compare pointer values;
  // the actual objects pointed to are not
  if (!impl::comms_equal(a.comm_,b.comm_)){ return false; }
  if (a.numElementsPerProc_ != b.numElementsPerProc_){ return false; }
  if (a.offsets_ != b.offsets_){ return false; }
  if (a.indexBegin_ != b.indexBegin_){ return false;}
  if (a.indexEnd_ != b.indexEnd_){ return false;}
  if (a.localNumEntries_ != b.localNumEntries_){ return false; }
  if (a.globalNumEntries_ != b.globalNumEntries_){ return false; }
  return true;
}

bool operator!=(const Map& a, const Map& b){
  return !(a==b);
}

Map::Map(int globalNumEntries, const MPI_Comm& comm) :
  comm_(new MPI_Comm(comm)),
  globalNumEntries_(globalNumEntries)
{
  // Get the number of MPI processes
  int myRank, nprocs;
  MPI_Comm_rank(comm,&myRank);
  MPI_Comm_size(comm,&nprocs);

  // Assert that the global number of entries is bigger
  // than the number of MPI processes
  // assert(globalNumEntries > nprocs);

  // Determine the number of entries owned by each process
  numElementsPerProc_.resize(nprocs);

  for(int rank=0; rank<nprocs; rank++) {
    numElementsPerProc_[rank] = globalNumEntries/nprocs;
    if(rank < globalNumEntries%nprocs)
      numElementsPerProc_[rank]++;
  }

  // Determine the row offsets for each process
  offsets_.resize(nprocs+1);

  offsets_[0] = 0;
  for(int rank=1; rank<=nprocs; rank++) {
    offsets_[rank] = offsets_[rank-1] + numElementsPerProc_[rank-1];
  }

  // Determine the starting and ending indices for THIS process
  indexBegin_ = offsets_[myRank];
  indexEnd_ = offsets_[myRank+1]-1;
  localNumEntries_ = 1 + indexEnd_ - indexBegin_;
}

Map::Map(int globalNumEntries, int localNumEntries, const MPI_Comm& comm) :
  comm_(new MPI_Comm(comm)),
  localNumEntries_(localNumEntries),
  globalNumEntries_(globalNumEntries)
{
  // Get the number of MPI processes
  int myRank, nprocs;
  MPI_Comm_rank(comm,&myRank);
  MPI_Comm_size(comm,&nprocs);

  // Assert that the global number of entries is bigger
  // than the number of MPI processes
  // assert(globalNumEntries > nprocs);

  // Determine the number of entries owned by each process
  numElementsPerProc_.resize(nprocs);
  MPI_Allgather(
    &localNumEntries_, 1, MPI_INT, numElementsPerProc_.data(), 1, MPI_INT, comm);

  // Determine the row offsets for each process
  offsets_.resize(nprocs+1);

  offsets_[0] = 0;
  for(int rank=1; rank<=nprocs; rank++) {
    offsets_[rank] = offsets_[rank-1] + numElementsPerProc_[rank-1];
  }

  // Determine the starting and ending indices for THIS process
  indexBegin_ = offsets_[myRank];
  indexEnd_ = offsets_[myRank+1]-1;
}

}
