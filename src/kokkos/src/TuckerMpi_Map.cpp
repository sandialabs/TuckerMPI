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
  if (a.removedEmptyProcs_ != b.removedEmptyProcs_){ return false; }

  return true;
}

bool operator!=(const Map& a, const Map& b){
  return !(a==b);
}

Map::Map(int globalNumEntries, const MPI_Comm& comm) :
  comm_(new MPI_Comm(comm)),
  globalNumEntries_(globalNumEntries),
  removedEmptyProcs_(false)
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
  globalNumEntries_(globalNumEntries),
  removedEmptyProcs_(false)
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

void Map::removeEmptyProcs()
{
  // Determine which processes are empty
  std::vector<int> emptyProcs;
  for(std::size_t rank=0; rank<numElementsPerProc_.size(); rank++) {
    if(numElementsPerProc_[rank] == 0) {
      emptyProcs.push_back(rank);
    }
  }

  // If none are empty, there's nothing to be done
  if(emptyProcs.size() == 0) {
    return;
  }

  // Remove those from numElementsPerProc
  std::size_t newNumProcs = numElementsPerProc_.size() - emptyProcs.size();
  std::size_t i=0;
  int src=0;
  assert(newNumProcs <= std::numeric_limits<std::size_t>::max());
  std::vector<int> newSize((int)newNumProcs);
  for(int dest=0; dest<(int)newNumProcs; dest++) {
    while(i < emptyProcs.size() && src == emptyProcs[i]) {
      src++;
      i++;
    }
    newSize[dest] = numElementsPerProc_[src];
    src++;
  }
  numElementsPerProc_ = newSize;

  // Remove them from offsets too
  std::vector<int> newOffsets((int)newNumProcs);
  i=0;
  src=0;
  for(int dest=0; dest<(int)newNumProcs; dest++) {
    while(i < emptyProcs.size() && src == emptyProcs[i]) {
      src++;
      i++;
    }
    newOffsets[dest] = offsets_[src];
    src++;
  }
  offsets_ = newOffsets;

  assert(emptyProcs.size() <= std::numeric_limits<std::size_t>::max());

  // Remove them from the communicator too
  MPI_Group old_group, new_group;
  MPI_Comm new_comm;
  MPI_Comm_group(*comm_, &old_group);
  MPI_Group_excl (old_group, (int)emptyProcs.size(),
      emptyProcs.data(), &new_group);
  MPI_Comm_create (*comm_, new_group, &new_comm);
  *comm_ = new_comm;
  removedEmptyProcs_ = true;
}

}
