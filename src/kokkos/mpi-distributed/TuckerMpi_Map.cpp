#include "TuckerMpi_Map.hpp"
#include <cassert>
#include <limits>

namespace TuckerMpiDistributed {

Map::Map(int globalNumEntries, const MPI_Comm& comm) :
  comm_(comm),
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
  numElementsPerProc_ = Tucker::SizeArray(nprocs);

  for(int rank=0; rank<nprocs; rank++) {
    numElementsPerProc_[rank] = globalNumEntries/nprocs;
    if(rank < globalNumEntries%nprocs)
      numElementsPerProc_[rank]++;
  }

  // Determine the row offsets for each process
  offsets_ = Tucker::SizeArray(nprocs+1);

  offsets_[0] = 0;
  for(int rank=1; rank<=nprocs; rank++) {
    offsets_[rank] = offsets_[rank-1] + numElementsPerProc_[rank-1];
  }

  // Determine the starting and ending indices for THIS process
  indexBegin_ = offsets_[myRank];
  indexEnd_ = offsets_[myRank+1]-1;

  localNumEntries_ = 1 + indexEnd_ - indexBegin_;
}

int Map::getLocalIndex(int globalIndex) const
{
  if(globalIndex > indexEnd_ || globalIndex < indexBegin_) {
    return -1;
  }
  return globalIndex - indexBegin_;
}

int Map::getGlobalIndex(int localIndex) const
{
  if(localIndex < 0 || localIndex >= localNumEntries_) {
    return -1;
  }
  return indexBegin_+localIndex;
}

int Map::getLocalNumEntries() const
{
  return localNumEntries_;
}

int Map::getGlobalNumEntries() const
{
  return globalNumEntries_;
}

int Map::getMaxNumEntries() const
{
  int maxNumEntries = 0;
  for(int i=0; i<numElementsPerProc_.size(); i++) {
    if(numElementsPerProc_[i] > maxNumEntries) {
      maxNumEntries = numElementsPerProc_[i];
    }
  }
  return maxNumEntries;
}

int Map::getNumEntries(int rank) const
{
  return numElementsPerProc_[rank];
}

int Map::getOffset(int rank) const
{
  return offsets_[rank];
}

const MPI_Comm& Map::getComm() const
{
  return comm_;
}

void Map::removeEmptyProcs()
{
  // Determine which processes are empty
  std::vector<int> emptyProcs;
  for(int rank=0; rank<numElementsPerProc_.size(); rank++) {
    if(numElementsPerProc_[rank] == 0) {
      emptyProcs.push_back(rank);
    }
  }

  // If none are empty, there's nothing to be done
  if(emptyProcs.size() == 0) {
    return;
  }

  // Remove those from numElementsPerProc
  size_t newNumProcs = numElementsPerProc_.size() - emptyProcs.size();
  size_t i=0;
  int src=0;
  assert(newNumProcs <= std::numeric_limits<int>::max());
  Tucker::SizeArray newSize = Tucker::SizeArray((int)newNumProcs);
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
  Tucker::SizeArray newOffsets = Tucker::SizeArray((int)newNumProcs);
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

  assert(emptyProcs.size() <= std::numeric_limits<int>::max());

  // Remove them from the communicator too
  MPI_Group old_group, new_group;
  MPI_Comm new_comm;
  MPI_Comm_group(comm_, &old_group);
  MPI_Group_excl (old_group, (int)emptyProcs.size(),
      emptyProcs.data(), &new_group);
  MPI_Comm_create (comm_, new_group, &new_comm);
  comm_ = new_comm;
  removedEmptyProcs_ = true;
}

} /* namespace TuckerMpiDistributed */
