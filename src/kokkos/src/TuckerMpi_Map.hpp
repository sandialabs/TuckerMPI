#ifndef TUCKERMPI_MAP_HPP_
#define TUCKERMPI_MAP_HPP_

#include "./impl/TuckerMpi_MPIWrapper.hpp"
#include "mpi.h"
#include <vector>
#include <cassert>
#include <memory>

namespace TuckerMpi {

class Map {
public:
  Map() = default;
  Map(int globalNumEntries, const MPI_Comm& comm);

  bool hasGlobalIndex(int globalIndex) const{
    return (globalIndex >= indexBegin_ && globalIndex <= indexEnd_);
  }

  int getLocalIndex(int globalIndex) const{
    assert(hasGlobalIndex(globalIndex));
    return globalIndex - indexBegin_;
  }

  int getGlobalIndex(int localIndex) const{
    assert(localIndex >= 0 && localIndex < localNumEntries_);
    return indexBegin_+localIndex;
  }

  int getLocalNumEntries() const{ return localNumEntries_; }
  int getGlobalNumEntries() const{ return globalNumEntries_; }

  int getMaxNumEntries() const{
    int maxNumEntries = 0;
    for(std::size_t i=0; i<numElementsPerProc_.size(); i++) {
      if(numElementsPerProc_[i] > maxNumEntries) {
	maxNumEntries = numElementsPerProc_[i];
      }
    }
    return maxNumEntries;
  }

  int getNumEntries(int rank) const{
    return numElementsPerProc_.empty() ? 0 : numElementsPerProc_[rank];
  }

  const auto & getOffsets() const{
    return offsets_;
  }

  int getOffset(int rank) const{
    return offsets_.empty() ? 0 : offsets_[rank];
  }

  const auto & getNumElementsPerProc() const{
    return numElementsPerProc_;
  }

  const MPI_Comm& getComm() const{ return *comm_; }
  void removeEmptyProcs();

  friend bool operator==(const Map&, const Map&);
  friend bool operator!=(const Map&, const Map&);

private:
  //! MPI communicator
  std::shared_ptr<MPI_Comm> comm_{new MPI_Comm(MPI_COMM_NULL)};
  //! Number of elements owned by each process
  std::vector<int> numElementsPerProc_ = {};
  //! Offset/displacement array
  std::vector<int> offsets_ = {};
  //! First index owned by this MPI process
  int indexBegin_ = {};
  //! Last index owned by this MPI process
  int indexEnd_ = {};
  //! Number of entries owned by this MPI process
  int localNumEntries_ = {};
  //! Total number of entries
  int globalNumEntries_ = {};
  bool removedEmptyProcs_ = {};
};

bool operator==(const Map& a, const Map& b);
bool operator!=(const Map& a, const Map& b);

}
#endif  // TUCKERMPI_MAP_HPP_
