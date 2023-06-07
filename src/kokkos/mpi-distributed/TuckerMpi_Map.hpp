#ifndef MPIKOKKOS_TUCKER_MAP_HPP_
#define MPIKOKKOS_TUCKER_MAP_HPP_

#include "mpi.h"
#include "Tucker_SizeArray.hpp"

namespace TuckerMpiDistributed {

class Map {

public:
  Map() = default;
  Map(int globalNumEntries, const MPI_Comm& comm);
  int getLocalIndex(int globalIndex) const;
  int getGlobalIndex(int localIndex) const;
  int getLocalNumEntries() const;
  int getGlobalNumEntries() const;
  int getMaxNumEntries() const;
  int getNumEntries(int rank) const;
  int getOffset(int rank) const;
  const MPI_Comm& getComm() const;
  void removeEmptyProcs();

private:
  //! MPI communicator
  MPI_Comm comm_;
  //! Number of elements owned by each process
  Tucker::SizeArray numElementsPerProc_;
  //! Offset/displacement array
  Tucker::SizeArray offsets_;
  //! First index owned by this MPI process
  int indexBegin_;
  //! Last index owned by this MPI process
  int indexEnd_;
  //! Number of entries owned by this MPI process
  int localNumEntries_;
  //! Total number of entries
  int globalNumEntries_;
  bool removedEmptyProcs_;
};

} /* namespace TuckerMpiDistributed */
#endif /* MPIKOKKOS_TUCKER_MAP_HPP_ */
