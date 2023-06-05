#ifndef MPIKOKKOS_TUCKER_PROCESSORGRID_HPP_
#define MPIKOKKOS_TUCKER_PROCESSORGRID_HPP_

#include "mpi.h"
#include "Tucker_SizeArray.hpp"

namespace TuckerMpiDistributed {

// Defines a parallel distribution of MPI processes on a Cartesian grid
class ProcessorGrid {
public:
  ProcessorGrid(const Tucker::SizeArray& sz, const MPI_Comm& comm);
  const MPI_Comm& getComm(bool squeezed) const;                 //! Returns the MPI communicator
  const MPI_Comm& getRowComm(const int d, bool squeezed) const; //! Returns the row communicator for dimension d
  const MPI_Comm& getColComm(const int d, bool squeezed) const; //! Returns the row communicator for dimension d
  int getNumProcs(int d, bool squeezed) const;                  //! Gets the number of MPI processors in a given dimension
  void squeeze(const Tucker::SizeArray& sz, const MPI_Comm& comm);
  const Tucker::SizeArray getSizeArray() const;

  // Just for debugging >>
  void getCoordinates(int* coords) const; //! Returns the cartesian coordinates of the calling process in the grid
  void getCoordinates(int* coords, int globalRank) const;
  int getRank(const int* coords) const; //! Returns the rank of the MPI process at a given coordinate

private:
  /* If false, return the entire communicator.
   * If true, return only a subset of MPI processes:
   * the ones that were not eliminated by the squeeze function.
   */
  bool squeezed_;
  Tucker::SizeArray size_;
  MPI_Comm cartComm_;           //! MPI communicator storing the Cartesian grid information
  MPI_Comm* rowcomms_;          //! Array of row communicators
  MPI_Comm* colcomms_;          //! Array of column communicators
  MPI_Comm cartComm_squeezed_;  //! MPI communicator storing the Cartesian grid information
  MPI_Comm* rowcomms_squeezed_; //! Array of row communicators
  MPI_Comm* colcomms_squeezed_; //! Array of column communicators
};

}
#endif
