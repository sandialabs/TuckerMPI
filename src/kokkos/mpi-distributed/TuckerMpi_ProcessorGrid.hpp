#ifndef MPIKOKKOS_TUCKER_PROCESSORGRID_HPP_
#define MPIKOKKOS_TUCKER_PROCESSORGRID_HPP_

#include "mpi.h"
#include <vector>

namespace TuckerMpiDistributed {

class ProcessorGrid {
public:
  ProcessorGrid() = default;
  ProcessorGrid(const std::vector<int>& sz, const MPI_Comm& comm);

  const MPI_Comm& getComm(bool squeezed) const;                 //! Returns the MPI communicator
  const MPI_Comm& getRowComm(int d, bool squeezed) const; //! Returns the row communicator for dimension d
  const MPI_Comm& getColComm(int d, bool squeezed) const; //! Returns the row communicator for dimension d
  int getNumProcs(int d, bool squeezed) const;                  //! Gets the number of MPI processors in a given dimension
  void squeeze(const std::vector<int>& sz, const MPI_Comm& comm);
  const std::vector<int> & getSizeArray() const;

  // Just for debugging >>
  int getRank(const std::vector<int> & coords) const;   //! Returns the rank of the MPI process at a given coordinate
  void getCoordinates(std::vector<int> & coords) const; //! Returns the cartesian coordinates of the calling process in the grid
  void getCoordinates(std::vector<int> & coords, int globalRank) const;

private:
  bool squeezed_;
  std::vector<int> size_;
  MPI_Comm cartComm_;                       //! MPI communicator storing the Cartesian grid information
  std::vector<MPI_Comm> rowcomms_;          //! Array of row communicators
  std::vector<MPI_Comm> colcomms_;          //! Array of column communicators
  MPI_Comm cartComm_squeezed_;              //! MPI communicator storing the Cartesian grid information
  std::vector<MPI_Comm> rowcomms_squeezed_; //! Array of row communicators
  std::vector<MPI_Comm> colcomms_squeezed_; //! Array of column communicators
};

}
#endif
