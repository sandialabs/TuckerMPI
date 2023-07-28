#ifndef MPIKOKKOS_TUCKER_PROCESSORGRID_HPP_
#define MPIKOKKOS_TUCKER_PROCESSORGRID_HPP_

#include "TuckerMpi_MPIWrapper.hpp"
#include <vector>
#include <memory>

namespace TuckerMpi {

class ProcessorGrid {
public:
  ProcessorGrid() = default;
  ProcessorGrid(const std::vector<int>& sz, const MPI_Comm& comm);

  //! Returns the MPI communicator
  const MPI_Comm& getComm(bool squeezed) const{
    if(squeezed && squeezed_) { return *cartComm_squeezed_; }
    return *cartComm_;
  }

  //! Returns the row communicator for dimension d
  const MPI_Comm& getRowComm(int d, bool squeezed) const{
    if(squeezed && squeezed_) { return rowcomms_squeezed_[d]; }
    return rowcomms_[d];
  }

  //! Returns the col communicator for dimension d
  const MPI_Comm& getColComm(int d, bool squeezed) const{
    if(squeezed && squeezed_) { return colcomms_squeezed_[d]; }
    return colcomms_[d];
  }

  //! Gets the number of MPI processors in a given dimension
  int getNumProcs(int d, bool squeezed) const{
    int nprocs;
    if(squeezed && squeezed_) {
      MPI_Comm_size(colcomms_squeezed_[d],&nprocs);
    }
    else {
      MPI_Comm_size(colcomms_[d],&nprocs);
    }
    return nprocs;
  }

  void squeeze(const std::vector<int>& sz, const MPI_Comm& comm);

  const std::vector<int> & getSizeArray() const{ return size_; }

  //! Returns the rank of the MPI process at a given coordinate
  int getRank(const std::vector<int> & coords) const{
    int rank;
    MPI_Cart_rank(*cartComm_, coords.data(), &rank);
    return rank;
  }

  //! Returns the cartesian coordinates of the calling process in the grid
  void getCoordinates(std::vector<int> & coords) const{
    int globalRank;
    MPI_Comm_rank(*cartComm_, &globalRank);
    getCoordinates(coords, globalRank);
  }

  void getCoordinates(std::vector<int> & coords,
		      int globalRank) const{
    int ndims = size_.size();
    MPI_Cart_coords(*cartComm_, globalRank, ndims, coords.data());
  }

  /*
   * operators overloading
   */
  friend bool operator==(const ProcessorGrid&, const ProcessorGrid&);
  friend bool operator!=(const ProcessorGrid&, const ProcessorGrid&);

private:
  bool squeezed_;
  std::vector<int> size_ = {};
  //! MPI communicator storing the Cartesian grid information
  std::shared_ptr<MPI_Comm> cartComm_{new MPI_Comm(MPI_COMM_NULL)};
  //! Array of row communicators
  std::vector<MPI_Comm> rowcomms_ = {};
  //! Array of column communicators
  std::vector<MPI_Comm> colcomms_ = {};
  //! MPI communicator storing the Cartesian grid information
  std::shared_ptr<MPI_Comm> cartComm_squeezed_{new MPI_Comm(MPI_COMM_NULL)};
  //! Array of row communicators
  std::vector<MPI_Comm> rowcomms_squeezed_ = {};
  //! Array of column communicators
  std::vector<MPI_Comm> colcomms_squeezed_= {};
};

bool operator==(const ProcessorGrid& a, const ProcessorGrid& b);
bool operator!=(const ProcessorGrid& a, const ProcessorGrid& b);

}
#endif
