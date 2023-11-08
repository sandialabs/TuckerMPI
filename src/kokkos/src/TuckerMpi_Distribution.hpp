#ifndef TUCKERMPI_DISTRIBUTION_HPP_
#define TUCKERMPI_DISTRIBUTION_HPP_

#include "TuckerMpi_Map.hpp"
#include "TuckerMpi_ProcessorGrid.hpp"
#include <memory>

namespace TuckerMpi{

//! Describes the distribution of a Cartesian grid over a set of MPI processes
class Distribution {

public:
  Distribution() = default;

  Distribution(const std::vector<int>& dims,
               const std::vector<int>& procs);

  Distribution(const std::vector<int>& dims,
               const ProcessorGrid& grid);

  //! Returns the dimensions of the locally owned portion of the N-dimensional grid
  const std::vector<int>& getLocalDims() const{ return localDims_; }

  //! Returns the dimensions of the N-dimensional grid
  const std::vector<int>& getGlobalDims() const{ return globalDims_; }

  //! Returns the processor grid
  const ProcessorGrid& getProcessorGrid() const{ return grid_; }

  //! Returns the map of a given dimension
  const Map* getMap(int dimension) const{ return &maps_[dimension]; }

  const MPI_Comm& getComm() const{ return grid_.getComm(); }

  bool ownNothing() const{
    return ownNothing_;
  }

  bool empty() const { return empty_; }

  // Returns a new distribution by adding `p` slices along mode `mode` on each
  // processor, where p may be different on each processor.  The resulting map
  // on each processor is a superset of the prior map so that the resulting
  // tensor is a superset of the prior tensor on each processor
  Distribution growAlongMode(int mode, int p) const;

  // Returns a new distribution by replacing the global dimension of mode
  // `mode` with `R`.  All other dimensions are kept the same and the
  // distribution across mode `mode` uses the default algorithm
  Distribution replaceModeWithGlobalSize(int mode, int R) const;

  // Returns a new distribution by replacing the global dimension of mode
  // `mode` with `R_global` and local dimension with `R_local'.  All other
  // dimensions are kept the same
  Distribution replaceModeWithSizes(int mode, int R_global, int R_local) const;

  /*
   * operators overloading
   */
  friend bool operator==(const Distribution&, const Distribution&);
  friend bool operator!=(const Distribution&, const Distribution&);

private:
  void createMaps();

private:
  //! Size of the local grid; number of entries owned in each dimension.
  std::vector<int> localDims_ = {};

  //! The global Cartesian grid size
  std::vector<int> globalDims_ = {};

  //! Maps MPI processes to a grid
  ProcessorGrid grid_ = {};

  //! The maps describing the parallel distribution in each dimension
  std::vector<Map> maps_ = {};

  bool ownNothing_ = false;
  bool empty_ = true;
};

bool operator==(const Distribution& a, const Distribution& b);
bool operator!=(const Distribution& a, const Distribution& b);

}
#endif  // TUCKERMPI_DISTRIBUTION_HPP_
