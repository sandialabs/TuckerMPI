#ifndef MPIKOKOS_TUCKER_DISTRIBUTION_HPP_
#define MPIKOKOS_TUCKER_DISTRIBUTION_HPP_

#include <memory>
#include "TuckerMpi_Map.hpp"
#include "TuckerMpi_ProcessorGrid.hpp"

namespace TuckerMpiDistributed {

//! Describes the distribution of a Cartesian grid over a set of MPI processes
class Distribution {
public:
  Distribution(const Tucker::SizeArray& dims,
    const Tucker::SizeArray& procs);
  const Tucker::SizeArray& getLocalDims() const;          //! Returns the dimensions of the locally owned portion of the N-dimensional grid  
  const Tucker::SizeArray& getGlobalDims() const;         //! Returns the dimensions of the N-dimensional grid
  const ProcessorGrid & getProcessorGrid() const;         //! Returns the processor grid
  const Map* getMap(int dimension, bool squeezed) const;  //! Returns the map of a given dimension
  const MPI_Comm& getComm(bool squeezed) const;
  bool ownNothing() const;

private:
  void createMaps();
  void findAndEliminateEmptyProcs(MPI_Comm& comm);  //! Finds and eliminates processes that don't have work
  void updateProcessorGrid(const MPI_Comm& comm);   //! Creates new processor grid without the processes that don't have work
  Tucker::SizeArray localDims_;                     //! Size of the local grid; number of entries owned in each dimension.
  Tucker::SizeArray globalDims_;                    //! The global Cartesian grid size
  ProcessorGrid grid_;                              //! Maps MPI processes to a grid
  //! The maps describing the parallel distribution in each dimension
  std::vector<Map> maps_;
  std::vector<Map> maps_squeezed_;
  bool ownNothing_;
  bool squeezed_;
};

} /* namespace TuckerMpiDistributed */
#endif /* MPIKOKOS_TUCKER_DISTRIBUTION_HPP_ */
