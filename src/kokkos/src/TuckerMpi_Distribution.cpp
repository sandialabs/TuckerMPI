#include <algorithm>
#include "TuckerMpi_Distribution.hpp"
#include "mpi.h"
#include <cassert>
#include <limits>

namespace TuckerMpi{

bool operator==(const Distribution& a, const Distribution& b)
{
  if (a.localDims_     != b.localDims_){ return false; }
  if (a.globalDims_    != b.globalDims_){ return false; }
  if (a.grid_          != b.grid_){ return false; }
  if (a.maps_          != b.maps_){ return false; }
  if (a.maps_squeezed_ != b.maps_squeezed_){ return false; }
  if (a.ownNothing_    != b.ownNothing_){ return false; }
  if (a.squeezed_      != b.squeezed_){ return false; }
  if (a.empty_         != b.empty_){ return false; }

  return true;
}

bool operator!=(const Distribution& a, const Distribution& b){
  return !(a==b);
}

Distribution::Distribution(const std::vector<int>& dims,
			   const std::vector<int>& procs)
  : localDims_(dims.size()),
    globalDims_(dims),
    grid_(procs, MPI_COMM_WORLD),
    empty_(false)
{
  const int ndims = dims.size();
  createMaps();
  // Copy local dimensions to localDims_
  for(int d=0; d<ndims; d++) {
    localDims_[d] = maps_[d].getLocalNumEntries();
  }

  MPI_Comm comm;
  findAndEliminateEmptyProcs(comm);
  if(squeezed_) {
    // Create a map for each dimension
    maps_squeezed_ = std::vector<Map>(ndims);
    for(int d=0; d<ndims; d++) {
      const MPI_Comm& comm = grid_.getColComm(d,false);
      maps_squeezed_[d] = TuckerMpi::Map(globalDims_[d], comm);
    }

    // Remove the empty processes from the map communicators
    for(int i=0; i<ndims; i++) { maps_squeezed_[i].removeEmptyProcs(); }

    // Determine whether I own nothing
    ownNothing_ = false;
    for(int i=0; i<ndims; i++) {
      if(maps_[i].getLocalNumEntries() == 0) {
        ownNothing_ = true;
        break;
      }
    }

    // Re-create the processor grid without lazy processes
    if(!ownNothing_)
      updateProcessorGrid(comm);
  } // end if(squeezed_)
  else {
    ownNothing_ = false;
  }

  if(comm != MPI_COMM_WORLD && !ownNothing()) {
    MPI_Comm_free(&comm);
  }
}


void Distribution::createMaps()
{
  int ndims = globalDims_.size();

  // Create a map for each dimension
  maps_ = std::vector<Map>(ndims);
  for(int d=0; d<ndims; d++) {
    const MPI_Comm& comm = grid_.getColComm(d,false);
    maps_[d] = TuckerMpi::Map(globalDims_[d], comm);
  }
}

void Distribution::findAndEliminateEmptyProcs(MPI_Comm& newcomm)
{
  int ndims = globalDims_.size();

  // Determine which processes GET NOTHING
  std::vector<int> emptyProcs;
  for(int d=0; d<ndims; d++) {
    const MPI_Comm& comm = grid_.getColComm(d,false);
    int nprocs;
    MPI_Comm_size(comm,&nprocs);
    for(int rank=0; rank<nprocs; rank++) {
      // This part of the map is empty
      if(maps_[d].getNumEntries(rank) == 0) {
        int nprocsToAdd = 1;
        for(int i=0; i<ndims; i++) {
          if(i == d) continue;
          nprocsToAdd *= grid_.getNumProcs(i,false);
        }
        std::vector<int> coords(ndims);
        coords[d] = rank;
        for(int i=0; i<nprocsToAdd; i++) {
          int divnum = 1;
          for(int j=ndims-1; j>=0; j--) {
            if(j == d) continue;
            coords[j] = (i/divnum) % grid_.getNumProcs(j,false);
            divnum *= grid_.getNumProcs(j,false);
          }
          emptyProcs.push_back(grid_.getRank(coords));
        }
      }
    }
  }

  // Create a communicator without the slacker MPI processes
  if(emptyProcs.size() > 0) {
    std::sort(emptyProcs.begin(),emptyProcs.end());
    std::vector<int>::iterator it =
        std::unique(emptyProcs.begin(), emptyProcs.end());
    emptyProcs.resize(std::distance(emptyProcs.begin(),it));

    // Get the group corresponding to MPI_COMM_WORLD
    MPI_Group group, newgroup;
    MPI_Comm_group(MPI_COMM_WORLD, &group);

    assert(emptyProcs.size() <= std::numeric_limits<std::size_t>::max());

    // Create a new group without the slacker MPI processors
    MPI_Group_excl(group, (int)emptyProcs.size(),
        emptyProcs.data(), &newgroup);

    // Create a new MPI_Comm without the slacker MPI processes
    MPI_Comm_create (MPI_COMM_WORLD, newgroup, &newcomm);
    squeezed_ = true;
  }
  else {
    newcomm = MPI_COMM_WORLD;
  }
}

void Distribution::updateProcessorGrid(const MPI_Comm& newcomm)
{
  int ndims = globalDims_.size();

  // Count the new number of processes in each dimension
  std::vector<int> newProcs(ndims);
  for(int i=0; i<ndims; i++) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const MPI_Comm& comm = maps_squeezed_[i].getComm();
    int nprocs;
    MPI_Comm_size(comm,&nprocs);
    newProcs[i] = nprocs;
  }
  grid_.squeeze(newProcs,newcomm);
}

}
