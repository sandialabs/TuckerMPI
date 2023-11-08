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
  //if (a.maps_squeezed_ != b.maps_squeezed_){ return false; }
  //if (a.ownNothing_    != b.ownNothing_){ return false; }
  //if (a.squeezed_      != b.squeezed_){ return false; }
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
  createMaps();

  // Copy local dimensions to localDims_
  const int ndims = dims.size();
  for(int d=0; d<ndims; d++) {
    localDims_[d] = maps_[d].getLocalNumEntries();
  }

  createSqueezedMaps();
}

Distribution::Distribution(const std::vector<int>& dims,
                           const ProcessorGrid& grid)
  : localDims_(dims.size()),
    globalDims_(dims),
    grid_(grid),
    empty_(false)
{
  createMaps();

  // Copy local dimensions to localDims_
  const int ndims = dims.size();
  for(int d=0; d<ndims; d++) {
    localDims_[d] = maps_[d].getLocalNumEntries();
  }

  createSqueezedMaps();
}

Distribution Distribution::growAlongMode(int mode, int p) const
{
  Distribution new_dist;
  new_dist.empty_ = empty_;

  // We use the same processor grid
  new_dist.grid_ = grid_;

  // Check p is the same for all processors in the given slice
  int p_tot_row = 0;
  int n_row_procs = 0;
  const MPI_Comm& row_comm = grid_.getRowComm(mode,false);
  MPI_Allreduce(&p, &p_tot_row, 1, MPI_INT, MPI_SUM, row_comm);
  MPI_Comm_size(row_comm, &n_row_procs);
  if (p_tot_row != p*n_row_procs) {
    throw std::logic_error(
      "number of new slices must be the same on each row processor!");
  }

  // Compute total number of new slices
  int p_tot_col = 0;
  const MPI_Comm& col_comm = grid_.getColComm(mode,false);
  MPI_Allreduce(&p, &p_tot_col, 1, MPI_INT, MPI_SUM, col_comm);

  // New global dims
  new_dist.globalDims_ = globalDims_;
  new_dist.globalDims_[mode] += p_tot_col;

  // New local dims
  new_dist.localDims_ = localDims_;
  new_dist.localDims_[mode] += p;

  // New maps
  int ndims = globalDims_.size();
  new_dist.maps_ = std::vector<Map>(ndims);
  for (int d=0; d<ndims; d++) {
    const MPI_Comm& comm = new_dist.grid_.getColComm(d,false);
    new_dist.maps_[d] = TuckerMpi::Map(
      new_dist.globalDims_[d], new_dist.localDims_[d], comm);
  }

  // New squeezed maps
  new_dist.createSqueezedMaps();

  return new_dist;
}

Distribution Distribution::replaceModeWithGlobalSize(int mode, int R) const
{
  Distribution new_dist;
  new_dist.empty_ = empty_;

  // We use the same processor grid
  new_dist.grid_ = grid_;

  // New global dims, only changing mode `mode`
  new_dist.globalDims_ = globalDims_;
  new_dist.globalDims_[mode] = R;

  // New local dims (we'll overwrite localDims_[mode] later)
  new_dist.localDims_ = localDims_;

  // New maps, use prescribed local/global dims for all modes except mode `mode`
  int ndims = globalDims_.size();
  new_dist.maps_ = std::vector<Map>(ndims);
  for (int d=0; d<ndims; d++) {
    const MPI_Comm& comm = new_dist.grid_.getColComm(d,false);
    if (d == mode)
       new_dist.maps_[d] = TuckerMpi::Map(R, comm);
    else
      new_dist.maps_[d] = TuckerMpi::Map(
        new_dist.globalDims_[d], new_dist.localDims_[d], comm);
  }

  // Update localDims_
  new_dist.localDims_[mode] = new_dist.maps_[mode].getLocalNumEntries();

  // New squeezed maps
  new_dist.createSqueezedMaps();

  return new_dist;
}

Distribution Distribution::replaceModeWithSizes(int mode, int R_global,
                                                int R_local) const
{
  Distribution new_dist;
  new_dist.empty_ = empty_;

  // We use the same processor grid
  new_dist.grid_ = grid_;

  // New global dims, only changing mode `mode`
  new_dist.globalDims_ = globalDims_;
  new_dist.globalDims_[mode] = R_global;

  // New local dims
  new_dist.localDims_ = localDims_;
  new_dist.localDims_[mode] = R_local;

  // New maps, use prescribed local/global dims for all modes except mode `mode`
  int ndims = globalDims_.size();
  new_dist.maps_ = std::vector<Map>(ndims);
  for (int d=0; d<ndims; d++) {
    const MPI_Comm& comm = new_dist.grid_.getColComm(d,false);
    new_dist.maps_[d] = TuckerMpi::Map(
      new_dist.globalDims_[d], new_dist.localDims_[d], comm);
  }

  // New squeezed maps
  new_dist.createSqueezedMaps();

  return new_dist;
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

void Distribution::createSqueezedMaps()
{
  MPI_Comm comm;
  findAndEliminateEmptyProcs(comm);
  if(squeezed_) {
    const int ndims = globalDims_.size();

    // Create a map for each dimension
    maps_squeezed_ = std::vector<Map>(ndims);
    for(int d=0; d<ndims; d++) {
      const MPI_Comm& col_comm = grid_.getColComm(d,false);
      maps_squeezed_[d] =
        TuckerMpi::Map(globalDims_[d], localDims_[d], col_comm);
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
