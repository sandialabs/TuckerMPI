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
  if (a.ownNothing_    != b.ownNothing_){ return false; }
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
  const MPI_Comm& row_comm = grid_.getRowComm(mode);
  MPI_Allreduce(&p, &p_tot_row, 1, MPI_INT, MPI_SUM, row_comm);
  MPI_Comm_size(row_comm, &n_row_procs);
  if (p_tot_row != p*n_row_procs) {
    throw std::logic_error(
      "number of new slices must be the same on each row processor!");
  }

  // Compute total number of new slices
  int p_tot_col = 0;
  const MPI_Comm& col_comm = grid_.getColComm(mode);
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
    const MPI_Comm& comm = new_dist.grid_.getColComm(d);
    new_dist.maps_[d] = TuckerMpi::Map(
      new_dist.globalDims_[d], new_dist.localDims_[d], comm);
  }

  // Determine whether new_dist owns nothing
  new_dist.ownNothing_ = false;
  for(int i=0; i<ndims; i++) {
    if(new_dist.maps_[i].getLocalNumEntries() == 0) {
      new_dist.ownNothing_ = true;
      break;
    }
  }

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
    const MPI_Comm& comm = new_dist.grid_.getColComm(d);
    if (d == mode)
       new_dist.maps_[d] = TuckerMpi::Map(R, comm);
    else
      new_dist.maps_[d] = TuckerMpi::Map(
        new_dist.globalDims_[d], new_dist.localDims_[d], comm);
  }

  // Update localDims_
  new_dist.localDims_[mode] = new_dist.maps_[mode].getLocalNumEntries();

  // Determine whether new_dist owns nothing
  new_dist.ownNothing_ = false;
  for(int i=0; i<ndims; i++) {
    if(new_dist.maps_[i].getLocalNumEntries() == 0) {
      new_dist.ownNothing_ = true;
      break;
    }
  }

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
    const MPI_Comm& comm = new_dist.grid_.getColComm(d);
    new_dist.maps_[d] = TuckerMpi::Map(
      new_dist.globalDims_[d], new_dist.localDims_[d], comm);
  }

  // Determine whether new_dist owns nothing
  new_dist.ownNothing_ = false;
  for(int i=0; i<ndims; i++) {
    if(new_dist.maps_[i].getLocalNumEntries() == 0) {
      new_dist.ownNothing_ = true;
      break;
    }
  }

  return new_dist;
}

void Distribution::createMaps()
{
  int ndims = globalDims_.size();

  // Create a map for each dimension
  maps_ = std::vector<Map>(ndims);
  for(int d=0; d<ndims; d++) {
    const MPI_Comm& comm = grid_.getColComm(d);
    maps_[d] = TuckerMpi::Map(globalDims_[d], comm);
  }

  // Determine whether I own nothing
  ownNothing_ = false;
  for(int i=0; i<ndims; i++) {
    if(maps_[i].getLocalNumEntries() == 0) {
      ownNothing_ = true;
      break;
    }
  }
}

}
