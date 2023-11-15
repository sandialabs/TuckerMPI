
#ifndef IMPL_TUCKERMPI_MATRIX_HPP_
#define IMPL_TUCKERMPI_MATRIX_HPP_

#include "mpi.h"
#include <Kokkos_Core.hpp>

namespace TuckerMpi {
namespace impl{

template <class scalar_t, class MemorySpace = typename Kokkos::DefaultExecutionSpace::memory_space>
class Matrix {

  using view_type = Kokkos::View<scalar_t**, Kokkos::LayoutLeft, MemorySpace>;

public:
  Matrix(int nrows, int ncols, const MPI_Comm& comm, bool isBlockRow)
    : globalRows_(nrows), globalCols_(ncols), comm_(&comm)
  {
    int rank, nprocs;
    MPI_Comm_rank(*comm_, &rank);
    MPI_Comm_size(*comm_, &nprocs);
    if(isBlockRow) { map_ = Map(nrows,comm); }
    else { map_ = Map(ncols,comm); }

    int localRows, localCols;
    if(isBlockRow) {
      localRows = map_.getLocalNumEntries();
      localCols = ncols;
    }
    else {
      localRows = nrows;
      localCols = map_.getLocalNumEntries();
    }
    M_ = view_type(Kokkos::ViewAllocateWithoutInitializing("M_"), localRows,localCols);
  }

  Matrix() = default;
  ~Matrix() = default;

  auto & getLocalMatrix() { return M_; }
  std::size_t localSize() const{ return M_.size(); }
  int getGlobalNumRows() const { return globalRows_; }
  int getLocalNumRows() const  { return M_.extent(0); }
  int getGlobalNumCols() const { return globalCols_; }
  int getLocalNumCols() const  { return M_.extent(1); }
  const Map* getMap() const{ return &map_; }

private:
  view_type M_ = {};
  Map map_ = {};
  int globalRows_ = {};
  int globalCols_ = {};
  const MPI_Comm * comm_ = nullptr;
};

}} // end namespace impl::TuckerMpi
#endif  // IMPL_TUCKERMPI_MATRIX_HPP_
