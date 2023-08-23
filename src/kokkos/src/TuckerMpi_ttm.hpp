#ifndef TUCKERMPI_TTM_HPP_
#define TUCKERMPI_TTM_HPP_

#include "TuckerMpi_Tensor.hpp"
#include "./impl/TuckerMpi_ttm_impl.hpp"

namespace TuckerMpi{

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
auto ttm(Tensor<ScalarType, TensorProperties...> Xtensor,
         int n,
         Kokkos::View<ScalarType**, ViewProperties...> Umatrix,
         bool Utransp,
         std::size_t nnz_limit)
{

  // constraints
  using tensor_type   = Tensor<ScalarType, TensorProperties...>;
  using local_tensor_type = typename tensor_type::traits::onnode_tensor_type;
  using tensor_layout = typename tensor_type::traits::onnode_layout;
  using tensor_memory_space = typename tensor_type::traits::memory_space;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>,
		"TuckerMpi::ttm: currently supports tensors with LayoutLeft");

  using u_view_t = Kokkos::View<ScalarType**, ViewProperties...>;
  static_assert(std::is_same_v<tensor_memory_space, typename u_view_t::memory_space>,
		"TuckerMpi::ttm: tensor and matrix arguments must have matching memory spaces");

  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  // Compute the number of rows for the resulting "matrix"
  const int nrows = Utransp ? Umatrix.extent(1) : Umatrix.extent(0);

  // Get the size of the new tensor
  const int ndims = Xtensor.rank();
  std::vector<int> newSize(ndims);
  for(int i=0; i<ndims; i++) {
    newSize[i] = (i == n) ? nrows : Xtensor.globalExtent(i);
  }

  Distribution resultDist(newSize, Xtensor.getDistribution().getProcessorGrid().getSizeArray());
  tensor_type result(resultDist);

  auto preferSingleReduceScatter = [=]() -> bool
  {
    auto & Xdist = Xtensor.getDistribution();
    std::size_t max_lcl_nnz_x = 1;
    for(int i=0; i<Xtensor.rank(); i++) {
      max_lcl_nnz_x *= Xdist.getMap(i,false)->getMaxNumEntries();
    }
    // Compute the nnz required for the reduce_scatter
    std::size_t nnz_reduce_scatter = 1;
    for(int i=0; i<Xtensor.rank(); i++) {
      if(i == n){
        nnz_reduce_scatter *= result.globalExtent(n);
      }
      else{
        nnz_reduce_scatter *= Xdist.getMap(i,false)->getMaxNumEntries();
      }
    }

    return (nnz_reduce_scatter <= std::max(max_lcl_nnz_x, nnz_limit));
  };

  const int Pn = Xtensor.getDistribution().getProcessorGrid().getNumProcs(n, false);
  if(Pn == 1)
  {
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    if (mpiRank == 0){ std::cout << "MPITTM: Pn==1 \n"; }
#endif

    if(!Xtensor.getDistribution().ownNothing()) {
      auto localX = Xtensor.localTensor();
      auto localresult = result.localTensor();
      TuckerOnNode::ttm(localX, n, Umatrix, localresult, Utransp);
    }
  }
  else
  {
    // FRIZZI: NOTE: reduce scatter has bad performance, needs to figure out why
    // so always calling the series of reduction
    // if (preferSingleReduceScatter()){
    //   impl::ttm_impl_use_single_reduce_scatter(mpiRank, Xtensor, result, n, Umatrix, Utransp);
    //} else{
    impl::ttm_impl_use_series_of_reductions(mpiRank, Xtensor, result, n, Umatrix, Utransp);
    //}
  }
  return result;
}

} // end namespace TuckerMpi

#endif  // TUCKERMPI_TTM_HPP_
