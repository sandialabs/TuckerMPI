#ifndef TUCKER_KOKKOS_MPI_TTM_HPP_
#define TUCKER_KOKKOS_MPI_TTM_HPP_

#include "TuckerMpi_Tensor.hpp"
#include "./impl/TuckerMpi_ttm_impl.hpp"

namespace TuckerMpi{

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
auto ttm(Tensor<ScalarType, TensorProperties...> X,
	 int n,
	 Kokkos::View<ScalarType**, ViewProperties...> U,
	 bool Utransp,
	 std::size_t nnz_limit)
{
  return impl::ttm_impl(X, n, U, Utransp, nnz_limit);
}

} // end namespace TuckerMpi

#endif /* MPI_TUCKERMPI_TTM_HPP_ */
