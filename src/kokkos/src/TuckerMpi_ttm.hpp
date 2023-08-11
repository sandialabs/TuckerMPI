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
  using tensor_layout = typename tensor_type::traits::onnode_layout;
  using tensor_memory_space = typename tensor_type::traits::memory_space;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>,
		"TuckerMpi::ttm: currently supports tensors with LayoutLeft");

  using u_view_t = Kokkos::View<ScalarType**, ViewProperties...>;
  static_assert(std::is_same_v<tensor_memory_space, typename u_view_t::memory_space>,
		"TuckerMpi::ttm: tensor and matrix arguments must have matching memory spaces");

  return impl::ttm_impl(Xtensor, n, Umatrix, Utransp, nnz_limit);
}

} // end namespace TuckerMpi

#endif  // TUCKERMPI_TTM_HPP_
