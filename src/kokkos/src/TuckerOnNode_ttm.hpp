#ifndef TTM_TOPLEVEL_HPP_
#define TTM_TOPLEVEL_HPP_

#include "TuckerOnNode_Tensor.hpp"
#if defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)
#include "./impl/TuckerOnNode_ttm_using_host_blas_impl.hpp"
#endif
#include "./impl/TuckerOnNode_ttm_using_kokkos_kernels_impl.hpp"

namespace TuckerOnNode{

template <
  class ScalarType, class ...TensorProperties,
  class ViewDataType, class ...ViewProps
  >
void ttm(Tensor<ScalarType, TensorProperties...> X,
	 std::size_t mode,
	 Kokkos::View<ViewDataType, ViewProps ...> U,
	 Tensor<ScalarType, TensorProperties...> Y,
	 bool Utransp)
{

  // constraints
  using tensor_type   = Tensor<ScalarType, TensorProperties...>;
  using tensor_layout = typename tensor_type::traits::array_layout;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>,
		"TuckerOnNode::ttm: currently supports tensors with LayoutLeft");

  // preconditions
  if(Utransp) {
    assert(U.extent(0) == X.extent(mode));
    assert(U.extent(1) == Y.extent(mode));
  }
  else {
    assert(U.extent(1) == X.extent(mode));
    assert(U.extent(0) == Y.extent(mode));
  }

  if(mode == 0) {
    impl::ttm_kker_mode_zero(X, mode, U, Y, Utransp);
  } else {
    impl::ttm_kker_mode_greater_than_zero(X, mode, U, Y, Utransp);
  }
}

template <
  class ScalarType, class ...TensorProperties,
  class ViewDataType, class ...ViewProps
  >
auto ttm(Tensor<ScalarType, TensorProperties...> X,
	 std::size_t mode,
	 Kokkos::View<ViewDataType, ViewProps ...> U,
	 bool Utransp)
{

  // constraints
  using tensor_type       = Tensor<ScalarType, TensorProperties...>;
  using tensor_layout     = typename tensor_type::traits::array_layout;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>,
		"TuckerOnNode::ttm: supports tensors with LayoutLeft");

  const std::size_t nrows = Utransp ? U.extent(1) : U.extent(0);
  std::vector<int> I(X.rank());
  for(std::size_t i=0; i< (std::size_t)I.size(); i++) {
    I[i] = (i != mode) ? X.extent(i) : nrows;
  }
  tensor_type Y(I);
  ttm(X, mode, U, Y, Utransp);
  return Y;
}

}
#endif
