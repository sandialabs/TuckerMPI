#ifndef TUCKERONNODE_TTM_HPP_
#define TUCKERONNODE_TTM_HPP_

#include "TuckerOnNode_Tensor.hpp"
#if defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)
#include "./impl/Tucker_view_clone_utils.hpp"
#include "./impl/TuckerOnNode_ttm_using_host_blas_impl.hpp"
#else
#include "./impl/TuckerOnNode_ttm_using_kokkos_kernels_impl.hpp"
#endif

namespace TuckerOnNode{

template <
  class ScalarType, class ...TensorProperties,
  class ViewDataType, class ...ViewProperties
  >
void ttm(Tensor<ScalarType, TensorProperties...> Xtensor,
	 std::size_t mode,
	 Kokkos::View<ViewDataType, ViewProperties ...> Umatrix,
	 Tensor<ScalarType, TensorProperties...> Ytensor,
	 bool Utransp)
{

  // constraints
  using tensor_type   = Tensor<ScalarType, TensorProperties...>;
  using tensor_layout = typename tensor_type::traits::array_layout;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>,
		"TuckerOnNode::ttm: currently supports tensors with LayoutLeft");

  // preconditions
  if(Utransp) {
    assert(Umatrix.extent(0) == Xtensor.extent(mode));
    assert(Umatrix.extent(1) == Ytensor.extent(mode));
  }
  else {
    assert(Umatrix.extent(1) == Xtensor.extent(mode));
    assert(Umatrix.extent(0) == Ytensor.extent(mode));
  }

#if defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)
  // NOTE that here we cannot just do a regular create_mirror because
  // the matrix U can be strided (see the MPI case) so we need to account for that case too
  auto Umatrix_dc = ::Tucker::impl::create_deep_copyable_compatible_clone<Kokkos::LayoutLeft>(Umatrix);
  auto U_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Umatrix_dc);
  impl::ttm_hostblas(Xtensor, mode, U_h.data(), Umatrix.extent(0), Ytensor, Utransp);

#else
  if(mode == 0) {
    impl::ttm_mode_zero_use_kkernels_gemm(Xtensor, mode, Umatrix, Ytensor, Utransp);
  } else {
    impl::ttm_nonzero_mode_use_kkernels_team_gemm(Xtensor, mode, Umatrix, Ytensor, Utransp);
    //impl::ttm_nonzero_mode_sequentially_call_kkernels_gemm(Xtensor, mode, Umatrix, Ytensor, Utransp);
  }
#endif
}

template <
  class ScalarType, class ...TensorProperties,
  class ViewDataType, class ...ViewProperties
  >
[[nodiscard]] auto ttm(Tensor<ScalarType, TensorProperties...> Xtensor,
		       std::size_t mode,
		       Kokkos::View<ViewDataType, ViewProperties ...> Umatrix,
		       bool Utransp)
{

  // constraints
  using tensor_type       = Tensor<ScalarType, TensorProperties...>;
  using tensor_layout     = typename tensor_type::traits::array_layout;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>,
		"TuckerOnNode::ttm: supports tensors with LayoutLeft");

  const std::size_t nrows = Utransp ? Umatrix.extent(1) : Umatrix.extent(0);
  std::vector<int> I(Xtensor.rank());
  for(std::size_t i=0; i< (std::size_t)I.size(); i++) {
    I[i] = (i != mode) ? Xtensor.extent(i) : nrows;
  }
  tensor_type Ytensor(I);
  ttm(Xtensor, mode, Umatrix, Ytensor, Utransp);
  return Ytensor;
}

}
#endif  // TUCKERONNODE_TTM_HPP_
