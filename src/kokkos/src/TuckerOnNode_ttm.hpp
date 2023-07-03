#ifndef TTM_TOPLEVEL_HPP_
#define TTM_TOPLEVEL_HPP_

#if defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)
#include "./impl/TuckerOnNode_ttm_using_host_blas_impl.hpp"
#endif
#include "./impl/TuckerOnNode_ttm_using_kokkos_kernels_impl.hpp"

namespace TuckerOnNode{

template <class ScalarType, class ...TensorProperties, class UType>
void ttm(Tensor<ScalarType, TensorProperties...> X,
	 std::size_t mode,
	 UType U,
	 Tensor<ScalarType, TensorProperties...> Y,
	 bool Utransp)
{
  if(Utransp) {
    assert(U.extent(0) == X.extent(n));
    assert(U.extent(1) == Y.extent(n));
  }
  else {
    assert(U.extent(1) == X.extent(n));
    assert(U.extent(0) == Y.extent(n));
  }

  if(mode == 0) {
    impl::ttm_kker_mode_zero(X, mode, U, Y, Utransp);
  } else {
    impl::ttm_kker_mode_greater_than_zero(X, mode, U, Y, Utransp);
  }
}

template <class ScalarType, class ...TensorProperties, class UType>
auto ttm(Tensor<ScalarType, TensorProperties...> X,
	 std::size_t mode,
	 UType U,
	 bool Utransp)
{
  const std::size_t nrows = Utransp ? U.extent(1) : U.extent(0);
  std::vector<int> I(X.rank());
  for(std::size_t i=0; i< (std::size_t)I.size(); i++) {
    I[i] = (i != mode) ? X.extent(i) : nrows;
  }
  Tensor<ScalarType, TensorProperties...> Y(I);
  ttm(X, mode, U, Y, Utransp);
  return Y;
}

template <class ScalarType, class ...TensorProperties, class AType>
void ttm(Tensor<ScalarType, TensorProperties...> X,
	 int mode,
	 AType A,
	 Tensor<ScalarType, TensorProperties...> Y,
	 bool Utransp)
{
  if(mode == 0) {
    impl::ttm_kker_mode_zero(X, mode, A, Y, Utransp);
  } else {
    impl::ttm_kker_mode_greater_than_zero(X, mode, A, Y, Utransp);
  }
}

}
#endif
