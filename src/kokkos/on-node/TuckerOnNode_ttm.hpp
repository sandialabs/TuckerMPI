#ifndef TTM_TOPLEVEL_HPP_
#define TTM_TOPLEVEL_HPP_

#include "./impl/TuckerOnNode_ttm_using_host_blas.hpp"
#include "./impl/TuckerOnNode_ttm_using_kokkos_kernels.hpp"

namespace TuckerOnNode{

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
void ttm(Tensor<ScalarType, TensorProperties...> X,
	 std::size_t mode,
	 Kokkos::View<ScalarType**, ViewProperties...> U,
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

  // FIXME: need to guard this somehow to make it work
  // for example for comparison purposes or when kernels is not on
  // impl::ttm_hostblas(X, n, U, Y, Utransp);

  if(mode == 0) {
    impl::ttm_kker_mode_zero(X, mode, U, Y, Utransp);
  } else {
    impl::ttm_kker_mode_greater_than_zero(X, mode, U, Y, Utransp);
  }
}

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
auto ttm(Tensor<ScalarType, TensorProperties...> X,
	 std::size_t mode,
	 Kokkos::View<ScalarType**, ViewProperties...> U,
	 bool Utransp)
{
  const std::size_t nrows = Utransp ? U.extent(1) : U.extent(0);
  Tucker::SizeArray I(X.rank());
  for(std::size_t i=0; i< (std::size_t)I.size(); i++) {
    I[i] = (i != mode) ? X.extent(i) : nrows;
  }
  Tensor<ScalarType, TensorProperties...> Y(I);
  ttm(X, mode, U, Y, Utransp);
  return Y;
}

}
#endif
