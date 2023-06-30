#ifndef TTM_TOPLEVEL_HPP_
#define TTM_TOPLEVEL_HPP_

#include "./impl/TuckerOnNode_ttm_using_host_blas_impl.hpp"
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

template <class ScalarType, class ...TensorProperties>
void ttm(Tensor<ScalarType, TensorProperties...> X,
	 int mode,
	 ScalarType* Uptr,
	 int strideU,
	 Tensor<ScalarType, TensorProperties...> Y,
	 bool Utransp)
{
  using tensor_type  = Tensor<ScalarType, TensorProperties...>;
  using memory_space = typename tensor_type::traits::memory_space;
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible,
		"TuckerOnNode::ttm: this overload is only for a tensor that is host accessible");
  impl::ttm_hostblas(X, mode, Uptr, strideU, Y, Utransp);
}

template <class ScalarType, class ...TensorProperties>
auto ttm(Tensor<ScalarType, TensorProperties...> X,
	 const int n,
	 ScalarType* Uptr,
	 const int dimU,
	 int strideU,
	 bool Utransp)
{
  using tensor_type  = Tensor<ScalarType, TensorProperties...>;
  using memory_space = typename tensor_type::traits::memory_space;
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible,
		"TuckerOnNode::ttm: this overload is only for a tensor that is host accessible");

  std::vector<int> I(X.rank());
  for(int i=0; i<I.size(); i++) {
    I[i] = (i != n) ? X.extent(i) : dimU;
  }
  Tensor<ScalarType, TensorProperties...> Y(I);
  ttm(X, n, Uptr, strideU, Y, Utransp);
  return Y;
}

}
#endif
