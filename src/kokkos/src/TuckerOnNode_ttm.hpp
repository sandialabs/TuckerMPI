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
  using umv_ls_type = Kokkos::View<ScalarType**, Kokkos::LayoutStride,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  /**
   * TTM do: Y = beta*Y + alpha*op(A)*op(X)
   * Need to create a Kokkos's View A from:
   * - Uptr (data) and
   * - strideU (way to read data)
   */

  // Need to create a strided layout
  Kokkos::LayoutStride layout(
    X.extent(mode),     // get the 1st dim of X = 2nd dim of A
    1,                  // way to read data on 2nd dim
    Y.extent(mode),     // get the 1st dim of Y = 1st dim of A
    strideU             // way to read data on 1st dim
  );

  // Create Kokkos's View with Uptr and our layout
  umv_ls_type Aumvls(Uptr, layout);

  if(mode == 0) {
    impl::ttm_kker_mode_zero(X, mode, Aumvls, Y, Utransp);
  } else {
    impl::ttm_kker_mode_greater_than_zero(X, mode, Aumvls, Y, Utransp);
  }
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
