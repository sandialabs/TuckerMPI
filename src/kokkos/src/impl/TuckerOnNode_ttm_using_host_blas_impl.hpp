#ifndef TTM_IMPL_USING_HOST_BLAS_HPP_
#define TTM_IMPL_USING_HOST_BLAS_HPP_

#include "Tucker_BlasWrapper.hpp"
#include "TuckerOnNode_Tensor.hpp"
#include <Kokkos_Core.hpp>

namespace TuckerOnNode{
namespace impl{

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
void ttm_hostblas(Tensor<ScalarType, TensorProperties...> X,
		  int n,
		  ScalarType* Uptr_h,
		  int strideU,
		  Tensor<ScalarType, TensorProperties...> Y,
		  bool Utransp)
{
  using tensor_type  = Tensor<ScalarType, TensorProperties...>;
  static_assert(std::is_same_v<typename tensor_type::traits::array_layout, Kokkos::LayoutLeft>,
		"TuckerOnNode::ttm_hostblas: tensor must be layoutleft");

  std::size_t Unrows, Uncols;
  if(Utransp) {
    Unrows = X.extent(n);
    Uncols = Y.extent(n);
  }
  else {
    Uncols = X.extent(n);
    Unrows = Y.extent(n);
  }
  auto X_view_d = X.data();
  auto X_view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X_view_d);
  auto X_ptr_h = X_view_h.data();

  auto Y_view_d = Y.data();
  auto Y_view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y_view_d);
  auto Y_ptr_h = Y_view_h.data();

  // n = 0 is a special case: Y_0 is stored column major
  if(n == 0)
  {
    size_t ncols = X.prod(1,X.rank()-1);
    char transa = Utransp ? 'T' : 'N';
    char transb = 'N';
    int m =  (int)Y.extent(n);
    int blas_n = (int)ncols;
    int k =  (int)X.extent(n);
    int lda = strideU;
    int ldb = k;
    int ldc = m;
    ScalarType alpha = 1;
    ScalarType beta = 0;
    Tucker::gemm(&transa, &transb, &m, &blas_n, &k, &alpha, Uptr_h,
		 &lda, X_ptr_h, &ldb, &beta, Y_ptr_h, &ldc);
  }
  else
  {
    size_t ncols = X.prod(0,n-1);
    size_t nmats = X.prod(n+1,X.rank()-1,1);

    // For each matrix...
    for(size_t i=0; i<nmats; i++) {
      char transa = 'N';
      char transb = Utransp ? 'N' : 'T';
      int m = (int)ncols;
      int blas_n = (int)Y.extent(n);
      int k = Utransp ? Unrows : Uncols;
      int lda = (int)ncols;
      int ldb = strideU;
      int ldc = (int)ncols;
      ScalarType alpha = 1;
      ScalarType beta = 0;
      Tucker::gemm(&transa, &transb, &m, &blas_n, &k, &alpha,
		   X_ptr_h+i*k*m, &lda, Uptr_h, &ldb, &beta,
		   Y_ptr_h+i*m*blas_n, &ldc);
    }
  }

  Kokkos::deep_copy(X_view_d, X_view_h);
  Kokkos::deep_copy(Y_view_d, Y_view_h);
}

} //end namespace impl
} //endm namespace Tucker
#endif
