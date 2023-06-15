#ifndef TTM_IMPL_USING_HOST_BLAS_HPP_
#define TTM_IMPL_USING_HOST_BLAS_HPP_

#include "Tucker_BlasWrapper.hpp"
#include "TuckerOnNode_Tensor.hpp"
#include <Kokkos_Core.hpp>

namespace TuckerOnNode{
namespace impl{

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
std::enable_if_t<
  std::is_same_v<typename Tensor<ScalarType, TensorProperties...>::traits::array_layout, Kokkos::LayoutLeft>
  && std::is_same_v<typename Kokkos::View<ScalarType**, ViewProperties...>::array_layout, Kokkos::LayoutLeft>
  >
ttm_hostblas(Tensor<ScalarType, TensorProperties...> X,
	     int n,
	     Kokkos::View<ScalarType**, ViewProperties...> A,
	     Tensor<ScalarType, TensorProperties...> Y,
	     bool Utransp)
{
  assert(n >= 0 && n < X.rank());
  for(std::size_t i=0; i<X.rank(); i++) {
    if(i != (size_t)n) {
      assert(X.extent(i) == Y.extent(i));
    }
  }

  // Obtain the number of rows and columns of U
  std::size_t Unrows, Uncols;
  if(Utransp) {
    Unrows = X.extent(n);
    Uncols = Y.extent(n);
  }
  else {
    Uncols = X.extent(n);
    Unrows = Y.extent(n);
  }

  int strideU = A.extent(0);
  auto U_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto U_ptr_h = U_h.data();

  auto X_view_d = X.data();
  auto X_view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X_view_d);
  auto X_ptr_h = X_view_h.data();

  auto Y_view_d = Y.data();
  auto Y_view_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y_view_d);
  auto Y_ptr_h = Y_view_h.data();

  // n = 0 is a special case
  // Y_0 is stored column major
  if(n == 0)
  {

    // Compute number of columns of Y_n
    // Technically, we could divide the total number of entries by n,
    // but that seems like a bad decision
    size_t ncols = X.prod(1,X.rank()-1);

    if(ncols > std::numeric_limits<std::size_t>::max()) {
      std::ostringstream oss;
      oss << "Error in Tucker::ttm: " << ncols
          << " is larger than std::numeric_limits<std::size_t>::max() ("
          << std::numeric_limits<std::size_t>::max() << ")";
      throw std::runtime_error(oss.str());
    }

    // Call matrix matrix multiply
    // call gemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    // C := alpha*op( A )*op( B ) + beta*C
    // A, B and C are matrices, with op( A ) an m by k matrix,
    // op( B ) a k by n matrix and C an m by n matrix.
    char transa;
    char transb = 'N';
    int m =  (int)Y.extent(n);
    int blas_n = (int)ncols;
    int k =  (int)X.extent(n);
    int lda = strideU;
    int ldb = k;
    int ldc = m;
    ScalarType alpha = 1;
    ScalarType beta = 0;

    if(Utransp) {
      transa = 'T';
    } else {
      transa = 'N';
    }
    Tucker::gemm(&transa, &transb, &m, &blas_n, &k, &alpha, U_ptr_h,
		 &lda, X_ptr_h, &ldb, &beta, Y_ptr_h, &ldc);

  }
  else
  {
    // Count the number of columns
    size_t ncols = X.prod(0,n-1);

    // Count the number of matrices
    size_t nmats = X.prod(n+1,X.rank()-1,1);

    if(ncols > std::numeric_limits<std::size_t>::max()) {
      std::ostringstream oss;
      oss << "Error in Tucker::ttm: " << ncols
          << " is larger than std::numeric_limits<std::size_t>::max() ("
          << std::numeric_limits<std::size_t>::max() << ")";
      throw std::runtime_error(oss.str());
    }

    // For each matrix...
    for(size_t i=0; i<nmats; i++) {
      // Call matrix matrix multiply
      // call dgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
      // C := alpha*op( A )*op( B ) + beta*C
      // A, B and C are matrices, with op( A ) an m by k matrix,
      // op( B ) a k by n matrix and C an m by n matrix.
      char transa = 'N';
      char transb;
      int m = (int)ncols;
      int blas_n = (int)Y.extent(n);
      int k;
      int lda = (int)ncols;
      int ldb = strideU;
      int ldc = (int)ncols;
      ScalarType alpha = 1;
      ScalarType beta = 0;
      if(Utransp) {
        transb = 'N';
        k = Unrows;
      } else {
        transb = 'T';
        k = Uncols;
      }
      Tucker::gemm(&transa, &transb, &m, &blas_n, &k, &alpha,
		   X_ptr_h+i*k*m, &lda, U_ptr_h, &ldb, &beta,
		   Y_ptr_h+i*m*blas_n, &ldc);
    }
  }

  Kokkos::deep_copy(X_view_d, X_view_h);
  Kokkos::deep_copy(Y_view_d, Y_view_h);
  Kokkos::deep_copy(A, U_h);
}

} //end namespace impl
} //endm namespace Tucker
#endif
