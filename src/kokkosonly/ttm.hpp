#ifndef TTM_HPP_
#define TTM_HPP_

#include "Tucker_BlasWrapper.hpp"

namespace TuckerKokkos{

template <class ScalarType, class MemorySpace>
void ttm_impl(const Tensor<ScalarType, MemorySpace>* const X,
	      const int n,
	      const ScalarType* const Uptr,
	      const int strideU,
	      Tensor<ScalarType, MemorySpace> & Y,
	      bool Utransp)
{
  // Check that the input is valid
  assert(Uptr != 0);
  //assert(Y != 0);
  assert(n >= 0 && n < X->rank());
  for(int i=0; i<X->rank(); i++) {
    if(i != n) {
      assert(X->size(i) == Y.size(i));
    }
  }

  // Obtain the number of rows and columns of U
  int Unrows, Uncols;
  if(Utransp) {
    Unrows = X->size(n);
    Uncols = Y.size(n);
  }
  else {
    Uncols = X->size(n);
    Unrows = Y.size(n);
  }

  auto X_view_d = X->data();
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
    size_t ncols = X->sizeArray().prod(1,X->rank()-1);

    if(ncols > std::numeric_limits<int>::max()) {
      std::ostringstream oss;
      oss << "Error in Tucker::ttm: " << ncols
          << " is larger than std::numeric_limits<int>::max() ("
          << std::numeric_limits<int>::max() << ")";
      throw std::runtime_error(oss.str());
    }

    // Call matrix matrix multiply
    // call gemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    // C := alpha*op( A )*op( B ) + beta*C
    // A, B and C are matrices, with op( A ) an m by k matrix,
    // op( B ) a k by n matrix and C an m by n matrix.
    char transa;
    char transb = 'N';
    int m = Y.size(n);
    int blas_n = (int)ncols;
    int k = X->size(n);
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
    Tucker::gemm(&transa, &transb, &m, &blas_n, &k, &alpha, Uptr,
		 &lda, X_ptr_h, &ldb, &beta, Y_ptr_h, &ldc);

  }
  else
  {
    // Count the number of columns
    size_t ncols = X->sizeArray().prod(0,n-1);

    // Count the number of matrices
    size_t nmats = X->sizeArray().prod(n+1,X->rank()-1,1);

    if(ncols > std::numeric_limits<int>::max()) {
      std::ostringstream oss;
      oss << "Error in Tucker::ttm: " << ncols
          << " is larger than std::numeric_limits<int>::max() ("
          << std::numeric_limits<int>::max() << ")";
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
      int blas_n = Y.size(n);
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
		   X_ptr_h+i*k*m, &lda, Uptr, &ldb, &beta,
		   Y_ptr_h+i*m*blas_n, &ldc);
    }
  }

  Kokkos::deep_copy(X_view_d, X_view_h);
  Kokkos::deep_copy(Y_view_d, Y_view_h);
}


template <class ScalarType, class MemorySpace>
void ttm(const Tensor<ScalarType, MemorySpace>* const X,
	 const int n,
	 Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> U,
	 Tensor<ScalarType, MemorySpace> & Y,
	 bool Utransp)
{
  // Check that the input is valid
  //assert(U != 0);
  if(Utransp) {
    assert(U.extent(0) == X->size(n));
    assert(U.extent(1) == Y.size(n));
  }
  else {
    assert(U.extent(1) == X->size(n));
    assert(U.extent(0) == Y.size(n));
  }

  auto U_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U);
  ttm_impl(X, n, U_h.data(), U.extent(0), Y, Utransp);
  Kokkos::deep_copy(U, U_h);
}


template <class ScalarType, class ...Props, class ...Props2>
auto ttm(const Tensor<ScalarType, Props...>* X,
	 const int n,
	 Kokkos::View<ScalarType**, Kokkos::LayoutLeft, Props2...> U,
	 bool Utransp)
{
  // Compute the number of rows for the resulting "matrix"
  int nrows;
  if(Utransp)
    nrows = U.extent(1);
  else
    nrows = U.extent(0);

  // Allocate space for the new tensor
  TuckerKokkos::SizeArray I(X->rank());
  for(int i=0; i<I.size(); i++) {
    if(i != n) {
      I[i] = X->size(i);
    }
    else {
      I[i] = nrows;
    }
  }

  Tensor<ScalarType, Props...> Y(I);
  ttm(X, n, U, Y, Utransp);
  return Y;
}


}
#endif
