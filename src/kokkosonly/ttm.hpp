#ifndef TTM_HPP_
#define TTM_HPP_

#include "Tucker_BlasWrapper.hpp"

#include<Kokkos_Core.hpp>
#include<KokkosBlas3_gemm.hpp>

namespace TuckerKokkos{

template <class ScalarType, class MemorySpace>
void ttm_kokkosblas_impl(const Tensor<ScalarType, MemorySpace>* const X,
        const int n,
        const Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> A,
	      Tensor<ScalarType, MemorySpace> & Y,
	      bool Utransp)
{
  // Check that the input is valid
  // Assert if tensor Y not null
  assert(Y != 0);
  // Assert if n (Mode) has a valid value
  assert(n >= 0 && n < X->N());
  for(int i=0; i<X->N(); i++) {
    if(i != n) {
      // Assert if each slices of Tensor X and Y have the same size
      assert(X->size(i) == Y.size(i));
    }
  }

  // Obtain the number of rows and columns of A
  int Unrows, Uncols;
  if(Utransp) {
    Unrows = X->size(n);
    Uncols = Y.size(n);
  }
  else {
    Uncols = X->size(n);
    Unrows = Y.size(n);
  }

  // View B
  auto X_view_d = X->data();
  auto X_ptr_d = X_view_d.data();

  // View C
  auto Y_view_d = Y.data();
  auto Y_ptr_d = Y_view_d.data();

  // n = 0 is a special case
  /** Column-major order (Fortran):
   * 
   *  A = | a11 a12 a13 |
   *      | a21 a22 a23 |
   * 
   *  address | access  | values
   *    0     | A(0,0)  |   a11
   *    1     | A(1,0)  |   a21
   *    2     | A(0,1)  |   a12
   *    3     | A(1,1)  |   a22
   *    4     | A(0,2)  |   a13
   *    5     | A(1,2)  |   a23
   */
  // Y_0 is stored column major
  if(n == 0)
  {
    // Compute number of columns of Y_n
    // Technically, we could divide the total number of entries by n,
    // but that seems like a bad decision
    size_t ncols = X->size().prod(1,X->N()-1);

    if(ncols > std::numeric_limits<int>::max()) {
      std::ostringstream oss;
      oss << "Error in Tucker::ttm: " << ncols
          << " is larger than std::numeric_limits<int>::max() ("
          << std::numeric_limits<int>::max() << ")";
      throw std::runtime_error(oss.str());
    }

    /** Compute dense matrix-matrix multiply
     * call KokkosBlas::gemm(modeA, modeB, alpha, A, B, beta, C);
     * C = beta*C + alpha*op(A)*op(B)
     * 
     * A, B and C are 2-D Kokkos::View
     * 
     * A and B are input
     * C is output
     * 
     * A is m by k
     * B is k by blas_n
     * C is m by blas_n
     * Keep in mind: dimensions are set for a given Mode n
     */
    char transa = Utransp ? 'T' : 'N';      // "T" for Transpose
    const char transb = 'N';                // "N" for Non-tranpose
    int m = Y.size(n);                      // 1st dim of A and C
    int blas_n = (int)ncols;                // 2nd dim of B and C
    int k = X->size(n);                     // 1st dim of B
    const ScalarType alpha = ScalarType(1); // input coef. of op(A)*op(B)
    const ScalarType beta = ScalarType(0);  // input coef. of C
 
    Kokkos::View<ScalarType**, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>> B(X_ptr_d, k, blas_n);
    // C must have a LayoutLeft (column-major order)
    Kokkos::View<ScalarType**, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>> C(Y_ptr_d, m, blas_n);
    KokkosBlas::gemm(&transa,&transb,alpha,A,B,beta,C);
  }
  else
  {
    // Count the number of columns
    size_t ncols = X->size().prod(0,n-1);

    // Count the number of matrices
    size_t nmats = X->size().prod(n+1,X->N()-1,1);

    if(ncols > std::numeric_limits<int>::max()) {
      std::ostringstream oss;
      oss << "Error in Tucker::ttm: " << ncols
          << " is larger than std::numeric_limits<int>::max() ("
          << std::numeric_limits<int>::max() << ")";
      throw std::runtime_error(oss.str());
    }

    // For each matrix...
    for(size_t i=0; i<nmats; i++) {
      /** Compute dense matrix-matrix multiply
       * call KokkosBlas::gemm(modeA, modeB, alpha, B, A, beta, C);
       * C = beta*C + alpha*op(B)*op(A)
       * 
       * A, B and C are 2-D Kokkos::View
       * 
       * A and B are input
       * C is output
       * 
       * B is m by k
       * A is k by blas_n
       * C is m by blas_n
       * Warning: A and B are reversed
       */
      char transa = 'N';                      // "N" for Non-tranpose
      char transb = Utransp ? 'N' : 'T';      // "T" for Transpose
      int m = (int)ncols;                     // 1st dim of B and C
      int blas_n = Y.size(n);                 // 2nd dim of A and C
      int k = Utransp ? Unrows : Uncols;      // 2nd dim of B
      const ScalarType alpha = ScalarType(1); // input coef. of op(A)*op(B)
      const ScalarType beta = ScalarType(0);  // input coef. of C

      Kokkos::View<ScalarType**, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>> B(X_ptr_d+i*k*m, m, k);
      Kokkos::View<ScalarType**, Kokkos::LayoutLeft, Kokkos::MemoryTraits<Kokkos::Unmanaged>> C(Y_ptr_d+i*m*blas_n, m, blas_n);
      // Warning: call gemm(modeA, modeB, alpha, B, A, beta, C);
      KokkosBlas::gemm(&transa,&transb,alpha,B,A,beta,C);
    }
  }

}

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
  assert(n >= 0 && n < X->N());
  for(int i=0; i<X->N(); i++) {
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
    size_t ncols = X->size().prod(1,X->N()-1);

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
    size_t ncols = X->size().prod(0,n-1);

    // Count the number of matrices
    size_t nmats = X->size().prod(n+1,X->N()-1,1);

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
  ttm_kokkosblas_impl(X, n, U_h, Y, Utransp);
  Kokkos::deep_copy(U, U_h);
}


template <class ScalarType, class MemorySpace>
auto ttm(const Tensor<ScalarType, MemorySpace>* X,
	 const int n,
	 Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemorySpace> U,
	 bool Utransp)
{
  // Compute the number of rows for the resulting "matrix"
  int nrows;
  if(Utransp)
    nrows = U.extent(1);
  else
    nrows = U.extent(0);

  // Allocate space for the new tensor
  TuckerKokkos::SizeArray I(X->N());
  for(int i=0; i<I.size(); i++) {
    if(i != n) {
      I[i] = X->size(i);
    }
    else {
      I[i] = nrows;
    }
  }
  Tensor<ScalarType, MemorySpace> Y(I);
  ttm(X, n, U, Y, Utransp);
  return Y;
}


}
#endif
