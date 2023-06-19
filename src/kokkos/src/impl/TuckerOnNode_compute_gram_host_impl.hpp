#ifndef TUCKERKOKKOS_COMP_GRAM_HOST_IMPL_HPP_
#define TUCKERKOKKOS_COMP_GRAM_HOST_IMPL_HPP_

#include "Tucker_BlasWrapper.hpp"
#include "TuckerOnNode_Tensor.hpp"

namespace TuckerOnNode{
namespace impl{

template<class ScalarType, class MemorySpace>
void compute_gram_host(const Tensor<ScalarType, MemorySpace> & Y,
		       const std::size_t n,
		       ScalarType* gram,
		       const int stride)
{
  // if(Y == 0) {
  //   throw std::runtime_error("Tucker::computeGram(const Tensor<ScalarType>* Y, const int n, ScalarType* gram, const int stride): Y is a null pointer");
  // }
  if(gram == 0) {
    throw std::runtime_error("Tucker::computeGram(const Tensor<ScalarType>* Y, const int n, ScalarType* gram, const int stride): gram is a null pointer");
  }
  if(Y.size() == 0) {
    throw std::runtime_error("Tucker::computeGram(const Tensor<ScalarType>* Y, const int n, ScalarType* gram, const int stride): Y.size() == 0");
  }
  if(stride < 1) {
    std::ostringstream oss;
    oss << "Tucker::computeGram(const Tensor<ScalarType>* Y, const int n, ScalarType* gram, "
        << "const int stride): stride = " << stride << " < 1";
    throw std::runtime_error(oss.str());
  }
  if(n < 0 || n >= Y.rank()) {
    std::ostringstream oss;
    oss << "Tucker::computeGram(const Tensor<ScalarType>* Y, const int n, ScalarType* gram, "
        << "const int stride): n = " << n << " is not in the range [0,"
        << Y.rank() << ")";
    throw std::runtime_error(oss.str());
  }

  const int nrows = (int)Y.extent(n);

  auto Y_v = Y.data();
  auto Y_v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y_v);
  auto Y_rawPtr = Y_v_h.data();

  // // n = 0 is a special case
  // // Y_0 is stored column major
  if(n == 0)
  {

    // Compute number of columns of Y_n
    // Technically, we could divide the total number of entries by n,
    // but that seems like a bad decision
    int ncols = 1;
    for(int i=0; i<(int)Y.rank(); i++) {
      if((std::size_t)i != n) {
        ncols *= (int)Y.extent(i);
      }
    }

    // Call symmetric rank-k update
    // call syrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    // C := alpha*A*A' + beta*C
    char uplo = 'U';
    char trans = 'N';
    ScalarType alpha = 1;
    ScalarType beta = 0;
    Tucker::syrk(&uplo, &trans, &nrows, &ncols, &alpha,
		 Y_rawPtr, &nrows, &beta, gram, &stride);

  }
  else
  {
    int ncols = 1;
    int nmats = 1;

    // Count the number of columns
    for(std::size_t i=0; i<n; i++) {
      ncols *= (int)Y.extent(i);
    }

    // Count the number of matrices
    for(int i=n+1; i<(int)Y.rank(); i++) {
      nmats *= (int)Y.extent(i);
    }

    // For each matrix...
    for(int i=0; i<nmats; i++) {
      // Call symmetric rank-k update
      // call dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
      // C := alpha*A'*A + beta*C
      char uplo = 'U';
      char trans = 'T';
      ScalarType alpha = 1;
      ScalarType beta;
      if(i==0)
        beta = 0;
      else
        beta = 1;

      Tucker::syrk(&uplo, &trans, &nrows, &ncols, &alpha,
		   Y_rawPtr+i*nrows*ncols, &ncols, &beta,
		   gram, &stride);
    }
  }
}

}}
#endif
