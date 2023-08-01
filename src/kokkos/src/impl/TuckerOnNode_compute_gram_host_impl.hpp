#ifndef TUCKERKOKKOS_COMP_GRAM_HOST_IMPL_HPP_
#define TUCKERKOKKOS_COMP_GRAM_HOST_IMPL_HPP_

#include "Tucker_BlasWrapper.hpp"

namespace TuckerOnNode{
namespace impl{

template <class ScalarType, class DataType, class ...ViewProps, class ...Properties>
void compute_gram_host(Tensor<ScalarType, Properties...> Y,
		       const std::size_t n,
		       Kokkos::View<DataType, ViewProps...> gram)
{
  using tensor_type = Tensor<ScalarType, Properties...>;
  using tensor_memory_space = typename tensor_type::traits::memory_space;
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, tensor_memory_space>::accessible,
                "compute_gram_host: tensor must be accessible on the host");

  using gram_view_type = Kokkos::View<DataType, ViewProps...>;
  using gram_mem_space = typename gram_view_type::memory_space;
  using gram_layout = typename gram_view_type::array_layout;
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, gram_mem_space>::accessible,
                "compute_gram_host: gram view must be accessible on the host");
  static_assert(std::is_same_v<Kokkos::LayoutLeft, gram_layout>,
                "compute_gram_host: gram view must have LayoutLeft");

  // if(Y == 0) {
  //   throw std::runtime_error("Tucker::computeGram(const Tensor<ScalarType>* Y, const int n, ScalarType* gram, const int stride): Y is a null pointer");
  // }
  // if(Y.size() == 0) {
  //   throw std::runtime_error("compute_gram_host: Y.size() == 0");
  // }
  // if(stride < 1) {
  //   std::ostringstream oss;
  //   oss << "compute_gram_host: stride = " << stride << " < 1";
  //   throw std::runtime_error(oss.str());
  // }
  // if(n < 0 || n >= Y.rank()) {
  //   std::ostringstream oss;
  //   oss << "compute_gram_host(const Tensor<ScalarType>* Y, const int n, ScalarType* gram, "
  //       << "const int stride): n = " << n << " is not in the range [0,"
  //       << Y.rank() << ")";
  //   throw std::runtime_error(oss.str());
  // }

  const int nrows = (int)Y.extent(n);
  auto Y_rawPtr = Y.data().data();
  auto gramPtr = gram.data();

  // n = 0 is a special case, Y_0 is stored column major
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
    int ldc = gram.extent(0);
    Tucker::syrk(&uplo, &trans, &nrows, &ncols, &alpha,
		 Y_rawPtr, &nrows, &beta, gramPtr, &ldc);
  }
  else
  {
    int ncols = 1;
    int nmats = 1;
    for(std::size_t i=0; i<n; i++) {
      ncols *= (int)Y.extent(i);
    }
    for(int i=n+1; i<(int)Y.rank(); i++) {
      nmats *= (int)Y.extent(i);
    }

    for(int i=0; i<nmats; i++) {
      // Call symmetric rank-k update
      // call dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
      // C := alpha*A'*A + beta*C
      char uplo = 'U';
      char trans = 'T';
      ScalarType alpha = 1;
      ScalarType beta = (i==0) ? 0 : 1;
      int ldc = gram.extent(0);
      Tucker::syrk(&uplo, &trans, &nrows, &ncols, &alpha,
		   Y_rawPtr+i*nrows*ncols, &ncols, &beta,
		   gramPtr, &ldc);
    }
  }
}

}}
#endif
