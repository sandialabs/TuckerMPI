#ifndef TUCKERKOKKOS_COMP_GRAM_KOKKOS_IMPL_HPP_
#define TUCKERKOKKOS_COMP_GRAM_KOKKOS_IMPL_HPP_

#include "Tucker_BlasWrapper.hpp"
#include "TuckerOnNode_Tensor.hpp"

namespace TuckerOnNode{
namespace impl{

template <class ScalarType, class DataType, class ...ViewProps, class ...Properties>
void compute_gram_kokkos(Tensor<ScalarType, Properties...> Y,
			 const std::size_t n,
			 Kokkos::View<DataType, ViewProps...> C)
{

  const int nrows = (int)Y.extent(n);
  auto Y_rawPtr = Y.data().data();
  auto gramPtr = C.data();

  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft,
				Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

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

    // symmetric rank-k update
    //   C := alpha*A*A' + beta*C
    // which corresponds to:
    //   call syrk('U', 'N', nrows, ncols, alpha, Aptr, nrows, beta, Cptr, C.extent(0))
    char uplo = 'U';
    char trans = 'N';
    const ScalarType alpha = 1;
    const ScalarType beta = 0;
    umv_type Aview(Y.data().data(), Y.extent(0), ncols);
    Kokkos::parallel_for(Kokkos::RangePolicy(0, C.extent(1)),
			 KOKKOS_LAMBDA(const std::size_t j) {
			   for (std::size_t i = 0; i <= j; ++i) {
			     ScalarType sum = {};
			     for (std::size_t k = 0; k < Aview.extent(1); ++k) {
			       sum += Aview(i,k) * Aview(j,k);
			     }
			     C(i,j) = beta*C(i,j) + alpha*sum;
			   }
			 });

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
      // symmetric rank-k update as follows:
      //   C := alpha*A'*A + beta*C
      // which corresponds to:
      //   dsyrk('U', 'T', nrows, ncols, alpha,
      //         Aptr, ncols, beta, Cptr, C.extent(0))

      const ScalarType alpha = 1;
      const ScalarType beta = (i==0) ? 0 : 1;

      auto Aptr = Y_rawPtr+i*nrows*ncols;
      umv_type Aview(Aptr, ncols, nrows);
      Kokkos::parallel_for(Kokkos::RangePolicy(0, C.extent(1)),
			   KOKKOS_LAMBDA(const std::size_t j) {
 			     for (std::size_t i = 0; i <= j; ++i) {
			       ScalarType sum = {};
			       for (std::size_t k = 0; k < Aview.extent(0); ++k) {
				 sum += Aview(k,i) * Aview(k,j);
			       }
			       C(i,j) = beta*C(i,j) + alpha*sum;
			     }
			   });

    }
  }
}

}}
#endif
