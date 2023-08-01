#ifndef TUCKERKOKKOS_COMP_GRAM_KOKKOS_IMPL_HPP_
#define TUCKERKOKKOS_COMP_GRAM_KOKKOS_IMPL_HPP_

#include "Tucker_syrk_kokkos.hpp"

namespace TuckerOnNode{
namespace impl{

template <class ScalarType, class DataType, class ...ViewProps, class ...Properties>
void compute_gram_kokkos(Tensor<ScalarType, Properties...> Y,
			 const std::size_t n,
			 Kokkos::View<DataType, ViewProps...> C)
{
  using tensor_type   = Tensor<ScalarType, Properties...>;
  using tensor_layout = typename tensor_type::traits::array_layout;
  using view_type   = Kokkos::View<DataType, ViewProps...>;
  using view_layout = typename view_type::array_layout;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
		&& std::is_same_v<view_layout, Kokkos::LayoutLeft>);

  const int nrows = (int)Y.extent(n);
  auto Y_rawPtr = Y.data().data();
  auto gramPtr = C.data();
  using C_view_type = Kokkos::View<DataType, ViewProps...>;
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
      if((std::size_t)i != n) { ncols *= (int)Y.extent(i); }
    }

    // symmetric rank-k update: C := alpha*A*A' + beta*C
    // which corresponds to:
    //   call syrk('U', 'N', nrows, ncols, alpha, Aptr, nrows, beta, Cptr, C.extent(0))
    const ScalarType alpha = 1;
    const ScalarType beta = 0;
    umv_type Aview(Y.data().data(), Y.extent(0), ncols);
    Tucker::impl::syrk_kokkos("U", "N", alpha, Aview, beta, C);
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
      // symmetric rank-k update as follows: C := alpha*A'*A + beta*C
      // which corresponds to:
      //   dsyrk('U', 'T', nrows, ncols, alpha, Aptr, ncols, beta, Cptr, C.extent(0))
      const ScalarType alpha = 1;
      const ScalarType beta = (i==0) ? 0 : 1;
      auto Aptr = Y_rawPtr+i*nrows*ncols;
      umv_type Aview(Aptr, ncols, nrows);
      Tucker::impl::syrk_kokkos("U", "T", alpha, Aview, beta, C);
   }
  }
}

}}
#endif
