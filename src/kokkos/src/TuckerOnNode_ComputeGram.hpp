#ifndef TUCKERKOKKOS_COMP_GRAM_HPP_
#define TUCKERKOKKOS_COMP_GRAM_HPP_

#include "./impl/TuckerOnNode_compute_gram_host_impl.hpp"
#include "./impl/TuckerOnNode_compute_gram_kokkos_impl.hpp"

namespace TuckerOnNode{

template<class ScalarType, class ...Properties>
auto compute_gram(Tensor<ScalarType, Properties...> Y,
		  const std::size_t n)
{
  using tensor_type = Tensor<ScalarType, Properties...>;
  using memory_space = typename tensor_type::traits::memory_space;

  const std::size_t nrows = Y.extent(n);
  Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space> S_d("S", nrows, nrows);

#if defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)
  /* this code below works for any backend, even if obviously inefficiently,
     and is left here for now as a backup case if we need it to verify things */
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible){
    auto S_h = Kokkos::create_mirror(S_d);
    impl::compute_gram_host(Y, n, S_h);
    Kokkos::deep_copy(S_d, S_h);
  }
  else
  {
    const auto Ydims = Tucker::impl::create_stdvec_from_view(Y.dimensionsOnHost());
    auto Y_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y.data());
    Tensor<ScalarType, Kokkos::HostSpace> Y_h(Ydims);
    Kokkos::deep_copy(Y_h.data(), Y_view);
    auto S_h = Kokkos::create_mirror(S_d);
    impl::compute_gram_host(Y_h, n, S_h);
    Kokkos::deep_copy(S_d, S_h);
   }

#else
  impl::compute_gram_kokkos(Y, n, S_d);
#endif

  return S_d;
}


}
#endif
