#ifndef TUCKERKOKKOS_COMP_GRAM_HPP_
#define TUCKERKOKKOS_COMP_GRAM_HPP_

#include "./impl/TuckerOnNode_compute_gram_host_impl.hpp"

namespace TuckerOnNode{

template<class ScalarType, class ...Properties>
auto compute_gram(Tensor<ScalarType, Properties...> Y,
		 const std::size_t n)
{
  using tensor_type = Tensor<ScalarType, Properties...>;
  using memory_space = typename tensor_type::traits::memory_space;
  const std::size_t nrows = Y.extent(n);
  Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space> S_d("S", nrows, nrows);
  auto S_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), S_d);
  impl::compute_gram_host(Y, n, S_h.data(), nrows);
  Kokkos::deep_copy(S_d, S_h);
  return S_d;
}


}
#endif
