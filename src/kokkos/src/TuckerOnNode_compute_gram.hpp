#ifndef TUCKERONNODE_COMPUTE_GRAM_HPP_
#define TUCKERONNODE_COMPUTE_GRAM_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include "./impl/TuckerOnNode_compute_gram_host_impl.hpp"
#include "./impl/TuckerOnNode_compute_gram_kokkos_impl.hpp"

namespace TuckerOnNode{

template<class ScalarType, class ...Properties>
auto compute_gram(Tensor<ScalarType, Properties...> tensor,
		  const std::size_t n)
{

  using tensor_type       = Tensor<ScalarType, Properties...>;
  using memory_space      = typename tensor_type::traits::memory_space;
  using tensor_layout     = typename tensor_type::traits::array_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;

  // constraints
  static_assert(   std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
    && std::is_same_v<std::remove_cv_t<tensor_value_type>, double>,
		   "TuckerOnNode::compute_gram: supports tensors with LayoutLeft" \
		   "and double scalar type");

  const std::size_t nrows = tensor.extent(n);
  Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space> S_d("S", nrows, nrows);

#if defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)
  /* this code below works for any backend, even if obviously inefficiently,
     and is left here for now as a backup case if we need it to verify things */
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible){
    auto S_h = Kokkos::create_mirror(S_d);
    impl::compute_gram_host(tensor, n, S_h);
    Kokkos::deep_copy(S_d, S_h);
  }
  else
  {
    const auto tensordims = Tucker::impl::create_stdvec_from_view(tensor.dimensionsOnHost());
    auto tensor_view = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), tensor.data());
    Tensor<ScalarType, Kokkos::HostSpace> tensor_h(tensordims);
    Kokkos::deep_copy(tensor_h.data(), tensor_view);
    auto S_h = Kokkos::create_mirror(S_d);
    impl::compute_gram_host(tensor_h, n, S_h);
    Kokkos::deep_copy(S_d, S_h);
   }

#else
  impl::compute_gram_kokkos(tensor, n, S_d);
#endif

  return S_d;
}

}
#endif  // TUCKERONNODE_COMPUTE_GRAM_HPP_
