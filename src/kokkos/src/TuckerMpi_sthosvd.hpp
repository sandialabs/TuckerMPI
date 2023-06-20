#ifndef TUCKER_KOKKOS_MPI_STHOSVD_HPP_
#define TUCKER_KOKKOS_MPI_STHOSVD_HPP_

#include "./impl/TuckerMpi_sthosvd_newgram_impl.hpp"

namespace TuckerMpi{

struct TagNewGram{};

template <class TagType, class ScalarType, class ...Properties, class TruncatorType>
[[nodiscard]] auto STHOSVD(TagType tag,
			   ::TuckerMpi::Tensor<ScalarType, Properties...> X,
			   TruncatorType && truncator,
			   const std::vector<int> & modeOrder,
			   bool flipSign)
{
  // constraints
  static_assert(std::is_same_v<TagType, TagNewGram>,
		"TuckerMpi::STHOSVD: invalid tag");

  using tensor_type = ::TuckerMpi::Tensor<ScalarType, Properties...>;
  using onnode_layout = typename tensor_type::traits::onnode_layout;
  using memory_space  = typename tensor_type::traits::memory_space;
  static_assert(std::is_same_v<onnode_layout, Kokkos::LayoutLeft>,
		"TuckerMpi::STHOSVD: currently only supporting tensor with layoutleft");
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, memory_space>::accessible,
		"TuckerMpi::STHOSVD: currently only supporting tensor accssible on host");

  // compute
  return impl::sthosvd_newgram(X, std::forward<TruncatorType>(truncator),
			       modeOrder, flipSign);
}

}
#endif
