#ifndef TUCKER_KOKKOS_MPI_STHOSVD_HPP_
#define TUCKER_KOKKOS_MPI_STHOSVD_HPP_

#include "./impl/TuckerMpi_sthosvd_newgram_impl.hpp"

namespace TuckerMpi{

enum class Method{ NewGram };

template <class ScalarType, class ...Properties, class TruncatorType>
[[nodiscard]] auto sthosvd(Method method,
			   ::TuckerMpi::Tensor<ScalarType, Properties...> tensor,
			   TruncatorType && truncator,
			   const std::vector<int> & modeOrder,
			   bool flipSign)
{
  // constraints
  using tensor_type = ::TuckerMpi::Tensor<ScalarType, Properties...>;
  using onnode_layout = typename tensor_type::traits::onnode_layout;
  static_assert(   std::is_same_v<onnode_layout, Kokkos::LayoutLeft>
		&& std::is_floating_point_v<ScalarType>,
		   "TuckerMpi::sthosvd: currently only supporting a tensor with LayoutLeft" \
		   "and floating point scalar");

  // preconditions
  assert(modeOrder.empty() || modeOrder.size() == (std::size_t) tensor.rank());

  // execute
  if (method == Method::NewGram){
    return impl::sthosvd_newgram(tensor, std::forward<TruncatorType>(truncator), modeOrder, flipSign);
  }
  else{
    throw std::runtime_error("TuckerMpi: sthosvd: invalid or unsupported method");
  }
}

}
#endif
