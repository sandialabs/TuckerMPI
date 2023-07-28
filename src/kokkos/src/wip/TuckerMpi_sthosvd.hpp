#ifndef TUCKER_KOKKOS_MPI_STHOSVD_HPP_
#define TUCKER_KOKKOS_MPI_STHOSVD_HPP_

#include "./impl/TuckerMpi_sthosvd_newgram_impl.hpp"

namespace TuckerMpi{

enum class Method{
  NewGram
};

template <class ScalarType, class ...Properties, class TruncatorType>
[[nodiscard]] auto sthosvd(Method method,
			   ::TuckerMpi::Tensor<ScalarType, Properties...> X,
			   TruncatorType && truncator,
			   const std::vector<int> & modeOrder,
			   bool flipSign)
{
  // constraints
  using tensor_type = ::TuckerMpi::Tensor<ScalarType, Properties...>;
  using onnode_layout = typename tensor_type::traits::onnode_layout;
  static_assert(std::is_same_v<onnode_layout, Kokkos::LayoutLeft>,
		"TuckerMpi::sthosvd: currently only supporting a tensor with LayoutLeft");

  // preconditions
  assert(modeOder.size() == (std::size_t) X.rank());

  // execute
  if (method == Method::NewGram){
    return impl::sthosvd_newgram(X, std::forward<TruncatorType>(truncator), modeOrder, flipSign);
  }
  else{
    throw std::runtime_error("TuckerMpi: sthosvd: invalid or unsupported method");
  }
}

}
#endif
