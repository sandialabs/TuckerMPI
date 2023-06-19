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
  if constexpr(std::is_same_v<TagType, TagNewGram>){
    return impl::sthosvd_newgram(X, std::forward<TruncatorType>(truncator),
				 modeOrder, flipSign);
  }
  else{
    throw std::runtime_error("TuckerMpi: sthosvd: invalid or unsupported tag specified");
    return {};
  }
}

}
#endif
