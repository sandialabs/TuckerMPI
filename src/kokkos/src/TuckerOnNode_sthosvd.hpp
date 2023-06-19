#ifndef TUCKER_ONNODE_KOKKOS_STHOSVD_HPP_
#define TUCKER_ONNODE_KOKKOS_STHOSVD_HPP_

#include "./impl/TuckerOnNode_sthosvd_gram_impl.hpp"

namespace TuckerOnNode{

struct TagGram{};

template <class TagType, class ScalarType, class ...Properties, class TruncatorType>
[[nodiscard]] auto STHOSVD(TagType tag,
			   ::TuckerOnNode::Tensor<ScalarType, Properties...> X,
			   TruncatorType && truncator,
			   bool flipSign)
{
  if constexpr(std::is_same_v<TagType, TagGram>){
    return impl::sthosvd_gram(X, std::forward<TruncatorType>(truncator), flipSign);
  }
  else{
    throw std::runtime_error("TuckerOnNode: sthosvd: invalid or unsupported tag specified");
    return {};
  }
}

}
#endif
