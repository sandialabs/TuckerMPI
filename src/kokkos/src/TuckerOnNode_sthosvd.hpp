#ifndef TUCKER_ONNODE_KOKKOS_STHOSVD_HPP_
#define TUCKER_ONNODE_KOKKOS_STHOSVD_HPP_

#include "./impl/TuckerOnNode_sthosvd_gram_impl.hpp"

namespace TuckerOnNode{

enum class Method{
  Gram
};

template <class ScalarType, class ...Properties, class TruncatorType>
[[nodiscard]] auto sthosvd(Method method,
			   ::TuckerOnNode::Tensor<ScalarType, Properties...> X,
			   TruncatorType && truncator,
			   bool flipSign)
{
  if (method == Method::Gram){
    return impl::sthosvd_gram(X, std::forward<TruncatorType>(truncator), flipSign);
  }
  else{
    throw std::runtime_error("TuckerOnNode::sthosvd: invalid or unsupported method");
  }
}

}
#endif
