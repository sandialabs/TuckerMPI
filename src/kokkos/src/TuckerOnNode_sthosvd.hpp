#ifndef TUCKER_ONNODE_KOKKOS_STHOSVD_HPP_
#define TUCKER_ONNODE_KOKKOS_STHOSVD_HPP_

#include "./impl/TuckerOnNode_sthosvd_gram_impl.hpp"

namespace TuckerOnNode{

enum class Method{ Gram };

template <class ScalarType, class ...Properties, class TruncatorType>
[[nodiscard]] auto sthosvd(Method method,
			   ::TuckerOnNode::Tensor<ScalarType, Properties...> X,
			   TruncatorType && truncator,
			   bool flipSign)
{
  // constraints
  using tensor_type       = ::TuckerOnNode::Tensor<ScalarType, Properties...>;
  using tensor_layout     = typename tensor_type::traits::array_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;
  static_assert(   std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
		&& std::is_floating_point_v<tensor_value_type>,
		   "TuckerOnNode::sthosvd: currently supports tensors with LayoutLeft" \
		   "and floating point scalar");

  if (method == Method::Gram){
    return impl::sthosvd_gram(X, std::forward<TruncatorType>(truncator), flipSign);
  }
  else{
    throw std::runtime_error("TuckerOnNode::sthosvd: invalid or unsupported method");
  }
}

}
#endif
