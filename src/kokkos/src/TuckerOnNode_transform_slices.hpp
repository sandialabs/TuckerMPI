#ifndef TUCKERONNODE_TRANSFORM_SLICES_HPP_
#define TUCKERONNODE_TRANSFORM_SLICES_HPP_

#include "./impl/TuckerOnNode_transform_slices.hpp"

namespace TuckerOnNode{

template<
  class ScalarType, class ... TensorProps,
  class ViewDataType1, class ... ViewParams1,
  class ViewDataType2, class ... ViewParams2
  >
void transform_slices(Tensor<ScalarType, TensorProps...> tensor,
                      int mode,
                      const Kokkos::View<ViewDataType1, ViewParams1...> & dividing_scales,
                      const Kokkos::View<ViewDataType2, ViewParams2...> & pre_scaling_shifts)
{
  using tensor_type = Tensor<ScalarType, TensorProps...>;
  using tensor_layout = typename tensor_type::traits::array_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;

  // constraints
  static_assert(   std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
		&& std::is_same_v<std::remove_cv_t<tensor_value_type>, double>,
		   "TuckerOnNode::transform_slices: supports tensors with LayoutLeft" \
		   "and double scalar");

  // preconditions
  if(tensor.extent(mode) <= 0) {
    std::ostringstream oss;
    oss << "TuckerOnNode::transform_slices: "
        << "for mode = " << mode << " we have tensor.extent(mode) = " << tensor.extent(mode) << " <= 0";
    throw std::runtime_error(oss.str());
  }

  if(mode < 0) {
    throw std::runtime_error("mode must be non-negative");
  }

  // execute
  if(tensor.size() > 0) {
    impl::transform_slices(tensor, mode, dividing_scales, pre_scaling_shifts);
  }
}

} // end of tucker
#endif  // TUCKERONNODE_TRANSFORM_SLICES_HPP_
