#ifndef IMPL_TUCKERONNODE_TRANSFORM_SLICES_HPP_
#define IMPL_TUCKERONNODE_TRANSFORM_SLICES_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <cuchar>

namespace TuckerOnNode{
namespace impl{

template<class ValueType, class ShiftValuesBeforeScaleView, class DividingScaleView>
struct ShiftThenDivide
{
  ShiftValuesBeforeScaleView pre_scaling_shifts_;
  DividingScaleView dividing_scales_;

  ShiftThenDivide(ShiftValuesBeforeScaleView pre_scaling_shifts,
		  DividingScaleView dividing_scales)
    : pre_scaling_shifts_(pre_scaling_shifts), dividing_scales_(dividing_scales){}

  KOKKOS_FUNCTION void operator()(int index, ValueType & valueInOut) const
  {
    valueInOut = (valueInOut + pre_scaling_shifts_(index)) / dividing_scales_(index);
  }
};


// FIXME: this needs to be improved with a team level and nested reductions
template<class ItType, class ModifyingOp>
struct TransformSlicesFunc{
  ItType itBegin_;
  std::size_t numContig_;
  std::size_t numSetsContig_;
  std::size_t distBetweenSets_;
  ModifyingOp op_;

  TransformSlicesFunc(ItType it, std::size_t numContig, std::size_t numSetsContig,
       std::size_t distBetweenSets, ModifyingOp op)
    : itBegin_(it), numContig_(numContig), numSetsContig_(numSetsContig),
      distBetweenSets_(distBetweenSets), op_(op){}

  KOKKOS_FUNCTION void operator()(int sliceIndex) const
  {
    auto it = itBegin_ + sliceIndex*numContig_;
    for(std::size_t c=0; c<numSetsContig_; c++){
      for(std::size_t i=0; i<numContig_; i++){
	op_(sliceIndex, *(it+i));
      }
      it += distBetweenSets_;
    }
  }
};

template<
  class ScalarType, class ... TensorProps,
  class ViewDataType1, class ... ViewParams1,
  class ViewDataType2, class ... ViewParams2>
void transform_slices(TuckerOnNode::Tensor<ScalarType, TensorProps...> tensor,
                      int mode,
                      const Kokkos::View<ViewDataType1, ViewParams1...> & dividing_scales,
                      const Kokkos::View<ViewDataType2, ViewParams2...> & pre_scaling_shifts)
{
  using tensor_type = Tensor<ScalarType, TensorProps...>;
  using tensor_layout = typename tensor_type::traits::array_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;

  // constraints
  static_assert(   std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
		&& std::is_floating_point_v<tensor_value_type>,
		   "TuckerOnNode::impl::transform_slices: supports tensors with LayoutLeft" \
		   "and floating point scalar");

  const int ndims = tensor.rank();
  // Number of contiguous elements in a slice
  std::size_t numContig = tensor.prod(0,mode-1,1);
  // Number of sets of contiguous elements per slice
  std::size_t numSetsContig = tensor.prod(mode+1,ndims-1,1);
  // Distance between sets of contiguous elements
  std::size_t distBetweenSets = tensor.prod(0,mode);

  if (tensor.size() == 0){
    return;
  }

  auto itBegin = Kokkos::Experimental::begin(tensor.data());

  using scaling_view_t = Kokkos::View<ViewDataType1, ViewParams1...>;
  using shifts_view_t  = Kokkos::View<ViewDataType2, ViewParams2...>;
  using op_t = ShiftThenDivide<tensor_value_type, shifts_view_t, scaling_view_t>;
  Kokkos::parallel_for(tensor.extent(mode),
		       TransformSlicesFunc(itBegin, numContig, numSetsContig, distBetweenSets,
					   op_t(pre_scaling_shifts, dividing_scales) ));

};

} // end namespace impl
} // end namespace TuckerOnNode
#endif  // IMPL_TUCKERONNODE_TRANSFORM_SLICES_HPP_
