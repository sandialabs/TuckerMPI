#ifndef TUCKER_KOKKOS_TRANSFORM_SLICES_HPP_
#define TUCKER_KOKKOS_TRANSFORM_SLICES_HPP_

#include "TuckerOnNode_Tensor.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include <cuchar>

namespace Tucker {

namespace impl{

// FIXME: this needs to be improved with a team level and nested reductions
template<
  class ItType, class DividingScaleView, class ShiftValuesBeforeScaleView>
struct Func{
  ItType itBegin_;
  std::size_t numContig_;
  std::size_t numSetsContig_;
  std::size_t distBetweenSets_;
  DividingScaleView dividing_scales_;
  ShiftValuesBeforeScaleView shift_values_before_scale_;

  Func(ItType it, std::size_t numContig, std::size_t numSetsContig,
       std::size_t distBetweenSets, DividingScaleView dividing_scales,
       ShiftValuesBeforeScaleView shift_values_before_scale)
    : itBegin_(it), numContig_(numContig), numSetsContig_(numSetsContig),
      distBetweenSets_(distBetweenSets), dividing_scales_(dividing_scales), shift_values_before_scale_(shift_values_before_scale){}

  KOKKOS_FUNCTION void operator()(int sliceIndex) const
  {
    auto it = itBegin_ + sliceIndex*numContig_;
    for(std::size_t c=0; c<numSetsContig_; c++)
      {
        for(std::size_t i=0; i<numContig_; i++)
        {
          auto & value = *(it+i);
          value = (value + shift_values_before_scale_(sliceIndex)) / dividing_scales_(sliceIndex);
        } // end for i
        it += distBetweenSets_;
      } // end for c
  }
};


template<
  class ScalarType, class ... TensorParams,
  class ViewDataType, class ... ViewParams>
void transform_slices(TuckerOnNode::Tensor<ScalarType, TensorParams...> Y,
                      int mode,
                      const Kokkos::View<ViewDataType, ViewParams...> dividing_scales,
                      const Kokkos::View<ViewDataType, ViewParams...> shift_values_before_scale)
{

  const int numSlices = Y.extent(mode);
  const int ndims = Y.rank();
  // Number of contiguous elements in a slice
  std::size_t numContig = Y.prod(0,mode-1,1);
  // Number of sets of contiguous elements per slice
  std::size_t numSetsContig = Y.prod(mode+1,ndims-1,1);
  // Distance between sets of contiguous elements
  std::size_t distBetweenSets = Y.prod(0,mode);

  auto itBegin = Kokkos::Experimental::begin(Y.data());
  Kokkos::parallel_for(numSlices,
		       Func(itBegin, numContig, numSetsContig, distBetweenSets,
			    dividing_scales, shift_values_before_scale));

};

} // end of impl


template<
  class ScalarType, class ... TensorParams,
  class ViewDataType, class ... ViewParams>
void transform_slices(TuckerOnNode::Tensor<ScalarType, TensorParams...> tensor,
                      int mode,
                      const Kokkos::View<ViewDataType, ViewParams...> dividing_scales,
                      const Kokkos::View<ViewDataType, ViewParams...> shift_values_before_scale)
{

  impl::transform_slices(tensor, mode, dividing_scales, shift_values_before_scale);
}

} // end of tucker
#endif

/*













  // If the tensor has no entries, no transformation is necessary
  size_t numEntries = tensor.size();
  if(numEntries == 0)
    return;

  // Compute the result
  int ndims = tensor.rank();
  int numSlices = tensor.extent(mode);
  // Number of contiguous elements in a slice
  size_t numContig = tensor.prod(0,mode-1,1);
  // Number of sets of contiguous elements per slice
  size_t numSetsContig = tensor.prod(mode+1,ndims-1,1);
  // Distance between sets of contiguous elements
  size_t distBetweenSets = tensor.prod(0,mode);

  ScalarType* dataPtr;
  int slice;
  size_t i, c;
  for(slice=0; slice<numSlices; slice++) // parallel_for
  {
    dataPtr = tensor.data().data() + slice*numContig;
    for(size_t c=0; c<numSetsContig; c++)
    {
      for(size_t i=0; i<numContig; i++){
        dataPtr[i] = (dataPtr[i] + shifts[slice]) / scales[slice];
      }
      dataPtr += distBetweenSets;
    }
  }

  // WIP
  // Kokkos idea
  /*
  Kokkos::View<ViewDataType, ViewParams...> dataPtr("data view", numEntries);
  Kokkos::RangePolicy<> policy_1(0, numSlices);
  Kokkos::parallel_for(
    "Loop",
    policy_1,
    KOKKOS_LAMBDA(const int slice) {
      std::cout << "slice: " << slice << std::endl;

    }
  );

  // typedef Kokkos::TeamPolicy<>::member_type team_handle;
  // Kokkos::TeamPolicy<> policy_1(numSlices, Kokkos::impl::AUTO, numSetsContig);

  // WIP
}

}*/
