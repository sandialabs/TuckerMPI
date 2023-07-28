#ifndef TUCKER_KOKKOS_TRABNSFORM_SLICES_HPP_
#define TUCKER_KOKKOS_TRABNSFORM_SLICES_HPP_

#include <fstream> // TO REMOVE

namespace Tucker {

// Shift is applied before scale
// We divide by scaleVals, not multiply
template<
  class ScalarType, class ... TensorParams,
  class ViewDataType, class ... ViewParams>
void transform_slices(TuckerOnNode::Tensor<ScalarType, TensorParams...> tensor,
                      int mode,
                      const Kokkos::View<ViewDataType, ViewParams...> scales,
                      const Kokkos::View<ViewDataType, ViewParams...> shifts)
{
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
  for(slice=0; slice<numSlices; slice++)
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
  */
  // typedef Kokkos::TeamPolicy<>::member_type team_handle;
  // Kokkos::TeamPolicy<> policy_1(numSlices, Kokkos::impl::AUTO, numSetsContig);

  // WIP
}

}

#endif