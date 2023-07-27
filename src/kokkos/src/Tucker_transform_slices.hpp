#ifndef TUCKER_KOKKOS_TRABNSFORM_SLICES_HPP_
#define TUCKER_KOKKOS_TRABNSFORM_SLICES_HPP_

#include <fstream> // TO REMOVE

namespace Tucker {

// Shift is applied before scale
// We divide by scaleVals, not multiply
template<class ScalarType, class ... TensorParams, class ViewDataType, class ... ViewParams>
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
  //size_t numContig = Y->size().prod(0,mode-1,1); // Number of contiguous elements in a slice
  //size_t numSetsContig = Y->size().prod(mode+1,ndims-1,1); // Number of sets of contiguous elements per slice
  //size_t distBetweenSets = Y->size().prod(0,mode); // Distance between sets of contiguous elements

  //

  // parallel_for(
  //   just on tensor data
  // )

}

}

#endif