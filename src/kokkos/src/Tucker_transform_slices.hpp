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

  std::cout << "init" << std::endl;

}

}

#endif


//=============================
/*
template <class scalar_t>
void transformSlices(Tensor<scalar_t>* Y, int mode, const scalar_t* scales, const scalar_t* shifts)
{
  // If the tensor has no entries, no transformation is necessary
  size_t numEntries = Y->getNumElements();
  if(numEntries == 0)
    return;

  // Compute the result
  int ndims = Y->N();
  int numSlices = Y->size(mode);
  size_t numContig = Y->size().prod(0,mode-1,1); // Number of contiguous elements in a slice
  size_t numSetsContig = Y->size().prod(mode+1,ndims-1,1); // Number of sets of contiguous elements per slice
  size_t distBetweenSets = Y->size().prod(0,mode); // Distance between sets of contiguous elements

  scalar_t* dataPtr;
  int slice;
  size_t i, c;
  #pragma omp parallel for default(shared) private(slice,i,c,dataPtr)
  for(slice=0; slice<numSlices; slice++)
  {
    dataPtr = Y->data() + slice*numContig;
    for(c=0; c<numSetsContig; c++)
    {
      for(i=0; i<numContig; i++)
        dataPtr[i] = (dataPtr[i] + shifts[slice]) / scales[slice];
      dataPtr += distBetweenSets;
    }
  }
}*/