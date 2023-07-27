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
  // Number of contiguous elements in a slice
  size_t numContig = tensor.prod(0,mode-1,1);
  std::cout << "numContig: " << numContig << std::endl;
  // Number of sets of contiguous elements per slice
  size_t numSetsContig = tensor.prod(mode+1,ndims-1,1);
  std::cout << "numSetsContig: " << numSetsContig << std::endl;
  // Distance between sets of contiguous elements
  size_t distBetweenSets = tensor.prod(0,mode);
  std::cout << "distBetweenSets: " << distBetweenSets << std::endl;

  ScalarType* dataPtr;
  int slice;
  size_t i, c;
  #pragma omp parallel for default(shared) private(slice,i,c,dataPtr)
  for(slice=0; slice<numSlices; slice++)
  {
    dataPtr = tensor.data().data() + slice*numContig;
    for(c=0; c<numSetsContig; c++)
    {
      for(i=0; i<numContig; i++)
        dataPtr[i] = (dataPtr[i] + shifts[slice]) / scales[slice];
      dataPtr += distBetweenSets;
    }
  }
  //

  // parallel_for(
  //   just on tensor data
  // )


  // DONE 1) FINIR CODE
  // DONE 2) TEST PASSE NORMALEMENT
  // 3) METTRE FOOR LOOP IN CMAKELISTS
  // 4) AJOUTER ANCIEN TESTS

}

}

#endif