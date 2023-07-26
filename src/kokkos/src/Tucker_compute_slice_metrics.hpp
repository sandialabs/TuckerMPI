
#ifndef TUCKER_KOKKOSONLY_COMPUTE_SLICE_METRICS_HPP_
#define TUCKER_KOKKOSONLY_COMPUTE_SLICE_METRICS_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include "Tucker_MetricData.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace TuckerOnNode {

template <class ScalarType, class ...Properties>
auto compute_slice_metrics(const Tensor<ScalarType, Properties...> & Y,
			   const int mode,
			   const std::vector<Tucker::Metric> & metrics)
{
  using tensor_type = Tensor<ScalarType, Properties...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;

  //
  // preconditions
  //
  if(Y.extent(mode) <= 0) {
    std::ostringstream oss;
    oss << "TuckerOnNode::compute_slice_metrics: "
        << "for mode = " << mode << " we have Y.extent(mode) = " << Y.extent(mode) << " <= 0";
    throw std::runtime_error(oss.str());
  }

  if(mode < 0) {
    throw std::runtime_error("mode must be non-negative");
  }

  //
  // execute
  //
  const int numSlices = Y.extent(mode);
  auto result = TuckerOnNode::MetricData<ScalarType, tensor_mem_space>(metrics, numSlices);
  if(Y.size() == 0) { return result; }

  // // Initialize the result
  // std::vector<ScalarType> delta;
  // std::vector<int> nArray;
  // if((metrics & MEAN) || (metrics & VARIANCE)) {
  //   delta = std::vector<ScalarType>(numSlices);
  //   nArray = std::vector<int>(numSlices);
  // }
  // if(metrics & MIN) {
  //   Kokkos::fill(result.minData(), std::numeric_limits<ScalarType>::max());
  // }

  const int ndims = Y.rank();
  std::size_t numContig = Y.prod(0,mode-1,1); // Number of contiguous elements in a slice
  std::size_t numSetsContig = Y.prod(mode+1,ndims-1,1); // Number of sets of contiguous elements per slice
  std::size_t distBetweenSets = Y.prod(0,mode); // Distance between sets of contiguous elements

  auto itBegin = Kokkos::Experimental::cbegin(Y.data());
  Kokkos::parallel_for("computeResultData", numSlices,
	KOKKOS_LAMBDA(const int& sliceIndex) {
	  auto it = itBegin + sliceIndex*numContig;
	  for(std::size_t c=0; c<numSetsContig; c++)
	    {
	      for(std::size_t i=0; i<numContig; i++){
		if (result.contains(Tucker::Metric::MIN)){
		  auto view = result.get(Tucker::Metric::MIN);
		  auto & value = view(sliceIndex);
		  value = Kokkos::min(value, *(it+i));
		}
	      } // end for(i=0; i<numContig; i++)

	      it += distBetweenSets;
	    } // end for(c=0; c<numSetsContig; c++)
	});

  // if(metrics & VARIANCE) {
  //   size_t sizeOfSlice = numContig*numSetsContig;
  //   for(int i=0; i<numSlices; i++){
  //     result.getVarianceData()[i] /= (ScalarType)sizeOfSlice;
  //   }
  // }

  return result;
}

}
#endif
