
#ifndef TUCKER_KOKKOSONLY_COMPUTE_SLICE_METRICS_HPP_
#define TUCKER_KOKKOSONLY_COMPUTE_SLICE_METRICS_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include "Tucker_MetricData.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace TuckerOnNode {

// FIXME: this needs to be improved with a team level and nested reductions
template<
  class ItType, class ScalarType, class MemSpace,
  class DeltaView, class nArrayView>
struct Func{
  using metric_data_t = TuckerOnNode::MetricData<ScalarType, MemSpace>;

  ItType itBegin_;
  std::size_t numContig_;
  std::size_t numSetsContig_;
  std::size_t distBetweenSets_;
  DeltaView delta_;
  nArrayView nArray_;
  metric_data_t metricData_;

  Func(ItType it, std::size_t numContig, std::size_t numSetsContig,
       std::size_t distBetweenSets, DeltaView delta,
       nArrayView nArray, metric_data_t metricData)
    : itBegin_(it), numContig_(numContig), numSetsContig_(numSetsContig),
      distBetweenSets_(distBetweenSets), delta_(delta), nArray_(nArray),
      metricData_(metricData){}

  KOKKOS_FUNCTION void operator()(int sliceIndex) const
  {
    auto it = itBegin_ + sliceIndex*numContig_;
    for(std::size_t c=0; c<numSetsContig_; c++)
      {
	for(std::size_t i=0; i<numContig_; i++)
	{
	  if (metricData_.contains(Tucker::Metric::MIN)){
	    auto view = metricData_.get(Tucker::Metric::MIN);
	    auto & value = view(sliceIndex);
	    value = Kokkos::min(value, *(it+i));
	  }

	  if (metricData_.contains(Tucker::Metric::MAX)){
	    auto view = metricData_.get(Tucker::Metric::MAX);
	    auto & value = view(sliceIndex);
	    value = Kokkos::max(value, *(it+i));
	  }

	  if (metricData_.contains(Tucker::Metric::SUM)){
	    auto view = metricData_.get(Tucker::Metric::SUM);
	    auto & value = view(sliceIndex);
	    value += *(it+i);
	  }

	  const auto mean = metricData_.contains(Tucker::Metric::MEAN);
	  const auto var = metricData_.contains(Tucker::Metric::VARIANCE);
	  if (mean || var){
	    auto view_mean = metricData_.get(Tucker::Metric::MEAN);
	    auto & view_mean_val = view_mean(sliceIndex);
	    delta_(sliceIndex) = *(it+i) - view_mean_val;
	    nArray_(sliceIndex)++;
	    view_mean_val += delta_(sliceIndex)/nArray_(sliceIndex);
	  }

	  if (var){
	    auto view_var = metricData_.get(Tucker::Metric::VARIANCE);
	    auto view_mean = metricData_.get(Tucker::Metric::MEAN);
	    const auto incr = *(it+i)-view_mean(sliceIndex);
	    view_var(sliceIndex) += (delta_[sliceIndex]*incr);
	  }

	} // end for i
	it += distBetweenSets_;

      } // end for c
  }
};

template<class ScalarType, class MemSpace>
void normalize_variance(MetricData<ScalarType, MemSpace> metricData,
			int numSlices, int sizeOfSlice)
{
  Kokkos::parallel_for(numSlices,
		       KOKKOS_LAMBDA(int i){
			 auto view_var = metricData.get(Tucker::Metric::VARIANCE);
			 view_var(i) /= (ScalarType) sizeOfSlice;
		       });
}

template <class ScalarType, class ...Properties>
auto compute_slice_metrics(Tensor<ScalarType, Properties...> Y,
			   const int mode,
			   const std::vector<Tucker::Metric> & metrics)
{
  using tensor_type = Tensor<ScalarType, Properties...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;
  using tensor_layout = typename tensor_type::traits::array_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;

  //
  // constraints
  static_assert(   std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
		&& std::is_floating_point_v<tensor_value_type>,
		   "TuckerOnNode::compute_slice_metrics: supports tensors with LayoutLeft" \
		   "and floating point scalar");

  //
  // preconditions
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
  const int numSlices = Y.extent(mode);

  auto result = TuckerOnNode::MetricData<ScalarType, tensor_mem_space>(metrics, numSlices);
  if(Y.size() == 0) { return result; }

  const int ndims = Y.rank();
  // Number of contiguous elements in a slice
  std::size_t numContig = Y.prod(0,mode-1,1);
  // Number of sets of contiguous elements per slice
  std::size_t numSetsContig = Y.prod(mode+1,ndims-1,1);
  // Distance between sets of contiguous elements
  std::size_t distBetweenSets = Y.prod(0,mode);

  Kokkos::View<ScalarType*, tensor_mem_space> delta("delta", numSlices);
  Kokkos::View<int*, tensor_mem_space> nArray("nArray", numSlices);
  auto itBegin = Kokkos::Experimental::cbegin(Y.data());
  Kokkos::parallel_for(numSlices,
		       Func(itBegin, numContig, numSetsContig, distBetweenSets,
			    delta, nArray, result));

  // if variance is present, normalize
  if (std::find(metrics.begin(), metrics.end(), Tucker::Metric::VARIANCE) != metrics.end()){
    size_t sizeOfSlice = numContig*numSetsContig;
    normalize_variance(result, numSlices, sizeOfSlice);
  }

  return result;
}

}
#endif
