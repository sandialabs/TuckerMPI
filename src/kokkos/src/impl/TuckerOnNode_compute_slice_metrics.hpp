
#ifndef IMPL_TUCKER_COMPUTE_SLICE_METRICS_HPP_
#define IMPL_TUCKER_COMPUTE_SLICE_METRICS_HPP_

#include "TuckerOnNode_MetricData.hpp"
#include <Kokkos_StdAlgorithms.hpp>

namespace TuckerOnNode {
namespace impl{

// FIXME: this needs to be improved with a team level and nested reductions
template<
  class ItType, class ScalarType, class MemSpace,
  class DeltaView, class nArrayView>
struct Func{
  using metric_data_t = MetricData<ScalarType, MemSpace>;

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

template <std::size_t n, class ScalarType, class ...Properties, class MemSpace>
void compute_slice_metrics(::TuckerOnNode::Tensor<ScalarType, Properties...> Y,
			   const int mode,
			   const std::array<Tucker::Metric, n> & metrics,
			   MetricData<ScalarType, MemSpace> metricsData)
{

  const int numSlices = Y.extent(mode);
  const int ndims = Y.rank();
  // Number of contiguous elements in a slice
  std::size_t numContig = Y.prod(0,mode-1,1);
  // Number of sets of contiguous elements per slice
  std::size_t numSetsContig = Y.prod(mode+1,ndims-1,1);
  // Distance between sets of contiguous elements
  std::size_t distBetweenSets = Y.prod(0,mode);

  Kokkos::View<ScalarType*, MemSpace> delta("delta", numSlices);
  Kokkos::View<int*, MemSpace> nArray("nArray", numSlices);
  auto itBegin = Kokkos::Experimental::cbegin(Y.data());
  Kokkos::parallel_for(numSlices,
		       Func(itBegin, numContig, numSetsContig, distBetweenSets,
			    delta, nArray, metricsData));

  // if variance is present, normalize
  if (std::find(metrics.begin(), metrics.end(), Tucker::Metric::VARIANCE) != metrics.end()){
    std::size_t sizeOfSlice = numContig*numSetsContig;
    normalize_variance(metricsData, numSlices, sizeOfSlice);
  }
}

}}
#endif  // IMPL_TUCKER_COMPUTE_SLICE_METRICS_HPP_
