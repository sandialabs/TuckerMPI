
#ifndef IMPL_TUCKER_COMPUTE_SLICE_METRICS_HPP_
#define IMPL_TUCKER_COMPUTE_SLICE_METRICS_HPP_

#include "TuckerOnNode_MetricData.hpp"
#include <Kokkos_StdAlgorithms.hpp>

namespace TuckerOnNode {
namespace impl{

template<class ItType, class ScalarType, class MemSpace>
struct MatricsHiParFunctor{
  using metric_data_t = MetricData<ScalarType, MemSpace>;

  ItType itBegin_;
  std::size_t numContig_;
  std::size_t numSetsContig_;
  std::size_t distBetweenSets_;
  metric_data_t metricData_;

  MatricsHiParFunctor(ItType it, std::size_t numContig, std::size_t numSetsContig,
       std::size_t distBetweenSets, metric_data_t metricData)
    : itBegin_(it), numContig_(numContig), numSetsContig_(numSetsContig),
      distBetweenSets_(distBetweenSets), metricData_(metricData){}

  template<class member_type>
  KOKKOS_FUNCTION void operator()(const member_type & member) const
  {
    const int sliceIndex = member.league_rank();

    using minmax_reducer_t = Kokkos::MinMax<ScalarType, typename member_type::execution_space>;
    using minmax_value_t = typename minmax_reducer_t::value_type;

    ScalarType sum = {};
    minmax_value_t minMaxRes = {};
    ScalarType mean = {};
    ScalarType var = {};
    std::size_t isum = {};
    std::size_t loopCount = numSetsContig_ * numContig_;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, loopCount),
			    [=] (std::size_t k,
				 ScalarType& lsum, minmax_value_t & lminmax,
				 ScalarType& lmean, ScalarType & lvar,
				 std::size_t & lisum)
			    {
			      const std::size_t c = k / numContig_;
			      const std::size_t i = k % numContig_;
			      auto it = itBegin_ + sliceIndex*numContig_ + c * distBetweenSets_ + i;
			      const auto myvalue = *it;

			      lsum += myvalue;
			      lminmax.min_val = Kokkos::min(lminmax.min_val, myvalue);
			      lminmax.max_val = Kokkos::max(lminmax.max_val, myvalue);

			      lisum++;
			      const ScalarType delta = myvalue - lmean;
			      lmean += delta/(ScalarType) lisum;
			      lvar += delta * (myvalue - lmean);
			    },
			    sum, minmax_reducer_t(minMaxRes), mean, var, isum);

    if (metricData_.contains(Tucker::Metric::SUM)){
      auto view = metricData_.get(Tucker::Metric::SUM);
      view(sliceIndex) = sum;
    }
    if (metricData_.contains(Tucker::Metric::MIN)){
      auto view = metricData_.get(Tucker::Metric::MIN);
      view(sliceIndex) = minMaxRes.min_val;
    }
    if (metricData_.contains(Tucker::Metric::MAX)){
      auto view = metricData_.get(Tucker::Metric::MAX);
      view(sliceIndex) = minMaxRes.max_val;
    }
    if (metricData_.contains(Tucker::Metric::MEAN)){
      auto view = metricData_.get(Tucker::Metric::MEAN);
      view(sliceIndex) = mean;
    }
    if (metricData_.contains(Tucker::Metric::VARIANCE)){
      auto view = metricData_.get(Tucker::Metric::VARIANCE);
      view(sliceIndex) = var/ (ScalarType) loopCount;
    }
  }
};

template<
  class ItType, class ScalarType, class MemSpace,
  class DeltaView, class nArrayView>
struct MetricsNaiveFunctor{
  using metric_data_t = MetricData<ScalarType, MemSpace>;

  ItType itBegin_;
  std::size_t numContig_;
  std::size_t numSetsContig_;
  std::size_t distBetweenSets_;
  DeltaView delta_;
  nArrayView nArray_;
  metric_data_t metricData_;

  MetricsNaiveFunctor(ItType it, std::size_t numContig, std::size_t numSetsContig,
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
void compute_slice_metrics_use_hierarc_par(Tensor<ScalarType, Properties...> Y,
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

  using tensor_type = Tensor<ScalarType, Properties...>;
  using exespace = typename tensor_type::traits::execution_space;
  using policy_t = Kokkos::TeamPolicy<exespace>;
  policy_t policy(numSlices, Kokkos::AUTO);
  auto itBegin = Kokkos::Experimental::cbegin(Y.data());
  Kokkos::parallel_for(policy,
		       MatricsHiParFunctor(itBegin, numContig, numSetsContig,
					   distBetweenSets, metricsData));
}

template <std::size_t n, class ScalarType, class ...Properties, class MemSpace>
void compute_slice_metrics_naive(Tensor<ScalarType, Properties...> Y,
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
		       MetricsNaiveFunctor(itBegin, numContig, numSetsContig,
					   distBetweenSets, delta, nArray, metricsData));
  // if variance is present, normalize
  if (std::find(metrics.begin(), metrics.end(), Tucker::Metric::VARIANCE) != metrics.end()){
    std::size_t sizeOfSlice = numContig*numSetsContig;
    normalize_variance(metricsData, numSlices, sizeOfSlice);
  }
}

}}
#endif  // IMPL_TUCKER_COMPUTE_SLICE_METRICS_HPP_
