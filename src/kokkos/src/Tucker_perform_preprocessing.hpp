#ifndef TUCKER_PERFORM_PREPROCESSING_HPP_
#define TUCKER_PERFORM_PREPROCESSING_HPP_

#include "Tucker_fwd.hpp"
#include "Tucker_compute_slice_metrics.hpp"
#include "Tucker_transform_slices.hpp"
#include "Kokkos_Core.hpp"

namespace TuckerOnNode{

struct UseMax{};
struct UseMinMax{};
struct UseStandardCentering{};

template<
  class MetricsDataType, class ViewScales, class ViewShifts>
struct NormalizeFunc{
  MetricsDataType metrics_;
  ViewScales scales_;
  ViewShifts shifts_;
  typename ViewScales::const_value_type stdThresh_ = {};

  NormalizeFunc(MetricsDataType metrics, ViewScales scales, ViewShifts shifts)
    : metrics_(metrics), scales_(scales), shifts_(shifts){}

  NormalizeFunc(MetricsDataType metrics, ViewScales scales,
		ViewShifts shifts, typename ViewScales::const_value_type stdThresh)
    : metrics_(metrics), scales_(scales), shifts_(shifts), stdThresh_(stdThresh){}

  KOKKOS_FUNCTION void operator()(const UseMax /*tag*/, int i) const{
    auto view_min = metrics_.get(Tucker::Metric::MIN);
    auto view_max = metrics_.get(Tucker::Metric::MAX);
    scales_(i) = Kokkos::max(Kokkos::abs(view_min(i)), Kokkos::abs(view_max(i)));
    shifts_(i) = 0;
  }

  KOKKOS_FUNCTION void operator()(const UseMinMax /*tag*/, int i) const{
    auto view_min = metrics_.get(Tucker::Metric::MIN);
    auto view_max = metrics_.get(Tucker::Metric::MAX);
    scales_(i) = view_max(i) - view_min(i);
    shifts_(i) = -view_min(i);
  }

  KOKKOS_FUNCTION void operator()(const UseStandardCentering /*tag*/, int i) const{
    auto view_var  = metrics_.get(Tucker::Metric::VARIANCE);
    auto view_mean = metrics_.get(Tucker::Metric::MEAN);
    scales_(i) = Kokkos::sqrt(view_var(i));

    if(scales_(i) < stdThresh_) { scales_(i) = 1; }
    else{ shifts_(i) = -view_mean(i); }
  }
};

auto set_target_metrics(const std::string & scalingType)
{
  using result_type = std::vector<Tucker::Metric>;
  if(scalingType == "Max") {
    return result_type{Tucker::Metric::MIN, Tucker::Metric::MAX};
  }
  else if(scalingType == "MinMax") {
    return result_type{Tucker::Metric::MIN, Tucker::Metric::MAX};
  }
  else if(scalingType == "StandardCentering") {
    return result_type{Tucker::Metric::MEAN, Tucker::Metric::VARIANCE};
  }
  else {
    throw std::runtime_error("Error: invalid scaling type");
    return result_type{}; // just to make compiler happy
  }
}

template <class ScalarType, class ...Props>
auto normalize_tensor(const TuckerOnNode::Tensor<ScalarType, Props...> & X,
		      const std::string & scalingType,
		      const int scaleMode,
		      const ScalarType stdThresh)
{
  /* normalizing a tensor involves these steps:
     1. compute metrics for the given scaleMode
     2. use metrics to fill scales and shifts depending on the scalingType
     3. use scales and shifts to normalize tensor data
  */

  using tensor_type = TuckerOnNode::Tensor<ScalarType, Props...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;

  //
  // 1. compute metrics for the given scaleMode
  const auto targetMetrics = set_target_metrics(scalingType);
  auto dataMetrics = TuckerOnNode::compute_slice_metrics(X, scaleMode, targetMetrics);

  //
  // 2. use metrics to fill scales and shifts depending on the scalingType
  Kokkos::View<ScalarType*, tensor_mem_space> scales("scales", X.extent(scaleMode));
  Kokkos::View<ScalarType*, tensor_mem_space> shifts("shifts", X.extent(scaleMode));
  if(scalingType == "Max") {
    NormalizeFunc func(dataMetrics, scales, shifts);
    std::cout << "Normalizing the tensor by maximum entry - mode " << scaleMode << std::endl;
    Kokkos::parallel_for(Kokkos::RangePolicy<UseMax>(0, X.extent(scaleMode)), func);
  }

  else if(scalingType == "MinMax") {
    NormalizeFunc func(dataMetrics, scales, shifts);
    std::cout << "Normalizing the tensor using minmax scaling - mode " << scaleMode << std::endl;
    Kokkos::parallel_for(Kokkos::RangePolicy<UseMinMax>(0, X.extent(scaleMode)), func);
  }

  else if(scalingType == "StandardCentering") {
    NormalizeFunc func(dataMetrics, scales, shifts, stdThresh);
    std::cout << "Normalizing the tensor using standard centering - mode " << scaleMode << std::endl;
    Kokkos::parallel_for(Kokkos::RangePolicy<UseStandardCentering>(0, X.extent(scaleMode)), func);
  }

  //
  // 3. use scales and shifts to normalize tensor data
  transform_slices(X, scaleMode, scales, shifts);

  return std::pair(scales, shifts);
}

}//end namespace TuckerOnNode
#endif
