#ifndef TUCKER_TENSOR_NORMALIZATION_HPP_
#define TUCKER_TENSOR_NORMALIZATION_HPP_

#include "Tucker_fwd.hpp"
#include "Kokkos_Core.hpp"

namespace TuckerOnNode{
namespace impl{

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

}} //end namespace TuckerOnNode::impl
#endif
