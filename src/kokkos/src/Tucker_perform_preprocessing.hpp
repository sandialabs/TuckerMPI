#ifndef TUCKER_PERFORM_PREPROCESSING_HPP_
#define TUCKER_PERFORM_PREPROCESSING_HPP_

#include "Tucker_fwd.hpp"
#include "Tucker_compute_slice_metrics.hpp"
#include "Tucker_transform_slices.hpp"
#include "./impl/Tucker_tensor_normalization.hpp"

namespace TuckerOnNode{

void check_scaling_type_else_throw(const std::string & scalingType){
  if(scalingType != "Max" &&
     scalingType != "MinMax" &&
     scalingType != "StandardCentering")
  {
    throw std::runtime_error("Error: invalid scaling type");
  }
}

template <class ScalarType, class ...Props>
auto normalize_tensor(const TuckerOnNode::Tensor<ScalarType, Props...> & X,
		      const std::string & scalingType,
		      const int scaleMode,
		      const ScalarType stdThresh)
{
  // preconditions
  check_scaling_type_else_throw(scalingType);

  /* normalizing a tensor involves these steps:
     1. compute metrics for the given scaleMode
     2. use metrics to fill scales and shifts depending on the scalingType
     3. use scales and shifts to normalize tensor data
  */

  using tensor_type = TuckerOnNode::Tensor<ScalarType, Props...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;

  //
  // 1. compute metrics for the given scaleMode
  const auto targetMetrics = impl::set_target_metrics(scalingType);
  auto dataMetrics = TuckerOnNode::compute_slice_metrics(X, scaleMode, targetMetrics);

  //
  // 2. use metrics to fill scales and shifts depending on the scalingType
  Kokkos::View<ScalarType*, tensor_mem_space> scales("scales", X.extent(scaleMode));
  Kokkos::View<ScalarType*, tensor_mem_space> shifts("shifts", X.extent(scaleMode));
  if(scalingType == "Max") {
    impl::NormalizeFunc func(dataMetrics, scales, shifts);
    std::cout << "Normalizing the tensor by maximum entry - mode " << scaleMode << std::endl;
    Kokkos::parallel_for(Kokkos::RangePolicy<impl::UseMax>(0, X.extent(scaleMode)), func);
  }

  else if(scalingType == "MinMax") {
    impl::NormalizeFunc func(dataMetrics, scales, shifts);
    std::cout << "Normalizing the tensor using minmax scaling - mode " << scaleMode << std::endl;
    Kokkos::parallel_for(Kokkos::RangePolicy<impl::UseMinMax>(0, X.extent(scaleMode)), func);
  }

  else if(scalingType == "StandardCentering") {
    impl::NormalizeFunc func(dataMetrics, scales, shifts, stdThresh);
    std::cout << "Normalizing the tensor using standard centering - mode " << scaleMode << std::endl;
    Kokkos::parallel_for(Kokkos::RangePolicy<impl::UseStandardCentering>(0, X.extent(scaleMode)), func);
  }

  //
  // 3. use scales and shifts to normalize tensor data
  transform_slices(X, scaleMode, scales, shifts);

  return std::pair(scales, shifts);
}

}//end namespace TuckerOnNode
#endif
