#ifndef TUCKERONNODE_NORMALIZE_TENSOR_HPP_
#define TUCKERONNODE_NORMALIZE_TENSOR_HPP_

#include "Tucker_fwd.hpp"
#include "TuckerOnNode_compute_slice_metrics.hpp"
#include "TuckerOnNode_transform_slices.hpp"
#include "./impl/TuckerOnNode_tensor_normalization.hpp"

namespace TuckerOnNode{

template <class ScalarType, class MetricMemSpace, class ...Props>
auto normalize_tensor(const TuckerOnNode::Tensor<ScalarType, Props...> & tensor,
		      const TuckerOnNode::MetricData<ScalarType, MetricMemSpace> & metricsData,
		      const std::string & scalingType,
		      const int scaleMode,
		      const ScalarType stdThresh,
		      int mpiRank = 0)
{
  // preconditions
  impl::check_scaling_type_else_throw(scalingType);

  auto metricsData_h = Tucker::create_mirror(metricsData);
  Tucker::deep_copy(metricsData_h, metricsData);
  ::TuckerOnNode::impl::check_metricdata_usable_for_scaling_else_throw(metricsData_h, scalingType);

  using tensor_type = TuckerOnNode::Tensor<ScalarType, Props...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;

  //
  // 1. use metrics to fill scales and shifts depending on the scalingType
  Kokkos::View<ScalarType*, tensor_mem_space> scales("scales", tensor.extent(scaleMode));
  Kokkos::View<ScalarType*, tensor_mem_space> shifts("shifts", tensor.extent(scaleMode));
  if(scalingType == "Max") {
    impl::NormalizeFunc func(metricsData, scales, shifts);
#if defined TUCKER_ENABLE_MPI
    if (mpiRank == 0){
#endif
      std::cout << "Normalizing the tensor by maximum entry - mode " << scaleMode << std::endl;
#if defined TUCKER_ENABLE_MPI
    }
#endif

    Kokkos::parallel_for(Kokkos::RangePolicy<impl::UseMax>(0, tensor.extent(scaleMode)), func);
  }

  else if(scalingType == "MinMax") {
    impl::NormalizeFunc func(metricsData, scales, shifts);
#if defined TUCKER_ENABLE_MPI
    if (mpiRank == 0){
#endif
      std::cout << "Normalizing the tensor using minmax scaling - mode " << scaleMode << std::endl;
#if defined TUCKER_ENABLE_MPI
    }
#endif

    Kokkos::parallel_for(Kokkos::RangePolicy<impl::UseMinMax>(0, tensor.extent(scaleMode)), func);
  }

  else if(scalingType == "StandardCentering") {
    impl::NormalizeFunc func(metricsData, scales, shifts, stdThresh);
#if defined TUCKER_ENABLE_MPI
    if (mpiRank == 0){
#endif
      std::cout << "Normalizing the tensor using standard centering - mode " << scaleMode << std::endl;
#if defined TUCKER_ENABLE_MPI
    }
#endif

    Kokkos::parallel_for(Kokkos::RangePolicy<impl::UseStandardCentering>(0, tensor.extent(scaleMode)), func);
  }

  //
  // 2. use scales and shifts to normalize tensor data
  transform_slices(tensor, scaleMode, scales, shifts);

  return std::pair(scales, shifts);
}

}//end namespace TuckerOnNode
#endif  // TUCKERONNODE_NORMALIZE_TENSOR_HPP_
