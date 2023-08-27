#ifndef TUCKERMPI_NORMALIZE_TENSOR_HPP_
#define TUCKERMPI_NORMALIZE_TENSOR_HPP_

#include "Tucker_fwd.hpp"
#include "TuckerMpi_compute_slice_metrics.hpp"
#include "TuckerOnNode_transform_slices.hpp"
#include "TuckerOnNode_normalize_tensor.hpp"

namespace TuckerMpi{

template <class ScalarType, class MetricMemSpace, class ...Props>
[[nodiscard]] auto normalize_tensor(const int mpiRank,
				    ::TuckerMpi::Tensor<ScalarType, Props...> & tensor,
				    const TuckerOnNode::MetricData<ScalarType, MetricMemSpace> & metricData,
				    const std::string & scalingType,
				    const int scaleMode,
				    const ScalarType stdThresh)
{

  using tensor_type = ::TuckerMpi::Tensor<ScalarType, Props...>;
  using onnode_layout = typename tensor_type::traits::onnode_layout;
  using tensor_mem_space = typename tensor_type::traits::memory_space;

  // constraints
  static_assert(   std::is_same_v<onnode_layout, Kokkos::LayoutLeft>
    && std::is_same_v<std::remove_cv_t<ScalarType>, double>,
       "TuckerMpi::normalize_tensor: supports tensors with LayoutLeft" \
       "and double scalar type");

  // preconditions
  ::TuckerOnNode::impl::check_scaling_type_else_throw(scalingType);

  Kokkos::View<ScalarType*, tensor_mem_space> scales;
  Kokkos::View<ScalarType*, tensor_mem_space> shifts;
  if (tensor.localExtent(scaleMode) == 0){
    return std::pair(scales, shifts);
  }
  else{
    ::TuckerOnNode::impl::check_metricdata_usable_for_scaling_else_throw(metricData, scalingType);
    return ::TuckerOnNode::normalize_tensor(tensor.localTensor(), metricData,
					    scalingType, scaleMode, stdThresh);
  }
}

}//end namespace TuckerOnNode
#endif  // TUCKERMPI_NORMALIZE_TENSOR_HPP_
