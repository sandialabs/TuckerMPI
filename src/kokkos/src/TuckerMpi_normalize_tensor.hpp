#ifndef TUCKERMPI_NORMALIZE_TENSOR_HPP_
#define TUCKERMPI_NORMALIZE_TENSOR_HPP_

#include "Tucker_fwd.hpp"
#include "TuckerMpi_compute_slice_metrics.hpp"
#include "TuckerOnNode_transform_slices.hpp"
#include "TuckerOnNode_normalize_tensor.hpp"

namespace TuckerMpi{

template <class ScalarType, class MetricMemSpace, class ...Props>
auto normalize_tensor(const int mpiRank,
		      ::TuckerMpi::Tensor<ScalarType, Props...> & tensor,
		      const TuckerOnNode::MetricData<ScalarType, MetricMemSpace> & metricData,
		      const std::string & scalingType,
		      const int scaleMode,
		      const ScalarType stdThresh)
{
  // preconditions
  ::TuckerOnNode::impl::check_scaling_type_else_throw(scalingType);

  auto metricData_h = Tucker::create_mirror(metricData);
  Tucker::deep_copy(metricData_h, metricData);
  ::TuckerOnNode::impl::check_metricdata_usable_for_scaling_else_throw(metricData_h, scalingType);

  return ::TuckerOnNode::normalize_tensor(tensor.localTensor(), metricData,
					  scalingType, scaleMode, stdThresh, mpiRank);
}

}//end namespace TuckerOnNode
#endif  // TUCKERMPI_NORMALIZE_TENSOR_HPP_
