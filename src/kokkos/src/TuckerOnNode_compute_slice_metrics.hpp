
#ifndef TUCKERONNODE_COMPUTE_SLICE_METRICS_HPP_
#define TUCKERONNODE_COMPUTE_SLICE_METRICS_HPP_

#include "./impl/TuckerOnNode_compute_slice_metrics.hpp"

namespace TuckerOnNode{

template <std::size_t n, class ScalarType, class ...Properties>
[[nodiscard]] auto compute_slice_metrics(Tensor<ScalarType, Properties...> tensor,
					 const int mode,
					 const std::array<Tucker::Metric, n> & metrics)
{
  using tensor_type = Tensor<ScalarType, Properties...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;
  using tensor_layout = typename tensor_type::traits::array_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;

  //
  // constraints
  static_assert(   std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
    && std::is_same_v<std::remove_cv_t<tensor_value_type>, double>,
		   "TuckerOnNode::compute_slice_metrics: supports tensors with LayoutLeft" \
		   "and double scalar type");

  //
  // preconditions
  if(mode < 0) { throw std::runtime_error("mode must be non-negative"); }

  if(tensor.extent(mode) == 0) {
    std::ostringstream oss;
    oss << "TuckerOnNode::compute_slice_metrics: "
        << "for mode = " << mode << " we have tensor.extent(mode) = " << tensor.extent(mode) << " <= 0";
    throw std::runtime_error(oss.str());
  }

  // run
  const int numSlices = tensor.extent(mode);
  TuckerOnNode::MetricData<ScalarType, tensor_mem_space> result(metrics, numSlices);
  if (tensor.size() > 0){
#if 1
    impl::compute_slice_metrics_use_hierarc_par(tensor, mode, metrics, result);
#else
    // the naive impl is a simple porting of the original code
    impl::compute_slice_metrics_naive(tensor, mode, metrics, result);
#endif
  }
  return result;
}

}
#endif  // TUCKERONNODE_COMPUTE_SLICE_METRICS_HPP_
