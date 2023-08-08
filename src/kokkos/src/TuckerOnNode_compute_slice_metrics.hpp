
#ifndef TUCKER_KOKKOSONLY_COMPUTE_SLICE_METRICS_HPP_
#define TUCKER_KOKKOSONLY_COMPUTE_SLICE_METRICS_HPP_

#include "./impl/Tucker_compute_slice_metrics.hpp"

namespace TuckerOnNode{

template <std::size_t n, class ScalarType, class ...Properties>
auto compute_slice_metrics(Tensor<ScalarType, Properties...> Y,
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
  if(Y.size() > 0) {
    impl::compute_slice_metrics(Y, mode, metrics, result);
  }

  return result;
}

}
#endif
