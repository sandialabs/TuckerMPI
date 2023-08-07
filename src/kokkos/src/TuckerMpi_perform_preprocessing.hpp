#ifndef TUCKER_KOKKOS_MPI_PERFORM_PREPROCESSING_HPP_
#define TUCKER_KOKKOS_MPI_PERFORM_PREPROCESSING_HPP_

#include "Tucker_fwd.hpp"
#include "TuckerMpi_compute_slice_metrics.hpp"
#include "TuckerOnNode_transform_slices.hpp"
#include "./impl/Tucker_tensor_normalization.hpp"

namespace TuckerMpi{

void check_scaling_type_else_throw(const std::string & scalingType){
  if(scalingType != "Max" &&
     scalingType != "MinMax" &&
     scalingType != "StandardCentering")
  {
    throw std::runtime_error("Error: invalid scaling type");
  }
}

template <class ScalarType, class ...Props>
auto normalize_tensor(const int mpiRank,
		      const ::TuckerMpi::Tensor<ScalarType, Props...> & X,
		      const std::string & scalingType,
		      const int scaleMode,
		      const ScalarType stdThresh)
{
  // preconditions
  check_scaling_type_else_throw(scalingType);

  if(X.localSize(scaleMode) == 0){
    // I don't have to do any work because I don't own any data
    return;
  }

  using tensor_type = ::TuckerMpi::Tensor<ScalarType, Props...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;

  //
  // 1. compute metrics for the given scaleMode
  const auto targetMetrics = TuckerOnNode::impl::set_target_metrics(scalingType);
  auto dataMetrics = TuckerOnNode::compute_slice_metrics(mpiRank, X, scaleMode, targetMetrics);

  //
  // 2. use metrics to fill scales and shifts depending on the scalingType
  Kokkos::View<ScalarType*, tensor_mem_space> scales("scales", X.localExtent(scaleMode));
  Kokkos::View<ScalarType*, tensor_mem_space> shifts("shifts", X.localExtent(scaleMode));
  if(scalingType == "Max") {
    TuckerOnNode::impl::NormalizeFunc func(dataMetrics, scales, shifts);
    std::cout << "Normalizing the tensor by maximum entry - mode " << scaleMode << std::endl;
    Kokkos::RangePolicy<TuckerOnNode::impl::UseMax> policy(0, X.localExtent(scaleMode));
    Kokkos::parallel_for(policy, func);
  }

  else if(scalingType == "MinMax") {
    TuckerOnNode::impl::NormalizeFunc func(dataMetrics, scales, shifts);
    std::cout << "Normalizing the tensor using minmax scaling - mode " << scaleMode << std::endl;
    Kokkos::RangePolicy<TuckerOnNode::impl::UseMinMax> policy(0, X.localExtent(scaleMode));
    Kokkos::parallel_for(policy, func);
  }

  else if(scalingType == "StandardCentering") {
    TuckerOnNode::impl::NormalizeFunc func(dataMetrics, scales, shifts, stdThresh);
    std::cout << "Normalizing the tensor using standard centering - mode " << scaleMode << std::endl;
    Kokkos::RangePolicy<TuckerOnNode::impl::UseStandardCentering> policy(0, X.localExtent(scaleMode));
    Kokkos::parallel_for(policy, func);
  }

  //
  // 3. use scales and shifts to normalize tensor data
  transform_slices(X.localTensor(), scaleMode, scales, shifts);

  return std::pair(scales, shifts);
}

}//end namespace TuckerOnNode
#endif
