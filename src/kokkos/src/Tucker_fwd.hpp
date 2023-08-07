#ifndef TUCKER_KOKKOS_FWD_DECL_HPP_
#define TUCKER_KOKKOS_FWD_DECL_HPP_

namespace Tucker{
enum class Metric {
  MIN, MAX, SUM, NORM1, NORM2, MEAN, VARIANCE
};
}//end namespace Tucker

namespace TuckerOnNode{
template<class ScalarType, class ...Properties> class Tensor;
template<class ScalarType, class MemorySpace> class MetricData;
}//end namespace TuckerOnNode

namespace TuckerMpi{
template<class ScalarType, class ...Properties> class Tensor;
}

namespace Tucker{
template<class ScalarType, class MemorySpace>
auto create_mirror(::TuckerOnNode::MetricData<ScalarType, MemorySpace>);

template<class ScalarType, class MemorySpaceFrom, class MemorySpaceDest>
void deep_copy(const ::TuckerOnNode::MetricData<ScalarType, MemorySpaceDest> & dest,
	       const ::TuckerOnNode::MetricData<ScalarType, MemorySpaceFrom> & from);
}

#endif
