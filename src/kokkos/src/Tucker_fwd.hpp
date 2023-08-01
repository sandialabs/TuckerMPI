#ifndef TUCKER_KOKKOS_FWD_DECL_HPP_
#define TUCKER_KOKKOS_FWD_DECL_HPP_

namespace Tucker{
enum class Metric {
  MIN, MAX, SUM, NORM1, NORM2, MEAN, VARIANCE
};
}//end namespace Tucker

namespace TuckerOnNode{
template<class ScalarType, class ...Properties> class Tensor;

namespace impl{
template<class ScalarType, class MemorySpace> class MetricData;
}//end namespace impl

}//end namespace TuckerOnNode

namespace TuckerMpi{
template<class ScalarType, class ...Properties> class Tensor;
}

#endif
