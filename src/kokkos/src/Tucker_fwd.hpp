#ifndef TUCKER_FWD_HPP_
#define TUCKER_FWD_HPP_

#include <array>
#include <vector>

namespace Tucker
{
enum class Metric {
  MIN, MAX, SUM, NORM1, NORM2, MEAN, VARIANCE
};

constexpr std::array<Tucker::Metric, 4> defaultMetrics{
  Tucker::Metric::MIN,
  Tucker::Metric::MAX,
  Tucker::Metric::MEAN,
  Tucker::Metric::VARIANCE};

template<class CoreTensorType> class TuckerTensor;
}//end namespace Tucker

namespace TuckerOnNode
{
template<class ScalarType, class ...Properties> class Tensor;
template<class ScalarType, class MemorySpace> class MetricData;
template<class ScalarType, class MemorySpace> class TensorGramEigenvalues;

namespace impl{
template <class ScalarTypeIn, class ...Properties, class TruncatorType>
auto sthosvd_gram(Tensor<ScalarTypeIn, Properties...> X,
		  TruncatorType && truncator,
		  bool flipSign);
}
}//end namespace TuckerOnNode

namespace TuckerMpi
{
template<class ScalarType, class ...Properties> class Tensor;

namespace impl{
template <class ScalarType, class ...Properties, class TruncatorType>
auto sthosvd_newgram(Tensor<ScalarType, Properties...> X,
		     TruncatorType && truncator,
		     const std::vector<int> & modeOrder,
		     bool flipSign);
}
}//end namespace TuckerMpi

namespace Tucker
{
// NOTE: the following funcs are forward declared here because
// TuckerOnNode::MetricData has private constructors and they are its friend funcs
template<class ScalarType, class MemorySpace>
auto create_mirror(::TuckerOnNode::MetricData<ScalarType, MemorySpace>);

template<class ScalarType, class MemorySpaceFrom, class MemorySpaceDest>
void deep_copy(const ::TuckerOnNode::MetricData<ScalarType, MemorySpaceDest> & dest,
	       const ::TuckerOnNode::MetricData<ScalarType, MemorySpaceFrom> & from);
}

#endif  // TUCKER_FWD_HPP_
