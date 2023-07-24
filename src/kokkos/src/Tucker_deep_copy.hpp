#ifndef TUCKER_DEEP_COPY_HPP_
#define TUCKER_DEEP_COPY_HPP_

#include <Kokkos_Core.hpp>

// fwd declaration
namespace TuckerOnNode{
template<class ScalarType, class ...Properties> class Tensor;
}

namespace Tucker{

template<
  class ScalarTypeDest, class ...PropertiesDest,
  class ScalarTypeSrc, class ...PropertiesSrc
  >
void deep_copy(const TuckerOnNode::Tensor<ScalarTypeDest, PropertiesDest...> & dest,
	       const TuckerOnNode::Tensor<ScalarTypeSrc, PropertiesSrc...> & src)
{
  assert(dest.dimensionsOnHost() == serc.dimensionsOnHost() );
  Kokkos::deep_copy(dest.data(), src.data());
}

}

#endif
