#ifndef TUCKER_DEEP_COPY_HPP_
#define TUCKER_DEEP_COPY_HPP_

#include "Tucker_fwd.hpp"
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Core.hpp>

namespace Tucker{

template<
  class ScalarTypeDest, class ...PropertiesDest,
  class ScalarTypeSrc, class ...PropertiesSrc
  >
void deep_copy(const TuckerOnNode::Tensor<ScalarTypeDest, PropertiesDest...> & dest,
	       const TuckerOnNode::Tensor<ScalarTypeSrc, PropertiesSrc...> & src)
{
  Kokkos::deep_copy(dest.data(), src.data());
}

template<class ScalarType, class MemorySpaceFrom, class MemorySpaceDest>
void deep_copy(const TuckerOnNode::MetricData<ScalarType, MemorySpaceDest> & dest,
	       const TuckerOnNode::MetricData<ScalarType, MemorySpaceFrom> & from)
{
  auto vals_dest = dest.getValues();
  auto map_dest  = dest.getMap();
  auto vals_from = from.getValues();
  auto map_from  = from.getMap();
  Kokkos::deep_copy(map_dest, map_from);
  Kokkos::deep_copy(vals_dest, vals_from);
}

}

#endif  // TUCKER_DEEP_COPY_HPP_
