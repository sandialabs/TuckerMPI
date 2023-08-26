#ifndef TUCKER_CREATE_MIRROR_HPP_
#define TUCKER_CREATE_MIRROR_HPP_

#include "Tucker_fwd.hpp"
#include "./impl/Tucker_stdvec_view_conversion_helpers.hpp"
#if defined TUCKER_ENABLE_MPI
#include "TuckerMpi_Distribution.hpp"
#endif
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace Tucker{

//
// overloads accepting a TuckerOnNode::Tensor
//
template<class ScalarType, class ...Properties>
[[nodiscard]] auto create_mirror(const TuckerOnNode::Tensor<ScalarType, Properties...> & tensor)
{
  using tensor_type = TuckerOnNode::Tensor<ScalarType, Properties...>;
  using tensor_mirror_type = typename tensor_type::traits::HostMirror;

  auto tensor_view_h = Kokkos::create_mirror(tensor.data());
  auto tensor_dims_h = tensor.dimensionsOnHost();
  tensor_mirror_type tensor_h(tensor_dims_h, tensor_view_h);
  return tensor_h;
}

template<class SpaceT, class ScalarType, class ...Properties>
[[nodiscard]] auto create_mirror_and_copy(const SpaceT & space,
					  const TuckerOnNode::Tensor<ScalarType, Properties...> & tensor)
{
  using out_tensor_type = TuckerOnNode::Tensor<ScalarType, SpaceT>;

  auto tensor_view = Kokkos::create_mirror_view_and_copy(space, tensor.data());
  auto tensor_dims_h = tensor.dimensionsOnHost();
  out_tensor_type Tout(tensor_dims_h, tensor_view);
  return Tout;
}

//
// overloads accepting a TuckerOnNode::MetricData
//
template<class ScalarType, class MemorySpace>
[[nodiscard]] auto create_mirror(TuckerOnNode::MetricData<ScalarType, MemorySpace> d)
{
  using T = TuckerOnNode::MetricData<ScalarType, MemorySpace>;
  using T_mirror = typename T::HostMirror;

  auto vals = d.getValues();
  auto map  = d.getMap();
  auto vals_h = Kokkos::create_mirror(vals);
  typename T_mirror::map_t map_h(map.capacity());
  // we need this or deep copy below won't work
  Kokkos::deep_copy(map_h, map);

  return T_mirror(map_h, vals_h);
}

#if defined TUCKER_ENABLE_MPI
//
// overloads accepting a TuckerMpi::Tensor
//
template<class SpaceT, class ScalarType, class ...Properties>
[[nodiscard]] auto create_mirror_and_copy(const SpaceT & space,
					  ::TuckerMpi::Tensor<ScalarType, Properties...> tensor)
{
  using out_tensor_type = ::TuckerMpi::Tensor<ScalarType, SpaceT>;

  const ::TuckerMpi::Distribution & tensor_dist = tensor.getDistribution();
  auto tensor_local_tensor = tensor.localTensor();

  out_tensor_type Tout(tensor_dist);
  auto Tout_local_tensor = Tout.localTensor();
  Tucker::deep_copy(Tout_local_tensor, tensor_local_tensor);

  return Tout;
}
#endif

} // end namespace Tucker
#endif  // TUCKER_CREATE_MIRROR_HPP_
