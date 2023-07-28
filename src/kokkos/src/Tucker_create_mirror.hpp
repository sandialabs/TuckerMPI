#ifndef TUCKER_CREATE_MIRROR_HPP_
#define TUCKER_CREATE_MIRROR_HPP_

#include <Kokkos_Core.hpp>

// fwd declaration
namespace TuckerOnNode{
template<class ScalarType, class ...Properties> class Tensor;
template<class ScalarType, class MemorySpace> class MetricData;
}

namespace Tucker{

// Creates a new host accessible View() with the same layout and padding as src

template<class ScalarType, class ...Properties>
auto create_mirror(const TuckerOnNode::Tensor<ScalarType, Properties...> & T)
{
  using tensor_type = TuckerOnNode::Tensor<ScalarType, Properties...>;
  using tensor_mirror_type = typename tensor_type::traits::HostMirror;

  auto T_view_h = Kokkos::create_mirror(T.data());
  auto T_dims_h = T.dimensionsOnHost();
  auto dims_vec = Tucker::impl::create_stdvec_from_view(T_dims_h);
  tensor_mirror_type T_h(dims_vec, T_view_h);
  return T_h;
}

template<class ScalarType, class MemorySpace>
auto create_mirror(TuckerOnNode::MetricData<ScalarType, MemorySpace> d){
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

template<class SpaceT, class ScalarType, class ...Properties>
auto create_mirror_and_copy(const SpaceT & space,
				   const TuckerOnNode::Tensor<ScalarType, Properties...> & Tin)
{
  using in_tensor_type = TuckerOnNode::Tensor<ScalarType, Properties...>;
  using out_tensor_type = TuckerOnNode::Tensor<ScalarType, SpaceT>;

  auto T_view = Kokkos::create_mirror_view_and_copy(space, Tin.data());
  auto T_dims_h = Tin.dimensionsOnHost();
  auto dims_vec = Tucker::impl::create_stdvec_from_view(T_dims_h);
  out_tensor_type Tout(dims_vec, T_view);
  return Tout;
}

} // end namespace Tucker

#endif
