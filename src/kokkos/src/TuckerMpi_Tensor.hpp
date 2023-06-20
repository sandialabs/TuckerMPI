#ifndef TUCKER_KOKKOS_MPI_TENSOR_HPP_
#define TUCKER_KOKKOS_MPI_TENSOR_HPP_

#include "Tucker_Utils.hpp"
#include "TuckerOnNode_Tensor.hpp"
#include "TuckerMpi_MPIWrapper.hpp"
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Core.hpp>
#include <numeric>

namespace TuckerMpi{

namespace impl{
template<class Enable, class ScalarType, class ...Properties>
struct TensorTraits;

template<class ScalarType> struct TensorTraits<void, ScalarType>{
  using memory_space       = typename Kokkos::DefaultExecutionSpace::memory_space;
  using onnode_tensor_type = TuckerOnNode::Tensor<ScalarType, memory_space>;
  using value_type         = typename onnode_tensor_type::traits::data_view_type::value_type;
  using onnode_layout      = typename onnode_tensor_type::traits::array_layout;
};

template<class ScalarType, class MemSpace>
struct TensorTraits<
  std::enable_if_t< Kokkos::is_memory_space_v<MemSpace> >, ScalarType, MemSpace >
{
  using memory_space = MemSpace;
  using onnode_tensor_type = TuckerOnNode::Tensor<ScalarType, memory_space>;
  using value_type = typename onnode_tensor_type::traits::data_view_type::value_type;
  using onnode_layout      = typename onnode_tensor_type::traits::array_layout;
};
}//end namespace impl

template<class ScalarType, class ...Properties>
class Tensor
{
  static_assert(std::is_floating_point_v<ScalarType>, "");

  // need for the copy/move constr/assign accepting a compatible tensor
  template <class, class...> friend class Tensor;

  using dims_view_type            = Kokkos::View<int*>;
  using dims_host_view_type       = typename dims_view_type::HostMirror;
  using dims_const_view_type      = typename dims_view_type::const_type;
  using dims_host_const_view_type = typename dims_host_view_type::const_type;

public:
  // ----------------------------------------
  // Type aliases
  // ----------------------------------------
  using traits = impl::TensorTraits<void, ScalarType, Properties...>;

public:
  // ----------------------------------------
  // Regular constructors, destructor, and assignment
  // ----------------------------------------
  Tensor() = default;
  ~Tensor() = default;

  explicit Tensor(const Distribution & dist)
    : dist_(dist),
      globalDims_("globalDims", dist.getGlobalDims().size()),
      localDims_("localDims", dist.getLocalDims().size()),
      localTensor_(dist.getLocalDims())
  {
    auto std_gd = dist.getGlobalDims();
    auto std_ld = dist.getLocalDims();
    Tucker::impl::copy_stdvec_to_view(std_gd, globalDims_);
    Tucker::impl::copy_stdvec_to_view(std_ld, localDims_);
  }

  // Tensor(const Tensor& o) = default;
  // Tensor(Tensor&&) = default;

  // // FIXME: missing or incomplete some special mem functions because
  // // we need to define semantics for when distributions are different etc
  // Tensor& operator=(const Tensor&) = default;
  // Tensor& operator=(Tensor&&) = default;

public:
  int rank() const{ return localTensor_.rank(); }
  auto & getLocalTensor(){ return localTensor_; }
  //auto & getLocalTensor() const{ return localTensor_; }
  dims_const_view_type getGlobalSize() const{ return globalDims_; }
  dims_const_view_type getLocalSize() const { return localDims_; }

  int getGlobalSize(int n) const{ return dist_.getGlobalDims()[n]; }
  int getLocalSize(int n) const{ return dist_.getLocalDims()[n]; }
  int getNumDimensions() const{ return dist_.getGlobalDims().size(); }
  const Distribution & getDistribution() const{ return dist_; }
  size_t getLocalNumEntries() const{ return localTensor_.size(); }
  size_t getGlobalNumEntries() const{
    auto dims = dist_.getGlobalDims();
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<std::size_t>());
  }

  auto frobeniusNormSquared() const{
    const ScalarType myvalue = localTensor_.frobeniusNormSquared();
    ScalarType globalNorm2;
    MPI_Allreduce_(&myvalue, &globalNorm2, 1, MPI_SUM, MPI_COMM_WORLD);
    return globalNorm2;
  }

  // missing some methods

private:
  Distribution dist_ = {};
  dims_view_type globalDims_ = {};
  dims_view_type localDims_ = {};
  typename traits::onnode_tensor_type localTensor_ = {};
};

} // end namespace TuckerMpi
#endif
