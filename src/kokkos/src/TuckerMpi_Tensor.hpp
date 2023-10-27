#ifndef TUCKERMPI_TENSOR_HPP_
#define TUCKERMPI_TENSOR_HPP_

#include "./impl/TuckerMpi_MPIWrapper.hpp"
#include "./impl/Tucker_stdvec_view_conversion_helpers.hpp"
#include "TuckerMpi_Distribution.hpp"
#include "TuckerOnNode_Tensor.hpp"
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
  using execution_space    = Kokkos::DefaultExecutionSpace;
  using onnode_tensor_type = TuckerOnNode::Tensor<ScalarType, memory_space>;
  using value_type         = typename onnode_tensor_type::traits::data_view_type::value_type;
  using onnode_layout      = typename onnode_tensor_type::traits::array_layout;
};

template<class ScalarType, class MemSpace>
struct TensorTraits<
  std::enable_if_t< Kokkos::is_memory_space_v<MemSpace> >, ScalarType, MemSpace >
{
  using memory_space       = MemSpace;
  using execution_space    = typename memory_space::execution_space;
  using onnode_tensor_type = TuckerOnNode::Tensor<ScalarType, memory_space>;
  using value_type         = typename onnode_tensor_type::traits::data_view_type::value_type;
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
      globalDims_h_("globalDims_h", dist.getGlobalDims().size()),
      localDims_h_("localDims_h", dist.getLocalDims().size()),
      localTensor_(dist.getLocalDims())
  {
    auto std_gd = dist.getGlobalDims();
    auto std_ld = dist.getLocalDims();
    Tucker::impl::copy_stdvec_to_view(std_gd, globalDims_h_);
    Tucker::impl::copy_stdvec_to_view(std_ld, localDims_h_);
    Kokkos::deep_copy(globalDims_, globalDims_h_);
    Kokkos::deep_copy(localDims_, localDims_h_);
  }

  Tensor(const std::vector<int>& dims,
  	 const std::vector<int>& procs)
    : Tensor( Distribution(dims, procs) ){}

  Tensor(const Tensor& o) = default;
  Tensor(Tensor&&) = default;

  Tensor& operator=(const Tensor& o){
    is_assignable_else_throw(o);
    dist_        = o.dist_;
    globalDims_  = o.globalDims_;
    localDims_   = o.localDims_;
    globalDims_h_  = o.globalDims_h_;
    localDims_h_   = o.localDims_h_;
    localTensor_ = o.localTensor_;
    return *this;
  }

  Tensor& operator=(Tensor&& o){
    is_assignable_else_throw(o);
    dist_        = std::move(o.dist_);
    globalDims_  = std::move(o.globalDims_);
    localDims_   = std::move(o.localDims_);
    globalDims_h_  = std::move(o.globalDims_h_);
    localDims_h_   = std::move(o.localDims_h_);
    localTensor_ = std::move(o.localTensor_);
    return *this;
  }

  // ----------------------------------------
  // copy/move constr, assignment for compatible Tensor
  // ----------------------------------------
  template<class ST, class ... PS>
  Tensor(const Tensor<ST,PS...> & o)
    : dist_(o.dist_),
      globalDims_(o.globalDims_),
      localDims_(o.localDims_),
      globalDims_h_(o.globalDims_h_),
      localDims_h_(o.localDims_h_),
      localTensor_(o.localTensor_)
  {}

  template<class ST, class ... PS>
  Tensor& operator=(const Tensor<ST,PS...> & o){
    is_assignable_else_throw(o);
    dist_        = o.dist_;
    globalDims_  = o.globalDims_;
    localDims_   = o.localDims_;
    globalDims_h_  = o.globalDims_h_;
    localDims_h_   = o.localDims_h_;
    localTensor_ = o.localTensor_;
    return *this;
  }

  template<class ST, class ... PS>
  Tensor(Tensor<ST,PS...> && o)
    : dist_(std::move(o.dist_)),
      globalDims_(std::move(o.globalDims_)),
      localDims_(std::move(o.localDims_)),
      globalDims_h_(std::move(o.globalDims_h_)),
      localDims_h_(std::move(o.localDims_h_)),
      localTensor_(std::move(o.localTensor_))
  {}

  template<class ST, class ... PS>
  Tensor& operator=(Tensor<ST,PS...> && o){
    is_assignable_else_throw(o);
    dist_        = std::move(o.dist_);
    globalDims_  = std::move(o.globalDims_);
    localDims_   = std::move(o.localDims_);
    globalDims_h_  = std::move(o.globalDims_h_);
    localDims_h_   = std::move(o.localDims_h_);
    localTensor_ = std::move(o.localTensor_);
    return *this;
  }


public:
  int rank() const{ return localTensor_.rank(); }
  auto localTensor() const { return localTensor_; }
  const Distribution & getDistribution() const{ return dist_; }

  dims_const_view_type globalDimensions() const{ return globalDims_; }
  dims_const_view_type localDimensions() const { return localDims_; }
  dims_host_const_view_type globalDimensionsOnHost() const{ return globalDims_h_; }
  dims_host_const_view_type localDimensionsOnHost() const { return localDims_h_; }

  int globalExtent(int n) const{ return dist_.getGlobalDims()[n]; }
  int localExtent(int n) const{ return dist_.getLocalDims()[n]; }

  std::size_t localSize() const{ return localTensor_.size(); }
  std::size_t globalSize() const{
    auto dims = dist_.getGlobalDims();
    const std::size_t init = 1;
    return std::accumulate(dims.begin(), dims.end(), init, std::multiplies<std::size_t>());
  }

  auto frobeniusNormSquared() const{
    const ScalarType myvalue = localTensor_.frobeniusNormSquared();
    ScalarType globalNorm2;
    MPI_Allreduce_(&myvalue, &globalNorm2, 1, MPI_SUM, MPI_COMM_WORLD);
    return globalNorm2;
  }

  // missing some methods

private:
  template<class ST, class ... PS>
  void is_assignable_else_throw(const Tensor<ST,PS...> & o)
  {
    // This check is highly impractical as it doesn't allow you to overwrite
    // an existing tensor with another one, which is required in the streaming
    // st-hosvd algorithm, so it is disabled

    // if (!dist_.empty() && !o.dist_.empty() && (dist_ != o.dist_)){
    //   throw std::runtime_error("TuckerMpi::Tensor: mismatching distributions for copy assignemnt");
    // }
  }

private:
  Distribution dist_ = {};
  dims_view_type globalDims_ = {};
  dims_view_type localDims_ = {};
  dims_host_view_type globalDims_h_ = {};
  dims_host_view_type localDims_h_ = {};

  typename traits::onnode_tensor_type localTensor_ = {};
};

} // end namespace TuckerMpi
#endif  // TUCKERMPI_TENSOR_HPP_
