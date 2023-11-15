#ifndef TUCKERONNODE_TENSOR_HPP_
#define TUCKERONNODE_TENSOR_HPP_

#include "./impl/Tucker_stdvec_view_conversion_helpers.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <numeric>

namespace TuckerOnNode{

template<class ScalarType, class ...Properties>
class Tensor;

namespace impl{
template<class Enable, class ScalarType, class ...Properties>
struct TensorTraits;

template<class ScalarType>
struct TensorTraits<void, ScalarType>{
  using array_layout   = Kokkos::LayoutLeft;
  using execution_space = Kokkos::DefaultExecutionSpace;
  using memory_space   = typename Kokkos::DefaultExecutionSpace::memory_space;
  using data_view_type = Kokkos::View<ScalarType*, array_layout, memory_space>;
  using value_type     = typename data_view_type::value_type;
  using HostMirror = Tensor<ScalarType, Kokkos::HostSpace>;
};

template<class ScalarType, class MemSpace>
struct TensorTraits<
  std::enable_if_t< Kokkos::is_memory_space_v<MemSpace> >, ScalarType, MemSpace >
{
  using array_layout = Kokkos::LayoutLeft;
  using memory_space = MemSpace;
  using execution_space = typename memory_space::execution_space;
  using data_view_type = Kokkos::View<ScalarType*, array_layout, memory_space>;
  using value_type     = typename data_view_type::value_type;
  using HostMirror = Tensor<ScalarType, Kokkos::HostSpace>;
};
}//end namespace impl


/* NOTE:
   rank-0 tensor = we do not allow that

   rank-1 tensor = vector
   rank-2 tensor = matrix
   rank-3 tensor = ...
*/

template<class ScalarType, class ...Properties>
class Tensor
{
  static_assert(std::is_floating_point_v<ScalarType>, "");
  using data_view_type = typename impl::TensorTraits<
    void, ScalarType, Properties...>::data_view_type;

  // need for the copy/move constr/assign accepting a compatible tensor
  template <class, class...> friend class Tensor;

public:
  using dims_view_type            = Kokkos::View<int*>;
  using dims_host_view_type       = typename dims_view_type::HostMirror;
  using dims_const_view_type      = typename dims_view_type::const_type;
  using dims_host_const_view_type = typename dims_host_view_type::const_type;

  // ----------------------------------------
  // Type aliases

  using traits = impl::TensorTraits<void, ScalarType, Properties...>;

  // ----------------------------------------
  // Regular constructors, destructor, and assignment
  // ----------------------------------------

  Tensor() = default;
  ~Tensor() = default;

  explicit Tensor(std::initializer_list<int> list)
    : Tensor(std::vector<int>(list)){}

  explicit Tensor(const std::vector<int> & dimsIn)
    : rank_(dimsIn.size()),
      dims_("dims", rank_),
      dims_h_("dims_h", rank_)
  {
    Tucker::impl::copy_stdvec_to_view(dimsIn, dims_h_);
    Kokkos::deep_copy(dims_, dims_h_);
    const std::size_t init = 1;
    const std::size_t numEl = std::accumulate(dimsIn.cbegin(), dimsIn.cend(),
					      init, std::multiplies<std::size_t>());
    // allocate for data
    data_ = data_view_type(Kokkos::ViewAllocateWithoutInitializing("tensorData"), numEl);
  }

  template<class DataView>
  Tensor(const std::vector<int> & dimsIn, DataView dataIn)
    : rank_(dimsIn.size()),
      dims_("dims", rank_),
      dims_h_("dims_h", rank_),
      data_(dataIn)
  {
    Tucker::impl::copy_stdvec_to_view(dimsIn, dims_h_);
    Kokkos::deep_copy(dims_, dims_h_);

    // check that the data extent is consistent with the dimsIn
    const std::size_t init = 1;
    const std::size_t numEl = std::accumulate(dimsIn.cbegin(), dimsIn.cend(),
					      init, std::multiplies<std::size_t>());
    if (data_.extent(0) != numEl){
      Kokkos::abort("TuckerOnNode:: data.extent(0) is not compatible with the dimensions provided");
    }
  }

  explicit Tensor(const dims_host_const_view_type & dimsIn)
    : rank_(dimsIn.size()),
      dims_("dims", rank_),
      dims_h_("dims_h", rank_)
  {
    namespace KE = Kokkos::Experimental;
    KE::copy(Kokkos::DefaultHostExecutionSpace(), dimsIn, dims_h_);
    Kokkos::deep_copy(dims_, dims_h_);
    const std::size_t init = 1;
    const std::size_t numEl = std::accumulate(KE::cbegin(dimsIn), KE::cend(dimsIn),
					      init, std::multiplies<std::size_t>());
    data_ = data_view_type(Kokkos::ViewAllocateWithoutInitializing("tensorData"), numEl);
  }

  template<class DataView>
  explicit Tensor(const dims_host_const_view_type & dimsIn,
		  DataView dataIn)
    : rank_(dimsIn.size()),
      dims_("dims", rank_),
      dims_h_("dims_h", rank_),
      data_(dataIn)
  {
    namespace KE = Kokkos::Experimental;
    KE::copy(Kokkos::DefaultHostExecutionSpace(), dimsIn, dims_h_);
    Kokkos::deep_copy(dims_, dims_h_);
    const std::size_t init = 1;
    const std::size_t numEl = std::accumulate(KE::cbegin(dimsIn), KE::cend(dimsIn),
					      init, std::multiplies<std::size_t>());
    if (data_.extent(0) != numEl){
      Kokkos::abort("TuckerOnNode:: data.extent(0) is not compatible with the dimensions provided");
    }
  }

  Tensor(const Tensor& o) = default;
  Tensor(Tensor&&) = default;

  Tensor& operator=(const Tensor& o){
    is_assignable_else_throw(o);
    rank_ = o.rank_;
    data_ = o.data_;
    dims_ = o.dims_;
    dims_h_ = o.dims_h_;
    return *this;
  }

  Tensor& operator=(Tensor&& o){
    is_assignable_else_throw(o);
    rank_ = std::move(o.rank_);
    data_ = std::move(o.data_);
    dims_ = std::move(o.dims_);
    dims_h_ = std::move(o.dims_h_);
    return *this;
  }

  // ----------------------------------------
  // copy/move constr, assignment for compatible Tensor
  // ----------------------------------------

  template<class ST, class ... PS>
  Tensor(const Tensor<ST,PS...> & o)
    : rank_(o.rank_), dims_(o.dims_),
      dims_h_(o.dims_h_), data_(o.data_)
  {}

  template<class ST, class ... PS>
  Tensor& operator=(const Tensor<ST,PS...> & o){
    is_assignable_else_throw(o);
    rank_ = o.rank_;
    data_ = o.data_;
    dims_ = o.dims_;
    dims_h_ = o.dims_h_;
    return *this;
  }

  template<class ST, class ... PS>
  Tensor(Tensor<ST,PS...> && o)
    : rank_(std::move(o.rank_)),
      data_(std::move(o.data_)),
      dims_(std::move(o.dims_)),
      dims_h_(std::move(o.dims_h_))
  {}

  template<class ST, class ... PS>
  Tensor& operator=(Tensor<ST,PS...> && o){
    is_assignable_else_throw(o);
    rank_ = std::move(o.rank_);
    data_ = std::move(o.data_);
    dims_ = std::move(o.dims_);
    dims_h_ = std::move(o.dims_h_);
    return *this;
  }

  //----------------------------------------
  // "shape" things
  // ----------------------------------------

  int rank() const{ return rank_; }
  dims_const_view_type dimensions() const{ return dims_; }
  dims_host_const_view_type dimensionsOnHost() const{ return dims_h_; }

  std::size_t extent(std::size_t mode) const {
    assert(mode < (std::size_t)rank_);
    return dims_h_[mode];
  }

  size_t size() const{
    return (rank_ == -1) ? 0 : prod(0, rank_-1);
  }

  size_t prod(const int low, const int high, const int defaultReturnVal = -1) const
  {
    assert(low >=0 && high < rank_);
    if(low > high) { return defaultReturnVal; }

    size_t result = 1;
    for(int j = low; j <= high; j++){ result *= dims_h_[j]; }
    return result;
  }

  // ----------------------------------------
  // other
  // ----------------------------------------
  data_view_type data() const{ return data_; }

  auto frobeniusNormSquared() const{
    const auto v = ::KokkosBlas::nrm2(data_);
    return v*v;
  }

  void fillRandom(ScalarType a, ScalarType b){
    Kokkos::Random_XorShift64_Pool<> pool(4543423);
    Kokkos::fill_random(data_, pool, a, b);
  }

private:
  template<class ST, class ... PS>
  void is_assignable_else_throw(const Tensor<ST,PS...> & o){
    /* need to check ranks are compatible and that this or other are not empty tensors */
    if (rank_ != -1 && o.rank_ != -1 && (rank_ != o.rank_)){
      throw std::runtime_error("Tensor: mismatching ranks for copy assignemnt");
    }
  }

private:
  int rank_ = -1;
  dims_view_type dims_ = {};
  dims_host_view_type dims_h_ = {};
  data_view_type data_;
};

} // end namespace TuckerOnNode
#endif  // TUCKERONNODE_TENSOR_HPP_
