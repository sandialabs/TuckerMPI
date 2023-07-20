#ifndef TUCKER_KOKKOS_MPI_TUCKERTENSOR_HPP_
#define TUCKER_KOKKOS_MPI_TUCKERTENSOR_HPP_

#include "Tucker_TuckerTensorSliceHelpers.hpp"
#include "TuckerMpi_Tensor.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace TuckerMpi{

namespace impl{
template<class Enable, class ...Args>
struct TuckerTensorTraits;

template<class ScalarType>
struct TuckerTensorTraits<
  std::enable_if_t<std::is_floating_point_v<ScalarType>>, ScalarType
  >
{
  using memory_space             = typename Kokkos::DefaultExecutionSpace::memory_space;
  using core_tensor_type         = Tensor<ScalarType, memory_space>;
  using value_type               = typename core_tensor_type::traits::value_type;
  using factors_store_view_t     = Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space>;
};

template<class ScalarType, class ...Props>
struct TuckerTensorTraits<void, Tensor<ScalarType, Props...> >
{
  using core_tensor_type         = Tensor<ScalarType, Props...>;
  using value_type               = typename core_tensor_type::traits::value_type;
  using memory_space             = typename core_tensor_type::traits::memory_space;
  using factors_store_view_t     = Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space>;
};

}//end namespace impl

template<class ...Args>
class TuckerTensor
{
  // the slicing info is stored on the host
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

  // need for the copy/move constr/assign accepting a compatible tensor
  template <class...> friend class TuckerTensor;

public:
  using traits = impl::TuckerTensorTraits<void, Args...>;

  // ----------------------------------------
  // Regular constructors, destructor, and assignment
  // ----------------------------------------

  ~TuckerTensor() = default;

  TuckerTensor()
    : rank_(-1),
      coreTensor_{},
      factors_("factors", 0),
      perModeSlicingInfo_("info", 0)
  {}

  template<class FactorsViewType>
  TuckerTensor(typename traits::core_tensor_type coreTensor,
	       FactorsViewType factors,
	       slicing_info_view_t slicingInfo)
    : rank_(slicingInfo.extent(0)),
      coreTensor_(coreTensor),
      factors_("factors", factors.extent(0)),
      perModeSlicingInfo_(slicingInfo)
  {
    namespace KEX = Kokkos::Experimental;
    using exespace = typename FactorsViewType::execution_space;
    KEX::copy(exespace(), factors, factors_);
  }

  TuckerTensor(const TuckerTensor& o) = default;
  TuckerTensor(TuckerTensor&&) = default;

  // can default these because semantics of this class
  // are the same as tensor so we can just rely on those
  TuckerTensor& operator=(const TuckerTensor&) = default;
  TuckerTensor& operator=(TuckerTensor&&) = default;

  // ----------------------------------------
  // copy/move constr, assignment for compatible TuckerTensor
  // ----------------------------------------

  template<class ... LocalArgs>
  TuckerTensor(const TuckerTensor<LocalArgs...> & o)
    : rank_(o.rank_),
      coreTensor_(o.coreTensor_),
      factors_(o.factors_),
      perModeSlicingInfo_(o.perModeSlicingInfo_)
  {}

  template<class ... LocalArgs>
  TuckerTensor& operator=(const TuckerTensor<LocalArgs...> & o){
    rank_ = o.rank_;
    coreTensor_ = o.coreTensor_;
    factors_ = o.factors_;
    perModeSlicingInfo_ = o.perModeSlicingInfo;
    return *this;
  }

  template<class ... LocalArgs>
  TuckerTensor(TuckerTensor<LocalArgs...> && o)
    : rank_(std::move(o.rank_)),
      coreTensor_(std::move(o.coreTensor_)),
      factors_(std::move(o.factors_)),
      perModeSlicingInfo_(std::move(o.perModeSlicingInfo_))
  {}

  template<class ... LocalArgs>
  TuckerTensor& operator=(TuckerTensor<LocalArgs...> && o){
    rank_ = std::move(o.rank_);
    coreTensor_ = std::move(o.coreTensor_);
    factors_ = std::move(o.factors_);
    perModeSlicingInfo_ = std::move(o.perModeSlicingInfo_);
    return *this;
  }

  //----------------------------------------
  // methods
  // ----------------------------------------

  int rank() const{ return rank_; }

  typename traits::core_tensor_type coreTensor(){ return coreTensor_; }

  auto factorMatrix(int mode)
  {
    //FIXME: adapt this to support striding
    if (!factors_.span_is_contiguous()){
      throw std::runtime_error("factors: currently, span must be contiguous");
    }

    using factors_layout = typename traits::factors_store_view_t::array_layout;
    using umv_type = Kokkos::View<typename traits::value_type**, factors_layout,
				  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    if (rank_ == -1){
      return umv_type(factors_.data(), 0, 0);
    }

    const auto & sliceInfo = perModeSlicingInfo_(mode);
    auto ptr = factors_.data() + sliceInfo.startIndex;
    return umv_type(ptr, sliceInfo.extent0, sliceInfo.extent1);
  }

private:
  int rank_ = {};

  /** core tensor */
  typename traits::core_tensor_type coreTensor_ = {};

  /** Factors matrices: factor matrices in "linearized" form for each mode */
  typename traits::factors_store_view_t factors_ = {};

  /** Slicing info: info needed to access mode-specific factors */
  slicing_info_view_t perModeSlicingInfo_ = {};
};

}
#endif
