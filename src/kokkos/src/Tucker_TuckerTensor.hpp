#ifndef TUCKER_TUCKERTENSOR_HPP_
#define TUCKER_TUCKERTENSOR_HPP_

#include "Tucker_fwd.hpp"
#include "./impl/Tucker_TuckerTensorSliceHelpers.hpp"
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Core.hpp>

namespace Tucker{

template<class CoreTensorType>
struct TuckerTensorTraits
{
  using core_tensor_type     = CoreTensorType;
  using value_type           = typename core_tensor_type::traits::value_type;
  using memory_space         = typename core_tensor_type::traits::memory_space;
  using factors_store_view_t = Kokkos::View<value_type*, Kokkos::LayoutLeft, memory_space>;
};

template<class CoreTensorType>
class TuckerTensor
{
  // the slicing info is stored on the host
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

  // need for the copy/move constr/assign accepting a compatible tensor
  template <class> friend class TuckerTensor;

#if !defined TUCKER_ALLOW_PRIVATE_CONSTRUCTORS_TO_BE_PUBLIC_FOR_TESTING
  template <class ScalarTypeIn, class ...Properties, class TruncatorType>
  friend auto ::TuckerOnNode::impl::sthosvd_gram(::TuckerOnNode::Tensor<ScalarTypeIn, Properties...> X,
						 TruncatorType && truncator,
						 bool flipSign);

#if defined TUCKER_ENABLE_MPI
  template <class ScalarTypeIn, class ...Properties, class TruncatorType>
  friend auto ::TuckerMpi::impl::sthosvd_newgram(::TuckerMpi::Tensor<ScalarTypeIn, Properties...> X,
						 TruncatorType && truncator,
						 const std::vector<int> & modeOrder,
						 bool flipSign);
#endif
#endif

public:
  using traits = TuckerTensorTraits<CoreTensorType>;

  // ----------------------------------------
  // Regular constructors, destructor, and assignment
  // ----------------------------------------
#if defined TUCKER_ALLOW_PRIVATE_CONSTRUCTORS_TO_BE_PUBLIC_FOR_TESTING
public:
#else
private:
#endif

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

public:
  ~TuckerTensor() = default;

  TuckerTensor(const TuckerTensor& o) = default;
  TuckerTensor(TuckerTensor&&) = default;

  // can default these because semantics of this class
  // are the same as tensor so we can just rely on those
  TuckerTensor& operator=(const TuckerTensor&) = default;
  TuckerTensor& operator=(TuckerTensor&&) = default;

  // ----------------------------------------
  // copy/move constr, assignment for compatible TuckerTensor
  // ----------------------------------------

  template<class LocalArg>
  TuckerTensor(const TuckerTensor<LocalArg> & o)
    : rank_(o.rank_),
      coreTensor_(o.coreTensor_),
      factors_(o.factors_),
      perModeSlicingInfo_(o.perModeSlicingInfo_)
  {}

  template<class LocalArg>
  TuckerTensor& operator=(const TuckerTensor<LocalArg> & o){
    rank_ = o.rank_;
    coreTensor_ = o.coreTensor_;
    factors_ = o.factors_;
    perModeSlicingInfo_ = o.perModeSlicingInfo;
    return *this;
  }

  template<class LocalArg>
  TuckerTensor(TuckerTensor<LocalArg> && o)
    : rank_(std::move(o.rank_)),
      coreTensor_(std::move(o.coreTensor_)),
      factors_(std::move(o.factors_)),
      perModeSlicingInfo_(std::move(o.perModeSlicingInfo_))
  {}

  template<class LocalArg>
  TuckerTensor& operator=(TuckerTensor<LocalArg> && o){
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
#endif  // TUCKER_TUCKERTENSOR_HPP_
