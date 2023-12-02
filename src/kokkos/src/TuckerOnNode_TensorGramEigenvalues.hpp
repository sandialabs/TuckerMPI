#ifndef TUCKERONNODE_TENSORGRAMEIGENVALUES_HPP_
#define TUCKERONNODE_TENSORGRAMEIGENVALUES_HPP_

#include "Tucker_fwd.hpp"
#include "./impl/Tucker_TuckerTensorSliceHelpers.hpp"
#include <Kokkos_StdAlgorithms.hpp>
#include <Kokkos_Core.hpp>

namespace TuckerOnNode{

template<class ScalarType, class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
class TensorGramEigenvalues
{
  // the slicing info is stored on the host
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

  // need for the copy/move constr/assign accepting a compatible tensor
  template <class, class> friend class TensorGramEigenvalues;

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

  // ----------------------------------------
  // Regular constructors, destructor, and assignment
  // ----------------------------------------
#if defined TUCKER_ALLOW_PRIVATE_CONSTRUCTORS_TO_BE_PUBLIC_FOR_TESTING
public:
#else
private:
#endif

  TensorGramEigenvalues()
    : rank_(-1),
      eigenvalues_("eigenvalues", 0),
      perModeSlicingInfo_("info", 0)
  {}

  template<class EigvalsViewType>
  TensorGramEigenvalues(EigvalsViewType eigvals,
			slicing_info_view_t slicingInfo)
    : rank_(slicingInfo.extent(0)),
      eigenvalues_("eigenvalues", eigvals.extent(0)),
      perModeSlicingInfo_(slicingInfo)
  {
    namespace KEX = Kokkos::Experimental;
    using exespace = typename EigvalsViewType::execution_space;
    KEX::copy(exespace(), eigvals, eigenvalues_);
  }

public:
  ~TensorGramEigenvalues() = default;

  TensorGramEigenvalues(const TensorGramEigenvalues& o) = default;
  TensorGramEigenvalues(TensorGramEigenvalues&&) = default;

  // can default these because semantics of this class
  // are the same as tensor so we can just rely on those
  TensorGramEigenvalues& operator=(const TensorGramEigenvalues&) = default;
  TensorGramEigenvalues& operator=(TensorGramEigenvalues&&) = default;

  // ----------------------------------------
  // copy/move constr, assignment for compatible TensorGramEigenvalues
  // ----------------------------------------

  template<class ... LocalArgs>
  TensorGramEigenvalues(const TensorGramEigenvalues<LocalArgs...> & o)
    : rank_(o.rank_),
      eigenvalues_(o.eigenvalues_),
      perModeSlicingInfo_(o.perModeSlicingInfo_)
  {}

  template<class ... LocalArgs>
  TensorGramEigenvalues& operator=(const TensorGramEigenvalues<LocalArgs...> & o){
    rank_ = o.rank_;
    eigenvalues_ = o.eigenvalues_;
    perModeSlicingInfo_ = o.perModeSlicingInfo;
    return *this;
  }

  template<class ... LocalArgs>
  TensorGramEigenvalues(TensorGramEigenvalues<LocalArgs...> && o)
    : rank_(std::move(o.rank_)),
      eigenvalues_(std::move(o.eigenvalues_)),
      perModeSlicingInfo_(std::move(o.perModeSlicingInfo_))
  {}

  template<class ... LocalArgs>
  TensorGramEigenvalues& operator=(TensorGramEigenvalues<LocalArgs...> && o){
    rank_ = std::move(o.rank_);
    eigenvalues_ = std::move(o.eigenvalues_);
    perModeSlicingInfo_ = std::move(o.perModeSlicingInfo_);
    return *this;
  }

  //----------------------------------------
  // methods
  // ----------------------------------------

  int rank() const{ return rank_; }

  auto operator[](int mode){
    if (rank_ == -1){
      return Kokkos::subview(eigenvalues_, std::pair{0, 0});
    }

    const auto & sliceInfo = perModeSlicingInfo_(mode);
    const std::size_t a = sliceInfo.startIndex;
    const std::size_t b = sliceInfo.endIndexExclusive;
    return Kokkos::subview(eigenvalues_, std::pair{a, b});
  }

private:
  int rank_ = {};

  /** Eigenvalues: eigenvalues for the Gram matrix for each mode */
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, MemorySpace> eigenvalues_ = {};

  /** Slicing info: info needed to access mode-specific eigevalues/factors */
  slicing_info_view_t perModeSlicingInfo_ = {};
};

}
#endif  // TUCKERONNODE_TENSORGRAMEIGENVALUES_HPP_
