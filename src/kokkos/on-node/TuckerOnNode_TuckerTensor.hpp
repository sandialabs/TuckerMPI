#ifndef TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_
#define TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <iomanip>
#include <fstream>

namespace TuckerOnNode{

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
  using value_type               = typename core_tensor_type::traits::data_view_type::value_type;
  using eigenvalues_store_view_t = Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space>;
  using factors_store_view_t     = Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space>;
};

template<class ScalarType, class ...Props>
struct TuckerTensorTraits<void, Tensor<ScalarType, Props...> >
{
  using core_tensor_type         = Tensor<ScalarType, Props...>;
  using value_type               = typename core_tensor_type::traits::data_view_type::value_type;
  using memory_space             = typename core_tensor_type::traits::memory_space;
  using eigenvalues_store_view_t = Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space>;
  using factors_store_view_t     = Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space>;
};

struct PerModeSliceInfo{
  std::size_t eigvalsStartIndex        = 0;
  std::size_t eigvalsEndIndexExclusive = 0;
  std::size_t factorsStartIndex        = 0;
  std::size_t factorsEndIndexExclusive = 0;
  std::size_t factorsExtent0           = 0;
  std::size_t factorsExtent1	       = 0;
};
}//end namespace impl

template<class ...Args>
class TuckerTensor
{
  // the slicing info is stored on the host
  using slicing_info_view_t = Kokkos::View<impl::PerModeSliceInfo*, Kokkos::HostSpace>;

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
      eigenvalues_("eigenvalues", 0),
      factors_("factors", 0),
      perModeSlicingInfo_("info", 0)
  {}

  template<class EigvalsViewType, class FactorsViewType>
  TuckerTensor(typename traits::core_tensor_type coreTensor,
	       EigvalsViewType eigvals,
	       FactorsViewType factors,
	       slicing_info_view_t slicingInfo)
    : rank_(slicingInfo.extent(0)),
      coreTensor_(coreTensor),
      eigenvalues_("eigenvalues", eigvals.extent(0)),
      factors_("factors", factors.extent(0)),
      perModeSlicingInfo_(slicingInfo)
  {
    namespace KEX = Kokkos::Experimental;
    using exespace = typename EigvalsViewType::execution_space;
    KEX::copy(exespace(), eigvals, eigenvalues_);
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
      eigenvalues_(o.eigenvalues_),
      factors_(o.factors_),
      perModeSlicingInfo_(o.perModeSlicingInfo_)
  {}

  template<class ... LocalArgs>
  TuckerTensor& operator=(const TuckerTensor<LocalArgs...> & o){
    rank_ = o.rank_;
    coreTensor_ = o.coreTensor_;
    eigenvalues_ = o.eigenvalues_;
    factors_ = o.factors_;
    perModeSlicingInfo_ = o.perModeSlicingInfo;
    return *this;
  }

  template<class ... LocalArgs>
  TuckerTensor(TuckerTensor<LocalArgs...> && o)
    : rank_(std::move(o.rank_)),
      coreTensor_(std::move(o.coreTensor_)),
      eigenvalues_(std::move(o.eigenvalues_)),
      factors_(std::move(o.factors_)),
      perModeSlicingInfo_(std::move(o.perModeSlicingInfo_))
  {}

  template<class ... LocalArgs>
  TuckerTensor& operator=(TuckerTensor<LocalArgs...> && o){
    rank_ = std::move(o.rank_);
    coreTensor_ = std::move(o.coreTensor_);
    eigenvalues_ = std::move(o.eigenvalues_);
    factors_ = std::move(o.factors_);
    perModeSlicingInfo_ = std::move(o.perModeSlicingInfo_);
    return *this;
  }

  //----------------------------------------
  // methods
  // ----------------------------------------

  int rank() const{ return rank_; }

  typename traits::core_tensor_type coreTensor(){ return coreTensor_; }

  auto eigenvalues(int mode){
    if (rank_ == -1){
      return Kokkos::subview(eigenvalues_, std::pair{0, 0});
    }

    const auto & sliceInfo = perModeSlicingInfo_(mode);
    const std::size_t a = sliceInfo.eigvalsStartIndex;
    const std::size_t b = sliceInfo.eigvalsEndIndexExclusive;
    return Kokkos::subview(eigenvalues_, std::pair{a, b});
  }

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
    auto ptr = factors_.data() + sliceInfo.factorsStartIndex;
    return umv_type(ptr, sliceInfo.factorsExtent0, sliceInfo.factorsExtent1);
  }

private:
  int rank_ = {};

  /** core tensor */
  typename traits::core_tensor_type coreTensor_ = {};

  /** Eigenvalues: eigenvalues for the Gram matrix for each mode */
  typename traits::eigenvalues_store_view_t eigenvalues_ = {};

  /** Factors matrices: factor matrices in "linearized" form for each mode */
  typename traits::factors_store_view_t factors_ = {};

  /** Slicing info: info needed to access mode-specific eigevalues/factors */
  slicing_info_view_t perModeSlicingInfo_ = {};
};


template <class ScalarType, class ...Props>
void print_eigenvalues(TuckerTensor<ScalarType, Props...> factorization,
		       const std::string& filePrefix,
		       bool squareBeforeWriting)
{
  const int nmodes = factorization.rank();

  for(int mode=0; mode<nmodes; mode++) {
    std::ostringstream ss;
    ss << filePrefix << mode << ".txt";
    std::ofstream ofs(ss.str());
    // Determine the number of eigenvalues for this mode
    auto eigvals = factorization.eigenvalues(mode);
    const int nevals = eigvals.extent(0);

    if (squareBeforeWriting){
      for(int i=0; i<nevals; i++) {
        ofs << std::setprecision(16) << std::pow(eigvals(i), 2) << std::endl;
      }
    }
    else{
      for(int i=0; i<nevals; i++) {
        ofs << std::setprecision(16) << eigvals(i) << std::endl;
      }
    }
    ofs.close();
  }
}

}
#endif
