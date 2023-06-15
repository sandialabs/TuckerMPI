#ifndef TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_
#define TUCKER_KOKKOSONLY_TUCKERTENSOR_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include <Kokkos_Core.hpp>

namespace TuckerOnNode{

namespace impl{
template<class Enable, class ...Args>
struct TuckerTensorTraits;

template<class ScalarType, class ...Props>
struct TuckerTensorTraits<void, Tensor<ScalarType, Props...> >
{
  using core_tensor_type          = Tensor<ScalarType, Props...>;
  using value_type                = typename core_tensor_type::traits::data_view_type::value_type;
  using memory_space              = typename core_tensor_type::traits::memory_space;
  using eigenvalues_store_view_t  = Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space>;
  using eigenvectors_store_view_t = Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space>;
};

struct PerModeSliceInfo{
  std::size_t eigvalsStartIndex        = 0;
  std::size_t eigvalsEndIndexExclusive = 0;
  std::size_t eigvecsStartIndex        = 0;
  std::size_t eigvecsEndIndexExclusive = 0;
  std::size_t eigvecsExtent0           = 0;
  std::size_t eigvecsExtent1	       = 0;
};
}//end namespace impl

template<class ...Args>
class TuckerTensor
{
  // the slicing info is stored on the host, most likely we do not need it on device
  // based on similar arguments as to SizeArray for the Tensor class
  using slicing_info_view_t = Kokkos::View<impl::PerModeSliceInfo*, Kokkos::HostSpace>;

public:
  using traits = impl::TuckerTensorTraits<void, Args...>;

  TuckerTensor()
    : rank_(-1),
      coreTensor_{},
      eigenvalues_("eigenvalues", 0),
      eigenvectors_("eigenvectors", 0),
      perModeSlicingInfo_("info", 0)
  {}

  template<class EigvalsViewType, class EigvecsViewType>
  TuckerTensor(typename traits::core_tensor_type coreTensor,
	       EigvalsViewType eigvals,
	       EigvecsViewType eigvecs,
	       slicing_info_view_t slicingInfo)
    : rank_(slicingInfo.extent(0)),
      coreTensor_(coreTensor),
      eigenvalues_("eigenvalues", eigvals.extent(0)),
      eigenvectors_("eigenvectors", eigvecs.extent(0)),
      perModeSlicingInfo_(slicingInfo)
  {
    namespace KEX = Kokkos::Experimental;
    using exespace = typename EigvalsViewType::execution_space;
    KEX::copy(exespace(), eigvals, eigenvalues_);
    KEX::copy(exespace(), eigvecs, eigenvectors_);
  }

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

  auto eigenvectors(int mode)
  {
    //FIXME: adapt this to support striding
    if (!eigenvectors_.span_is_contiguous()){
      throw std::runtime_error("eigenvectors: currently, span must be contiguous");
    }

    using eigenvectors_layout = typename traits::eigenvectors_store_view_t::array_layout;
    using umv_type = Kokkos::View<typename traits::value_type**, eigenvectors_layout,
				  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    if (rank_ == -1){
      return umv_type(eigenvectors_.data(), 0, 0);
    }

    const auto & sliceInfo = perModeSlicingInfo_(mode);
    auto ptr = eigenvectors_.data() + sliceInfo.eigvecsStartIndex;
    return umv_type(ptr, sliceInfo.eigvecsExtent0, sliceInfo.eigvecsExtent1);
  }

private:
  int rank_ = {};
  typename traits::core_tensor_type coreTensor_ = {};
  typename traits::eigenvalues_store_view_t eigenvalues_ = {};
  typename traits::eigenvectors_store_view_t eigenvectors_ = {};
  slicing_info_view_t perModeSlicingInfo_ = {};
};

template <class ScalarType, class ...Props>
void print_eigenvalues(TuckerTensor<ScalarType, Props...> factorization,
		       const std::string& filePrefix,
		       bool useLQ)
{
  const int nmodes = factorization.rank();

  for(int mode=0; mode<nmodes; mode++) {
    std::ostringstream ss;
    ss << filePrefix << mode << ".txt";
    std::ofstream ofs(ss.str());
    // Determine the number of eigenvalues for this mode
    auto eigvals = factorization.eigenvalues(mode);
    const int nevals = eigvals.extent(0);

    // if(useLQ){
    //   for(int i=0; i<nevals; i++) {
    //     ofs << std::setprecision(16)
    //      << std::pow(factorization->singularValues[mode][i], 2)
    //      << std::endl;
    //   }
    // }
    // else{
      for(int i=0; i<nevals; i++) {
        ofs << std::setprecision(16)
      << eigvals(i)
      << std::endl;
      }
   //}
    ofs.close();
  }
}

}
#endif
