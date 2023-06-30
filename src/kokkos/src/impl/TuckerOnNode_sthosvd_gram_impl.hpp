#ifndef TUCKER_KOKKOSONLY_STHOSVD_GRAM_IMPL_HPP_
#define TUCKER_KOKKOSONLY_STHOSVD_GRAM_IMPL_HPP_

#include "Tucker_ComputeEigValsEigVecs.hpp"
#include "TuckerOnNode_Tensor.hpp"
#include "TuckerOnNode_TuckerTensor.hpp"
#include "TuckerOnNode_ComputeGram.hpp"
#include "TuckerOnNode_ttm.hpp"
#include <Kokkos_Core.hpp>

namespace TuckerOnNode{
namespace impl{

template<
  class DataType1, class ...Props1,
  class DataType2, class ...Props2
>
void appendEigenvaluesAndUpdateSliceInfo(int mode,
					 Kokkos::View<DataType1, Props1...> & dest,
					 Kokkos::View<DataType2, Props2...> & src,
					 ::Tucker::impl::PerModeSliceInfo & sliceInfo)
{
  namespace KEX = Kokkos::Experimental;

  // constraints
  using dest_view =  Kokkos::View<DataType1, Props1...>;
  // both must be rank1, same mem space

  // preconditions
  assert(mode>=0);

  // use Kokkos::resize to preserve the current content of the view
  const std::size_t currentExt = dest.extent(0);
  Kokkos::resize(dest, currentExt + src.extent(0));

  // copy the data
  auto it0 = KEX::begin(dest);
  auto outItBegin = it0 + currentExt;
  using exespace = typename dest_view::execution_space;
  auto resIt = KEX::copy(exespace(), KEX::cbegin(src), KEX::cend(src), outItBegin);

  // update slicing info
  sliceInfo.eigvalsStartIndex = KEX::distance(it0, outItBegin);
  sliceInfo.eigvalsEndIndexExclusive = KEX::distance(it0, resIt);
}

template <class IteratorType, class ViewType>
struct CopyFactorData
{
  IteratorType outIt_;
  ViewType src_;
  CopyFactorData(IteratorType it, ViewType view) : outIt_(it), src_(view){}

  KOKKOS_FUNCTION void operator()(std::size_t k) const{
    const std::size_t nR = src_.extent(0);
    const std::size_t row = k % nR;
    const std::size_t col = k / nR;
    *(outIt_ + k) = src_(row, col);
  }
};

template<class DataType, class ...Props, class SourceViewType>
void appendFactorsAndUpdateSliceInfo(int mode,
				     Kokkos::View<DataType, Props...> dest,
				     SourceViewType src,
				     ::Tucker::impl::PerModeSliceInfo & sliceInfo)
{
  namespace KEX = Kokkos::Experimental;

  // use Kokkos::resize to preserve the current content of the view
  const std::size_t currentExt = dest.extent(0);
  Kokkos::resize(dest, currentExt + src.size());

  // copy the data
  auto it0 = KEX::begin(dest);
  auto outItBegin = it0 + currentExt;
  Kokkos::parallel_for(src.size(), CopyFactorData(outItBegin, src));

  // update slicing info
  sliceInfo.factorsStartIndex = currentExt;
  sliceInfo.factorsEndIndexExclusive = currentExt + src.size();
  sliceInfo.factorsExtent0 = src.extent(0);
  sliceInfo.factorsExtent1 = src.extent(1);
}

template <class ScalarType, class ...Properties, class TruncatorType>
auto sthosvd_gram(Tensor<ScalarType, Properties...> X,
		  TruncatorType && truncator,
		  bool flipSign)
{

  using tensor_type         = Tensor<ScalarType, Properties...>;
  using tucker_tensor_type  = TuckerTensor<tensor_type>;
  using memory_space        = typename tensor_type::traits::memory_space;
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> eigvals;
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> factors;
  slicing_info_view_t perModeSlicingInfo("pmsi", X.rank());
  tensor_type Y = X;
  for (std::size_t n=0; n<X.rank(); n++)
  {
    std::cout << "\tAutoST-HOSVD::Starting Mode(" << n << ")...\n";

    /*
     * gram
     */
    std::cout << "\n\tAutoST-HOSVD::Gram(" << n << ") \n";
    auto S = compute_gram(Y, n);
    /* check postconditions on the S
     * - S must be a rank-2 view
     * - S is a gram matrix so by definition should be symmetric
     * - S should have leading extent = Y.extent(n) */
    using S_type = decltype(S);
    static_assert(Kokkos::is_view_v<S_type> && S_type::rank == 2);
    assert(S.extent(0) == S.extent(1));
    assert(S.extent(0) == Y.extent(n));
    // use S
    Tucker::write_view_to_stream(std::cout, S);

    /*
     * eigenvalues and eigenvectors
     */
    std::cout << "\n\tAutoST-HOSVD::Eigen{vals,vecs}(" << n << ")...\n";
    // Note: eigenvals are returned, but S is being overwritten with eigenvectors
    auto currEigvals = Tucker::impl::compute_and_sort_descending_eigvals_and_eigvecs_inplace(S, flipSign);
    /* check postconditions */
    using ev_ret_type = decltype(currEigvals);
    static_assert(Kokkos::is_view_v<ev_ret_type> && ev_ret_type::rank == 1);
    assert(currEigvals.extent(0) == S.extent(0));
    assert(S.extent(0) == S.extent(1));
    // use the curreEigvals
    impl::appendEigenvaluesAndUpdateSliceInfo(n, eigvals, currEigvals, perModeSlicingInfo(n));
    Tucker::write_view_to_stream(std::cout, currEigvals);

    /*
     * truncation
     */
    // S now contains the eigenvectors and we need to extract only
    // a subset of them depending on the truncation method
    std::cout << "\n\tAutoST-HOSVD::Truncating\n";
    const std::size_t numEvecs = truncator(n, currEigvals);
    auto currEigVecs = Kokkos::subview(S, Kokkos::ALL, std::pair<std::size_t,std::size_t>{0, numEvecs});
    impl::appendFactorsAndUpdateSliceInfo(n, factors, currEigVecs, perModeSlicingInfo(n));
    Tucker::write_view_to_stream(std::cout, currEigVecs);

    /*
     * ttm
     */
    std::cout << "\tAutoST-HOSVD::Starting TTM(" << n << ")...\n";
    tensor_type temp = ttm(Y, n, currEigVecs, true);

    Y = temp;
    std::cout << "Local tensor size after STHOSVD iteration " << n << ": ";
    const auto sizeInfo = Y.dimensionsOnHost();
    for (int i=0; i<sizeInfo.extent(0); ++i){ std::cout << sizeInfo(i) << " "; }
    std::cout << "\n";
   }

  return tucker_tensor_type(Y, eigvals, factors, perModeSlicingInfo);
}

}} //end namespace TuckerOnNode::impl
#endif
