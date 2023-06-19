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
  class DataType2, class ...Props2>
void appendEigenvaluesAndUpdateSliceInfo(
         int mode,
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

template<
  class DataType1, class ...Props1,
  class DataType2, class ...Props2>
void appendFactorsAndUpdateSliceInfo(
         int mode,
  	 Kokkos::View<DataType1, Props1...> & dest,
	 Kokkos::View<DataType2, Props2...> & src,
	 ::Tucker::impl::PerModeSliceInfo & sliceInfo)
{
  namespace KEX = Kokkos::Experimental;

  // use Kokkos::resize to preserve the current content of the view
  const std::size_t currentExt = dest.extent(0);
  Kokkos::resize(dest, currentExt + src.size());

  // copy the data
  auto it0 = KEX::begin(dest);
  auto outItBegin = it0 + currentExt;
  const std::size_t nR = src.extent(0);
  Kokkos::parallel_for(src.size(),
		       KOKKOS_LAMBDA(std::size_t k){
			 const std::size_t row = k % nR;
			 const std::size_t col = k / nR;
			 *(outItBegin+k) = src(row, col);
		       });

  // update slicing info
  sliceInfo.factorsStartIndex = currentExt;
  sliceInfo.factorsEndIndexExclusive = currentExt + src.size();
  sliceInfo.factorsExtent0 = nR;
  sliceInfo.factorsExtent1 = src.extent(1);
}

template <class ScalarType, class ...Properties, class TruncatorType>
auto sthosvd_gram(const Tensor<ScalarType, Properties...> & X,
		  TruncatorType && truncator,
		  bool flipSign)
{

  using tensor_type        = Tensor<ScalarType, Properties...>;
  using tucker_tensor_type = TuckerTensor<tensor_type>;
  using memory_space       = typename tensor_type::traits::memory_space;
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> eigvals;
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> factors;
  slicing_info_view_t perModeSlicingInfo("pmsi", X.rank());
  tensor_type Y = X;
  for (std::size_t n=0; n<X.rank(); n++)
  {
    std::cout << "\tAutoST-HOSVD::Starting Mode(" << n << ")...\n";

    std::cout << "\n\tAutoST-HOSVD::Gram(" << n << ") \n";
    auto S = compute_gram(Y, n);
    Tucker::write_view_to_stream(std::cout, S);

    std::cout << "\n\tAutoST-HOSVD::Eigen{vals,vecs}(" << n << ")...\n";
    auto currEigvals = Tucker::compute_eigenvals_and_eigenvecs_inplace(S, flipSign);
    impl::appendEigenvaluesAndUpdateSliceInfo(n, eigvals, currEigvals, perModeSlicingInfo(n));
    Tucker::write_view_to_stream(std::cout, currEigvals);

    std::cout << "\n\tAutoST-HOSVD::Truncating\n";
    const std::size_t numEvecs = truncator(n, currEigvals);
    using eigvec_rank2_view_t = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space>;
    eigvec_rank2_view_t currEigVecs("currEigVecs", Y.extent(n), numEvecs);
    const int nToCopy = Y.extent(n)*numEvecs;
    const int ONE = 1;
    Tucker::copy(&nToCopy, S.data(), &ONE, currEigVecs.data(), &ONE);
    impl::appendFactorsAndUpdateSliceInfo(n, factors, currEigVecs, perModeSlicingInfo(n));
    Tucker::write_view_to_stream(std::cout, currEigVecs);

    std::cout << "\tAutoST-HOSVD::Starting TTM(" << n << ")...\n";
    tensor_type temp = ttm(Y, n, currEigVecs, true);
    output_tensor_to_stream(temp, std::cout);

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
