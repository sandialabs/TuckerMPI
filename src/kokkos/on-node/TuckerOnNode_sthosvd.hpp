#ifndef TUCKER_KOKKOSONLY_STHOSVD_HPP_
#define TUCKER_KOKKOSONLY_STHOSVD_HPP_

#include "TuckerOnNode_CoreTensorTruncator.hpp"
#include "TuckerOnNode_Tensor.hpp"
#include "TuckerOnNode_TuckerTensor.hpp"
#include "TuckerOnNode_ComputeGram.hpp"
#include "TuckerOnNode_ttm.hpp"
#include <Kokkos_Core.hpp>

namespace TuckerOnNode{
namespace impl{

template<class ScalarType, class ... Properties>
auto compute_eigenvalues(Kokkos::View<ScalarType**, Properties...> G,
			 const bool flipSign)
{
  using view_type = Kokkos::View<ScalarType**, Properties...>;
  using mem_space = typename view_type::memory_space;
  static_assert(std::is_same_v< typename view_type::array_layout, Kokkos::LayoutLeft>);

  const int nrows = (int) G.extent(0);
  auto G_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), G);
  Kokkos::View<ScalarType*, mem_space> eigenvalues_d("EIG", nrows);
  auto eigenvalues_h = Kokkos::create_mirror_view(eigenvalues_d);

  char jobz = 'V';
  char uplo = 'U';
  int lwork = (int) 8*nrows;
  std::vector<ScalarType> work(lwork);
  int info;
  Tucker::syev(&jobz, &uplo, &nrows, G_h.data(), &nrows,
	       eigenvalues_h.data(), work.data(), &lwork, &info);

  // Check the error code
  if(info != 0){
    std::cerr << "Error: invalid error code returned by dsyev (" << info << ")\n";
  }

  // The user will expect the eigenvalues to be sorted in descending order
  // LAPACK gives us the eigenvalues in ascending order
  for(int esubs=0; esubs<nrows-esubs-1; esubs++) {
    ScalarType temp = eigenvalues_h[esubs];
    eigenvalues_h[esubs] = eigenvalues_h[nrows-esubs-1];
    eigenvalues_h[nrows-esubs-1] = temp;
  }

  // Sort the eigenvectors too
  ScalarType* Gptr = G_h.data();
  const int ONE = 1;
  for(int esubs=0; esubs<nrows-esubs-1; esubs++) {
    Tucker::swap(&nrows, Gptr+esubs*nrows, &ONE, Gptr+(nrows-esubs-1)*nrows, &ONE);
  }

  if(flipSign){
    for(int c=0; c<nrows; c++)
    {
      int maxIndex=0;
      ScalarType maxVal = std::abs(Gptr[c*nrows]);
      for(int r=1; r<nrows; r++)
      {
        ScalarType testVal = std::abs(Gptr[c*nrows+r]);
        if(testVal > maxVal) {
          maxIndex = r;
          maxVal = testVal;
        }
      }

      if(Gptr[c*nrows+maxIndex] < 0) {
        const ScalarType NEGONE = -1;
	      Tucker::scal(&nrows, &NEGONE, Gptr+c*nrows, &ONE);
      }
    }
  }

  Kokkos::deep_copy(G, G_h);
  Kokkos::deep_copy(eigenvalues_d, eigenvalues_h);
  return eigenvalues_d;
}

template<
  class DataType1, class ...Props1,
  class DataType2, class ...Props2>
void appendEigenvaluesAndUpdateSliceInfo(
         int mode,
  	 Kokkos::View<DataType1, Props1...> & dest,
	 Kokkos::View<DataType2, Props2...> & src,
	 PerModeSliceInfo & sliceInfo)
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
void appendEigenvectorsAndUpdateSliceInfo(
         int mode,
  	 Kokkos::View<DataType1, Props1...> & dest,
	 Kokkos::View<DataType2, Props2...> & src,
	 PerModeSliceInfo & sliceInfo)
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
  sliceInfo.eigvecsStartIndex = currentExt;
  sliceInfo.eigvecsEndIndexExclusive = currentExt + src.size();
  sliceInfo.eigvecsExtent0 = nR;
  sliceInfo.eigvecsExtent1 = src.extent(1);
}

} //end namespace impl

template <class ScalarType, class ...Properties, class TruncatorType>
auto STHOSVD(const Tensor<ScalarType, Properties...> & X,
	     TruncatorType && truncator,
	     bool useQR    = false,
	     bool flipSign = false)
{

  using tensor_type        = Tensor<ScalarType, Properties...>;
  using tucker_tensor_type = TuckerTensor<tensor_type>;
  using memory_space       = typename tensor_type::traits::memory_space;
  using slicing_info_view_t = Kokkos::View<impl::PerModeSliceInfo*, Kokkos::HostSpace>;

  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> eigvals;
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> eigvecs;
  slicing_info_view_t perModeSlicingInfo("pmsi", X.rank());
  tensor_type Y = X;
  for (std::size_t n=0; n<X.rank(); n++){
    std::cout << "\tAutoST-HOSVD::Starting Mode(" << n << ")...\n";

    std::cout << "\n\tAutoST-HOSVD::Gram(" << n << ") \n";
    auto S = compute_gram(Y, n);
    Tucker::write_view_to_stream(std::cout, S);

    std::cout << "\n\tAutoST-HOSVD::Eigen{vals,vecs}(" << n << ")...\n";
    auto currEigvals = impl::compute_eigenvalues(S, flipSign);
    impl::appendEigenvaluesAndUpdateSliceInfo(n, eigvals, currEigvals, perModeSlicingInfo(n));
    Tucker::write_view_to_stream(std::cout, currEigvals);

    std::cout << "\n\tAutoST-HOSVD::Truncating\n";
    const std::size_t numEvecs = truncator(n, currEigvals);
    using eigvec_rank2_view_t = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space>;
    eigvec_rank2_view_t currEigVecs("currEigVecs", Y.extent(n), numEvecs);
    const int nToCopy = Y.extent(n)*numEvecs;
    const int ONE = 1;
    Tucker::copy(&nToCopy, S.data(), &ONE, currEigVecs.data(), &ONE);
    impl::appendEigenvectorsAndUpdateSliceInfo(n, eigvecs, currEigVecs, perModeSlicingInfo(n));
    Tucker::write_view_to_stream(std::cout, currEigVecs);

    std::cout << "\tAutoST-HOSVD::Starting TTM(" << n << ")...\n";
    tensor_type temp = ttm(Y, n, currEigVecs, true);
    output_tensor_to_stream(temp, std::cout);

    Y = temp;
    std::cout << "Local tensor size after STHOSVD iteration " << n << ": ";
    const auto sizeInfo = Y.dimensions();
    for (int i=0; i<sizeInfo.extent(0); ++i){
      std::cout << sizeInfo(i) << " ";
    }
    std::cout << "\n";
   }

  return tucker_tensor_type(Y, eigvals, eigvecs, perModeSlicingInfo);
}

}
#endif
