#ifndef IMPL_TUCKERONNODE_STHOSVD_GRAM_IMPL_HPP_
#define IMPL_TUCKERONNODE_STHOSVD_GRAM_IMPL_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include "Tucker_ComputeEigValsEigVecs.hpp"
#include "TuckerOnNode_TensorGramEigenvalues.hpp"
#include "Tucker_TuckerTensorSliceHelpers.hpp"
#include "Tucker_TuckerTensor.hpp"
#include <Kokkos_Core.hpp>
#include <chrono>

namespace TuckerOnNode{
namespace impl{

template <class ScalarType, class ...Properties, class TruncatorType>
auto sthosvd_gram(Tensor<ScalarType, Properties...> X,
		  TruncatorType && truncator,
		  bool flipSign)
{

  // constraints
  using tensor_type       = Tensor<ScalarType, Properties...>;
  using tensor_layout     = typename tensor_type::traits::array_layout;
  using tensor_value_type = typename tensor_type::traits::value_type;
  static_assert(   std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
		&& std::is_floating_point_v<tensor_value_type>,
		   "TuckerOnNode::impl::sthosvd: supports tensors with LayoutLeft" \
		   "and floating point scalar");

  // aliases needed below
  using tucker_tensor_type  = Tucker::TuckerTensor<tensor_type>;
  using memory_space        = typename tensor_type::traits::memory_space;
  using gram_eigvals_type   = TensorGramEigenvalues<ScalarType, memory_space>;
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> eigvals;
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> factors;
  slicing_info_view_t perModeSlicingInfo_factors("pmsi_factors", X.rank());
  slicing_info_view_t perModeSlicingInfo_eigvals("pmsi_eigvals", X.rank());

  auto start = std::chrono::high_resolution_clock::now();
  
  tensor_type Y = X;
  for (std::size_t n=0; n<X.rank(); n++)
  {

    std::cout << "\n---------------------------------------------\n";
    std::cout << "--- AutoST-HOSVD::Starting Mode(" << n << ") --- \n";
    std::cout << "---------------------------------------------\n";

    /*
     * gram
     */
    std::cout << "  AutoST-HOSVD::Gram(" << n << ") \n";
    auto S = compute_gram(Y, n);
    /* check postconditions on the S
     * - S must be a rank-2 view
     * - S is a gram matrix so by definition should be symmetric
     * - S should have leading extent = Y.extent(n) */
    using S_type = decltype(S);
    static_assert(Kokkos::is_view_v<S_type> && S_type::rank == 2);
    assert(S.extent(0) == S.extent(1));
    assert(S.extent(0) == Y.extent(n));
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    std::cout << "\n";
    Tucker::write_view_to_stream(std::cout, S);
    std::cout << "\n";
#endif

    /*
     * eigenvalues and eigenvectors
     */
    std::cout << "  AutoST-HOSVD::Eigen{vals,vecs}(" << n << ")...\n";
    // Note: eigenvals are returned, but S is being overwritten with eigenvectors
    auto currEigvals = Tucker::impl::compute_and_sort_descending_eigvals_and_eigvecs_inplace(S, flipSign);
    /* check postconditions */
    using ev_ret_type = decltype(currEigvals);
    static_assert(Kokkos::is_view_v<ev_ret_type> && ev_ret_type::rank == 1);
    assert(currEigvals.extent(0) == S.extent(0));
    assert(S.extent(0) == S.extent(1));
    // use the curreEigvals
    appendEigenvaluesAndUpdateSliceInfo(n, eigvals, currEigvals, perModeSlicingInfo_eigvals(n));
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    std::cout << "\n";
    Tucker::write_view_to_stream(std::cout, currEigvals);
    std::cout << "\n";
#endif

    /*
     * truncation
     */
    // S now contains the eigenvectors and we need to extract only
    // a subset of them depending on the truncation method
    std::cout << "  AutoST-HOSVD::Truncating\n";
    const std::size_t numEvecs = truncator(n, currEigvals);
    auto currEigVecs = Kokkos::subview(S, Kokkos::ALL, std::pair<std::size_t,std::size_t>{0, numEvecs});
    appendFactorsAndUpdateSliceInfo(n, factors, currEigVecs, perModeSlicingInfo_factors(n));
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    std::cout << "\n";
    Tucker::write_view_to_stream(std::cout, currEigVecs);
    std::cout << "\n";
#endif

    /*
     * ttm
     */
    std::cout << "  AutoST-HOSVD::Starting TTM(" << n << ")...\n";
    tensor_type temp = ttm(Y, n, currEigVecs, true);

    Kokkos::fence("onnode gram: fencing after mode" + std::to_string(n));

    Y = temp;
    std::cout << "  Tensor size after STHOSVD iteration " << n << ": ";
    const auto sizeInfo = Y.dimensionsOnHost();
    for (int i=0; i<sizeInfo.extent(0); ++i){ std::cout << sizeInfo(i) << " "; }
    std::cout << "\n";
   }

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  std::cout << "STHOSVD time: " << duration.count() << std::endl;
  
  return std::pair( tucker_tensor_type(Y, factors, perModeSlicingInfo_factors),
		    gram_eigvals_type(eigvals, perModeSlicingInfo_eigvals) );
}

}} //end namespace TuckerOnNode::impl
#endif  // IMPL_TUCKERONNODE_STHOSVD_GRAM_IMPL_HPP_
