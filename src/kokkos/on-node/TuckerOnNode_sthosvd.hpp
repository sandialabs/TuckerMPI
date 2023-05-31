#ifndef TUCKER_KOKKOSONLY_STHOSVD_HPP_
#define TUCKER_KOKKOSONLY_STHOSVD_HPP_

#include "TuckerOnNode_CoreTensorTruncator.hpp"
#include "TuckerOnNode_Tensor.hpp"
#include "TuckerOnNode_TuckerTensor.hpp"
#include "TuckerOnNode_ComputeGram.hpp"
#include "TuckerOnNode_ttm.hpp"
#include <Kokkos_Core.hpp>

namespace TuckerOnNode{

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


template <class ScalarType, class ...Properties, class TruncatorType>
auto STHOSVD(Tensor<ScalarType, Properties...> & X,
	     TruncatorType && truncator,
	     bool useQR = false,
	     bool flipSign = false)
{
  using tensor_type  = Tensor<ScalarType, Properties...>;
  using memory_space = typename tensor_type::traits::memory_space;
  using factor_type  = TuckerTensor<ScalarType, memory_space>;
  using eigvec_view_t = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space>;

  const auto rank = X.rank();
  factor_type factorization(rank);
  tensor_type * Y = &X;
  tensor_type temp;
  for (std::size_t n=0; n<rank; n++)
  {
    std::cout << "\tAutoST-HOSVD::Starting Gram(" << n << ")...\n";

    std::cout << " \n ";
    std::cout << "\tAutoST-HOSVD::Gram(" << n << ") \n";
    auto S = compute_gram(*Y, n);
    auto S_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), S);
    for (std::size_t i=0; i<S_h.extent(0); ++i){
      for (std::size_t j=0; j<S_h.extent(1); ++j){
	      std::cout << S_h(i,j) << "  ";
      }
      std::cout << " \n ";
    }

    std::cout << " \n ";
    std::cout << "\tAutoST-HOSVD::Starting Evecs(" << n << ")...\n";
    auto eigvals = compute_eigenvalues(S, flipSign);
    factorization.eigValsAt(n) = eigvals;

    // need to copy back to S_h because of the reordering
    Kokkos::deep_copy(S_h, S);
    auto eigvals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigvals);
    for (std::size_t i=0; i<S.extent(0); ++i){ std::cout << eigvals_h(i) << "  "; }
    std::cout << " \n ";
    const std::size_t numEvecs = truncator(n, eigvals);

    std::cout << " \n ";
    eigvec_view_t eigVecs("eigVecs", Y->extent(n), numEvecs);
    auto eigVecs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigVecs);
    const int nToCopy = Y->extent(n)*numEvecs;
    const int ONE = 1;
    Tucker::copy(&nToCopy, S_h.data(), &ONE, eigVecs_h.data(), &ONE);
    for (std::size_t i=0; i<eigVecs_h.extent(0); ++i){
      for (std::size_t j=0; j<eigVecs_h.extent(1); ++j){
	      std::cout << eigVecs_h(i,j) << "  ";
      }
      std::cout << " \n ";
    }
    Kokkos::deep_copy(eigVecs, eigVecs_h);
    factorization.eigVecsAt(n) = eigVecs;

    std::cout << "\tAutoST-HOSVD::Starting TTM(" << n << ")...\n";
    std::cout << " \n ";
    temp = ttm(*Y, n, eigVecs, true);
    temp.writeToStream(std::cout);
    Kokkos::fence();

    Y = &temp;
    std::cout << "Local tensor size after STHOSVD iteration "
	      << n << ": " << Y->sizeArray() << ", or ";
  }

  factorization.getG() = *Y;

  return factorization;
}

}
#endif
