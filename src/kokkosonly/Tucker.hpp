#ifndef TUCKER_KOKKOSONLY_TUCKER_HPP_
#define TUCKER_KOKKOSONLY_TUCKER_HPP_

#include <Kokkos_Core.hpp>
#include "Tucker_Tensor.hpp"
#include "compute_gram.hpp"
#include "ttm.hpp"
#include <variant>

namespace TuckerKokkos{

struct CoreRankUserDefined{
  SizeArray value;
};

template<class ScalarType>
struct CoreRankViaThreshold{
  ScalarType value;
};


template<class ScalarType, class ...Properties>
auto computeGram(Tensor<ScalarType, Properties...> * Y, const int n)
{
  using tensor_type = TuckerKokkos::Tensor<ScalarType, Properties...>;
  using memory_space = typename tensor_type::traits::memory_space;

  const int nrows = (int)Y->extent(n);
  Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space> S_d("S", nrows, nrows);
  auto S_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), S_d);
  computeGramHost(Y, n, S_h.data(), nrows);
  Kokkos::deep_copy(S_d, S_h);
  return S_d;
}

template<class ScalarType, class ... Properties>
auto computeEigenvalues(Kokkos::View<ScalarType**, Properties...> G,
			const bool flipSign)
{
  using view_type = Kokkos::View<ScalarType**, Properties...>;
  using mem_space = typename view_type::memory_space;
  static_assert(std::is_same_v< typename view_type::array_layout, Kokkos::LayoutLeft>);

  const int nrows = G.extent(0);
  auto G_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), G);
  Kokkos::View<ScalarType*, mem_space> eigenvalues_d("EIG", nrows);
  auto eigenvalues_h = Kokkos::create_mirror_view(eigenvalues_d);

  char jobz = 'V';
  char uplo = 'U';
  int lwork = 8*nrows;
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

template <class ScalarType, class ...Properties>
int countEigValsUsingThreshold(Kokkos::View<ScalarType*, Properties...> eigvals,
			       const ScalarType thresh)
{
  using eigvals_view_type = Kokkos::View<ScalarType*, Properties...>;
  using mem_space = typename eigvals_view_type::memory_space;
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, mem_space>::accessible,
		"countEigValsUsingThreshold: view must be accessible on host");

  int nrows = eigvals.extent(0);
  int numEvecs = nrows;
  ScalarType sum = 0;
  for(int i=nrows-1; i>=0; i--) {
    sum += std::abs(eigvals[i]);
    if(sum > thresh) {
      break;
    }
    numEvecs--;
  }
  return numEvecs;
}

template <class ScalarType, class ...Properties, class ...Variants>
auto STHOSVD(Tensor<ScalarType, Properties...> & X,
	     const std::variant<Variants...> & coreTensorRankInfo,
	     bool useQR = false,
	     bool flipSign = false)
{
  using tensor_type  = Tensor<ScalarType, Properties...>;
  using memory_space = typename tensor_type::traits::memory_space;
  using factor_type  = TuckerTensor<ScalarType, memory_space>;
  using eigvec_view_t = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space>;

  const int rank = (int)X.rank();

  // decide truncation mechanism
  auto truncator = [&](int n, auto eigenValues) -> int
  {
    auto autoRank = std::holds_alternative<CoreRankViaThreshold<ScalarType>>(coreTensorRankInfo);
    if (autoRank)
    {
      const ScalarType epsilon = std::get<CoreRankViaThreshold<ScalarType>>(coreTensorRankInfo).value;
      const ScalarType tensorNorm = X.frobeniusNormSquared();
      const ScalarType threshold  = epsilon*epsilon*tensorNorm/rank;
      std::cout << "\tAutoST-HOSVD::Tensor Norm: "
		<< std::sqrt(tensorNorm)
		<< "...\n";
      std::cout << "\tAutoST-HOSVD::Relative Threshold: "
		<< threshold
		<< "...\n";
      auto eigVals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),
							   eigenValues);
      return countEigValsUsingThreshold(eigVals_h, threshold);
    }
    else{
      (void) eigenValues; // unused
      const auto & R_dims = std::get<CoreRankUserDefined>(coreTensorRankInfo).value;
      return R_dims[n];
    }
  };


  factor_type factorization(rank);
  tensor_type * Y = &X;
  tensor_type temp;
  for (int n=0; n<rank; n++)
  {
    std::cout << "\tAutoST-HOSVD::Starting Gram(" << n << ")...\n";

    std::cout << " \n ";
    std::cout << "\tAutoST-HOSVD::Gram(" << n << ") \n";
    auto S = computeGram(Y, n);
    auto S_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), S);
    for (int i=0; i<S_h.extent(0); ++i){
      for (int j=0; j<S_h.extent(1); ++j){
	      std::cout << S_h(i,j) << "  ";
      }
      std::cout << " \n ";
    }

    std::cout << " \n ";
    std::cout << "\tAutoST-HOSVD::Starting Evecs(" << n << ")...\n";
    auto eigvals = computeEigenvalues(S, flipSign);
    factorization.eigValsAt(n) = eigvals;

    // need to copy back to S_h because of the reordering
    Kokkos::deep_copy(S_h, S);
    auto eigvals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigvals);
    for (int i=0; i<S.extent(0); ++i){ std::cout << eigvals_h(i) << "  "; }
    std::cout << " \n ";
    const int numEvecs = truncator(n, eigvals);

    std::cout << " \n ";
    eigvec_view_t eigVecs("eigVecs", Y->extent(n), numEvecs);
    auto eigVecs_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), eigVecs);
    const int nToCopy = (int)Y->extent(n)*numEvecs;
    const int ONE = 1;
    Tucker::copy(&nToCopy, S_h.data(), &ONE, eigVecs_h.data(), &ONE);
    for (int i=0; i<eigVecs_h.extent(0); ++i){
      for (int j=0; j<eigVecs_h.extent(1); ++j){
	      std::cout << eigVecs_h(i,j) << "  ";
      }
      std::cout << " \n ";
    }
    Kokkos::deep_copy(eigVecs, eigVecs_h);
    factorization.eigVecsAt(n) = eigVecs;

    std::cout << "\tAutoST-HOSVD::Starting TTM(" << n << ")...\n";
    std::cout << " \n ";
    temp = ttm(Y, n, eigVecs, true);
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
