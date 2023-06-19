
#ifndef TUCKER_COMPUTE_EIGVALS_AND_EIGVECS_HPP_
#define TUCKER_COMPUTE_EIGVALS_AND_EIGVECS_HPP_

#include "Tucker_BlasWrapper.hpp"
#include <Kokkos_Core.hpp>

namespace Tucker{

template<class ScalarType, class ... Properties>
auto compute_eigenvals_and_eigenvecs_inplace(Kokkos::View<ScalarType**, Properties...> G,
					     const bool flipSign)
{
  if (G.extent(0) != G.extent(1)){
    throw std::runtime_error("G must be symmetric for calling syev");
  }

  using view_type = Kokkos::View<ScalarType**, Properties...>;
  using mem_space = typename view_type::memory_space;
  static_assert(std::is_same_v< typename view_type::array_layout, Kokkos::LayoutLeft>
		&& std::is_floating_point< typename view_type::value_type>::value,
		"G must have layoutleft and must be real");

  const int nrows = (int) G.extent(0);
  auto G_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), G);
  Kokkos::View<ScalarType*, mem_space> eigenvalues_d("EIG", nrows);
  auto eigenvalues_h = Kokkos::create_mirror_view(eigenvalues_d);

  char jobz = 'V'; // 'V' means Compute eigenvalues and eigenvectors.
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

}// end namespace Tucker
#endif
