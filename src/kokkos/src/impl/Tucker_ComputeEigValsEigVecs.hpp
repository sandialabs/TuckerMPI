
#ifndef TUCKER_COMPUTE_EIGVALS_AND_EIGVECS_HPP_
#define TUCKER_COMPUTE_EIGVALS_AND_EIGVECS_HPP_

#include "Tucker_BlasWrapper.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace Tucker{
namespace impl{

template<class HostViewType, class DevViewType>
void flip_sign_eigenvecs_columns_on_host(HostViewType G_h, DevViewType G)
{
  using mem_space = typename HostViewType::memory_space;
  static_assert(Kokkos::is_view_v<HostViewType> && HostViewType::rank() == 2,
		"ViewType must be a rank-2 Kokkos view");
  static_assert(Kokkos::is_view_v<DevViewType> && DevViewType::rank() == 2,
		"ViewType must be a rank-2 Kokkos view");
  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, mem_space>::accessible,
		"flip_sign_eigenvecs_columns_on_host: view must be accessible on the host");

  using scalar_type = typename HostViewType::non_const_value_type;
  const int nrows = (int) G.extent(0);
  scalar_type* Gptr = G_h.data();
  for(int c=0; c<nrows; c++)
    {
      int maxIndex=0;
      scalar_type maxVal = std::abs(Gptr[c*nrows]);
      for(int r=1; r<nrows; r++)
	{
	  scalar_type testVal = std::abs(Gptr[c*nrows+r]);
	  std::cout << c << " " << r << " " << maxVal << " " << testVal << std::endl;
	  if(testVal > maxVal) {
	    maxIndex = r;
	    maxVal = testVal;
	  }
	}

      if(Gptr[c*nrows+maxIndex] < 0) {
	std::cout << "scal : "
		  << maxIndex << " " << Gptr[c*nrows+maxIndex]
	   	  << " " << *(Gptr+c*nrows) << std::endl;
	const int ONE = 1;
        const scalar_type NEGONE = -1;
	Tucker::scal(&nrows, &NEGONE, Gptr+c*nrows, &ONE);
      }
    }

  Kokkos::deep_copy(G, G_h);
}

template<class Exespace, class ViewType>
void flip_sign_eigenvecs_columns(const Exespace & exespace, ViewType G)
{
  static_assert(Kokkos::is_view_v<ViewType> && (ViewType::rank == 2),
		"ViewType must be a rank-2 Kokkos view");

  using scalar_type = typename ViewType::non_const_value_type;
  using mem_space = typename ViewType::memory_space;
  using space_t   = Exespace;
  using policy_t  = Kokkos::TeamPolicy<space_t>;
  using reducer_t = Kokkos::MaxLoc<scalar_type, std::size_t, mem_space>;
  using reduction_value_t = typename reducer_t::value_type;

  const std::size_t numTeams = G.extent(1);
  policy_t policy(exespace, numTeams, Kokkos::AUTO());
  Kokkos::parallel_for(policy,
		       KOKKOS_LAMBDA(typename policy_t::member_type member)
		       {
			 const int colInd = member.league_rank();

			 // first we need to figure out if elements in this column
			 // must have sign flipped
			 reduction_value_t result = {};
			 const std::size_t numRows = G.extent(0);
			 Kokkos::parallel_reduce(Kokkos::TeamThreadRange(member, numRows),
						 [=] (const std::size_t i,
						      reduction_value_t & res)
						 {
						   scalar_type rawVal  = G(i,colInd);
						   const auto testVal = Kokkos::abs(rawVal);
						   if (res.val <= testVal){
						     res.val = testVal;
						     res.loc = i;
						   }
						 }, reducer_t(result));

			 const bool mustFlipSign = G(result.loc, colInd) < 0 ? true : false;
			 if (mustFlipSign){
			   Kokkos::parallel_for(Kokkos::TeamThreadRange(member, numRows),
						[=] (const std::size_t i){
						  G(i,colInd) *= -1;
						});
			 }
		       });

}

template<class ScalarType, class ... Properties>
auto compute_and_sort_descending_eigvals_and_eigvecs_inplace(Kokkos::View<ScalarType**, Properties...> G,
							     const bool flipSign)
{
  if (G.extent(0) != G.extent(1)){
    throw std::runtime_error("G must be symmetric for calling syev");
  }

  using view_type = Kokkos::View<ScalarType**, Properties...>;
  using exe_space = typename view_type::execution_space;
  using mem_space = typename view_type::memory_space;
  static_assert(std::is_same_v< typename view_type::array_layout, Kokkos::LayoutLeft>
		&& std::is_floating_point< typename view_type::value_type>::value,
		"G must have layoutleft and must be real");

  auto exespace = exe_space();

  /*
   * do the eigen decomposition
   */
  auto G_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), G);
  const int nrows = (int) G.extent(0);
  Kokkos::View<ScalarType*, mem_space> eigenvalues_d("EIG", nrows);
  auto eigenvalues_h = Kokkos::create_mirror_view(eigenvalues_d);

  // 'V' means Compute eigenvalues and eigenvectors.
  char jobz = 'V';
  char uplo = 'U';
  int lwork = (int) 8*nrows;
  std::vector<ScalarType> work(lwork);
  int info;
  Tucker::syev(&jobz, &uplo, &nrows, G_h.data(), &nrows,
	       eigenvalues_h.data(), work.data(), &lwork, &info);
  if(info != 0){
    std::cerr << "Error: invalid error code returned by dsyev (" << info << ")\n";
  }
  //FIXME: these deep copies are here because syev is done on host but
  // one we do the device call these will go away
  Kokkos::deep_copy(eigenvalues_d, eigenvalues_h);
  Kokkos::deep_copy(G, G_h);

  /*
    sorting
    -------
    Here, since jobz is V, if info == 0 it means LAPACK computes things in ascending order
    see here: https://netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga442c43fca5493590f8f26cf42fed4044.html
    This means that:

       eigvals(0) < eigvals(1) < ... < eigvals(N-1)

    and eigenvectors are ordered accordingly.
    We need to sort eigvvals and eigvec in !!!descending!! order to have:

       eigvals(0) > eigvals(1) > ...
  */
  const std::size_t n = eigenvalues_d.extent(0);
  Kokkos::Experimental::reverse(exespace, eigenvalues_d);

  // FIXME: this will need to change when we can run team-level swap_ranges
  const std::size_t nCols = G.extent(1);
  for (std::size_t j=0; j<nCols/2; ++j){
    auto a = Kokkos::subview(G, Kokkos::ALL, j);
    auto b = Kokkos::subview(G, Kokkos::ALL, nCols - j -1);
    Kokkos::Experimental::swap_ranges(exespace, a, b);
  }

  if (flipSign){
#if defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)
    // left here for the time being as backup
    flip_sign_eigenvecs_columns_on_host(G_h, G);
#else
    flip_sign_eigenvecs_columns(exespace, G);
#endif
  }
  exespace.fence();

  return eigenvalues_d;
}

}}
#endif
