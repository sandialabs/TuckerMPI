#ifndef TUCKERKOKKOS_COMP_GRAM_KOKKOS_IMPL_HPP_
#define TUCKERKOKKOS_COMP_GRAM_KOKKOS_IMPL_HPP_

#include "Tucker_BlasWrapper.hpp"
#include "TuckerOnNode_Tensor.hpp"

namespace TuckerOnNode{
namespace impl{

template <class ScalarType, class AViewType, class CViewType>
struct FunctorGramModeZero
{
  ScalarType alpha_;
  ScalarType beta_;
  AViewType Aview_;
  CViewType Cview_;
  FunctorGramModeZero(AViewType Av, CViewType Cv,
		      ScalarType alpha, ScalarType beta) :
    Aview_(Av), Cview_(Cv), alpha_(alpha), beta_(beta){}

  KOKKOS_FUNCTION void operator()(const std::size_t j) const
  {
    for (std::size_t i = 0; i <= j; ++i) {
      ScalarType sum = {};
      for (std::size_t k = 0; k < Aview_.extent(1); ++k) {
	sum += Aview_(i,k) * Aview_(j,k);
      }
      Cview_(i,j) = beta_*Cview_(i,j) + alpha_*sum;
    }

  }
};

template <class ScalarType, class AViewType, class CViewType>
struct FunctorGramModeNonZero
{
  ScalarType alpha_;
  ScalarType beta_;
  AViewType Aview_;
  CViewType Cview_;
  FunctorGramModeNonZero(AViewType Av, CViewType Cv,
		      ScalarType alpha, ScalarType beta) :
    Aview_(Av), Cview_(Cv), alpha_(alpha), beta_(beta){}

  KOKKOS_FUNCTION void operator()(const std::size_t j) const
  {
    for (std::size_t i = 0; i <= j; ++i) {
      ScalarType sum = {};
      for (std::size_t k = 0; k < Aview_.extent(0); ++k) {
	sum += Aview_(k,i) * Aview_(k,j);
      }
      Cview_(i,j) = beta_*Cview_(i,j) + alpha_*sum;
    }
  }
};

template <class ScalarType, class DataType, class ...ViewProps, class ...Properties>
void compute_gram_kokkos(Tensor<ScalarType, Properties...> Y,
			 const std::size_t n,
			 Kokkos::View<DataType, ViewProps...> C)
{

  const int nrows = (int)Y.extent(n);
  auto Y_rawPtr = Y.data().data();
  auto gramPtr = C.data();

  using C_view_type = Kokkos::View<DataType, ViewProps...>;
  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft,
				Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // n = 0 is a special case, Y_0 is stored column major
  if(n == 0)
  {
    // Compute number of columns of Y_n
    // Technically, we could divide the total number of entries by n,
    // but that seems like a bad decision
    int ncols = 1;
    for(int i=0; i<(int)Y.rank(); i++) {
      if((std::size_t)i != n) {
        ncols *= (int)Y.extent(i);
      }
    }

    // symmetric rank-k update
    //   C := alpha*A*A' + beta*C
    // which corresponds to:
    //   call syrk('U', 'N', nrows, ncols, alpha, Aptr, nrows, beta, Cptr, C.extent(0))
    const ScalarType alpha = 1;
    const ScalarType beta = 0;
    umv_type Aview(Y.data().data(), Y.extent(0), ncols);
    using func_t = FunctorGramModeZero<ScalarType, umv_type, C_view_type>;
    Kokkos::parallel_for(Kokkos::RangePolicy(0, C.extent(1)),
			 func_t(Aview, C, alpha, beta));
  }

  else
  {
    int ncols = 1;
    int nmats = 1;
    for(std::size_t i=0; i<n; i++) {
      ncols *= (int)Y.extent(i);
    }
    for(int i=n+1; i<(int)Y.rank(); i++) {
      nmats *= (int)Y.extent(i);
    }

    for(int i=0; i<nmats; i++) {
      // symmetric rank-k update as follows:
      //   C := alpha*A'*A + beta*C
      // which corresponds to:
      //   dsyrk('U', 'T', nrows, ncols, alpha,
      //         Aptr, ncols, beta, Cptr, C.extent(0))

      const ScalarType alpha = 1;
      const ScalarType beta = (i==0) ? 0 : 1;
      auto Aptr = Y_rawPtr+i*nrows*ncols;
      umv_type Aview(Aptr, ncols, nrows);
      using func_t = FunctorGramModeNonZero<ScalarType, umv_type, C_view_type>;
      Kokkos::parallel_for(Kokkos::RangePolicy(0, C.extent(1)),
			   func_t(Aview, C, alpha, beta));
   }
  }
}

}}
#endif
