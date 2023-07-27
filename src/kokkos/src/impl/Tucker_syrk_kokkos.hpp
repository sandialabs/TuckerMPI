
#ifndef TUCKER_SYRK_HPP_
#define TUCKER_SYRK_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <fstream>
#include <vector>
#include <iomanip>

namespace Tucker{
namespace impl{

template <class AlphaType, class BetaType, class AViewType, class CViewType>
struct SyrkFunctor1
{
  AlphaType alpha_;
  BetaType beta_;
  AViewType Aview_;
  CViewType Cview_;
  SyrkFunctor1(AViewType Av, CViewType Cv,
	       AlphaType alpha, BetaType beta)
    : Aview_(Av), Cview_(Cv), alpha_(alpha), beta_(beta){}

  KOKKOS_FUNCTION void operator()(const std::size_t j) const
  {
    for (std::size_t i = 0; i <= j; ++i) {
      typename CViewType::non_const_value_type sum = {};
      for (std::size_t k = 0; k < Aview_.extent(1); ++k) {
	sum += Aview_(i,k) * Aview_(j,k);
      }
      Cview_(i,j) = beta_*Cview_(i,j) + alpha_*sum;
    }
  }
};

template <class AlphaType, class BetaType, class AViewType, class CViewType>
struct SyrkFunctor2
{
  AlphaType alpha_;
  BetaType beta_;
  AViewType Aview_;
  CViewType Cview_;
  SyrkFunctor2(AViewType Av, CViewType Cv,
	       AlphaType alpha, BetaType beta)
    : Aview_(Av), Cview_(Cv), alpha_(alpha), beta_(beta){}

  KOKKOS_FUNCTION void operator()(const std::size_t j) const
  {
    for (std::size_t i = 0; i <= j; ++i) {
      typename CViewType::non_const_value_type sum = {};
      for (std::size_t k = 0; k < Aview_.extent(0); ++k) {
	sum += Aview_(k,i) * Aview_(k,j);
      }
      Cview_(i,j) = beta_*Cview_(i,j) + alpha_*sum;
    }
  }
};

template<class AViewType, class CViewType>
void syrk_kokkos(const char uplo[],
		 const char opA[],
		 typename AViewType::const_value_type & alpha,
		 const AViewType& A,
		 typename CViewType::const_value_type & beta,
		 const CViewType& C)
{
  static_assert(    std::is_floating_point_v<typename AViewType::const_value_type>
		 && std::is_floating_point_v<typename CViewType::const_value_type>,
		 "syrk_kokkos: A and C currently must be floating point matrices");

  static_assert(Kokkos::is_view_v<AViewType>,
                "syrk_kokkos: AViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view_v<CViewType>,
                "syrk_kokkos: CViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(AViewType::rank) == 2,
                "syrk_kokkos: AViewType must have rank 2.");
  static_assert(static_cast<int>(CViewType::rank) == 2,
                "syrk_kokkos: CViewType must have rank 2.");

  if (C.extent(0) != C.extent(1)){
    throw std::runtime_error("syrk_kokkos: C must be symmetric");
  }

  if (opA[0] != 'N' && opA[0] != 'T'){
    throw std::runtime_error("syrk_kokkos: opA must be 'N' or 'T'");
  }

  if (uplo[0] != 'U'){
    throw std::runtime_error("syrk_kokkos: currently uplo must be 'U'");
  }

  if (opA[0] == 'N' && C.extent(0) != A.extent(0)){
    throw std::runtime_error("syrk_kokkos: opA=N : A.extent(0) should equal C.extent(0)");
  }
  if (opA[0] == 'T' && C.extent(0) != A.extent(1)){
    throw std::runtime_error("syrk_kokkos: opA=T : A.extent(1) should equal C.extent(0)");
  }

  if (opA[0] == 'N'){
    // symmetric rank-k update: C := alpha*A*A' + beta*C
    using alpha_t = typename AViewType::const_value_type;
    using beta_t  = typename CViewType::const_value_type;
    using func_t = impl::SyrkFunctor1<alpha_t, beta_t, AViewType, CViewType>;
    Kokkos::parallel_for(Kokkos::RangePolicy(0, C.extent(1)),
			 func_t(A, C, alpha, beta));
  }

  else if (opA[0] == 'T'){
    using alpha_t = typename AViewType::const_value_type;
    using beta_t  = typename CViewType::const_value_type;
    using func_t = impl::SyrkFunctor2<alpha_t, beta_t, AViewType, CViewType>;
    Kokkos::parallel_for(Kokkos::RangePolicy(0, C.extent(1)),
			 func_t(A, C, alpha, beta));
  }
}

}}// end namespace Tucker::impl
#endif
