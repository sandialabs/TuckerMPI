#ifndef IMPL_TUCKERONNODE_COMPUTE_GRAM_KOKKOS_IMPL_HPP_
#define IMPL_TUCKERONNODE_COMPUTE_GRAM_KOKKOS_IMPL_HPP_

#include "Tucker_syrk_kokkos.hpp"

namespace TuckerOnNode{
namespace impl{

template<class AUmvType, class CViewType, class YDataPtr>
struct GramKokkosAtomicFunctor
{
  CViewType Cview_;
  YDataPtr YPtr_;
  int Anrows_;
  int Ancols_;

  GramKokkosAtomicFunctor(CViewType Cview, YDataPtr YPtr, int Anrows, int Ancols)
    : Cview_(Cview), YPtr_(YPtr), Anrows_(Anrows), Ancols_(Ancols){}

  KOKKOS_FUNCTION void operator()(int mIn) const{
    const int m = mIn+1;

    auto Aptr = YPtr_ + m*Anrows_*Ancols_;
    AUmvType Aview(Aptr, Anrows_, Ancols_);

    for (std::size_t j = 0; j < Cview_.extent(1); ++j) {
      for (std::size_t i = 0; i <= j; ++i) {
	double sum = {};
	for (std::size_t k = 0; k < Aview.extent(0); ++k) {
	  sum += Aview(k,i) * Aview(k,j);
	}
	Cview_(i,j) += sum;
      }
    }
  }
};

template <class ScalarType, class DataType, class ...ViewProps, class ...Properties>
void compute_gram_kokkos(Tensor<ScalarType, Properties...> Y,
			 const std::size_t n,
			 Kokkos::View<DataType, ViewProps...> C)
{
  using tensor_type   = Tensor<ScalarType, Properties...>;
  using tensor_layout = typename tensor_type::traits::array_layout;
  using tensor_mem_space = typename tensor_type::traits::memory_space;
  using view_type   = Kokkos::View<DataType, ViewProps...>;
  using view_layout = typename view_type::array_layout;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>
		&& std::is_same_v<view_layout, Kokkos::LayoutLeft>,
		"compute_gram_kokko:: tensor and view must have LayoutLeft");

  const int nrows = (int)Y.extent(n);
  auto Y_rawPtr = Y.data().data();
  using A_umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, tensor_mem_space, 
				Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // n = 0 is a special case, Y_0 is stored column major
  if(n == 0)
  {

    // Compute number of columns of Y_n
    // Technically, we could divide the total number of entries by n,
    // but that seems like a bad decision
    int ncols = 1;
    for(int i=0; i<(int)Y.rank(); i++) {
      if((std::size_t)i != n) { ncols *= (int)Y.extent(i); }
    }

    // here I can just do a single basic syrk call: C := alpha*A*A' + beta*C
    // which corresponds to:
    //   call syrk('U', 'N', nrows, ncols, alpha, Aptr, nrows, beta, Cptr, C.extent(0))
    const ScalarType alpha = 1;
    const ScalarType beta = 0;
    A_umv_type Aview(Y.data().data(), Y.extent(0), ncols);
    Tucker::impl::syrk_kokkos("U", "N", alpha, Aview, beta, C);
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

    /* IMPORTANT:

       The code below was originally implemented as a series of syrk
       similarly to what was done in the original Tucker code:

       for(int i=0; i<nmats; i++) {
	 const ScalarType alpha = 1;
	 const ScalarType beta = (i==0) ? 0 : 1;
	 auto Aptr = Y_rawPtr+i*nrows*ncols;
	 A_umv_type Aview(Aptr, ncols, nrows);
	 Tucker::impl::syrk_kokkos(exespace, "U", "T", alpha, Aview, beta, C);
       }
       Note that for i==0, C is being overwritten but for i>0 C is *updated*.

       That works, however it has pretty bad performance when using the GPU because
       this dispatches MANY kernels. One solution to this would be to use stream parallelism,
       or some sort of batched blas calls, but it does not seem like batched syrk is supported.
       For now, what we do instead is to manually rewrite this loop to take advantage of the device,
       by doing the following:
         1. first, we do a single syrk corresponding to i==0 above
	 2. second, we do a parfor where we update C atomically

       Potentially this can be further improved via team-level or similar.
     */

    // step 1., single syrk for alpha=1 and beta = 0
    const ScalarType alpha = 1;
    const ScalarType beta = 0;
    A_umv_type Aview(Y_rawPtr, ncols, nrows);
    Tucker::impl::syrk_kokkos("U", "T", alpha, Aview, beta, C);

    // step 2.: use parfor to update C
    using CViewType = Kokkos::View<DataType, ViewProps...>;
    using C_atom_type = Kokkos::View<ScalarType**, view_layout, typename CViewType::memory_space, Kokkos::MemoryTraits<Kokkos::Atomic> >;
    C_atom_type myC = C;
    // note that ncols and nrows are intentially passed in this order
    // because the matrix A has extents ncols x nrows in this branch of gram
    GramKokkosAtomicFunctor<A_umv_type, C_atom_type, decltype(Y_rawPtr)> func(myC, Y_rawPtr, ncols, nrows);
    Kokkos::parallel_for(nmats-1, func);
  }
}

}}
#endif  // IMPL_TUCKERONNODE_COMPUTE_GRAM_KOKKOS_IMPL_HPP_
