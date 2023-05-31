#ifndef TUCKER_TTM_USING_KOKKOS_KERNELS_HPP_
#define TUCKER_TTM_USING_KOKKOS_KERNELS_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include <Kokkos_Core.hpp>
#include <KokkosBlas3_gemm.hpp>

namespace TuckerOnNode{
namespace impl{

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
std::enable_if_t<
  std::is_same_v<typename Tensor<ScalarType, TensorProperties...>::traits::array_layout, Kokkos::LayoutLeft>
  && std::is_same_v<typename Kokkos::View<ScalarType**, ViewProperties...>::array_layout, Kokkos::LayoutLeft>
  >
ttm_kker_mode_zero(Tensor<ScalarType, TensorProperties...> B,
		   int n,
		   Kokkos::View<ScalarType**, ViewProperties...> A,
		   Tensor<ScalarType, TensorProperties...> C,
		   bool Atransp)
{
  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft,
				Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  /** C = beta*C + alpha*op(A)*op(B)
   * A is m by k
   * B is k by blas_n
   * C is m by blas_n
   * Keep in mind: dimensions are set for a given Mode n
   */
  const size_t ncols = B.sizeArray().prod(1,B.rank()-1);
  char transa = Atransp ? 'T' : 'N';
  const char transb = 'N';
  int m = C.extent(n);                    // 1st dim of A and C
  int blas_n = (int)ncols;                // 2nd dim of B and C
  int k = B.extent(n);                    // 1st dim of B
  const ScalarType alpha = ScalarType(1); // input coef. of op(A)*op(B)
  const ScalarType beta = ScalarType(0);  // input coef. of C

  umv_type Bumv(B.data().data(), k, blas_n);
  umv_type Cumv(C.data().data(), m, blas_n);
  KokkosBlas::gemm(&transa, &transb, alpha, A, Bumv, beta, Cumv);
}

template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
std::enable_if_t<
  std::is_same_v<typename Tensor<ScalarType, TensorProperties...>::traits::array_layout, Kokkos::LayoutLeft>
  && std::is_same_v<typename Kokkos::View<ScalarType**, ViewProperties...>::array_layout, Kokkos::LayoutLeft>
  >
ttm_kker_mode_greater_than_zero(Tensor<ScalarType, TensorProperties...> B,
				int n,
				Kokkos::View<ScalarType**, ViewProperties...> A,
				Tensor<ScalarType, TensorProperties...> C,
				bool Btransp)
{
  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft,
				Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const size_t ncols = B.sizeArray().prod(0,n-1);
  int Unrows, Uncols;
  if(Btransp) {
    Unrows = B.extent(n);
    Uncols = C.extent(n);
  } else {
    Uncols = B.extent(n);
    Unrows = C.extent(n);
  }

  const size_t nmats = B.sizeArray().prod(n+1,B.rank()-1,1);
  for(size_t i=0; i<nmats; i++) {
    /**C = beta*C + alpha*op(B)*op(A)
     * B is m by k
     * A is k by Uncols
     * C is m by blas_n
     * Warning: A and B are reversed
     */
    char transa = 'N';                      // "N" for Non-tranpose
    char transb = Btransp ? 'N' : 'T';      // "T" for Transpose
    int m = (int)ncols;                     // 1st dim of B and C
    int blas_n = C.extent(n);               // 2nd dim of C
    int k = Btransp ? Unrows : Uncols;      // 2nd dim of B
    const ScalarType alpha = ScalarType(1); // input coef. of op(B)*op(A)
    const ScalarType beta = ScalarType(0);  // input coef. of C

    auto B_ptr_d = B.data().data();
    auto C_ptr_d = C.data().data();
    umv_type Bumv(B_ptr_d+i*k*m, m, k);
    umv_type Cumv(C_ptr_d+i*m*blas_n, m, blas_n);
    KokkosBlas::gemm(&transa, &transb, alpha, Bumv, A, beta, Cumv);
  }
}

}//end namespace impl
}//end namespace Tucker
#endif
