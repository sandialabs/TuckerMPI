#ifndef IMPL_TUCKERONNODE_TTM_USING_KOKKOS_KERNELS_IMPL_HPP_
#define IMPL_TUCKERONNODE_TTM_USING_KOKKOS_KERNELS_IMPL_HPP_

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Team_Impl.hpp>
#include <KokkosBlas3_gemm.hpp>
#include <Kokkos_Core.hpp>

namespace TuckerOnNode{
namespace impl{

template <
  class ScalarType, class ...TensorProperties,
  class ViewDataType, class ...ViewProps>
void ttm_mode_zero_use_kkernels_gemm(Tensor<ScalarType, TensorProperties...> B,
				     int n,
				     Kokkos::View<ViewDataType, ViewProps ...> A,
				     Tensor<ScalarType, TensorProperties...> C,
				     bool Atransp)
{

  // constraints
  using tensor_type      = Tensor<ScalarType, TensorProperties...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;
  using tensor_layout    = typename tensor_type::traits::array_layout;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>,
		"tensor must have LayoutLeft");

  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, tensor_mem_space,
				Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  /** C = beta*C + alpha*op(A)*op(B)
   * A is m by k
   * B is k by blas_n
   * C is m by blas_n
   * Keep in mind: dimensions are set for a given Mode n
   */
  const size_t ncols = B.prod(1,B.rank()-1);
  char transa = Atransp ? 'T' : 'N';
  const char transb = 'N';
  const int m = C.extent(n);                    // 1st dim of A and C
  const int blas_n = (int)ncols;                // 2nd dim of B and C
  const int k = B.extent(n);                    // 1st dim of B
  const ScalarType alpha = ScalarType(1); // input coef. of op(A)*op(B)
  const ScalarType beta = ScalarType(0);  // input coef. of C

  umv_type Bumv(B.data().data(), k, blas_n);
  umv_type Cumv(C.data().data(), m, blas_n);
  KokkosBlas::gemm(&transa, &transb, alpha, A, Bumv, beta, Cumv);
}

template <
  class ScalarType, class ...TensorProperties,
  class ViewDataType, class ...ViewProps>
void ttm_nonzero_mode_use_kkernels_team_gemm(Tensor<ScalarType, TensorProperties...> B,
					     int n,
					     Kokkos::View<ViewDataType, ViewProps ...> A,
					     Tensor<ScalarType, TensorProperties...> C,
					     bool Btransp)
{

  // constraints
  using tensor_type      = Tensor<ScalarType, TensorProperties...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;
  using tensor_layout    = typename tensor_type::traits::array_layout;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>,
		"tensor must have LayoutLeft");

  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, tensor_mem_space,
				Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const size_t ncols = B.prod(0,n-1);
  const int Unrows = (Btransp) ? B.extent(n) : C.extent(n);
  const int Uncols = (Btransp) ? C.extent(n) : B.extent(n);

  const ScalarType alpha = ScalarType(1);
  const ScalarType beta = ScalarType(0);
  auto B_ptr_d = B.data().data();
  auto C_ptr_d = C.data().data();

  int m = (int)ncols;       // 1st dim of B and C
  int blas_n = C.extent(n); // 2nd dim of C
  const size_t nmats = B.prod(n+1,B.rank()-1,1);
  using exespace    = typename umv_type::execution_space;
  using policy_t    = Kokkos::TeamPolicy<exespace>;
  using member_type = typename policy_t::member_type;
  using kk_T        = KokkosBatched::Trans::Transpose;
  using kk_NT       = KokkosBatched::Trans::NoTranspose;
  using algo_flag   = KokkosBatched::Algo::Gemm::Unblocked;

  policy_t policy(nmats, Kokkos::AUTO);
  if (Btransp){
    const int k = Unrows;
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const member_type & member) {
	const int i = member.league_rank();

	umv_type Bumv(B_ptr_d+i*k*m, m, k);
	umv_type Cumv(C_ptr_d+i*m*blas_n, m, blas_n);
	KokkosBatched::TeamVectorGemm<member_type, kk_NT, kk_NT, algo_flag>
	  ::invoke(member, alpha, Bumv, A, beta, Cumv);
    });
  }
  else{

    const int k = Uncols;
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const member_type & member) {
	const int i = member.league_rank();

	umv_type Bumv(B_ptr_d+i*k*m, m, k);
	umv_type Cumv(C_ptr_d+i*m*blas_n, m, blas_n);
	KokkosBatched::TeamVectorGemm<member_type, kk_NT,kk_T, algo_flag>
	  ::invoke(member, alpha, Bumv, A, beta, Cumv);
    });
  }
}

template <
  class ScalarType, class ...TensorProperties,
  class ViewDataType, class ...ViewProps>
void ttm_nonzero_mode_sequentially_call_kkernels_gemm(Tensor<ScalarType, TensorProperties...> B,
						      int n,
						      Kokkos::View<ViewDataType, ViewProps ...> A,
						      Tensor<ScalarType, TensorProperties...> C,
						      bool Btransp)
{

  // constraints
  using tensor_type   = Tensor<ScalarType, TensorProperties...>;
  using tensor_mem_space = typename tensor_type::traits::memory_space;
  using tensor_layout = typename tensor_type::traits::array_layout;
  static_assert(std::is_same_v<tensor_layout, Kokkos::LayoutLeft>,
		"tensor must have LayoutLeft");

  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, tensor_mem_space,
				Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const size_t ncols = B.prod(0,n-1);
  const int Unrows = (Btransp) ? B.extent(n) : C.extent(n);
  const int Uncols = (Btransp) ? C.extent(n) : B.extent(n);

  const ScalarType alpha = ScalarType(1);
  const ScalarType beta = ScalarType(0);
  auto B_ptr_d = B.data().data();
  auto C_ptr_d = C.data().data();

  /**C = beta*C + alpha*op(B)*op(A)
   * B is m by k
   * A is k by Uncols
   * C is m by blas_n
   * Warning: A and B are reversed
   */
  char transa = 'N';                      // "N" for Non-tranpose
  char transb = Btransp ? 'N' : 'T';      // "T" for Transpose
  const int m = (int)ncols;                     // 1st dim of B and C
  const int blas_n = C.extent(n);               // 2nd dim of C
  const int k = Btransp ? Unrows : Uncols;      // 2nd dim of B

  const size_t nmats = B.prod(n+1,B.rank()-1,1);
  for(size_t i=0; i<nmats; i++) {
    umv_type Bumv(B_ptr_d+i*k*m, m, k);
    umv_type Cumv(C_ptr_d+i*m*blas_n, m, blas_n);
    KokkosBlas::gemm(&transa, &transb, alpha, Bumv, A, beta, Cumv);
  }
}

}//end namespace impl
}//end namespace Tucker
#endif  // IMPL_TUCKERONNODE_TTM_USING_KOKKOS_KERNELS_IMPL_HPP_
