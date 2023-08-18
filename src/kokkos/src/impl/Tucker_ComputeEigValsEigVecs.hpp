
#ifndef IMPL_TUCKER_COMPUTEEIGVALSEIGVECS_HPP_
#define IMPL_TUCKER_COMPUTEEIGVALSEIGVECS_HPP_

#include "Tucker_BlasWrapper.hpp"
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#if defined KOKKOS_ENABLE_HIP
#include <rocsolver/rocsolver.h>
#endif

#if defined KOKKOS_ENABLE_CUDA
#include <cusolverDn.h>
#endif

namespace Tucker{
namespace impl{

#if defined KOKKOS_ENABLE_CUDA
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

#endif

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
	  if(testVal > maxVal) {
	    maxIndex = r;
	    maxVal = testVal;
	  }
	}

      if(Gptr[c*nrows+maxIndex] < 0) {
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

template <class ExecutionSpace>
struct better_off_calling_host_syev : std::false_type {};

#if defined KOKKOS_ENABLE_SERIAL
template <>
struct better_off_calling_host_syev<Kokkos::Serial> : std::true_type {};
#endif

#if defined KOKKOS_ENABLE_OPENMP
template <>
struct better_off_calling_host_syev<Kokkos::OpenMP> : std::true_type {};
#endif

#if defined KOKKOS_ENABLE_THREADS
template <>
struct better_off_calling_host_syev<Kokkos::Threads> : std::true_type {};
#endif

#if defined KOKKOS_ENABLE_HPX
template <>
struct better_off_calling_host_syev<Kokkos::Experimental::HPX> : std::true_type {
};
#endif

template <class T>
inline constexpr bool better_off_calling_host_syev_v =
    better_off_calling_host_syev<T>::value;


template<class ScalarType, class ... AProperties, class ... EigvalProperties>
void compute_syev_use_host_lapack(Kokkos::View<ScalarType**, AProperties...> A,
			Kokkos::View<ScalarType*, EigvalProperties...> eigenvalues)
{

  using A_view_type  = Kokkos::View<ScalarType**, AProperties...>;
  using A_mem_space  = typename A_view_type::memory_space;
  using A_layout     = typename A_view_type::array_layout;
  using A_value_type = typename A_view_type::non_const_value_type;

  using ev_view_type  = Kokkos::View<ScalarType*, EigvalProperties...>;
  using ev_mem_space  = typename ev_view_type::memory_space;
  using ev_layout     = typename ev_view_type::array_layout;
  using ev_value_type = typename ev_view_type::non_const_value_type;

  static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, A_mem_space>::accessible
		&& Kokkos::SpaceAccessibility<Kokkos::HostSpace, ev_mem_space>::accessible,
		"do_syev_on_host: Views must be accessible on host");
  static_assert(std::is_same_v<A_layout, Kokkos::LayoutLeft>
		&& std::is_same_v<ev_layout, Kokkos::LayoutLeft>,
		"do_syev_on_host: Views must have LayoutLeft");
  static_assert(std::is_floating_point< typename A_view_type::value_type>::value
		&& std::is_floating_point< typename ev_view_type::value_type>::value,
		"do_syev_on_host: Views must have floating point value_type");

  const int nrows = (int) A.extent(0);

  // 'V' means Compute eigenvalues and eigenvectors.
  char jobz = 'V';
  char uplo = 'U';
  int lwork = (int) 8*nrows;
  std::vector<ScalarType> work(lwork);
  int info;
  Tucker::syev(&jobz, &uplo, &nrows, A.data(), &nrows,
	       eigenvalues.data(), work.data(), &lwork, &info);
  if(info != 0){
    std::cerr << "Error: invalid error code returned by dsyev (" << info << ")\n";
  }
}


#if defined(KOKKOS_ENABLE_HIP) && !defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)

template<class ScalarType, class ... AProperties, class ... EigvalProperties>
void compute_syev_on_device_views(const Kokkos::HIP & /*exec*/,
			  Kokkos::View<ScalarType**, AProperties...> A,
			  Kokkos::View<ScalarType*, EigvalProperties...> eigenvalues)
{  
  rocblas_handle handle;
  rocblas_create_handle(&handle);

  // https://rocsolver.readthedocs.io/en/latest/api/lapack.html#rocsolver-type-syevd
  const std::size_t nrows = A.extent(0);
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, Kokkos::HIPSpace> work("lwork", nrows);
  Kokkos::View<int, Kokkos::HIPSpace> info("info");

  rocblas_status status = {};
  if constexpr(std::is_same_v<ScalarType, double>){
    status = rocsolver_dsyev(handle,
			     rocblas_evect::rocblas_evect_original,
			     rocblas_fill::rocblas_fill_upper,
			     nrows, A.data(), nrows,
			     eigenvalues.data(),
			     work.data(), info.data());
  }
  if constexpr(std::is_same_v<ScalarType, float>){
    status = rocsolver_ssyev(handle,
			     rocblas_evect::rocblas_evect_original,
			     rocblas_fill::rocblas_fill_upper,
			     nrows, A.data(), nrows,
			     eigenvalues.data(),
			     work.data(), info.data());
  }

  if(status != rocblas_status_success){
    throw std::runtime_error("syev: status != rocblas_status_success");
  }

  auto info_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), info);
  if(info_h() != 0){
    throw std::runtime_error("syev: info != 0");
  }
  rocblas_destroy_handle(handle);
}
#endif


#if defined(KOKKOS_ENABLE_CUDA) && !defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)

template<class ScalarType, class ... AProperties, class ... EigvalProperties>
void compute_syev_on_device_views(const Kokkos::Cuda & exec,
      Kokkos::View<ScalarType**, AProperties...> A,
      Kokkos::View<ScalarType*, EigvalProperties...> eigenvalues)
{
  cusolverDnHandle_t cuDnHandle = nullptr;


  /* step 1: create cusolver handle, bind a stream */
  CUSOLVER_CHECK( cusolverDnCreate(&cuDnHandle) );

  auto stream = exec.cuda_stream();
  CUDA_CHECK( cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) );
  CUSOLVER_CHECK( cusolverDnSetStream(cuDnHandle, stream) );

// Doc for Reference
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSOLVER/Xsyevd/cusolver_Xsyevd_example.cu
// cusolverStatus_t
// cusolverDnXsyevd(
//     cusolverDnHandle_t handle,
//     cusolverDnParams_t params,
//     cusolverEigMode_t jobz,
//     cublasFillMode_t uplo,
//     int64_t n,
//     cudaDataType dataTypeA,
//     void *A,
//     int64_t lda,
//     cudaDataType dataTypeW,
//     void *W,
//     cudaDataType computeType,
//     void *bufferOnDevice,
//     size_t workspaceInBytesOnDevice,
//     void *bufferOnHost,
//     size_t workspaceInBytesOnHost,
//     int *info)

  const int64_t nrows = (int64_t) A.extent(0);
  Kokkos::View<int, Kokkos::CudaSpace> info("info");

  cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
  cublasFillMode_t uplo  = CUBLAS_FILL_MODE_UPPER;

  size_t d_lwork = 0;     /* size of workspace */
  void *d_work = nullptr; /* device workspace */
  size_t h_lwork = 0;     /* size of workspace */
  void *h_work = nullptr; /* host workspace */

  static_assert(std::is_floating_point_v<ScalarType>);
  constexpr cudaDataType cuda_data_type = std::is_same_v<ScalarType, double> ? CUDA_R_64F : CUDA_R_32F;

  CUSOLVER_CHECK( cusolverDnXsyevd_bufferSize(cuDnHandle, nullptr, jobz, uplo, nrows,
					      cuda_data_type, A.data(), nrows,
					      cuda_data_type, d_work,
					      cuda_data_type, &d_lwork, &h_lwork) );

  CUDA_CHECK( cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(cuda_data_type) * d_lwork) );
  h_work = (ScalarType*)malloc(h_lwork);

  CUSOLVER_CHECK( cusolverDnXsyevd(cuDnHandle, nullptr, jobz, uplo, nrows,
				   cuda_data_type, A.data(), nrows,
				   cuda_data_type, eigenvalues.data(),
				   cuda_data_type, d_work, d_lwork,
				   h_work, h_lwork,
				   info.data()) );

  auto info_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), info);
  if(info_h() != 0){ throw std::runtime_error("syev: info != 0"); }

  cudaFree(d_work);
  free(h_work);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUSOLVER_CHECK( cusolverDnDestroy(cuDnHandle) );
}
#endif


// fallback case if no better specialization is found
template<class ExecutionSpace, class ScalarType, class ... AProperties, class ... EigvalProperties>
void compute_syev_on_device_views(const ExecutionSpace& exec,
			  Kokkos::View<ScalarType**, AProperties...> A,
			  Kokkos::View<ScalarType*, EigvalProperties...> eigenvalues)
{
  auto A_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto eigenvalues_h = Kokkos::create_mirror_view(eigenvalues);
  compute_syev_use_host_lapack(A_h, eigenvalues_h);
  Kokkos::deep_copy(exec, eigenvalues, eigenvalues_h);
  Kokkos::deep_copy(exec, A, A_h);
  exec.fence();
}

template<class ScalarType, class ... Properties>
auto compute_and_sort_descending_eigvals_and_eigvecs_inplace(Kokkos::View<ScalarType**, Properties...> G,
							     const bool flipSign)
{

  // constraints
  using view_type = Kokkos::View<ScalarType**, Properties...>;
  using exe_space = typename view_type::execution_space;
  using mem_space = typename view_type::memory_space;
  static_assert(std::is_same_v< typename view_type::array_layout, Kokkos::LayoutLeft>
    && std::is_same_v<std::remove_cv_t<ScalarType>, double>,
		"G must have layoutleft and must have double scalar type");

  // preconditions
  if (G.extent(0) != G.extent(1)){
    throw std::runtime_error("G must be symmetric for calling syev");
  }

  /*
   * do the eigen decomposition
   */
  auto exespace = exe_space();
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, mem_space> eigenvalues_d("EIG", G.extent(0));
  if constexpr( better_off_calling_host_syev_v<exe_space> ){
    compute_syev_use_host_lapack(G, eigenvalues_d);
  }
  else{
    compute_syev_on_device_views(exespace, G,  eigenvalues_d);
  }

  /*
    sorting
    -------
    Here, since jobz is V, if info == 0 it means LAPACK computes things in ascending order
    see here: https://netlib.org/lapack/explore-html/d2/d8a/group__double_s_yeigen_ga442c43fca5493590f8f26cf42fed4044.html
    This means that: eigvals(0) < eigvals(1) < ... < eigvals(N-1)
    and eigenvectors are ordered accordingly.
    Sort eigvvals and eigvec in !!!descending!! order to have: eigvals(0) > eigvals(1) > ...
  */
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
#endif  // IMPL_TUCKER_COMPUTEEIGVALSEIGVECS_HPP_
