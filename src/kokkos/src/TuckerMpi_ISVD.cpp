/**
 * @file
 * @brief Incremental SVD implementation
 * @author Saibal De
 */

#include "TuckerMpi_ISVD.hpp"

#include <stdexcept>
#include <cmath>

#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_scal.hpp>

#include "TuckerMpi.hpp"
#include "impl/Tucker_SolverUtils.hpp"

#if defined KOKKOS_ENABLE_HIP
#include <rocsolver/rocsolver.h>
#endif

#if defined KOKKOS_ENABLE_CUDA
#include <cusolverDn.h>
#endif

// Helper Functions ============================================================

namespace {

template <class matrix_t, class vector_t>
std::enable_if_t<
  std::is_same_v<typename matrix_t::memory_space, Kokkos::HostSpace> &&
  std::is_same_v<typename vector_t::memory_space, Kokkos::HostSpace> >
svd(const matrix_t& A, matrix_t& U, vector_t& s, matrix_t& V)
{
  using scalar_t = typename matrix_t::non_const_value_type;

  // LAPACK SVD related variables
  const char jobu = 'S';
  const char jobvt = 'S';
  const int m = A.extent(0);
  const int n = A.extent(1);
  const int k = std::min(m, n);
  int lwork;
  int info;

  // create a copy of input data
  matrix_t A_copy("A_copy", m, n);
  Kokkos::deep_copy(A_copy, A);

  // allocate memory for U, s, V
  U = matrix_t("U", m, k);
  s = vector_t("s", k);
  V = matrix_t("V", k, n);

  // workspace size query
  {
    scalar_t work_query;
    const int lwork_query = -1;
    Tucker::gesvd(&jobu, &jobvt, &m, &n, A_copy.data(), &m, s.data(), U.data(),
                  &m, V.data(), &k, &work_query, &lwork_query, &info);
    if (info != 0) {
      throw std::runtime_error("gesvd work query did not exit successfully");
    }
    lwork = work_query;
  }

  // actual factorization
  {
    vector_t work("work", lwork);
    Tucker::gesvd(&jobu, &jobvt, &m, &n, A_copy.data(), &m, s.data(), U.data(),
                  &m, V.data(), &k, work.data(), &lwork, &info);
    if (info != 0) {
      throw std::runtime_error("gesvd computation did not exit successfully");
    }
  }
}

#if defined(KOKKOS_ENABLE_CUDA)

template <class matrix_t, class vector_t>
std::enable_if_t<
  std::is_same_v<typename matrix_t::memory_space, Kokkos::CudaSpace> &&
  std::is_same_v<typename vector_t::memory_space, Kokkos::CudaSpace> >
svd(const matrix_t& A, matrix_t& U, vector_t& s, matrix_t& V)
{
  using scalar_t = typename matrix_t::non_const_value_type;
  using mem_space_t = typename matrix_t::memory_space;

  // LAPACK SVD related variables
  const char jobu = 'S';
  const char jobvt = 'S';
  const std::int64_t m = A.extent(0);
  const std::int64_t n = A.extent(1);
  const std::int64_t k = std::min(m, n);

  // create a copy of input data
  matrix_t A_copy("A_copy", m, n);
  Kokkos::deep_copy(A_copy, A);

  // allocate memory for U, s, V
  U = matrix_t("U", m, k);
  s = vector_t("s", k);
  V = matrix_t("V", k, n);

  // leading dimensions (handles layout left and right)
  const std::int64_t lda = std::max(m, std::int64_t(A_copy.stride(1)));
  const std::int64_t ldu = std::max(m, std::int64_t(U.stride(1)));
  const std::int64_t ldv = std::max(k, std::int64_t(V.stride(1)));

  // cuda data type
  static_assert(std::is_floating_point_v<scalar_t>);
  constexpr cudaDataType cuda_data_type =
    std::is_same_v<scalar_t, double> ? CUDA_R_64F : CUDA_R_32F;

  // initialize solver library
  cusolverDnHandle_t cuDnHandle = Tucker::impl::CusolverHandle::get();

  // workspace size query
  std::size_t d_lwork = 0; /* size of workspace */
  std::size_t h_lwork = 0; /* size of workspace */
  CUSOLVER_CHECK( cusolverDnXgesvd_bufferSize(
                    cuDnHandle, nullptr, jobu, jobvt, m, n,
                    cuda_data_type, A_copy.data(), lda,
                    cuda_data_type, s.data(),
                    cuda_data_type, U.data(), ldu,
                    cuda_data_type, V.data(), ldv,
                    cuda_data_type, &d_lwork, &h_lwork) );

  // actual factorization
  Kokkos::View<char*,mem_space_t> d_work("d_work", d_lwork);
  Kokkos::View<char*,Kokkos::HostSpace> h_work("h_work", h_lwork);
  Kokkos::View<int,mem_space_t> d_info("d_info");
  CUSOLVER_CHECK( cusolverDnXgesvd(
                    cuDnHandle, nullptr, jobu, jobvt, m, n,
                    cuda_data_type, A_copy.data(), lda,
                    cuda_data_type, s.data(),
                    cuda_data_type, U.data(), ldu,
                    cuda_data_type, V.data(), ldv,
                    cuda_data_type,
                    reinterpret_cast<void*>(d_work.data()), d_lwork,
                    reinterpret_cast<void*>(h_work.data()), h_lwork,
                    d_info.data()) );
  auto h_info =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_info);
  if (h_info() != 0)
    throw std::runtime_error("gesvd computation did not exit successfully");
}

#endif

#if defined(KOKKOS_ENABLE_HIP)

template <class matrix_t, class vector_t>
std::enable_if_t<
  std::is_same_v<typename matrix_t::memory_space, Kokkos::HIPSpace> &&
  std::is_same_v<typename vector_t::memory_space, Kokkos::HIPSpace> >
svd(const matrix_t& A, matrix_t& U, vector_t& s, matrix_t& V)
{
  using scalar_t = typename matrix_t::non_const_value_type;
  using mem_space_t = typename matrix_t::memory_space;

  // LAPACK SVD related variables
  const std::int64_t m = A.extent(0);
  const std::int64_t n = A.extent(1);
  const std::int64_t k = std::min(m, n);

  // create a copy of input data
  matrix_t A_copy("A_copy", m, n);
  Kokkos::deep_copy(A_copy, A);

  // allocate memory for U, s, V
  U = matrix_t("U", m, k);
  s = vector_t("s", k);
  V = matrix_t("V", k, n);

  // leading dimensions (handles layout left and right)
  const std::int64_t lda = std::max(m, std::int64_t(A_copy.stride(1)));
  const std::int64_t ldu = std::max(m, std::int64_t(U.stride(1)));
  const std::int64_t ldv = std::max(k, std::int64_t(V.stride(1)));

  // initialize solver library
  rocblas_handle handle = Tucker::impl::RocblasHandle::get();

  // actual factorization
  Kokkos::View<scalar_t*,mem_space_t> d_work("d_work", k-1);
  Kokkos::View<int,mem_space_t> d_info("d_info");
  if constexpr(std::is_same_v<scalar_t, double>)
    ROCBLAS_CHECK( rocsolver_dgesvd(
                     handle, rocblas_svect_singular, rocblas_svect_singular,
                     m, n, A_copy.data(), lda, s.data(), U.data(), ldu,
                     V.data(), ldv, d_work.data(), rocblas_outofplace,
                     d_info.data()) );
  if constexpr(std::is_same_v<scalar_t, float>)
    ROCBLAS_CHECK( rocsolver_sgesvd(
                     handle, rocblas_svect_singular, rocblas_svect_singular,
                     m, n, A_copy.data(), lda, s.data(), U.data(), ldu,
                     V.data(), ldv, d_work.data(), rocblas_outofplace,
                     d_info.data()) );
  auto h_info =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_info);
  if (h_info() != 0)
    throw std::runtime_error("gesvd computation did not exit successfully");
}

#endif

template <class scalar_t, class matrix_t, class vector_t>
void
truncatedSvd(const matrix_t& A, scalar_t absolute_tolerance,
             scalar_t relative_tolerance, int r_min,
             matrix_t& U, vector_t& s, matrix_t& V,
             scalar_t &squared_frobenius_norm_data,
             scalar_t &squared_frobenius_norm_error)
{
  using exec_space = typename matrix_t::execution_space;

  // matrix sizes
  const int m = A.extent(0);
  const int n = A.extent(1);
  const int k = std::min(m, n);

  // thin SVD
  matrix_t U_thin;
  vector_t s_thin;
  matrix_t V_thin;
  svd(A, U_thin, s_thin, V_thin);

  // determine truncation rank
  squared_frobenius_norm_data = 0.0;
  Kokkos::parallel_reduce("squared_frobenius_norm_data",
                          Kokkos::RangePolicy<exec_space>(0,k),
                          KOKKOS_LAMBDA(const int i, scalar_t& t)
  {
    t += s_thin[i] * s_thin[i];
  }, squared_frobenius_norm_data);

  const scalar_t max_squared_frobenius_norm_error =
      absolute_tolerance * absolute_tolerance +
      relative_tolerance * relative_tolerance * squared_frobenius_norm_data;

  auto s_thin_host = Kokkos::create_mirror_view(s_thin);
  Kokkos::deep_copy(s_thin_host, s_thin);
  int r = k;
  squared_frobenius_norm_error = 0.0;
  while (r > r_min) {
    const scalar_t new_squared_frobenius_norm_error =
        squared_frobenius_norm_error + s_thin_host[r - 1] * s_thin_host[r - 1];

    if (new_squared_frobenius_norm_error > max_squared_frobenius_norm_error) {
      break;
    }

    --r;
    squared_frobenius_norm_error = new_squared_frobenius_norm_error;
  }

  // allocate memory for SVD factors and copy
  U = matrix_t("U", m, r);
  Kokkos::deep_copy(U, Kokkos::subview(U_thin, Kokkos::ALL, std::make_pair(0,r)));

  s = vector_t("S", r);
  Kokkos::deep_copy(s, Kokkos::subview(s_thin, std::make_pair(0,r)));

  V = matrix_t("V", r, n);
  Kokkos::deep_copy(V, Kokkos::subview(V_thin, std::make_pair(0,r), Kokkos::ALL));
}

// End Helper Functions ========================================================
}

namespace TuckerMpi {

template <class scalar_t, class mem_space_t>
scalar_t
ISVD<scalar_t,mem_space_t>::
getLeftSingularVectorsError() const
{
  checkIsAllocated();
  const int m = nrows();
  const int r = rank();
  matrix_t E("E", r, r);
  KokkosBlas::gemm("T", "N", scalar_t(1.0), U_, U_, scalar_t(0.), E);
  scalar_t error = 0.0;
  Kokkos::parallel_reduce("getLeftSingularVectorsError",
                          Kokkos::RangePolicy<exec_space>(0,r*r),
                          KOKKOS_LAMBDA(const int k, scalar_t& e)
  {
    const int j = k / r;
    const int i = k - j*r;
    if (i == j)
      e += (E(i,j)-scalar_t(1.0))*(E(i,j)-scalar_t(1.0));
    else
      e += E(i,j)*E(i,j);
  }, error);
  return std::sqrt(error);
}

template <class scalar_t, class mem_space_t>
scalar_t
ISVD<scalar_t,mem_space_t>::
getRightSingularVectorsError() const
{
  checkIsAllocated();
  const int ndim = V_.rank();
  matrix_t E = compute_gram(V_, ndim-1);
  const int r = E.extent(0);
  scalar_t error = 0.0;
  Kokkos::parallel_reduce("getRightSingularVectorsError",
                          Kokkos::RangePolicy<exec_space>(0,r*r),
                          KOKKOS_LAMBDA(const int k, scalar_t& e)
  {
    const int j = k / r;
    const int i = k - j*r;
    if (i == j)
      e += (E(i,j)-scalar_t(1.0))*(E(i,j)-scalar_t(1.0));
    else
      e += E(i,j)*E(i,j);
  }, error);
  return std::sqrt(error);
}

template <class scalar_t, class mem_space_t>
void
ISVD<scalar_t,mem_space_t>::
initializeFactors(const matrix_t& U,
                  const vector_t& s,
                  const tensor_t& X)
{
  // validate inputs -----------------------------------------------------------

  const int nrow = U.extent(0);
  const int rank = U.extent(1);
  const int ndim = X.rank();

  if (X.localTensor().extent(ndim - 1) != nrow) {
    throw std::invalid_argument(
        "number of rows in left singular vector matrix must match last mode "
        "size of tensor");
  }

  const int ncol = X.localSize() / nrow;

  if (nrow < 1 || rank < 1 || ncol < 1) {
    throw std::invalid_argument(
        "number of rows and columns of left/right singular vectors matrix must "
        "be positive");
  }

  // TODO: check U is orthogonal

  int nfail = 0;
  Kokkos::parallel_reduce("singular_values_positive",
                          Kokkos::RangePolicy<exec_space>(0,rank),
                          KOKKOS_LAMBDA(const int r, int& f)
  {
    if (s[r] <= static_cast<scalar_t>(0))
      ++f;
  }, nfail);
  if (nfail > 0)
    throw std::invalid_argument("singular values cannot be negative");

  nfail = 0;
  Kokkos::parallel_reduce("singular_values_nonincreasing",
                          Kokkos::RangePolicy<exec_space>(0,rank),
                          KOKKOS_LAMBDA(const int r, int& f)
  {
    if (r > 0 && s[r - 1] < s[r])
      ++f;
  }, nfail);
  if (nfail > 0)
    throw std::invalid_argument(
      "singular values must be in non-increasing order");

  // copy U and s --------------------------------------------------------------

  U_ = matrix_t("U_", nrow, rank);
  Kokkos::deep_copy(U_, U);

  s_ = vector_t("s_", rank);
  Kokkos::deep_copy(s_, s);

  // construct V and compute Y s.t. Y_{(d)} = X - U * s * V_{(d)} --------------

  // V = X x_d U.T
  V_ = ttm(X, ndim - 1, U_, /* trans = */ true);

  // Y = V x_d U
  auto Y = ttm(V_, ndim - 1, U_, /* trans = */ false);

  // V = V x_d inv(diag(s))
  auto Vd = V_.localTensor().data();
  auto s_h = Kokkos::create_mirror_view(s_);
  Kokkos::deep_copy(s_h, s_);
  for (int j = 0; j < rank; ++j) {
    auto sub = Kokkos::subview(Vd, std::make_pair(j*ncol,j*ncol+ncol));
    KokkosBlas::scal(sub, scalar_t(1.0)/s_h[j], sub);
  }

  // Y = -1 * X + Y
  assert(Y.getDistribution() == X.getDistribution());
  auto Yd = Y.localTensor().data();
  auto Xd = X.localTensor().data();
  Kokkos::parallel_for("Y = -1 * X + Y",
                       Kokkos::RangePolicy<exec_space>(0,X.localTensor().size()),
                       KOKKOS_LAMBDA(const int i)
  {
    Yd[i] -= Xd[i];
  });

  // compute norms -------------------------------------------------------------

  squared_frobenius_norm_data_  = X.frobeniusNormSquared();
  squared_frobenius_norm_error_ = Y.frobeniusNormSquared();

  // flag factorization as initialized -----------------------------------------

  is_allocated_ = true;
}

template <class scalar_t, class mem_space_t>
void
ISVD<scalar_t,mem_space_t>::
initializeFactors(const ttensor_t& X, const eigval_t& eig)
{
  using exec_space = typename mem_space_t::execution_space;

  const tensor_t G = X.coreTensor();
  const auto Gl = G.localTensor();
  const int ndim = G.rank();

  const int d = ndim - 1;
  const auto U_d = X.factorMatrix(d);
  const int I_d = U_d.extent(0);
  const int R_d = U_d.extent(1);
  const auto eig_d = eig[d];

  is_allocated_ = true;

  U_ = matrix_t("U_", I_d, R_d);
  Kokkos::deep_copy(U_, U_d);

  s_ = vector_t("s_", R_d);
  auto s = s_; // avoid implicit capture of this pointer
  Kokkos::parallel_for("compute_singular_values",
                       Kokkos::RangePolicy<exec_space>(0,R_d),
                       KOKKOS_LAMBDA(const int i)
  {
    s[i] = std::sqrt(std::abs(eig_d[i])); // eigenvalues should be positive, but very small eigenvalues may be negative due to roundoff error
  });

  V_ = tensor_t(G.getDistribution());
  auto Vd = V_.localTensor().data();
  Kokkos::deep_copy(Vd, Gl.data());
  const int ncol = Gl.prod(0, d-1, 1);
  auto s_h = Kokkos::create_mirror_view(s_);
  Kokkos::deep_copy(s_h, s_);
  for (int j=0; j<R_d; ++j) {
    auto sub = Kokkos::subview(Vd, std::make_pair(j*ncol,j*ncol+ncol));
    KokkosBlas::scal(sub, scalar_t(1.0)/s_h[j], sub);
  }

  squared_frobenius_norm_data_ = 0.0;
  squared_frobenius_norm_error_ = 0.0;
  Kokkos::parallel_reduce("squared_frobenius_norm",
                          Kokkos::RangePolicy<exec_space>(0,I_d),
                          KOKKOS_LAMBDA(const int i, scalar_t& t1, scalar_t& t2)
  {
    t1 += std::abs(eig_d[i]);
    if (i >= R_d)
      t2 += std::abs(eig_d[i]);
  }, squared_frobenius_norm_data_, squared_frobenius_norm_error_);
}

template <class scalar_t, class mem_space_t>
void ISVD<scalar_t,mem_space_t>::
padRightSingularVectorsAlongMode(int k, int p)
{
  V_ = padTensorAlongMode(V_, k, p);
}

template <class scalar_t, class mem_space_t>
void
ISVD<scalar_t,mem_space_t>::
updateFactorsWithNewSlice(const tensor_t& Y,
                          scalar_t tolerance)
{
  // Distributions are not the same for Y and V_ because Y has one less mode
  // but the other parallel maps should be the same
  assert(Y.rank()+1 == V_.rank());
  for (int i=0; i<Y.rank(); ++i)
    assert(*Y.getDistribution().getMap(i,false) == *V_.getDistribution().getMap(i,false));
  addSingleRowNaive(Y.localTensor().data(), tolerance);
}

template <class scalar_t, class mem_space_t>
void
ISVD<scalar_t,mem_space_t>::
checkIsAllocated() const
{
  if (!is_allocated_) {
    throw std::runtime_error("ISVD object is not initialized");
  }
}

template <class scalar_t, class mem_space_t>
void
ISVD<scalar_t,mem_space_t>::
addSingleRowNaive(const vector_t& c, scalar_t tolerance)
{
  // matrix sizes
  const int m = nrows();
  const int n = ncols();
  const int r = rank();

  auto Vl = V_.localTensor();
  const MPI_Comm& comm = V_.getDistribution().getComm(false);

  // projection: j[r] = V[nxr].T * c[n]
  vector_t jl("jl", r);
  matrix_t Vd(Vl.data().data(), n, r); // reshape tensor as n x r matrix
  KokkosBlas::gemv("T", scalar_t(1.0), Vd, c, scalar_t(0.0), jl);
  vector_t j("j", r);
  MPI_Allreduce_(jl.data(), j.data(), r, MPI_SUM, comm);

  // orthogonal complement: q[n] = (c[n] - V[nxr] * j[r]).normalized()
  //                        l    = (c[n] - V[nxr] * j[r]).norm()
  vector_t q("q", n);
  Kokkos::deep_copy(q, c);
  KokkosBlas::gemv("N", scalar_t(-1.0), Vd, j, scalar_t(1.0), q);
  scalar_t ll = KokkosBlas::nrm2(q);
  ll = ll*ll;
  scalar_t l = 0;
  MPI_Allreduce_(&ll, &l, 1, MPI_SUM, comm);
  l = std::sqrt(l);
  KokkosBlas::scal(q, scalar_t(1.0)/l, q);

  // assemble U1[(m+1)x(r+1)] = [ U[mxr] 0[mx1] ]
  //                            [ 0[1xr] I[1x1] ]
  // we rely on the constructor for U1 initializing to 0
  matrix_t U1("U1", m + 1, r + 1);
  Kokkos::deep_copy(Kokkos::subview(U1, std::make_pair(0,m), std::make_pair(0,r)), U_);
  Kokkos::deep_copy(Kokkos::subview(U1, std::make_pair(m,m+1), std::make_pair(r,r+1)), scalar_t(1.0));

  // assemble S1[(r+1)x(r+1)] = [ S[rxr] 0[rx1] ]
  //                            [ J[1xr] L[1x1] ]
  // here S = diag(s) and we rely on the constructor for S1 initializing to 0
  matrix_t S1("S1", r + 1, r + 1);
  auto s = s_; // avoid implicit capture of this pointer
  Kokkos::parallel_for("Assemble S1", Kokkos::RangePolicy<exec_space>(0,r),
                       KOKKOS_LAMBDA(const int i)
  {
    S1(i,i) = s[i];
    S1(r,i) = j[i];
  });
  Kokkos::deep_copy(Kokkos::subview(S1, std::make_pair(r,r+1), std::make_pair(r,r+1)), l);

  // assemble V1[nx(r+1)] = [ V[nxr] q[nx1] ] as a tensor
  // note:  we don't need to worry about updating the processor grid because
  // there is no parallelism over the streaming mode
  const int ndims = V_.rank();
  Distribution V1_dist = V_.getDistribution().replaceModeWithGlobalSize(
    ndims-1, V_.globalExtent(ndims-1)+1);
  tensor_t V1(V1_dist);
  auto V1d = V1.localTensor().data();
  Kokkos::deep_copy(Kokkos::subview(V1d, std::make_pair(0,n*r)), Vl.data());
  Kokkos::deep_copy(Kokkos::subview(V1d, std::make_pair(n*r,n*(r+1))), q);

  // SVD: S1 = U2 * diag(s2) * V2
  scalar_t c_norm_l = KokkosBlas::nrm2(c);
  c_norm_l = c_norm_l * c_norm_l;
  scalar_t c_norm = 0;
  MPI_Allreduce_(&c_norm_l, &c_norm, 1, MPI_SUM, comm);
  c_norm = std::sqrt(c_norm);

  matrix_t U2;
  vector_t s2;
  matrix_t V2;
  scalar_t new_squared_frobenius_norm_data;
  scalar_t new_squared_frobenius_norm_error;
  truncatedSvd(S1, tolerance * c_norm, scalar_t(0.0), r, U2, s2, V2,
               new_squared_frobenius_norm_data,
               new_squared_frobenius_norm_error);

  // U[(m+1)xr_new] = U1[(m+1)x(r+1)] * U2[(r+1)xr_new]
  const int r_new = s2.extent(0);
  U_ = matrix_t("U_", m + 1, r_new);
  KokkosBlas::gemm("N", "N", scalar_t(1.0), U1, U2, scalar_t(0.0), U_);

  // s[r_new] = s2[r_new]
  s_ = s2;

  // V = V1 x_d V2.T
  const int ndim = V_.rank();
  V_ = ttm(V1, ndim-1, V2, false);

  // update norm estimates
  squared_frobenius_norm_data_ += c_norm * c_norm;
  squared_frobenius_norm_error_ += new_squared_frobenius_norm_error;
}

template <class scalar_t, class mem_space_t>
Tensor<scalar_t,mem_space_t>
ISVD<scalar_t,mem_space_t>::
padTensorAlongMode(const Tensor<scalar_t,mem_space_t>& X, int n, int p)
{
  const std::string method_signature = "padTensorAlongMode(const Tensor<scalar_t,mem_space_t>& X, int n, int p)";

  const int d = X.rank();

  if (n < 0 || n >= d) {
    std::ostringstream str;
    str << method_signature << ": mode index n is out of range for dimensionality of tensor X";
    throw std::invalid_argument(str.str());
  }

  if (p < 0) {
    std::ostringstream str;
    str << method_signature << ": number of additional zero slices p must be non-negative";
    throw std::invalid_argument(str.str());
  }

  auto Xl = X.localTensor();
  const int nrow1 = Xl.prod(0, n);
  const int nrow2 = Xl.prod(0, n - 1, 1) * p;
  const int nrow = nrow1 + nrow2;
  const int ncol = Xl.prod(n + 1, d - 1);

  tensor_t X_new(X.getDistribution().growAlongMode(n,p));
  auto Xl_new = X_new.localTensor();

  for (int j = 0; j < ncol; ++j) {
    Kokkos::deep_copy(
      Kokkos::subview(Xl_new.data(), std::make_pair(j*nrow ,j*nrow +nrow1)),
      Kokkos::subview(Xl.data(),     std::make_pair(j*nrow1,j*nrow1+nrow1)));
    Kokkos::deep_copy(
      Kokkos::subview(Xl_new.data(), std::make_pair(j*nrow+nrow1,j*nrow+nrow1+nrow2)),
      scalar_t(0.0));
  }

  return X_new;
}

template <class scalar_t, class mem_space_t>
Tensor<scalar_t,mem_space_t>
ISVD<scalar_t,mem_space_t>::
concatenateTensorsAlongMode(const Tensor<scalar_t,mem_space_t>& X,
                            const Tensor<scalar_t,mem_space_t>& Y, int n)
{
  const std::string method_signature = "concatenateTensorsAlongMode(const Tensor<scalar_t,mem_space_t>& X, const Tensor<scalar_t,mem_space_t>& Y, int n)";

  const int d = X.rank();
  if (Y.rank() != d) {
    std::ostringstream str;
    str << method_signature << ": tensors X and Y must have the same dimensionality";
    throw std::invalid_argument(str.str());
  }

  if (n < 0 || n >= d) {
    std::ostringstream str;
    str << method_signature << ": mode index n is out of range for dimensionality of tensors X and Y";
    throw std::invalid_argument(str.str());
  }

  for (int k = 0; k < d; ++k) {
    if (k != n) {
      if (X.globalExtent(k) != Y.globalExtent(k)) {
        std::ostringstream str;
        str << method_signature << ": tensors X and Y must have the same global mode sizes except along mode n";
        throw std::invalid_argument(str.str());
      }
      if (X.localExtent(k) != Y.localExtent(k)) {
        std::ostringstream str;
        str << method_signature << ": tensors X and Y must have the same local mode sizes except along mode n";
        throw std::invalid_argument(str.str());
      }
    }
  }

  tensor_t Z(X.getDistribution().growAlongMode(n,Y.localExtent(n)));

  auto Xl = X.localTensor();
  auto Yl = Y.localTensor();
  auto Zl = Z.localTensor();
  const int nrowx = Xl.prod(0, n);
  const int nrowy = Yl.prod(0, n);
  const int nrowz = nrowx + nrowy;
  const int ncolz = Xl.prod(n + 1, d - 1);

  for (int j = 0; j < ncolz; ++j) {
    Kokkos::deep_copy(
      Kokkos::subview(Zl.data(), std::make_pair(j*nrowz,j*nrowz+nrowx)),
      Kokkos::subview(Xl.data(), std::make_pair(j*nrowx,j*nrowx+nrowx)));
    Kokkos::deep_copy(
      Kokkos::subview(Zl.data(), std::make_pair(j*nrowz+nrowx,j*nrowz+nrowx+nrowy)),
      Kokkos::subview(Yl.data(), std::make_pair(j*nrowy,j*nrowy+nrowy)));
  }

  return Z;
}

//template class ISVD<float>;
template class ISVD<double>;

}  // namespace Tucker
