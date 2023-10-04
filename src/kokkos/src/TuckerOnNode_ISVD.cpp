/**
 * @file
 * @brief Incremental SVD implementation
 * @author Saibal De
 */

#include "TuckerOnNode_ISVD.hpp"

#include <stdexcept>
#include <cmath>

#include <KokkosBlas3_gemm.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_scal.hpp>

#include "TuckerOnNode.hpp"

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

  const std::int64_t lda = std::max(m, std::int64_t(A_copy.stride(1)));
  const std::int64_t ldu = std::max(m, std::int64_t(U.stride(1)));
  const std::int64_t ldv = std::max(k, std::int64_t(V.stride(1)));

  cusolverDnHandle_t cuDnHandle = nullptr;
  cusolverDnCreate(&cuDnHandle);

  static_assert(std::is_floating_point_v<scalar_t>);
  constexpr cudaDataType cuda_data_type =
    std::is_same_v<scalar_t, double> ? CUDA_R_64F : CUDA_R_32F;

  // workspace size query
  std::size_t d_lwork = 0; /* size of workspace */
  std::size_t h_lwork = 0; /* size of workspace */
  {
    cusolverStatus_t err =
      cusolverDnXgesvd_bufferSize(cuDnHandle, nullptr, jobu, jobvt, m, n,
                                  cuda_data_type, A_copy.data(), lda,
                                  cuda_data_type, s.data(),
                                  cuda_data_type, U.data(), ldu,
                                  cuda_data_type, V.data(), ldv,
                                  cuda_data_type, &d_lwork, &h_lwork);
    if (err != CUSOLVER_STATUS_SUCCESS)
      throw std::runtime_error("cusolverDnXgesvd_bufferSize() error");
  }

  // actual factorization
  Kokkos::View<char*,mem_space_t> d_work("d_work", d_lwork);
  Kokkos::View<char*,Kokkos::HostSpace> h_work("h_work", h_lwork);
  Kokkos::View<int,mem_space_t> d_info("d_info");
  {
    cusolverStatus_t err =
      cusolverDnXgesvd(cuDnHandle, nullptr, jobu, jobvt, m, n,
                       cuda_data_type, A_copy.data(), lda,
                       cuda_data_type, s.data(),
                       cuda_data_type, U.data(), ldu,
                       cuda_data_type, V.data(), ldv,
                       cuda_data_type,
                       reinterpret_cast<void*>(d_work.data()), d_lwork,
                       reinterpret_cast<void*>(h_work.data()), h_lwork,
                       d_info.data());
    if (err != CUSOLVER_STATUS_SUCCESS)
      throw std::runtime_error("cusolverDnXgesvd_bufferSize() error");
  }
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

namespace TuckerOnNode {

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
  const int n = ncols();
  const int r = rank();
  matrix_t E("E", r, r);
  matrix_t Vm(V_.data().data(), n, r); // Reshape V_ to a n x r matrix
  KokkosBlas::gemm("T", "N", scalar_t(1.0), Vm, Vm, scalar_t(0.), E);
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

  if (X.extent(ndim - 1) != nrow) {
    throw std::invalid_argument(
        "number of rows in left singular vector matrix must match last mode "
        "size of tensor");
  }

  const int ncol = X.size() / nrow;

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
  auto Vd = V_.data();
  auto s_h = Kokkos::create_mirror_view(s_);
  Kokkos::deep_copy(s_h, s_);
  for (int j = 0; j < rank; ++j) {
    auto sub = Kokkos::subview(V_.data(), std::make_pair(j*ncol,j*ncol+ncol));
    KokkosBlas::scal(sub, scalar_t(1.0)/s_h[j], sub);
  }

  // Y = -1 * X + Y
  auto Yd = Y.data();
  auto Xd = X.data();
  Kokkos::parallel_for("Y = -1 * X + Y",
                       Kokkos::RangePolicy<exec_space>(0,X.size()),
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
  const int ndim = G.rank();
  const auto size = G.dimensionsOnHost();

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
    s[i] = std::sqrt(eig_d[i]);
  });

  V_ = tensor_t(size);
  Kokkos::deep_copy(V_.data(), G.data());
  const int ncol = G.prod(0, d-1, 1);
  auto s_h = Kokkos::create_mirror_view(s_);
  Kokkos::deep_copy(s_h, s_);
  for (int j=0; j<R_d; ++j) {
    auto sub = Kokkos::subview(V_.data(), std::make_pair(j*ncol,j*ncol+ncol));
    KokkosBlas::scal(sub, scalar_t(1.0)/s_h[j], sub);
  }

  squared_frobenius_norm_data_ = 0.0;
  squared_frobenius_norm_error_ = 0.0;
  Kokkos::parallel_reduce("squared_frobenius_norm",
                          Kokkos::RangePolicy<exec_space>(0,I_d),
                          KOKKOS_LAMBDA(const int i, scalar_t& t1, scalar_t& t2)
  {
    t1 += eig_d[i];
    if (i >= R_d)
      t2 += eig_d[i];
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
  addSingleRowNaive(Y.data(), tolerance);
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

  // projection: j[r] = V[nxr].T * c[n]
  vector_t j("j", r);
  matrix_t Vd(V_.data().data(), n, r); // reshape tensor as n x r matrix
  KokkosBlas::gemv("T", scalar_t(1.0), Vd, c, scalar_t(0.0), j);

  // orthogonal complement: q[n] = (c[n] - V[nxr] * j[r]).normalized()
  //                        l    = (c[n] - V[nxr] * j[r]).norm()
  vector_t q("q", n);
  Kokkos::deep_copy(q, c);
  KokkosBlas::gemv("N", scalar_t(-1.0), Vd, j, scalar_t(1.0), q);
  const scalar_t l = KokkosBlas::nrm2(q);
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

  // assemble V1[nx(r+1)] = [ V[nxr] q[nx1] ]
  matrix_t V1("V1", n, r + 1);
  Kokkos::deep_copy(Kokkos::subview(V1, std::make_pair(0,n), std::make_pair(0,r)), Vd);
  Kokkos::parallel_for("Assemble V1", Kokkos::RangePolicy<exec_space>(0,n),
                       KOKKOS_LAMBDA(const int i)
  {
    V1(i,r) = q[i];
  });

  // SVD: S1 = U2 * diag(s2) * V2
  const scalar_t c_norm = KokkosBlas::nrm2(c);

  matrix_t U2;
  vector_t s2;
  matrix_t V2;
  scalar_t new_squared_frobenius_norm_data;
  scalar_t new_squared_frobenius_norm_error;
  truncatedSvd(S1, tolerance * c_norm, scalar_t(0.0), r, U2, s2, V2,
               new_squared_frobenius_norm_data,
               new_squared_frobenius_norm_error);

  // memory allocation for update
  const int r_new = s2.extent(0);

  U_ = matrix_t("U_", m + 1, r_new);

  if (r_new != r) {
    const int ndim = V_.rank();

    typename tensor_t::dims_host_view_type size("size", ndim);
    for (int d=0; d<ndim-1; ++d) {
      size[d] = V_.extent(d);
    }
    size[ndim - 1] = r_new;

    V_ = tensor_t(size);
  }

  // U[(m+1)xr_new] = U1[(m+1)x(r+1)] * U2[(r+1)xr_new]
  KokkosBlas::gemm("N", "N", scalar_t(1.0), U1, U2, scalar_t(0.0), U_);

  // s[r_new] = s2[r_new]
  s_ = s2;

  // V[nxr_new] = V1[nx(r+1)] * V2[r_newx(r+1)].T
  Vd = matrix_t(V_.data().data(), n, r_new); // reshape tensor as n x r_new matrix
  KokkosBlas::gemm("N", "T", scalar_t(1.0), V1, V2, scalar_t(0.0), Vd);

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

  if (p <= 0) {
    std::ostringstream str;
    str << method_signature << ": number of additional zero slices p must be positive";
    throw std::invalid_argument(str.str());
  }

  const int nrow1 = X.prod(0, n);
  const int nrow2 = X.prod(0, n - 1) * p;
  const int nrow = nrow1 + nrow2;
  const int ncol = X.prod(n + 1, d - 1);

  typename tensor_t::dims_host_view_type size("size", d);
  for (int k = 0; k < d; ++k)
    size[k] = (n == k) ? X.extent(k) + p : X.extent(k);
  tensor_t X_new(size);

  for (int j = 0; j < ncol; ++j) {
    Kokkos::deep_copy(
      Kokkos::subview(X_new.data(), std::make_pair(j*nrow ,j*nrow +nrow1)),
      Kokkos::subview(X.data(),     std::make_pair(j*nrow1,j*nrow1+nrow1)));
    Kokkos::deep_copy(
      Kokkos::subview(X_new.data(), std::make_pair(j*nrow+nrow1,j*nrow+nrow1+nrow2)),
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

  typename tensor_t::dims_host_view_type size("size", d);
  for (int k = 0; k < d; ++k) {
    if (k != n) {
      if (X.extent(k) != Y.extent(k)) {
        std::ostringstream str;
        str << method_signature << ": tensors X and Y must have the same modes sizes except along mode n";
        throw std::invalid_argument(str.str());
      }
      size[k] = X.extent(k);
    } else {
      size[k] = X.extent(k) + Y.extent(k);
    }
  }
  tensor_t Z(size);

  const int nrowx = X.prod(0, n);
  const int nrowy = Y.prod(0, n);
  const int nrowz = nrowx + nrowy;
  const int ncolz = X.prod(n + 1, d - 1);

  for (int j = 0; j < ncolz; ++j) {
    Kokkos::deep_copy(
      Kokkos::subview(Z.data(), std::make_pair(j*nrowz,j*nrowz+nrowx)),
      Kokkos::subview(X.data(), std::make_pair(j*nrowx,j*nrowx+nrowx)));
    Kokkos::deep_copy(
      Kokkos::subview(Z.data(), std::make_pair(j*nrowz+nrowx,j*nrowz+nrowx+nrowy)),
      Kokkos::subview(Y.data(), std::make_pair(j*nrowy,j*nrowy+nrowy)));
  }

  return Z;
}

//template class ISVD<float>;
template class ISVD<double>;

}  // namespace Tucker
