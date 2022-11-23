/**
 * @file
 * @brief Incremental SVD implementation
 * @author Saibal De
 */

#include "Tucker_ISVD.hpp"

#include <stdexcept>

#include "Tucker.hpp"
#include "Tucker_BlasWrapper.hpp"
#include "Tucker_Util.hpp"

static const int INT_ZERO = 0;
static const int INT_ONE = 1;

namespace Tucker {

// Helper Functions ============================================================

template <class scalar_t>
static void svd(const Matrix<scalar_t> *A, Matrix<scalar_t> **U,
                Vector<scalar_t> **s, Matrix<scalar_t> **V) {
  // LAPACK SVD related variables
  const char jobu = 'S';
  const char jobvt = 'S';
  const int m = A->nrows();
  const int n = A->ncols();
  const int k = std::min(m, n);
  int lwork;
  int info;

  // create a copy of input data
  Matrix<scalar_t> *A_copy = MemoryManager::safe_new<Matrix<scalar_t>>(m, n);
  {
    const int nelm = m * n;
    copy(&nelm, A->data(), &INT_ONE, A_copy->data(), &INT_ONE);
  }

  // allocate memory for U, s, V
  *U = MemoryManager::safe_new<Matrix<scalar_t>>(m, k);
  *s = MemoryManager::safe_new<Vector<scalar_t>>(k);
  *V = MemoryManager::safe_new<Matrix<scalar_t>>(k, n);

  // workspace size query
  {
    scalar_t work_query;
    const int lwork_query = -1;
    gesvd(&jobu, &jobvt, &m, &n, A_copy->data(), &m, (*s)->data(), (*U)->data(),
          &m, (*V)->data(), &k, &work_query, &lwork_query, &info);
    if (info != 0) {
      throw std::runtime_error("gesvd work query did not exit successfully");
    }
    lwork = work_query;
  }

  // actual factorization
  {
    Vector<scalar_t> *work = MemoryManager::safe_new<Vector<scalar_t>>(lwork);
    gesvd(&jobu, &jobvt, &m, &n, A_copy->data(), &m, (*s)->data(), (*U)->data(),
          &m, (*V)->data(), &k, work->data(), &lwork, &info);
    if (info != 0) {
      throw std::runtime_error("gesvd computation did not exit successfully");
    }
    MemoryManager::safe_delete(work);
  }

  // clean up temporary memory allocations
  MemoryManager::safe_delete(A_copy);
}

template <class scalar_t>
static void truncatedSvd(const Matrix<scalar_t> *A, scalar_t absolute_tolerance,
                         scalar_t relative_tolerance, Matrix<scalar_t> **U,
                         Vector<scalar_t> **s, Matrix<scalar_t> **V,
                         scalar_t &squared_frobenius_error) {
  // matrix sizes
  const int m = A->nrows();
  const int n = A->ncols();
  const int k = std::min(m, n);

  // thin SVD
  Matrix<scalar_t> *U_thin = nullptr;
  Vector<scalar_t> *s_thin = nullptr;
  Matrix<scalar_t> *V_thin = nullptr;
  svd(A, &U_thin, &s_thin, &V_thin);

  // determine truncation rank
  scalar_t squared_frobenius_norm = static_cast<scalar_t>(0);
  for (int i = 0; i < k; ++i) {
    squared_frobenius_norm += (*s_thin)[i] * (*s_thin)[i];
  }

  const scalar_t squared_frobenius_max_error =
      absolute_tolerance * absolute_tolerance +
      relative_tolerance * relative_tolerance * squared_frobenius_norm;

  int r = k;
  squared_frobenius_error = static_cast<scalar_t>(0);
  while (r > 1) {
    const scalar_t squared_frobenius_new_error =
        squared_frobenius_error + (*s_thin)[r - 1] * (*s_thin)[r - 1];

    if (squared_frobenius_new_error > squared_frobenius_max_error) {
      break;
    }

    squared_frobenius_error = squared_frobenius_new_error;
    --r;
  }

  // allocate memory for SVD factors and copy
  *U = MemoryManager::safe_new<Matrix<scalar_t>>(m, r);
  {
    const int nelm = m * r;
    copy(&nelm, U_thin->data(), &INT_ONE, (*U)->data(), &INT_ONE);
  }

  *s = MemoryManager::safe_new<Vector<scalar_t>>(r);
  copy(&r, s_thin->data(), &INT_ONE, (*s)->data(), &INT_ONE);

  *V = MemoryManager::safe_new<Matrix<scalar_t>>(r, n);
  for (int j = 0; j < n; ++j) {
    copy(&r, V_thin->data() + j * k, &INT_ONE, (*V)->data() + j * r, &INT_ONE);
  }

  // clean up temporary memory allocations
  MemoryManager::safe_delete(U_thin);
  MemoryManager::safe_delete(s_thin);
  MemoryManager::safe_delete(V_thin);
}

// End Helper Functions ========================================================

template <class scalar_t>
ISVD<scalar_t>::ISVD() {
  is_allocated_ = false;
  U_ = nullptr;
  s_ = nullptr;
  V_ = nullptr;
  squared_frobenius_norm_data_ = static_cast<scalar_t>(0);
  squared_frobenius_norm_error_ = static_cast<scalar_t>(0);
}

template <class scalar_t>
ISVD<scalar_t>::~ISVD() {
  if (is_allocated_) {
    MemoryManager::safe_delete(U_);
    MemoryManager::safe_delete(s_);
    MemoryManager::safe_delete(V_);
  }
}

template <class scalar_t>
scalar_t ISVD<scalar_t>::getLeftSingularVectorsError() const {
  checkIsAllocated();
  const int m = nrows();
  const int r = rank();
  Matrix<scalar_t> *E = MemoryManager::safe_new<Matrix<scalar_t>>(r, r);
  {
    const char &transa = 'T';
    const char &transb = 'N';
    const scalar_t &alpha = static_cast<scalar_t>(1);
    const scalar_t &beta = static_cast<scalar_t>(0);
    gemm(&transa, &transb, &r, &r, &m, &alpha, U_->data(), &m, U_->data(), &m,
         &beta, E->data(), &r);
  }
  for (int i = 0; i < r; ++i) {
    *(E->data() + i + i * r) -= static_cast<scalar_t>(1);
  }
  scalar_t error = std::sqrt(E->norm2());
  MemoryManager::safe_delete(E);
  return error;
}

template <class scalar_t>
scalar_t ISVD<scalar_t>::getRightSingularVectorsError() const {
  checkIsAllocated();
  const int n = ncols();
  const int r = rank();
  Matrix<scalar_t> *E = MemoryManager::safe_new<Matrix<scalar_t>>(r, r);
  {
    const char &transa = 'T';
    const char &transb = 'N';
    const scalar_t &alpha = static_cast<scalar_t>(1);
    const scalar_t &beta = static_cast<scalar_t>(0);
    gemm(&transa, &transb, &r, &r, &n, &alpha, V_->data(), &n, V_->data(), &n,
         &beta, E->data(), &r);
  }
  for (int i = 0; i < r; ++i) {
    *(E->data() + i + i * r) -= static_cast<scalar_t>(1);
  }
  scalar_t error = std::sqrt(E->norm2());
  MemoryManager::safe_delete(E);
  return error;
}

template <class scalar_t>
void ISVD<scalar_t>::initializeFactors(const Matrix<scalar_t> *U,
                                       const scalar_t *s,
                                       const Tensor<scalar_t> *X) {
  if (is_allocated_) {
    MemoryManager::safe_delete(U_);
    MemoryManager::safe_delete(s_);
    MemoryManager::safe_delete(V_);
  }

  // validate inputs -----------------------------------------------------------

  if (U == nullptr || s == nullptr || X == nullptr) {
    throw std::invalid_argument(
        "pointer to left singular vector matrix, singular values or data "
        "tensor "
        "cannot be null");
  }

  const int nrow = U->nrows();
  const int rank = U->ncols();
  const int ndim = X->N();

  if (X->size(ndim - 1) != nrow) {
    throw std::invalid_argument(
        "number of rows in left singular vector matrix must match last mode "
        "size of tensor");
  }

  const int ncol = X->getNumElements() / nrow;

  if (nrow < 1 || rank < 1 || ncol < 1) {
    throw std::invalid_argument(
        "number of rows and columns of left/right singular vectors matrix must "
        "be positive");
  }

  // TODO: check U is orthogonal

  for (int r = 0; r < rank; ++r) {
    if (s[r] <= static_cast<scalar_t>(0)) {
      throw std::invalid_argument("singular values cannot be negative");
    }

    if (r > 0 && s[r - 1] < s[r]) {
      throw std::invalid_argument(
          "singular values must be in non-increasing order");
    }
  }

  // copy U and s --------------------------------------------------------------

  U_ = MemoryManager::safe_new<Matrix<scalar_t>>(nrow, rank);
  {
    const int nelm = nrow * rank;
    copy(&nelm, U->data(), &INT_ONE, U_->data(), &INT_ONE);
  }

  s_ = MemoryManager::safe_new<Vector<scalar_t>>(rank);
  copy(&rank, s, &INT_ONE, s_->data(), &INT_ONE);

  // construct V and compute Y s.t. Y_{(d)} = X - U * s * V_{(d)} --------------

  // V = X x_d U.T
  V_ = ttm(X, ndim - 1, U_, /* trans = */ true);

  // Y = V x_d U
  Tensor<scalar_t> *Y = ttm(V_, ndim - 1, U_, /* trans = */ false);

  // V = V x_d inv(diag(s))
  for (int j = 0; j < rank; ++j) {
    const scalar_t &alpha = 1 / (*s_)[j];
    scal(&ncol, &alpha, V_->data() + ncol * j, &INT_ONE);
  }

  // Y = -1 * X + Y
  {
    const int nelm = nrow * ncol;
    const scalar_t alpha = -static_cast<scalar_t>(1);
    axpy(&nelm, &alpha, X->data(), &INT_ONE, Y->data(), &INT_ONE);
  }

  // compute norms -------------------------------------------------------------

  squared_frobenius_norm_data_ = X->norm2();
  squared_frobenius_norm_error_ = Y->norm2();

  // deallocate tempoary memory ------------------------------------------------

  MemoryManager::safe_delete(Y);

  // flag factorization as initialized -----------------------------------------

  is_allocated_ = true;
}

template <class scalar_t>
void ISVD<scalar_t>::updateRightSingularVectors(int k,
                                                const Matrix<scalar_t> *U_new,
                                                const Matrix<scalar_t> *U_old) {
  const int n = U_new->nrows();
  const int r_new = U_new->ncols();
  const int r_old = U_old->ncols();

  if (U_old->nrows() != n) {
    throw std::invalid_argument(
        "old and new basis matrices must have the same number of rows");
  }

  Matrix<scalar_t> *M = MemoryManager::safe_new<Matrix<scalar_t>>(r_new, r_old);
  {
    const char &transa = 'T';
    const char &transb = 'N';
    const scalar_t &alpha = static_cast<scalar_t>(1);
    const scalar_t &beta = static_cast<scalar_t>(0);
    gemm(&transa, &transb, &r_new, &r_old, &n, &alpha, U_new->data(), &n,
         U_old->data(), &n, &beta, M->data(), &r_new);
  }

  Tensor<scalar_t> *V_new = ttm(V_, k, M, /* trans = */ false);
  std::swap(V_, V_new);

  MemoryManager::safe_delete(M);
  MemoryManager::safe_delete(V_new);
}

template <class scalar_t>
void ISVD<scalar_t>::updateFactorsWithNewSlice(const Tensor<scalar_t> *Y,
                                               scalar_t tolerance) {
  addSingleRowNaive(Y->data(), tolerance);
}

template <class scalar_t>
void ISVD<scalar_t>::checkIsAllocated() const {
  if (!is_allocated_) {
    throw std::runtime_error("ISVD object is not initialized");
  }
}

template <class scalar_t>
void ISVD<scalar_t>::addSingleRowNaive(const scalar_t *c, scalar_t tolerance) {
  const scalar_t SCALAR_ZERO = static_cast<scalar_t>(0);
  const scalar_t SCALAR_ONE = static_cast<scalar_t>(1);

  // matrix sizes
  const int m = nrows();
  const int n = ncols();
  const int r = rank();

  // projection: j[r] = V[nxr].T * c[n]
  Vector<scalar_t> *j = MemoryManager::safe_new<Vector<scalar_t>>(r);
  {
    const char &trans = 'T';
    const scalar_t &alpha = SCALAR_ONE;
    const scalar_t &beta = SCALAR_ZERO;
    gemv(&trans, &n, &r, &alpha, V_->data(), &n, c, &INT_ONE, &beta, j->data(),
         &INT_ONE);
  }

  // orthogonal complement: q[n] = (c[n] - V[nxr] * j[r]).normalized()
  //                        l    = (c[n] - V[nxr] * j[r]).norm()
  Vector<scalar_t> *q = MemoryManager::safe_new<Vector<scalar_t>>(n);
  {
    copy(&n, c, &INT_ONE, q->data(), &INT_ONE);

    const char &trans = 'N';
    const scalar_t &alpha = -SCALAR_ONE;
    const scalar_t &beta = SCALAR_ONE;
    gemv(&trans, &n, &r, &alpha, V_->data(), &n, j->data(), &INT_ONE, &beta,
         q->data(), &INT_ONE);
  }
  const scalar_t l = nrm2(&n, q->data(), &INT_ONE);
  {
    const scalar_t &alpha = SCALAR_ONE / l;
    scal(&n, &alpha, q->data(), &INT_ONE);
  }

  // assemble U1[(m+1)x(r+1)] = [ U[mxr] 0[mx1] ]
  //                            [ 0[1xr] I[1x1] ]
  Matrix<scalar_t> *U1 =
      MemoryManager::safe_new<Matrix<scalar_t>>(m + 1, r + 1);
  for (int j = 0; j < r; ++j) {
    copy(&m, U_->data() + j * m, &INT_ONE, U1->data() + j * (m + 1), &INT_ONE);
  }
  {
    const int &incr = m + 1;
    copy(&r, &SCALAR_ZERO, &INT_ZERO, U1->data() + m, &incr);
  }
  copy(&m, &SCALAR_ZERO, &INT_ZERO, U1->data() + (m + 1) * r, &INT_ONE);
  *(U1->data() + (m + 1) * (r + 1) - 1) = SCALAR_ONE;

  // assemble S1[(r+1)x(r+1)] = [ S[rxr] 0[rx1] ]
  //                            [ J[1xr] L[1x1] ]
  Matrix<scalar_t> *S1 =
      MemoryManager::safe_new<Matrix<scalar_t>>(r + 1, r + 1);
  for (int j = 0; j <= r; ++j) {
    copy(&r, &SCALAR_ZERO, &INT_ZERO, S1->data() + j * (r + 1), &INT_ONE);
  }
  {
    const int incr = r + 2;
    copy(&r, s_->data(), &INT_ONE, S1->data(), &incr);
  }
  {
    const int incr = r + 1;
    copy(&r, j->data(), &INT_ONE, S1->data() + r, &incr);
  }
  *(S1->data() + (r + 1) * (r + 1) - 1) = l;

  // assemble V1[nx(r+1)] = [ V[nxr] q[nx1] ]
  Matrix<scalar_t> *V1 = MemoryManager::safe_new<Matrix<scalar_t>>(n, r + 1);
  {
    const int nelm = n * r;
    copy(&nelm, V_->data(), &INT_ONE, V1->data(), &INT_ONE);
  }
  copy(&n, q->data(), &INT_ONE, V1->data() + n * r, &INT_ONE);

  // SVD: S1 = U2 * diag(s2) * V2
  const scalar_t c_norm = nrm2(&n, c, &INT_ONE);

  Matrix<scalar_t> *U2 = nullptr;
  Vector<scalar_t> *s2 = nullptr;
  Matrix<scalar_t> *V2 = nullptr;
  scalar_t new_squared_frobenius_norm_error;
  truncatedSvd(S1, tolerance * c_norm, SCALAR_ZERO, &U2, &s2, &V2,
               new_squared_frobenius_norm_error);

  // memory allocation for update
  const int r_new = s2->nrows();

  MemoryManager::safe_delete(U_);
  U_ = MemoryManager::safe_new<Matrix<scalar_t>>(m + 1, r_new);

  if (r_new != r) {
    const int ndim = V_->N();

    SizeArray size(ndim);
    for (int d = 0; d < ndim - 1; ++d) {
      size[d] = V_->size(d);
    }
    size[ndim - 1] = r_new;

    MemoryManager::safe_delete(V_);
    V_ = MemoryManager::safe_new<Tensor<scalar_t>>(size);
  }

  // U[(m+1)xr_new] = U1[(m+1)x(r+1)] * U2[(r+1)xr_new]
  {
    const char transa = 'N';
    const char transb = 'N';
    const int m_new = m + 1;
    const int &n_new = r_new;
    const int k_new = r + 1;
    const scalar_t &alpha = SCALAR_ONE;
    const scalar_t &beta = SCALAR_ZERO;
    gemm(&transa, &transb, &m_new, &n_new, &k_new, &alpha, U1->data(), &m_new,
         U2->data(), &k_new, &beta, U_->data(), &m_new);
  }

  // s[r_new] = s2[r_new]
  std::swap(s_, s2);

  // V[nxr_new] = V1[nx(r+1)] * V2[r_newx(r+1)].T
  {
    const char transa = 'N';
    const char transb = 'T';
    const int &m_new = n;
    const int &n_new = r_new;
    const int k_new = r + 1;
    const scalar_t &alpha = SCALAR_ONE;
    const scalar_t &beta = SCALAR_ZERO;
    gemm(&transa, &transb, &m_new, &n_new, &k_new, &alpha, V1->data(), &m_new,
         V2->data(), &n_new, &beta, V_->data(), &m_new);
  }

  // update norm estimates
  squared_frobenius_norm_data_ += c_norm * c_norm;
  squared_frobenius_norm_error_ += new_squared_frobenius_norm_error;

  // deallocate temporary variables
  MemoryManager::safe_delete(j);
  MemoryManager::safe_delete(q);

  MemoryManager::safe_delete(U1);
  MemoryManager::safe_delete(S1);
  MemoryManager::safe_delete(V1);

  MemoryManager::safe_delete(U2);
  MemoryManager::safe_delete(s2);
  MemoryManager::safe_delete(V2);
}

template class ISVD<float>;
template class ISVD<double>;

}  // namespace Tucker
