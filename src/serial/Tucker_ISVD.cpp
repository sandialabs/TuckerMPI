#include "Tucker_ISVD.hpp"

#include <stdexcept>

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
    const char &transa = 'N';
    const char &transb = 'T';
    const scalar_t &alpha = static_cast<scalar_t>(1);
    const scalar_t &beta = static_cast<scalar_t>(0);
    gemm(&transa, &transb, &r, &r, &n, &alpha, V_->data(), &r, V_->data(), &r,
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
                                       const Tensor<scalar_t> *X) {}


template <class scalar_t>
void updateRightSingularVectors(int k, const Matrix<scalar_t> *U_new,
                                const Matrix<scalar_t> *U_old) {}

template <class scalar_t>
void ISVD<scalar_t>::updateFactorsWithNewSlice(const Tensor<scalar_t> *Y,
                                               scalar_t tolerance) {}

template <class scalar_t>
void ISVD<scalar_t>::checkIsAllocated() const {
  if (!is_allocated_) {
    throw std::runtime_error("ISVD object is not initialized");
  }
}

template class ISVD<float>;
template class ISVD<double>;

}  // namespace Tucker
