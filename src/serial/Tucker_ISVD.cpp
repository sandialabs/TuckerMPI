#include "Tucker_ISVD.hpp"

#include <stdexcept>

#include "Tucker_BlasWrapper.hpp"
#include "Tucker_Util.hpp"

namespace Tucker {

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
