#include "Tucker_ISVD.hpp"

#include <stdexcept>

#include "Tucker_Util.hpp"

namespace Tucker {

template <class scalar_t>
ISVD<scalar_t>::ISVD() {
  U_ = nullptr;
  s_ = nullptr;
  Vt_ = nullptr;
  squared_frobenius_norm_data_ = static_cast<scalar_t>(0);
  squared_frobenius_norm_error_ = static_cast<scalar_t>(0);
}

template <class scalar_t>
ISVD<scalar_t>::~ISVD() {
  MemoryManager::safe_delete(U_);
  MemoryManager::safe_delete(s_);
  MemoryManager::safe_delete(Vt_);
}

template <class scalar_t>
int ISVD<scalar_t>::nrows() const {
  if (!is_allocated_) {
    throw std::runtime_error("ISVD object is not initialized");
  }

  return U_->nrows();
}

template <class scalar_t>
int ISVD<scalar_t>::ncols() const {
  if (!is_allocated_) {
    throw std::runtime_error("ISVD object is not initialized");
  }

  return Vt_->ncols();
}

template <class scalar_t>
int ISVD<scalar_t>::rank() const {
  if (!is_allocated_) {
    throw std::runtime_error("ISVD object is not initialized");
  }

  return s_->nrows();
}

template <class scalar_t>
void ISVD<scalar_t>::initializeFactors(const Matrix<scalar_t> *U,
                                       const scalar_t *s,
                                       const Tensor<scalar_t> *X) {}

template <class scalar_t>
void ISVD<scalar_t>::initializeFactors(const Matrix<scalar_t> *U,
                                       const scalar_t *s,
                                       const Matrix<scalar_t> *Vt,
                                       scalar_t squared_frobenius_norm_data,
                                       scalar_t squared_frobenius_norm_error) {}

template <class scalar_t>
void updateRightSingularVectors(int k, const Matrix<scalar_t> *U_new,
                                const Matrix<scalar_t> *U_old) {}

template <class scalar_t>
void ISVD<scalar_t>::updateFactorsWithNewSlice(const Tensor<scalar_t> *Y,
                                               scalar_t tolerance) {}

template class ISVD<float>;
template class ISVD<double>;

}  // namespace Tucker
