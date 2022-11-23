/**
 * @file
 * @brief Contains incremental SVD implementation
 * @author Saibal De
 */

#ifndef TUCKER_ISVD_HPP_
#define TUCKER_ISVD_HPP_

#include <cmath>
#include <limits>

#include "Tucker_Matrix.hpp"
#include "Tucker_Tensor.hpp"
#include "Tucker_Vector.hpp"

namespace Tucker {

/**
 * @brief Incremental singular value decomposition
 */
template <class scalar_t>
class ISVD {
public:
  /**
   * @brief Default constructor
   */
  ISVD();

  /**
   * @brief Default destructor
   */
  ~ISVD();

  /**
   * @brief Number of rows in ISVD factorization
   *
   * @exception std::runtime_error If the ISVD factors are not initialized/null
   *                               pointers
   */
  int nrows() const {
    checkIsAllocated();
    return U_->nrows();
  }

  /**
   * @brief Number of columns in ISVD factorization
   *
   * @exception std::runtime_error If the ISVD factors are not initialized/null
   *                               pointers
   */
  int ncols() const {
    checkIsAllocated();
    return V_->getNumElements() / rank();
  }

  /**
   * @brief Rank of ISVD factorization
   *
   * @exception std::runtime_error If the ISVD factors are not
   * initialized/null pointers
   */
  int rank() const {
    checkIsAllocated();
    return s_->nrows();
  }

  /**
   * @brief Constant pointer to singular values
   */
  const Vector<scalar_t> *getSingularValues() const {
    checkIsAllocated();
    return s_;
  }

  /**
   * @brief Constant pointer to left singular vectors
   */
  const Matrix<scalar_t> *getLeftSingularVectors() const {
    checkIsAllocated();
    return U_;
  }

  /**
   * @brief Constant pointer to right singular vectors
   */
  const Tensor<scalar_t> *getRightSingularVectors() const {
    checkIsAllocated();
    return V_;
  }

  /**
   * @brief Orthogonality error for left singular vectors
   */
  scalar_t getLeftSingularVectorsError() const;

  /**
   * @brief Orthogonality error for right singular vectors
   */
  scalar_t getRightSingularVectorsError() const;

  /**
   * @brief Absolute error estimate of approximation w.r.t. Frobenius norm
   */
  scalar_t getAbsoluteErrorEstimate() const {
    checkIsAllocated();
    return std::sqrt(squared_frobenius_norm_error_);
  }

  /**
   * @brief Relative error estimate of approximation w.r.t. Frobenius norm
   */
  scalar_t getRelativeErrorEstimate() const {
    checkIsAllocated();
    return std::sqrt(squared_frobenius_norm_error_ /
                     squared_frobenius_norm_data_);
  }

  /**
   * @brief Initialize ISVD
   *
   * @param[in] U Pointer to left singular vectors; column major matrix with
   *              orthonormal columns
   * @param[in] s Pointer to singular values array; memory in the range [s, s +
   *              U->ncols()) will be accessed
   * @param[in] X Pointer to tensor whose last unfolding is being factorized
   */
  void initializeFactors(const Matrix<scalar_t> *U, const scalar_t *s,
                         const Tensor<scalar_t> *X);

  /**
   * @brief Update right singular vectors corresponding to Tuker core updates
   *
   * @param[in] k Tensor mode
   * @param[in] U_new New basis vectors for tensor mode
   * @param[in] U_old Old basis vectors for tensor mode
   */
  void updateRightSingularVectors(int k, const Matrix<scalar_t> *U_new,
                                  const Matrix<scalar_t> *U_old);

  /**
   * @brief Update factorization given new data
   *
   * @param[in] C Pointer to tensor with new data; the entire tensor will be
   *              flattened and treated as a single row
   * @param[in] tolerance Approximation tolerance
   */
  void updateFactorsWithNewSlice(const Tensor<scalar_t> *Y, scalar_t tolerance);

private:
  /**
   * @brief Check if ISVD object is initialized
   */
  void checkIsAllocated() const;

  /**
   * @brief Update three-factor SVD given new rows
   */
  void addSingleRowNaive(const scalar_t *c, scalar_t tolerance);

private:
  bool is_allocated_;   /**< Flag specifying if memory is allocated */
  Matrix<scalar_t> *U_; /**< Pointer to left singular vectors */
  Vector<scalar_t> *s_; /**< Pointer to singular values */
  Tensor<scalar_t> *V_; /**< Pointer to right singular vectors */
  scalar_t squared_frobenius_norm_data_;  /**< Frobenius norm of data */
  scalar_t squared_frobenius_norm_error_; /**< Frobenius norm of error */
};

}  // namespace Tucker

#endif
