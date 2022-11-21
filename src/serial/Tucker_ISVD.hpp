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
   * @brief Initialize ISVD
   *
   * @param[in] U Pointer to left singular vectors; column major matrix with
   *              orthonormal columns
   * @param[in] s Pointer to singular values array; memory in the range [s, s +
   *              U->ncols()) will be accessed
   * @param[in] X Pointer to tensor whose last unfolding is being factorized
   */
  void initializeFactors(
      const Matrix<scalar_t> *U, const scalar_t *s, const Tensor<scalar_t> *X);

  /**
   * @brief Initialize ISVD
   *
   * @param[in] U Pointer to left singular vectors; column major matrix with
   *              orthonormal columns
   * @param[in] s Pointer to singular values array; memory in the range [s, s +
   *              U->ncols()) will be accessed
   * @param[in] Vt Pointer to right singular vectors (transposed); column major
   *               with U->ncols() orothonormal rows
   * @param[in] squared_frobenius_norm_data Squared Frobenius norm of the data
   *                                        from which initial SVD was computed
   * @param[in] squared_frobenius_norm_error Squared Frobenius norm of the
   *                                         approximation error of the initial
   *                                         SVD
   */
  void initializeFactors(
      const Matrix<scalar_t> *U, const scalar_t *s, const Matrix<scalar_t> *Vt,
      scalar_t squared_frobenius_norm_data = static_cast<scalar_t>(0),
      scalar_t squared_frobenius_norm_error = static_cast<scalar_t>(0));

  /**
   * @brief Update factorization given new data
   *
   * @param[in] C Pointer to tensor with new data; the entire tensor will be
   *              flattened and treated as a single row
   * @param[in] tolerance Approximation tolerance
   */
  void updateFactors(const Tensor<scalar_t> *C, scalar_t tolerance);

  /**
   * @brief Number of rows in ISVD factorization
   *
   * @exception std::runtime_error If the ISVD factors are not initialized/null
   *                               pointers
   */
  int nrows() const;

  /**
   * @brief Number of columns in ISVD factorization
   *
   * @exception std::runtime_error If the ISVD factors are not initialized/null
   *                               pointers
   */
  int ncols() const;

  /**
   * @brief Rank of ISVD factorization
   *
   * @exception std::runtime_error If the ISVD factors are not
   * initialized/null pointers
   */
  int rank() const;

  /**
   * @brief Constant pointer to left singular vectors
   */
  const Matrix<scalar_t> *leftSingularVectors() const { return U_; }

  /**
   * @brief Absolute error estimate of approximation w.r.t. Frobenius norm
   */
  scalar_t absoluteErrorEstimate() const {
    return std::sqrt(squared_frobenius_norm_error_);
  }

  /**
   * @brief Relative error estimate of approximation w.r.t. Frobenius norm
   */
  scalar_t relativeErrorEstimate() const {
    return std::sqrt(squared_frobenius_norm_error_ /
                     squared_frobenius_norm_data_);
  }

 private:
  Matrix<scalar_t> *U_;  /**< Pointer to left singular vectors */
  Vector<scalar_t> *s_;  /**< Pointer to singular values */
  Matrix<scalar_t> *Vt_; /**< Pointer to right singular vectors */
  scalar_t squared_frobenius_norm_data_;  /**< Frobenius norm of data */
  scalar_t squared_frobenius_norm_error_; /**< Frobenius norm of error */
};

}  // namespace Tucker

#endif
