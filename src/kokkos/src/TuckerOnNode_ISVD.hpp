/**
 * @file
 * @brief Incremental SVD definition
 * @author Saibal De
 */

#ifndef TUCKER_ISVD_HPP_
#define TUCKER_ISVD_HPP_

#include "TuckerOnNode_Tensor.hpp"
#include "Tucker_TuckerTensor.hpp"

namespace TuckerOnNode {

/**
 * @brief Incremental singular value decomposition
 */
template <class scalar_t, class mem_space_t = Kokkos::DefaultExecutionSpace::memory_space>
class ISVD {
public:
  using vector_t = Kokkos::View<scalar_t*, mem_space_t>;
  using matrix_t = Kokkos::View<scalar_t**, Kokkos::LayoutLeft, mem_space_t>;
  using tensor_t = Tensor<scalar_t, mem_space_t>;
  using ttensor_t = Tucker::TuckerTensor<tensor_t>;
  using eigval_t = TensorGramEigenvalues<scalar_t,mem_space_t>;
  /**
   * @brief Default constructor
   */
  ISVD() :
    is_allocated_(false),
    squared_frobenius_norm_data_(0.0),
    squared_frobenius_norm_error_(0.0) {}

  /**
   * @brief Default destructor
   */
  ~ISVD() = default;

  /**
   * @brief Number of rows in ISVD factorization
   *
   * @exception std::runtime_error If the ISVD factors are not initialized/null
   *                               pointers
   */
  int nrows() const {
    checkIsAllocated();
    return U_.extent(0);
  }

  /**
   * @brief Number of columns in ISVD factorization
   *
   * @exception std::runtime_error If the ISVD factors are not initialized/null
   *                               pointers
   */
  int ncols() const {
    checkIsAllocated();
    return V_.size() / rank();
  }

  /**
   * @brief Rank of ISVD factorization
   *
   * @exception std::runtime_error If the ISVD factors are not
   * initialized/null pointers
   */
  int rank() const {
    checkIsAllocated();
    return s_.extent(0);
  }

  /**
   * @brief Constant pointer to singular values
   */
  const vector_t& getSingularValues() const {
    checkIsAllocated();
    return s_;
  }

  /**
   * @brief Pointer to singular values
   */
  vector_t& getSingularValues() {
    checkIsAllocated();
    return s_;
  }

  /**
   * @brief Constant pointer to left singular vectors
   */
  const matrix_t& getLeftSingularVectors() const {
    checkIsAllocated();
    return U_;
  }

  /**
   * @brief Pointer to left singular vectors
   */
  matrix_t& getLeftSingularVectors() {
    checkIsAllocated();
    return U_;
  }

  /**
   * @brief Constant pointer to right singular vectors
   */
  const tensor_t& getRightSingularVectors() const {
    checkIsAllocated();
    return V_;
  }

  /**
   * @brief Pointer to right singular vectors
   */
  tensor_t& getRightSingularVectors() {
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
   * @brief Frobenius norm of data matrix
   */
  scalar_t getDataNorm() const {
    checkIsAllocated();
    return std::sqrt(squared_frobenius_norm_data_);
  }

  /**
   * @brief Forbenius norm of error matrix
   */
  scalar_t getErrorNorm() const {
    checkIsAllocated();
    return std::sqrt(squared_frobenius_norm_error_);
  }

  /**
   * @brief Relative error estimate of approximation w.r.t. Frobenius norm
   */
  scalar_t getRelativeError() const {
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
  void initializeFactors(const matrix_t& U, const vector_t& s,
                         const tensor_t& X);

  /**
   * @brief Initialize ISVD for last mode of tensor in Tucker format
   *
   * @param[in] X TuckerTensor
   */
  void initializeFactors(const ttensor_t& X, const eigval_t& eig);

  /**
   * @brief Pad right singular vectors with zeros along specified mode
   *
   * @param[in] k Tensor mode
   * @param[in] p Number of zero slices to add
   */
  void padRightSingularVectorsAlongMode(int k, int p);

  /**
   * @brief Update factorization given new data
   *
   * @param[in] C Pointer to tensor with new data; the entire tensor will be
   *              flattened and treated as a single row
   * @param[in] tolerance Approximation tolerance
   */
  void updateFactorsWithNewSlice(const tensor_t& Y, scalar_t tolerance);

  tensor_t padTensorAlongMode(const tensor_t& X, int n, int p);
  tensor_t concatenateTensorsAlongMode(const tensor_t& X, const tensor_t& Y,
                                       int n);

  // These can't be private on Cuda because they contain device lambdas
//private:
  /**
   * @brief Check if ISVD object is initialized
   */
  void checkIsAllocated() const;

  /**
   * @brief Update three-factor SVD given new rows
   */
  void addSingleRowNaive(const vector_t& c, scalar_t tolerance);

private:
  using exec_space = typename mem_space_t::execution_space;

  bool is_allocated_;   /**< Flag specifying if memory is allocated */
  matrix_t U_; /**< Pointer to left singular vectors */
  vector_t s_; /**< Pointer to singular values */
  tensor_t V_; /**< Pointer to right singular vectors */
  scalar_t squared_frobenius_norm_data_;  /**< Frobenius norm of data */
  scalar_t squared_frobenius_norm_error_; /**< Frobenius norm of error */
};

}  // namespace Tucker

#endif
