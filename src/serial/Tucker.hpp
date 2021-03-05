/** \copyright
 * Copyright (2016) Sandia Corporation. Under the terms of Contract
 * DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
 * certain rights in this software.
 * \n\n
 * BSD 2-Clause License
 * \n\n
 * Copyright (c) 2016, Sandia Corporation
 * All rights reserved.
 * \n\n
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * \n\n
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * \n\n
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * .
 * \n
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @file
 * \brief Contains functions relevant to computing a %Tucker decomposition.
 *
 * @author Alicia Klinvex
 */

#ifndef TUCKER_HPP_
#define TUCKER_HPP_

#include <iostream>
#include "string.h"
#include "Tucker_BlasWrapper.hpp"
#include "Tucker_TuckerTensor.hpp"
#include "Tucker_Metric.hpp"
#include "Tucker_SparseMatrix.hpp"
#include "Tucker_Vector.hpp"

namespace Tucker {

/** \brief Yn'(l by m)(The transpose of mode n unfolding of Y) consists of several column major submatrices stacked
 * vertically. R(h by m) should have the same number of columns as Yn'. R can be think of as a window that select 
 * elements from Yn'. This function then copies the elements of Yn' selected by the top window and reoders those 
 * elements in column major order 
 */
template <class scalar_t>
void combineColumnMajorBlocks(const Tensor<scalar_t>* Y, Matrix<scalar_t>* R, const int n, const int startingSubmatrix, const int numSubmatrices, const int variant);

/** \brief wrapper of computeLQ(const Tensor* Y, const int n, Matrix* L) which get the LQ of the mode n unfolding
 * of Y. If Yn is short and fat or square, the column major square matrix containing the lower triangle of Yn is returned.
 * We allow Yn to be tall and skinny, in which case the returned matrix is not square but a tall and skinny matrix containing
 * the lower trapezoid of Yn.
 */
template <class scalar_t>
Matrix<scalar_t>* computeLQ(const Tensor<scalar_t>* Y, const int n);

/** \brief Get the LQ of the mode n unfolding of Y. The result lower triangluar matrix is stored in L in column 
 * major. The upper triangle is filled with 0. Note if n=0, a call to dgelq is made. Otherwise the sequential
 * tsqr and transpose is used to get the L.
 */
template <class scalar_t>
void computeLQ(const Tensor<scalar_t>* Y, const int n, Matrix<scalar_t>* L);

/** \brief Compute the Gram matrix \f$Y_n Y_n^T\f$
 *
 * \f$Y_0\f$ is stored as a single column major matrix, and
 * \f$Y_{N-1}\f$ is stored as a single row major matrix.
 *
 * Otherwise, \f$Y_n\f$ can be written as \f$[M_1 M_2 M_3 ...]\f$,
 * where each \f$M_i\f$ is a row major matrix.  There will be
 * \f$\prod\limits_{i=n+1}^N d_i\f$ such row major matrices, each of which is of
 * dimension \f$d_k \times \prod\limits_{i=1}^{n-1} d_i\f$.
 *
 * We compute the solution as  \f$[M_1 M_2 M_3 ...] \left[\begin{array}{c}
 * M_1^T \\ M_2^T \\ M_3^T \\ \vdots\end{array}\right] = M_1 M_1^T +
 * M_2 M_2^T + M_3 M_3^T + ...\f$
 *
 * This function uses the BLAS
 * routine <tt>dsyrk</tt>, the symmetric rank-k update.  When \f$n=0\f$,
 * we call \n <tt>dsyrk('u', 'n', nrows, ncols, 1.0, Yptr, nrows, 0.0, Gptr,
 * nrows)</tt>.\n  When \f$n=N-1\f$, we call \n<tt>dsyrk('u', 't', nrows, ncols,
 * 1.0, Yptr, ncols, 0.0, Gptr, nrows)</tt>.\n
 *
 * Otherwise, we call \n
 * <tt>dsyrk('u', 't', nrows, ncontigcols, 1.0, Yptr, ncontigcols, 0.0,
 * Gptr, nrows)</tt>,\n then \n<tt>dsyrk('u', 't', nrows, ncontigcols, 1.0,
 * Yptr+nrows*ncontigcols, ncontigcols, 1.0, Gptr, nrows)</tt>, \n <tt>dsyrk('u',
 * 't', nrows, ncontigcols, 1.0, Yptr+2*nrows*ncontigcols, ncontigcols, 1.0,
 * Gptr, nrows)</tt>, \n <tt>dsyrk('u', 't', nrows, ncontigcols, 1.0,
 * Yptr+3*nrows*ncontigcols, ncontigcols, 1.0, Gptr, nrows)</tt>, \n etc.
 * The total number of calls to dsyrk is \f$\prod\limits_{i=n+1}^N d_i\f$,
 * and the number of contiguous columns is \f$\prod\limits_{i=1}^{n-1} d_i\f$
 *
 * \param[in] Y %Tensor to be multiplied
 * \param[in] n Mode
 *
 * \note This function allocates memory for the Gram matrix.  The user
 * is responsible for freeing that memory.
 *
 * \note The lower triangle of the Gram matrix is never initialized,
 * since the matrix is symmetric.
 *
 * \test Tucker_gram_test_file.cpp \n Reads a 4-dimensional tensor (3x5x7x11)
 * from a file and computes the Gram matrix of all 4 modes.  The tensor was
 * randomly generated by MATLAB, and the gold standard results (hard coded
 * within the test source code) were also
 * computed by MATLAB.  Every entry of the computed results is compared
 * with the gold standard, and if they differ by more than 1e-10, the
 * test fails.
 * \test Tucker_gram_test_nofile.cpp \n Computes the Gram matrix of all 4
 * modes of a 4-dimensional tensor (2x2x2x2). \n \f$Y_0 = \left[
 * \begin{array}{cccccccc}1 & 3 & 5 & 7 & 9 & 11 & 13 & 15 \\
 * 2 & 4 & 6 & 8 & 10 & 12 & 14 & 16\end{array}\right]\f$ \n
 * The results are compared with hard-coded gold-standard results that
 * were generated by MATLAB.  Every entry of the computed results is
 * compared with the gold standard, and if they differ by more than
 * 1e-10, the test fails.
 *
 * \exception std::runtime_error \a Y is a null-pointer
 * \exception std::runtime_error \a Y has no entries
 * \exception std::runtime_error \a n is not in the range [0,N)
 */
template <class scalar_t>
Matrix<scalar_t>* computeGram(const Tensor<scalar_t>* Y, const int n);

/** \brief Compute the Gram matrix \f$Y_n Y_n^T\f$
 *
 * This variant of computeGram is used by TuckerMPI.
 *
 * \param[in] Y %Tensor to be multiplied
 * \param[in] n Mode
 * \param[in,out] gram Gram matrix, stored in a single-dimensional
 * array.  This function assumes that the memory for the column-major
 * Gram matrix has already been allocated and fills in the values.
 * \param[in] stride Distance between consecutive rows of the matrix
 * in memory.  This is used by LAPACK.
 *
 * \note The lower triangle of the Gram matrix is never initialized,
 * since the matrix is symmetric.
 *
 * \exception std::runtime_error \a Y is a null-pointer
 * \exception std::runtime_error \a Y has no entries
 * \exception std::runtime_error \a n is not in the range [0,N)
 * \exception std::runtime_error \a gram is a null-pointer
 * \exception std::runtime_error \a stride < 1
 */
template <class scalar_t>
void computeGram(const Tensor<scalar_t>* Y, const int n,
    scalar_t* gram, const int stride);

/** \brief Compute all eigenpairs of G
 *
 * The eigenvalues are stored in the array eigenvalues, and the
 * eigenvectors overwrite G.
 * \param[in,out] G The matrix whose eigenvalues are being computed; overwritten
 * by the set of eigenvectors
 * \param[out] eigenvalues scalar_t array to store the computed eigenvalues.  This
 * function dynamically allocates space for this array.  The user is expected
 * to free that memory when done.
 * \param[in] flipSign If true, each eigenvector may have its sign flipped to be
 * consistent with the Matlab tensor toolbox; each vector's maximum magnitude
 * entry will be positive.
 *
 * \test Tucker_eig_test.cpp \n
 * Reads a symmetric matrix from a text file; the matrix was randomly
 * generated by MATLAB.  Computes the eigenpairs of the 5x5 matrix and
 * compares them to the hard-coded gold standard (also generated by
 * MATLAB).  If any of the values differ by more than 1e-10, the test
 * fails.
 *
 * \exception std::runtime_error \a G is a null-pointer
 * \exception std::runtime_error \a G has no entries
 */
template <class scalar_t>
void computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues,
    const bool flipSign=false);

/** \brief Compute all eigenpairs of \a G, and copy a subset to a separate
 * matrix
 *
 * The eigenvalues are stored in the array \a eigenvalues, and the
 * eigenvectors overwrite \a G.  The eigenvectors corresponding to the
 * \a numEvecs largest eigenvalues are copied to \a eigenvectors.
 * \param[in,out] G The matrix whose eigenvalues are being computed; overwritten
 * by the set of eigenvectors
 * \param[out] eigenvalues scalar_t array to store the computed eigenvalues.  This
 * function dynamically allocates space for this array.
 * \param[out] eigenvectors The matrix containing the eigenvectors corresponding to
 * the largest eigenvalues.  Dynamically allocated by this function; user
 * is expected to free the memory when done.
 * \param[in] numEvecs The number of vectors to be copied to \a eigenvectors
 * \param[in] flipSign If true, each eigenvector may have its sign flipped to be
 * consistent with the Matlab tensor toolbox; each vector's maximum magnitude
 * entry will be positive.
 *
 * \exception std::runtime_error \a G is a null-pointer
 * \exception std::runtime_error \a G has no entries
 * \exception std::runtime_error \a numEvecs is not in the range [1,nrows]
 */
template <class scalar_t>
void computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues,
    Matrix<scalar_t>*& eigenvectors, const int numEvecs, const bool flipSign=false);

/** \brief Compute all eigenpairs of \a G, and copy a subset to a separate
 * matrix
 *
 * The eigenvalues are stored in the array \a eigenvalues, and the
 * eigenvectors overwrite \a G.  The eigenvectors corresponding to the
 * smallest eigenvalues (those whose sum is smaller than \a thresh) are
 * discarded, and the rest are copied to \a eigenvectors.
 * \param[in,out] G The matrix whose eigenvalues are being computed; overwritten
 * by the set of eigenvectors
 * \param[out] eigenvalues scalar_t array to store the computed eigenvalues.  This
 * function dynamically allocates space for this array.
 * \param[out] eigenvectors The matrix containing the eigenvectors corresponding to
 * the largest eigenvalues.  Dynamically allocated by this function; user
 * is expected to free the memory when done.
 * \param[in] thresh The maximum sum of the eigenvalues to be discarded
 * \param[in] flipSign If true, each eigenvector may have its sign flipped to be
 * consistent with the Matlab tensor toolbox; each vector's maximum magnitude
 * entry will be positive.
 *
 * \exception std::runtime_error \a G is a null-pointer
 * \exception std::runtime_error \a G has no entries
 * \exception std::runtime_error \a thresh < 0
 */
template <class scalar_t>
void computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues,
    Matrix<scalar_t>*& eigenvectors, const scalar_t thresh, const bool flipSign=false);

/**This function does not return right singular vectors.
 */ 
template <class scalar_t>
void computeSVD(Matrix<scalar_t>* L, scalar_t* singularValues, 
  Matrix<scalar_t>* leftSingularVectors);

/**This function does not return right singular vectors.
 */ 
template <class scalar_t>
void computeSVD(Matrix<scalar_t>* L, scalar_t*& singularValues, 
  Matrix<scalar_t>*& leadingLeftSingularVectors, const scalar_t thresh);

/**This function does not return right singular vectors.
 * numSingularVector specifies how many left singular vectors to keep.
 */ 
template <class scalar_t>
void computeSVD(Matrix<scalar_t>* L, scalar_t*& singularValues, 
  Matrix<scalar_t>*& leadingLeftSingularVectors, const int numSingularVector);  
/** \brief Compute the Tucker decomposition of a tensor X
 *
 * This is an implementation of the sequentially truncated
 * higher-order singular value decomposition (ST_HOSVD).  The
 * algorithm is as follows:
 *
 * - \f$Y_1 = X\f$
 * - for n = 1..N
 *   - Compute the Gram matrix \f$S = Y_n Y_n^T\f$.
 *     This is accomplished by a call to computeGram
 *   - Compute all eigenpairs of S via a call to the LAPACK routine
 *     dsyev.
 *   - Discard the eigenvectors corresponding to the smallest eigenvalues.
 *     We discard the r smallest eigenvalues, where the sum of those
 *     eigenvalues is less than \f$\epsilon^2 \| X \| / N\f$
 *     We will refer to the block of eigenvectors corresponding to the largest
 *     eigenvalues as \f$U_n\f$
 *   - \f$Y_{n+1} = Y_n \times_n U_n\f$
 * - \f$G = Y_n\f$
 *
 * \param X The tensor to be factorized
 * \param epsilon The threshold defining which eigenvalues get discarded
 * \param flipSign See computeEigenpairs for details
 *
 * \exception std::runtime_error X is null
 * \exception std::runtime_error X has no entries
 * \exception std::runtime_error epsilon < 0
 */
template <class scalar_t>
const struct TuckerTensor<scalar_t>* STHOSVD(const Tensor<scalar_t>* X,
    const scalar_t epsilon, bool useQR=false, bool flipSign=false);

/** \brief Compute the Tucker decomposition of a tensor X
 *
 * This is an implementation of the sequentially truncated
 * higher-order singular value decomposition (ST_HOSVD).  The
 * algorithm is as follows:
 *
 * - \f$Y_1 = X\f$
 * - for n = 1..N
 *   - Compute the Gram matrix \f$S = Y_n Y_n^T\f$.
 *     This is accomplished by a call to computeGram
 *   - Compute all eigenpairs of S via a call to the LAPACK routine
 *     dsyev.
 *   - \f$U_n\f$ = the eigenvectors corresponding to the \f$reducedI_n\f$
 *     largest eigenvalues
 *   - \f$Y_{n+1} = Y_n \times_n U_n\f$
 * - \f$G = Y_n\f$
 *
 * \param X The tensor to be factorized
 * \param reducedI The dimension of the core tensor
 * \param flipSign See computeEigenpairs for details
 *
 * \exception std::runtime_error X is null
 * \exception std::runtime_error X has no entries
 * \exception std::runtime_error reducedI does not have the same number
 * of dimensions as X
 * \exception std::runtime_error reducedI has negative entries
 * \exception std::runtime_error reducedI specifies a size larger than X
 */
template <class scalar_t>
const struct TuckerTensor<scalar_t>* STHOSVD(const Tensor<scalar_t>* X,
    const SizeArray* reducedI, bool useQR=false, bool flipSign=false);

/** \brief Compute \f$Y := X \times_n U\f$, where X and Y are
 * tensors, and U is a small dense matrix
 *
 * \param X Tensor to be multiplied
 * \param n Mode
 * \param U Matrix to be multiplied
 * \param Utransp If true, compute \f$Y := X \times_n U^T\f$;
 * otherwise, compute \f$Y := X \times_n U\f$
 *
 * \test ttm_test_file.cpp
 * \test ttm_test_nofile.cpp
 */
template <class scalar_t>
Tensor<scalar_t>* ttm(const Tensor<scalar_t>* X, const int n,
    const Matrix<scalar_t>* U, bool Utransp=false);

template <class scalar_t>
Tensor<scalar_t>* ttm(const Tensor<scalar_t>* const X, const int n,
    const scalar_t* const Uptr, const int dimU,
    const int strideU, bool Utransp=false);

template <class scalar_t>
void ttm(const Tensor<scalar_t>* const X, const int n,
    const Matrix<scalar_t>* const U, Tensor<scalar_t>* Y, bool Utransp=false);

template <class scalar_t>
void ttm(const Tensor<scalar_t>* const X, const int n,
    const scalar_t* const Uptr, const int strideU,
    Tensor<scalar_t>* Y, bool Utransp=false);

/** \brief Compute some information about slices of a tensor
 *
 * \param[in] Y The tensor whose information is being computed
 * \param[in] mode The mode used to determine the slices
 * \param[in] metrics A sum of #Metric to be computed; currently supported options
 * are MAX, MEAN, MIN, SUM and VARIANCE.  If \a metrics = MAX + MIN, the minimum
 * and maximum entry will be computed for each slice.
 *
 * \note VARIANCE depends on MEAN, so if \a metrics & VARIANCE is true, the mean
 * will also be computed for each slice.
 *
 * \note This function will allocate memory for the metric data; the user is expected
 * to free it.
 *
 * \test slice_test_file.cpp
 * \test slice_test_nofile.cpp
 */
template <class scalar_t>
MetricData<scalar_t>* computeSliceMetrics(const Tensor<scalar_t>* Y, const int mode, const int metrics);

/** \brief Perform a transformation on each slice of a tensor
 *
 * If slice \a mode is denoted as S, the transformation is as follows
 * S = (S + \a shifts[\a mode]) / \a scales[\a mode]
 *
 * \param Y The tensor whose slices are being transformed
 * \param mode The mode which determines the slices
 * \param scales Array of numbers to divide by
 * \param shifts Array of numbers to add
 *
 * \test shift_scale_test.cpp
 */
template <class scalar_t>
void transformSlices(Tensor<scalar_t>* Y, int mode, const scalar_t* scales, const scalar_t* shifts);

/** \brief Normalize each slice of the tensor so its data lies in the range [0,1]
 *
 * \param Y The tensor whose slices are being normalized
 * \param mode The mode which determines the slices
 */
template <class scalar_t>
void normalizeTensorMinMax(Tensor<scalar_t>* Y, int mode, const char* scale_file=0);

template <class scalar_t>
void normalizeTensorMax(Tensor<scalar_t>* Y, int mode, const char* scale_file=0);

template <class scalar_t>
void normalizeTensorStandardCentering(Tensor<scalar_t>* Y, int mode, scalar_t stdThresh, const char* scale_file=0);

template <class scalar_t>
void writeScaleShift(const int mode, const int sizeOfModeDim, const scalar_t* scales,
    const scalar_t* shifts, const char* scale_file=0);

template <class scalar_t>
void readTensorBinary(Tensor<scalar_t>* Y, const char* filename);

/** \brief Imports a tensor from a text file
 *
 * The input file should be compatible with the MATLAB tensor
 * toolbox.
 *
 * Example: \n
 * <tt>
 * tensor\n
 * 3 [number of dimensions]\n
 * 2 3 4 [size of each mode]\n
 * [tensor entries, one per line]
 * </tt>
 *
 * \param[in] filename Name of file to be read
 */
template <class scalar_t>
Tensor<scalar_t>* importTensor(const char* filename);

/** \brief Imports a tensor from a binary file
 *
 * \param[in] filename Name of file to be read
 * \param[in,out] t Tensor to be read
 */
template <class scalar_t>
void importTensorBinary(Tensor<scalar_t>* t, const char* filename);

template <class scalar_t>
void importTimeSeries(Tensor<scalar_t>* Y, const char* filename);

/** \brief Imports a tensor from a text file
 *
 * \param[in] filename Name of file to be read
 */
template <class scalar_t>
Matrix<scalar_t>* importMatrix(const char* filename);

/** \brief Imports a sparse matrix from a text file
 *
 * \param[in] filename Name of file to be read
 */
template <class scalar_t>
SparseMatrix<scalar_t>* importSparseMatrix(const char* filename);

template <class scalar_t>
void writeTensorBinary(const Tensor<scalar_t>* Y, const char* filename);

/** \brief Writes a tensor to a text file
 *
 * The output file can be read by the MATLAB tensor toolbox.
 *
 * \param[in] Y tensor to be written to file
 * \param[in] filename Name of file to be written to
 */
template <class scalar_t>
void exportTensor(const Tensor<scalar_t>* Y, const char* filename);

/** \brief Writes a tensor to a binary file
 *
 * The output file can be read by MPI IO
 *
 * \param[in] Y tensor to be written to file
 * \param[in] filename Name of file to be written to
 */
template <class scalar_t>
void exportTensorBinary(const Tensor<scalar_t>* Y, const char* filename);

template <class scalar_t>
void exportTimeSeries(const Tensor<scalar_t>* Y, const char* filename);

/// Premultiply a dense matrix by a diagonal matrix
template <class scalar_t>
void premultByDiag(const Vector<scalar_t>* diag, Matrix<scalar_t>* mat);

/** \brief Generate core tensor and factor matrices. Combine them with a 
 * series of ttm and then add noise to generate the output tensor. 
 * 
 * \param[in] seed for the random generator
 * \param[in/out] TuckerTensor that will store the original core tensor and
 * factor matrices
 * \param[in] dimensions of the desired output tensor
 * \param[in] dimensions of the core tensor
 * \param[in] noise to be added. 
 */
template <class scalar_t>
Tensor<scalar_t>* generateTensor(int seed, TuckerTensor<scalar_t>* fact, SizeArray* tensor_dims,
 SizeArray* core_dims, scalar_t noise);
} // end of namespace Tucker

#endif /* TUCKER_HPP_ */
