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
 * \brief Contains functions relevant to computing a streaming %Tucker decomposition.
 *
 * \author Alicia Klinvex
 * \author Zitong Li
 * \author Saibal De
 * \author Hemanth Kolla
 */

#include "Tucker.hpp"
#include "Tucker_StreamingTuckerTensor.hpp"
#include <algorithm>
#include <limits>
#include <fstream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>

/** \namespace Tucker \brief Contains the data structures and functions
 * necessary for a sequential tucker decomposition
 */
namespace Tucker {

template <class scalar_t>
void updateStreamingGram(Matrix<scalar_t>* Gram, const Tensor<scalar_t>* Y, const int n)
{

  Matrix<scalar_t>* gram_to_add = computeGram(Y,n);
  size_t nnz = gram_to_add->getNumElements();
  for(int i=0; i<nnz; i++) {
    *(Gram->data()+i) += *(gram_to_add->data()+i);
  }
  MemoryManager::safe_delete<Matrix<scalar_t>>(gram_to_add);
}

template<class scalar_t>
void computeEigenpairs(const Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign)
{
  if(G == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(const Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign): G is a null pointer");
  }
  if(G->getNumElements() == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(const Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign): G has no entries");
  }
  if (old_eigenvectors == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(const Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign): old_eigenvectors is a null pointer");
  }
  if(thresh < 0) {
    std::ostringstream oss;
    oss << "Tucker::computeEigenpairs(const Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign): thresh = " << thresh << " < 0";
    throw std::runtime_error(oss.str());
  }

  // The Gram matrix is overwritten inside computeEigenpairs
  // Make a copy of it, retain it for later use
  int nrows = G->nrows();
  Matrix<scalar_t>* G_copy =
      MemoryManager::safe_new<Matrix<scalar_t>>(nrows, nrows);
  {
    const int& nelem = nrows * nrows;
    const int& ONE = 1;
    copy(&nelem, G->data(), &ONE, G_copy->data(), &ONE);
  }

  computeEigenpairs(G_copy, eigenvalues, flipSign);

  // Compute projection of new eigenvectors along old eigenvectors
  int nproj = old_eigenvectors->ncols();
  Vector<scalar_t> *projection = MemoryManager::safe_new<Vector<scalar_t>>(nproj);
  Vector<scalar_t> *projectionNorms = MemoryManager::safe_new<Vector<scalar_t>>(nrows);
  for (int i = 0; i < nrows; ++i) {
    const int ONE = 1;
    const char &trans = 'T';
    const scalar_t &alpha = 1;
    const scalar_t &beta = 0;
    gemv(&trans, &nrows, &nproj, &alpha, old_eigenvectors->data(), &nrows, G_copy->data() + i * nrows, &ONE, &beta, projection->data(), &ONE);
    (*projectionNorms)[i] = nrm2(&nproj, projection->data(), &ONE);
  }
  MemoryManager::safe_delete(projection);

  // Compute number of things to copy
  int numEvecs=nrows;
  scalar_t sum = 0;
  for(int i=nrows-1; i>=0; i--) {
    // TODO discuss the truncation criteria
    if ((*projectionNorms)[i] > std::sqrt(std::numeric_limits<scalar_t>::epsilon())) {
      break;
    }

    sum += std::abs(eigenvalues[i]);
    if(sum > thresh) {
      break;
    }
    numEvecs--;
  }
  MemoryManager::safe_delete(projectionNorms);

  // Allocate memory for eigenvectors
  int numRows = G->nrows();
  eigenvectors = MemoryManager::safe_new<Matrix<scalar_t>>(numRows,numEvecs);

  // Copy appropriate eigenvectors
  int nToCopy = numRows*numEvecs;
  const int ONE = 1;
  Tucker::copy(&nToCopy, G_copy->data(), &ONE, eigenvectors->data(), &ONE);
  MemoryManager::safe_delete(G_copy);
}

template <class scalar_t>
Tensor<scalar_t>* updateCore(Tensor<scalar_t>* G, const Matrix<scalar_t>* U_old, 
    const Matrix<scalar_t>* U_new, const int dim)
{

  // First the matrix multiplication U_new^T * U_old
  // Do sanity check of dimensions
  assert( U_new->nrows() == U_old->nrows() );

  int m = U_new->ncols();
  int n = U_old->ncols();
  int k = U_new->nrows();
  Matrix<scalar_t>* S = MemoryManager::safe_new<Matrix<scalar_t>>(m,n);

  char transa = 'T';
  char transb = 'N';
  int lda = k;
  int ldb = k;
  int ldc = m; 
  scalar_t alpha = 1.0;
  scalar_t beta = 0.0;
  gemm(&transa, &transb, &m, &n, &k, &alpha, U_new->data(),
        &lda, U_old->data(), &ldb, &beta, S->data(), &ldc);

  Tensor<scalar_t>* ttm_result = ttm(G,dim,S,false);

  MemoryManager::safe_delete<Matrix<scalar_t>>(S);
  return ttm_result;
}

template <class scalar_t>
const struct StreamingTuckerTensor<scalar_t>* StreamingHOSVD(const Tensor<scalar_t>* X, const TuckerTensor<scalar_t>* initial_factorization,
    const char* filename, const scalar_t epsilon, bool useQR, bool flipSign)
{

  // Create a struct to store the factorization
  struct StreamingTuckerTensor<scalar_t>* factorization = MemoryManager::safe_new<StreamingTuckerTensor<scalar_t>>(initial_factorization);

  // Construct and initialize ISVD object
  ISVD<scalar_t>* iSVD = MemoryManager::safe_new<ISVD<scalar_t>>();
  iSVD->initializeFactors(factorization->factorization);

  int ndims = X->N();
  scalar_t tensorNorm = X->norm2();
  scalar_t thresh = epsilon*epsilon*tensorNorm/X->N();
  Tucker::SizeArray* slice_dims = MemoryManager::safe_new<SizeArray>((ndims-1));

  // Compute the initial Gram matrices for all non-streaming modes
  for(int n=0; n<ndims-1; n++) {
    factorization->Gram[n] = computeGram(X,n);
    (*slice_dims)[n] = X->size(n); 
  }

  //Open the file containing names of stream of snapshot files
  //Loop over, read each snapshot, update the Tucker model
  std::ifstream inStream(filename);
  std::string snapshot_file;
  
  while(inStream >> snapshot_file) {
    // Create an object for the Tensor slice
    Tucker::Tensor<scalar_t>* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*slice_dims);

    std::cout<< "Reading snapshot " << snapshot_file << std::endl;
    importTensorBinary(Y,snapshot_file.c_str());

    // Line 1 of StreamingTuckerUpdate algorithm
    // Set the threshold for Gram SVDs
    const scalar_t delta = epsilon*std::sqrt(Y->norm2()/ndims);

    // Line 2 of StreamingTuckerUpdate algorithm
    // Set the threshold for ISVD
    thresh += delta * delta;

    // Loop over non-streaming modes
    for(int n=0; n<ndims-1; n++) {
      // Line 4 of StreamingTuckerUpdate algorithm
      // Update Gram of non-streaming modes
      updateStreamingGram(factorization->Gram[n], Y, n);

      // Lines 5-13 of StreamingTuckerUpdate algorithm
      // Update bases (factor matrices)
      // This function is overloaded, implemented above (line ~70)
      Matrix<scalar_t> *U_new;
      computeEigenpairs(factorization->Gram[n], factorization->factorization->eigenvalues[n],
          U_new, factorization->factorization->U[n], thresh, flipSign);

      // Line 14 of StreamingTuckerUpdate algorithm
      // Accumulate ttm products into existing core
      Tensor<scalar_t>* temp = updateCore(factorization->factorization->G, factorization->factorization->U[n], U_new, n);
      MemoryManager::safe_delete<Tensor<scalar_t>>(factorization->factorization->G);
      factorization->factorization->G = temp;

      // Line 15 of StreamingTuckerUpdate algorithm
      // Accumulate ttm products into new slice
      Tensor<scalar_t>* temp2 = ttm(Y,n,U_new,true);
      MemoryManager::safe_delete<Tensor<scalar_t>>(Y);
      Y = temp2;

      // Line 16 of StreamingTuckerUpdate algorithm
      // Update right singular vectors of ISVD factorization
      iSVD->updateRightSingularVectors(n, U_new, factorization->factorization->U[n]);

      // Line 17 of StreamingTuckerUpdate algorithm
      // Save the new factor matrix
      MemoryManager::safe_delete(factorization->factorization->U[n]);
      factorization->factorization->U[n] = U_new;
    }

    // Line 19 of StreamingTuckerUpdate algorithm
    // Add new row to ISVD factorization
    iSVD->updateFactorsWithNewSlice(Y, delta);

    // Lines 20-21 of StreamingTuckerUpdate algorithm
    // Retrieve updated left singular vectors from ISVD factorization
    Matrix<scalar_t> *U_new = nullptr;
    {
      const Matrix<scalar_t> *U_isvd = iSVD->getLeftSingularVectors();
      const int &nrows = U_isvd->nrows();
      const int &ncols = U_isvd->ncols();
      const int &nelem = nrows * ncols;
      const int &ONE = 1;
      U_new = MemoryManager::safe_new<Matrix<scalar_t>>(nrows, ncols);
      copy(&nelem, U_isvd->data(), &ONE, U_new->data(), &ONE);
    }

    // Line 22 of StreamingTuckerUpdate algorithm
    // Split U_new into two submatrices
    // Use first submatrix to update core
    {
      const int nrows = U_new->nrows();
      Tucker::Matrix<scalar_t> *U_sub = U_new->getSubmatrix(0, nrows - 2);
      Tensor<scalar_t>* temp3 = updateCore(factorization->factorization->G, factorization->factorization->U[ndims - 1], U_sub, ndims - 1);
      MemoryManager::safe_delete(U_sub);
      MemoryManager::safe_delete<Tensor<scalar_t>>(factorization->factorization->G);
      factorization->factorization->G = temp3;
    }

    // Lines 23-24 of StreamingTuckerUpdate algorithm
    // Use last row of Unew[d] to update scale new slice and update core in-place
    {
      const int &nrow = Y->getNumElements();
      const int &ncol = U_new->nrows();
      const int &rank = U_new->ncols();
      const int &ONE = 1;
      for (int j = 0; j < rank; ++j) {
        axpy(&nrow,
             U_new->data() + j * ncol - 1,
             Y->data(), &ONE,
             factorization->factorization->G->data() + j * nrow, &ONE);
      }
    }

    // Line 25 of StreamingTuckerUpdate algorithm
    // Save new factor matrix
    MemoryManager::safe_delete(factorization->factorization->U[ndims - 1]);
    factorization->factorization->U[ndims - 1] = U_new;

    // Free memory
    MemoryManager::safe_delete<Tucker::Tensor<scalar_t>>(Y);
  }

  // Close the file containing snapshot filenames
  inStream.close();

  // Free memory
  Tucker::MemoryManager::safe_delete<Tucker::SizeArray>(slice_dims);

  return factorization;
}

template void updateStreamingGram(Matrix<float>*, const Tensor<float>*, const int);
template Tensor<float>* updateCore(Tensor<float>*, const Matrix<float>*, const Matrix<float>*, const int);
template const struct StreamingTuckerTensor<float>* StreamingHOSVD(const Tensor<float>*, const TuckerTensor<float>*, 
             const char* filename, const float, bool, bool);

template void updateStreamingGram(Matrix<double>*, const Tensor<double>*, const int);
template Tensor<double>* updateCore(Tensor<double>*, const Matrix<double>*, const Matrix<double>*, const int);
template const struct StreamingTuckerTensor<double>* StreamingHOSVD(const Tensor<double>*, const TuckerTensor<double>*,
             const char* filename, const double, bool, bool);

} // end namespace Tucker
