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
void computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign)
{
  if(G == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign): G is a null pointer");
  }
  if(G->getNumElements() == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign): G has no entries");
  }
  if (old_eigenvectors == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign): old_eigenvectors is a null pointer");
  }
  if(thresh < 0) {
    std::ostringstream oss;
    oss << "Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const Matrix<scalar_t>* old_eigenvectors, const scalar_t thresh, const bool flipSign): thresh = " << thresh << " < 0";
    throw std::runtime_error(oss.str());
  }

  computeEigenpairs(G, eigenvalues, flipSign);

  // Compute projection of new eigenvectors along old eigenvectors
  int nproj = old_eigenvectors->ncols();
  int nrows = G->nrows();
  Vector<scalar_t> *projection = MemoryManager::safe_new<Vector<scalar_t>>(nproj);
  Vector<scalar_t> *projectionNorms = MemoryManager::safe_new<Vector<scalar_t>>(nrows);
  for (int i = 0; i < nrows; ++i) {
    const int ONE = 1;
    const char &trans = 'T';
    const scalar_t &alpha = 1;
    const scalar_t &beta = 0;
    gemv(&trans, &nrows, &nproj, &alpha, old_eigenvectors->data(), &nrows, G->data() + i * nrows, &ONE, &beta, projection->data(), &ONE);
    (*projectionNorms)[i] = nrm2(&nproj, projection->data(), &ONE);
  }
  MemoryManager::safe_delete(projection);

  // Compute number of things to copy
  int numEvecs=nrows;
  scalar_t sum = 0;
  for(int i=nrows-1; i>=0; i--) {
    if ((*projectionNorms)[i] > 10 * std::numeric_limits<scalar_t>::epsilon()) {
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
  Tucker::copy(&nToCopy, G->data(), &ONE, eigenvectors->data(), &ONE);
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

    //Create an object for the Tensor slice
    Tucker::Tensor<scalar_t>* Y = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(*slice_dims);

    std::cout<< "Reading snapshot " << snapshot_file << std::endl;
    importTensorBinary(Y,snapshot_file.c_str());

    // Update epsilon
    thresh += epsilon*epsilon*Y->norm2()/ndims;

    // Update Gram of non-streaming modes
    for(int n=0; n<ndims-1; n++) {
      updateStreamingGram(factorization->Gram[n], Y, n);
    }

    // Allocate memory for the new bases (factor matrices) along all modes
    Matrix<scalar_t>** U_new = MemoryManager::safe_new_array<Matrix<scalar_t>*>(ndims);

    Tensor<scalar_t>* core = factorization->factorization->G;

    // Loop over non-streaming modes
    for(int n=0; n<ndims-1; n++) {
      // Update bases (factor matrices)
      computeEigenpairs(factorization->Gram[n], factorization->factorization->eigenvalues[n],
          U_new[n], thresh, flipSign);
      // Accumulate ttm products into existing core
      Tensor<scalar_t>* temp = updateCore(core, factorization->factorization->U[n], U_new[n], n);
      MemoryManager::safe_delete<Tensor<scalar_t>>(core);
      core = temp;
      // Accumulate ttm products into new slice
      Tensor<scalar_t>* temp2 = ttm(Y,n,U_new[n],true);
      MemoryManager::safe_delete<Tensor<scalar_t>>(Y);
      Y = temp2;
    }
    factorization->factorization->G = core;

    // For the streaming mode initialize ISVD with full set of left singular vectors 
    Matrix<scalar_t>* gram_last_mode = computeGram(X,ndims-1);
    computeEigenpairs(gram_last_mode, factorization->factorization->eigenvalues[ndims-1],
        U_new[ndims-1], 0.0 /* thresh */, flipSign);  
   
    ISVD<scalar_t>* iSVD = MemoryManager::safe_new<ISVD<scalar_t>>();
    //

    // TO DO

    //
    // Free memory
    //
    for (int n=0; n<ndims; n++) {
      MemoryManager::safe_delete(factorization->factorization->U[n]);
    }
    MemoryManager::safe_delete_array(factorization->factorization->U,ndims);

    // Set the factor matrices to the updated ones
    factorization->factorization->U = U_new;
    for (int n=0; n<ndims; n++) {
      factorization->factorization->U[n] = U_new[n];
    }

    Tucker::MemoryManager::safe_delete<Tucker::Tensor<scalar_t>>(Y);
  }

  //Close the file containing snapshot filenames
  inStream.close();

  //
  //Free memory
  //
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
