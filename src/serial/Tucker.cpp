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

#include "Tucker.hpp"
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
void combineColumnMajorBlocks(const Tensor<scalar_t>* Y, Matrix<scalar_t>* R, const int n, const int startingSubmatrix, const int numSubmatrices){
  int submatrixNrows = 1;
  for(int i=0; i<n; i++) {
      submatrixNrows *= Y->size(i);
  }
  int Rncols = R->ncols();
  int Rnrows = R->nrows();
  int sizeOfSubmatrix = submatrixNrows * Rncols;
  int one = 1;
  for(int i=0; i<numSubmatrices; i++){
    for(int j=0; j<Rncols; j++){
      copy(&submatrixNrows, Y->data()+(i+startingSubmatrix)*sizeOfSubmatrix+j*submatrixNrows, &one, R->data()+j*Rnrows+i*submatrixNrows, &one);
    }
  }
}

template <class scalar_t>
Matrix<scalar_t>* computeLQ(const Tensor<scalar_t>* Y, const int n){
  int modeNDimension = Y->size(n);
  int YnNcols = 1;
  for(int i=0; i<Y->N(); i++) {
    if(i != n) {
      YnNcols *= Y->size(i);
    }
  }
  int LNrows = YnNcols > modeNDimension ? modeNDimension : YnNcols;
  Matrix<scalar_t>* L = MemoryManager::safe_new<Matrix<scalar_t>>(modeNDimension, LNrows);
  computeLQ(Y, n, L);
  return L;
}

template <class scalar_t>
void computeLQ(const Tensor <scalar_t>* Y, const int n, Matrix<scalar_t>* L){
  if(Y == 0) {
    throw std::runtime_error("Tucker::computeLQ(const Tensor<scalar_t>* Y, const int n): Y is a null pointer");
  }
  if(Y->getNumElements() == 0) {
    throw std::runtime_error("Tucker::computeLQ(const Tensor<scalar_t>* Y, const int n): Y->getNumElements() == 0");
  }
  if(n < 0 || n >= Y->N()) {
    std::ostringstream oss;
    oss << "Tucker::computeLQ(const Tensor* Y, const int n): n = "
        << n << " is not in the range [0," << Y->N() << ")";
    throw std::runtime_error(oss.str());
  }
  int one = 1;
  int negOne = -1;
  // int sizeOfR = Rnrows*Rncols;
  int modeNDimension = Y->size(n);
  if(n == 0){
    //get number of columns of Y0
    int Y0ncols = 1;
    for(int i=0; i<Y->N(); i++) {
      if(i != n) {
        Y0ncols *= Y->size(i);
      }
    }

    Matrix<scalar_t>* Y0 = MemoryManager::safe_new<Matrix<scalar_t>>(modeNDimension, Y0ncols);
    int Y0nrows = Y0->nrows();
    int sizeOfY0 = Y0nrows*Y0ncols;
    copy(&sizeOfY0, Y->data(), &one, Y0->data(), &one);
    int info;
    //workspace query
    scalar_t * work = MemoryManager::safe_new_array<scalar_t>(1);
    scalar_t * T = MemoryManager::safe_new_array<scalar_t>(5);
    Tucker::gelq(&Y0nrows, &Y0ncols, Y0->data(), &Y0nrows, T, &negOne, work, &negOne, &info);
    int lwork = work[0];
    int TSize = T[0];
    Tucker::MemoryManager::safe_delete_array(work, 1);
    Tucker::MemoryManager::safe_delete_array(T, 5);
    work = MemoryManager::safe_new_array<scalar_t>(lwork);
    T = MemoryManager::safe_new_array<scalar_t>(TSize);
    gelq(&Y0nrows, &Y0ncols, Y0->data(), &Y0nrows, T, &TSize, work, &lwork, &info);
    MemoryManager::safe_delete_array(T, TSize);
    MemoryManager::safe_delete_array(work, lwork);
    if(info != 0){
      std::ostringstream oss;
        oss << "the" << info*-1 << "th argument to gelq is invalid.";
      throw std::runtime_error(oss.str());
    }
    //copy the lower triangle of Y0 to L
    int sizeOfL = L->ncols()*L->nrows();
    copy(&sizeOfL, Y0->data(), &one, L->data(), &one);
  }
  else{//Serial TSQR:
    //get number of rows of each column major submatrix of Yn transpose.
    int submatrixNrows = 1;
    for(int i=0; i<n; i++) {
      submatrixNrows *= Y->size(i);
    }
    int YnTransposeNrows =1; 
    for(int i=0; i<Y->N(); i++) {
      if(i != n) {
        YnTransposeNrows *= Y->size(i);
      }
    }
    //total number of submatrices in Yn transpose
    int totalNumSubmatrices = 1;
    for(int i=n+1; i<Y->N(); i++) {
      totalNumSubmatrices *= Y->size(i);
    }
    //R would be the first block to do qr on.
    Matrix<scalar_t>* R;
    int Rnrows, Rncols, sizeOfR;
    //Handle edge case when the submatrices in YnTranspose are short and fat.
    if(submatrixNrows < modeNDimension){
      //Get the number of submatrices we need to stack to get a tall matrix.
      int numSubmatrices = (int)std::ceil((scalar_t)modeNDimension/(scalar_t)submatrixNrows);
      //When YnTranspose is short and fat this is true.
      if(numSubmatrices > totalNumSubmatrices)
        numSubmatrices = totalNumSubmatrices;
      Rnrows = numSubmatrices * submatrixNrows;
      R = MemoryManager::safe_new<Matrix<scalar_t>>(Rnrows, modeNDimension);
      Rncols = R->ncols();
      combineColumnMajorBlocks(Y, R, n, 0, numSubmatrices);


      int info;
      //workspace query
      scalar_t * work = MemoryManager::safe_new_array<scalar_t>(1);
      scalar_t * TforGeqr = Tucker::MemoryManager::safe_new_array<scalar_t>(5);
      geqr(&Rnrows, &Rncols, R->data(), &Rnrows, TforGeqr, &negOne, work, &negOne, &info);
      int lwork = work[0];
      int TSize = TforGeqr[0];
      Tucker::MemoryManager::safe_delete_array(work, 1);
      Tucker::MemoryManager::safe_delete_array(TforGeqr, 5);
      work = MemoryManager::safe_new_array<scalar_t>(lwork);
      TforGeqr = Tucker::MemoryManager::safe_new_array<scalar_t>(TSize);    
      geqr(&Rnrows, &Rncols, R->data(), &Rnrows, TforGeqr, &TSize, work, &lwork, &info);
      if(info != 0){
        std::ostringstream oss;
          oss << "the" << info*-1 << "th argument to geqr is invalid.";
        throw std::runtime_error(oss.str());
      }
      MemoryManager::safe_delete_array(work, lwork);
      MemoryManager::safe_delete_array(TforGeqr, TSize);
      
      
      int i; //This is the index for the first submatrix that hasn't been included in the TSQR.
      int numSubmatricesLeft = totalNumSubmatrices - numSubmatrices;
      Matrix<scalar_t>* B;
      int Bnrows, Bncols, sizeOfB;
      int nb = (modeNDimension > 32)? 32 : modeNDimension;
      scalar_t* TforTpqrt = MemoryManager::safe_new_array<scalar_t>(nb*modeNDimension);
      work = MemoryManager::safe_new_array<scalar_t>(nb*modeNDimension);
      int ZERO = 0;
      while(numSubmatricesLeft > 0){
        i = totalNumSubmatrices - numSubmatricesLeft;
        if(numSubmatricesLeft > numSubmatrices){
          B = Tucker::MemoryManager::safe_new<Matrix<scalar_t>>(submatrixNrows*numSubmatrices, modeNDimension);
          combineColumnMajorBlocks(Y, B, n, i, numSubmatrices);
          numSubmatricesLeft = numSubmatricesLeft - numSubmatrices;
        }
        else{
          B = Tucker::MemoryManager::safe_new<Matrix<scalar_t>>(submatrixNrows*numSubmatricesLeft, modeNDimension);
          combineColumnMajorBlocks(Y, B, n, i, numSubmatricesLeft);
          numSubmatricesLeft = numSubmatricesLeft - numSubmatricesLeft;
        }
        Bnrows = B->nrows();
        Bncols = B->ncols();
        sizeOfB = Bnrows*Bncols;
        tpqrt(&Bnrows, &Bncols, &ZERO, &nb, R->data(), &Rnrows, B->data(), &Bnrows, TforTpqrt, &nb, work, &info);
        Tucker::MemoryManager::safe_delete(B);
      }
      MemoryManager::safe_delete_array(work, nb*modeNDimension);
      MemoryManager::safe_delete_array(TforTpqrt, nb*modeNDimension);
    }
    else{
      R = MemoryManager::safe_new<Matrix<scalar_t>>(submatrixNrows, modeNDimension);
      Rnrows = R->nrows();
      Rncols = R->ncols();
      sizeOfR = Rnrows * Rncols;
      copy(&sizeOfR, Y->data(), &one, R->data(), &one);

      int info;
      //workspace query
      scalar_t * work = MemoryManager::safe_new_array<scalar_t>(1);
      scalar_t * TforGeqr = Tucker::MemoryManager::safe_new_array<scalar_t>(5);
      geqr(&Rnrows, &Rncols, R->data(), &Rnrows, TforGeqr, &negOne, work, &negOne, &info);
      int lwork = work[0];
      int TSize = TforGeqr[0];
      Tucker::MemoryManager::safe_delete_array(work, 1);
      Tucker::MemoryManager::safe_delete_array(TforGeqr, 5);
      work = MemoryManager::safe_new_array<scalar_t>(lwork);
      TforGeqr = Tucker::MemoryManager::safe_new_array<scalar_t>(TSize);    
      geqr(&Rnrows, &Rncols, R->data(), &Rnrows, TforGeqr, &TSize, work, &lwork, &info);
      if(info != 0){
        std::ostringstream oss;
          oss << "the" << info*-1 << "th argument to geqr is invalid.";
        throw std::runtime_error(oss.str());
      }
      MemoryManager::safe_delete_array(work, lwork);
      MemoryManager::safe_delete_array(TforGeqr, TSize);

      Matrix<scalar_t>* B = MemoryManager::safe_new<Matrix<scalar_t>>(submatrixNrows, modeNDimension);
      int sizeOfB = sizeOfR;
      int nb = (Rncols > 32)? 32 : Rncols;
      scalar_t* TforTpqrt = MemoryManager::safe_new_array<scalar_t>(nb*modeNDimension);
      work = MemoryManager::safe_new_array<scalar_t>(nb*modeNDimension);
      int ZERO = 0;
      for(int i = 1; i < totalNumSubmatrices; i++){
        copy(&sizeOfB, Y->data()+i*sizeOfB, &one, B->data(), &one);
        //call tpqrt(M, N, L, NB, A, LDA, B, LDB, T, LDT, WORK, INFO)
        tpqrt(&submatrixNrows, &Rncols, &ZERO, &nb, R->data(), &Rnrows, B->data(), &submatrixNrows, TforTpqrt, &nb, work, &info);
        if(info != 0){
          std::ostringstream oss;
          oss << "the " << info*-1 << "th argument to tpqrt is invalid.";
          throw std::runtime_error(oss.str());
        }
      }
      MemoryManager::safe_delete_array(work, nb*modeNDimension);
      MemoryManager::safe_delete(B);
      MemoryManager::safe_delete_array(TforTpqrt, nb*modeNDimension);
    }
    //copy R to L so that L becomes the transpose of the top  of R
    for(int r=0; r<L->ncols(); r++){
      copy(&Rncols, R->data()+r, &Rnrows, L->data()+(r*L->nrows()), &one);
    }
    MemoryManager::safe_delete(R);
  }
  //Final step of postprocessing: put 0s in the upper triangle of L
  for(int r=0; r<L->nrows(); r++){
    for(int c=r+1; c<L->ncols(); c++){
      L->data()[r+c*L->nrows()] = 0;
    }
  }
}


/** \example Tucker_gram_test_file.cpp
 * \example Tucker_gram_test_nofile.cpp
 */
template<class scalar_t>
Matrix<scalar_t>* computeGram(const Tensor<scalar_t>* Y, const int n)
{
  if(Y == 0) {
    throw std::runtime_error("Tucker::computeGram(const Tensor<scalar_t>* Y, const int n): Y is a null pointer");
  }
  if(Y->getNumElements() == 0) {
    throw std::runtime_error("Tucker::computeGram(const Tensor<scalar_t>* Y, const int n): Y->getNumElements() == 0");
  }
  if(n < 0 || n >= Y->N()) {
    std::ostringstream oss;
    oss << "Tucker::computeGram(const Tensor<scalar_t>* Y, const int n): n = "
        << n << " is not in the range [0," << Y->N() << ")";
    throw std::runtime_error(oss.str());
  }

  // Allocate memory for the Gram matrix
  int nrows = Y->size(n);
  Matrix<scalar_t>* S = MemoryManager::safe_new<Matrix<scalar_t>>(nrows,nrows);

  computeGram(Y, n, S->data(), nrows);

  return S;
}

template<class scalar_t>
void computeGram(const Tensor<scalar_t>* Y, const int n, scalar_t* gram,
    const int stride)
{
  if(Y == 0) {
    throw std::runtime_error("Tucker::computeGram(const Tensor<scalar_t>* Y, const int n, scalar_t* gram, const int stride): Y is a null pointer");
  }
  if(gram == 0) {
    throw std::runtime_error("Tucker::computeGram(const Tensor<scalar_t>* Y, const int n, scalar_t* gram, const int stride): gram is a null pointer");
  }
  if(Y->getNumElements() == 0) {
    throw std::runtime_error("Tucker::computeGram(const Tensor<scalar_t>* Y, const int n, scalar_t* gram, const int stride): Y->getNumElements() == 0");
  }
  if(stride < 1) {
    std::ostringstream oss;
    oss << "Tucker::computeGram(const Tensor<scalar_t>* Y, const int n, scalar_t* gram, "
        << "const int stride): stride = " << stride << " < 1";
    throw std::runtime_error(oss.str());
  }
  if(n < 0 || n >= Y->N()) {
    std::ostringstream oss;
    oss << "Tucker::computeGram(const Tensor<scalar_t>* Y, const int n, scalar_t* gram, "
        << "const int stride): n = " << n << " is not in the range [0,"
        << Y->N() << ")";
    throw std::runtime_error(oss.str());
  }

  int nrows = Y->size(n);

  // n = 0 is a special case
  // Y_0 is stored column major
  if(n == 0)
  {
    // Compute number of columns of Y_n
    // Technically, we could divide the total number of entries by n,
    // but that seems like a bad decision
    int ncols =1;
    for(int i=0; i<Y->N(); i++) {
      if(i != n) {
        ncols *= Y->size(i);
      }
    }

    // Call symmetric rank-k update
    // call syrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    // C := alpha*A*A' + beta*C
    char uplo = 'U';
    char trans = 'N';
    scalar_t alpha = 1;
    scalar_t beta = 0;
    syrk(&uplo, &trans, &nrows, &ncols, &alpha,
        Y->data(), &nrows, &beta, gram, &stride);
  }
  else
  {
    int ncols = 1;
    int nmats = 1;

    // Count the number of columns
    for(int i=0; i<n; i++) {
      ncols *= Y->size(i);
    }

    // Count the number of matrices
    for(int i=n+1; i<Y->N(); i++) {
      nmats *= Y->size(i);
    }

    // For each matrix...
    for(int i=0; i<nmats; i++) {
      // Call symmetric rank-k update
      // call dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
      // C := alpha*A'*A + beta*C
      char uplo = 'U';
      char trans = 'T';
      scalar_t alpha = 1;
      scalar_t beta;
      if(i==0)
        beta = 0;
      else
        beta = 1;
      syrk(&uplo, &trans, &nrows, &ncols, &alpha,
          Y->data()+i*nrows*ncols, &ncols, &beta,
          gram, &stride);
    }
  }
}

/// \example Tucker_eig_test.cpp
template<class scalar_t>
void computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues,
    const bool flipSign)
{
  if(G == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, const bool flipSign): G is a null pointer");
  }
  if(G->getNumElements() == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, const bool flipSign): G has no entries");
  }

  int nrows = G->nrows();
  eigenvalues = MemoryManager::safe_new_array<scalar_t>(nrows);

  // Compute the leading eigenvectors of S
  // call dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
  char jobz = 'V';
  char uplo = 'U';
  int lwork = 8*nrows;
  scalar_t* work = MemoryManager::safe_new_array<scalar_t>(lwork);
  int info;
  syev(&jobz, &uplo, &nrows, G->data(), &nrows,
      eigenvalues, work, &lwork, &info);

  // Check the error code
  if(info != 0)
    std::cerr << "Error: invalid error code returned by dsyev (" << info << ")\n";

  // The user will expect the eigenvalues to be sorted in descending order
  // LAPACK gives us the eigenvalues in ascending order
  for(int esubs=0; esubs<nrows-esubs-1; esubs++) {
    scalar_t temp = eigenvalues[esubs];
    eigenvalues[esubs] = eigenvalues[nrows-esubs-1];
    eigenvalues[nrows-esubs-1] = temp;
  }

  // Sort the eigenvectors too
  scalar_t* Gptr = G->data();
  const int ONE = 1;
  for(int esubs=0; esubs<nrows-esubs-1; esubs++) {
    Tucker::swap(&nrows, Gptr+esubs*nrows, &ONE,
        Gptr+(nrows-esubs-1)*nrows, &ONE);
  }

  // Flip the sign if necessary
  if(flipSign)
  {
    for(int c=0; c<nrows; c++)
    {
      int maxIndex=0;
      scalar_t maxVal = std::abs(Gptr[c*nrows]);
      for(int r=1; r<nrows; r++)
      {
        scalar_t testVal = std::abs(Gptr[c*nrows+r]);
        if(testVal > maxVal) {
          maxIndex = r;
          maxVal = testVal;
        }
      }

      if(Gptr[c*nrows+maxIndex] < 0) {
        const scalar_t NEGONE = -1;
        scal(&nrows, &NEGONE, Gptr+c*nrows, &ONE);
      }
    }
  }

  MemoryManager::safe_delete_array<scalar_t>(work,lwork);
}

template<class scalar_t>
void computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues,
    Matrix<scalar_t>*& eigenvectors, const int numEvecs, const bool flipSign)
{
  if(G == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const int numEvecs, const bool flipSign): G is a null pointer");
  }
  if(G->getNumElements() == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const int numEvecs, const bool flipSign): G has no entries");
  }
  if(numEvecs < 1) {
    std::ostringstream oss;
    oss << "Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, "
        << "Matrix<scalar_t>*& eigenvectors, const int numEvecs, "
        << "const bool flipSign): numEvecs = " << numEvecs << " < 1";
    throw std::runtime_error(oss.str());
  }
  if(numEvecs > G->nrows()) {
    std::ostringstream oss;
    oss << "Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, "
        << "Matrix<scalar_t>*& eigenvectors, const int numEvecs, "
        << "const bool flipSign): numEvecs = " << numEvecs
        << " > G->nrows() = " << G->nrows();
    throw std::runtime_error(oss.str());
  }

  computeEigenpairs(G, eigenvalues, flipSign);

  // Allocate memory for eigenvectors
  int numRows = G->nrows();
  eigenvectors = MemoryManager::safe_new<Matrix<scalar_t>>(numRows,numEvecs);

  // Copy appropriate eigenvectors
  int nToCopy = numRows*numEvecs;
  const int ONE = 1;
  Tucker::copy(&nToCopy, G->data(), &ONE, eigenvectors->data(), &ONE);
}

template<class scalar_t>
void computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues,
    Matrix<scalar_t>*& eigenvectors, const scalar_t thresh, const bool flipSign)
{
  if(G == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const int numEvecs, const bool flipSign): G is a null pointer");
  }
  if(G->getNumElements() == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const int numEvecs, const bool flipSign): G has no entries");
  }
  if(thresh < 0) {
    std::ostringstream oss;
    oss << "Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, "
        << "Matrix<scalar_t>*& eigenvectors, "
        << "const bool flipSign): thresh = " << thresh << " < 0";
    throw std::runtime_error(oss.str());
  }

  computeEigenpairs(G, eigenvalues, flipSign);

  // Compute number of things to copy
  int nrows = G->nrows();
  int numEvecs=nrows;
  scalar_t sum = 0;
  for(int i=nrows-1; i>=0; i--) {
    sum += std::abs(eigenvalues[i]);
    if(sum > thresh) {
      break;
    }
    numEvecs--;
  }

  // Allocate memory for eigenvectors
  int numRows = G->nrows();
  eigenvectors = MemoryManager::safe_new<Matrix<scalar_t>>(numRows,numEvecs);

  // Copy appropriate eigenvectors
  int nToCopy = numRows*numEvecs;
  const int ONE = 1;
  Tucker::copy(&nToCopy, G->data(), &ONE, eigenvectors->data(), &ONE);
}

template <class scalar_t>
void computeSVD(Matrix<scalar_t>* L, scalar_t* singularValues, 
  Matrix<scalar_t>* leftSingularVectors){
  if(L == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, const bool flipSign): G is a null pointer");
  }
  if(L->getNumElements() == 0) {
    throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, const bool flipSign): G has no entries");
  }
  char JOBU = 'A';
  char JOBVT = 'N';
  int Lnrows = L->nrows();
  int Lncols = L->ncols();
  scalar_t* VT = 0;
  scalar_t* work = Tucker::MemoryManager::safe_new_array<scalar_t>(1);
  const int negOne = -1; const int zero = 0; const int one = 1;
  int info;
  //workspace query
  gesvd(&JOBU, &JOBVT, &Lnrows, &Lncols, L->data(), &Lnrows, singularValues, 
    leftSingularVectors->data(), &Lnrows, VT, &one, work, &negOne, &info);
  if(info != 0){
    std::cout << "error in gesvd in computeSVD()" << std::endl;
  }
  int lwork = work[0];
  Tucker::MemoryManager::safe_delete_array(work, 1);
  work = Tucker::MemoryManager::safe_new_array<scalar_t>(lwork);
  gesvd(&JOBU, &JOBVT, &Lnrows, &Lncols, L->data(), &Lnrows, singularValues, 
    leftSingularVectors->data(), &Lnrows, VT, &one, work, &lwork, &info);
  Tucker::MemoryManager::safe_delete_array(work, lwork);
  if(info != 0){
    std::cout << "error in gesvd in computeSVD()" << std::endl;
  }
}

template <class scalar_t>
void computeSVD(Matrix<scalar_t>* L, scalar_t*& singularValues, 
  Matrix<scalar_t>*& leadingLeftSingularVectors, const scalar_t thresh){
    if(L == 0) {
      throw std::runtime_error("Tucker::computeSingularPairs(Matrix<scalar_t>* G, scalar_t*& singularValues, Matrix<scalar_t>*& singularVectors, const scalar_t thresh): L is a null pointer");
    }
    if(L->getNumElements() == 0) {
      throw std::runtime_error("Tucker::computeSingularPairs(Matrix<scalar_t>* G, scalar_t*& singularValues, Matrix<scalar_t>*& singularVectors, const scalar_t thresh): L has no entries");
    }
    if(thresh < 0) {
      std::ostringstream oss;
      oss << "Tucker::computeSingularPairs(Matrix<scalar_t>* G, scalar_t*& singularValues, "
          << "Matrix<scalar_t>*& singularVectors, const scalar_t thresh, "
          << "): thresh = " << thresh << " < 0";
      throw std::runtime_error(oss.str());
    }
    const int one = 1;
    int Lnrows = L->nrows();
    int Lncols = L->ncols();
    singularValues = Tucker::MemoryManager::safe_new_array<scalar_t>(std::min(Lnrows, Lncols));
    //TODO: Lnrows and Lncols should be the same as of now. If this remains so the min is useless.
    Matrix<scalar_t>* leftSingularVectors = Tucker::MemoryManager::safe_new<Matrix<scalar_t>>(Lnrows, Lnrows);
    computeSVD(L, singularValues, leftSingularVectors);
    int numSingularVector = Lnrows;
    scalar_t sum = 0;
    for(int i=Lnrows-1; i>=0; i--) {
      sum += std::pow(singularValues[i], 2);
      if(sum > thresh) {
        break;
      }
      numSingularVector--;
    }
    leadingLeftSingularVectors = Tucker::MemoryManager::safe_new<Matrix<scalar_t>>(Lnrows, numSingularVector);
    int nToCopy = Lnrows*numSingularVector;
    copy(&nToCopy, leftSingularVectors->data(), &one, leadingLeftSingularVectors->data(), &one);
    Tucker::MemoryManager::safe_delete(leftSingularVectors);
}

template <class scalar_t>
void computeSVD(Matrix<scalar_t>* L, scalar_t*& singularValues, 
  Matrix<scalar_t>*& leadingLeftSingularVectors, const int numSingularVector){
    if(L == 0) {
      throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const int numEvecs, const bool flipSign): G is a null pointer");
    }
    if(L->getNumElements() == 0) {
      throw std::runtime_error("Tucker::computeEigenpairs(Matrix<scalar_t>* G, scalar_t*& eigenvalues, Matrix<scalar_t>*& eigenvectors, const int numEvecs, const bool flipSign): G has no entries");
    }
    if(numSingularVector < 1) {
      std::ostringstream oss;
      oss << "Tucker::computeSingularPairs(Matrix<scalar_t>* L, scalar_t*& singularValues, "
          << "Matrix<scalar_t>*& singularVectors, const int numSingularVecotr, "
          << "): numSingularVecotr = " << numSingularVector << " < 1";
      throw std::runtime_error(oss.str());
    }
    if(numSingularVector > L->nrows()) {
      std::ostringstream oss;
      oss << "Tucker::computeSingularPairs(Matrix<scalar_t>* L, scalar_t*& singularValues, "
          << "Matrix<scalar_t>*& singularVectors, const int numSingularVecotr, "
          << "): numSingularVecotr = " << numSingularVector
          << " > L->nrows() = " << L->nrows();
      throw std::runtime_error(oss.str());
    }

    const int one = 1;
    int Lnrows = L->nrows();
    int Lncols = L->ncols();
    singularValues = Tucker::MemoryManager::safe_new_array<scalar_t>(std::min(Lnrows, Lncols));
    //TODO: Lnrows and Lncols should be the same as of now. If this remains so the min is useless.
    Matrix<scalar_t>* leftSingularVectors = Tucker::MemoryManager::safe_new<Matrix<scalar_t>>(Lnrows, Lnrows);
    computeSVD(L, singularValues, leftSingularVectors);
    leadingLeftSingularVectors = Tucker::MemoryManager::safe_new<Matrix<scalar_t>>(Lnrows, numSingularVector);
    int nToCopy = Lnrows*numSingularVector;
    copy(&nToCopy, leftSingularVectors->data(), &one, leadingLeftSingularVectors->data(), &one);
    Tucker::MemoryManager::safe_delete(leftSingularVectors);
}

template <class scalar_t>
const struct TuckerTensor<scalar_t>* STHOSVD(const Tensor<scalar_t>* X,
    const scalar_t epsilon, bool useQR, bool flipSign)
{
  if(X == 0) {
    throw std::runtime_error("Tucker::STHOSVD(const Tensor<scalar_t>* X, const scalar_t epsilon, bool flipSign): X is a null pointer");
  }
  if(X->getNumElements() == 0) {
    throw std::runtime_error("Tucker::STHOSVD(const Tensor<scalar_t>* X, const scalar_t epsilon, bool flipSign): X has no entries");
  }
  if(epsilon < 0) {
    std::ostringstream oss;
    oss << "Tucker::STHOSVD(const Tensor<scalar_t>* const X, const scalar_t epsilon, "
        << "bool flipSign): epsilon = " << epsilon << " < 0";
    throw std::runtime_error(oss.str());
  }
  int ndims = X->N();

  // Create a struct to store the factorization
  struct TuckerTensor<scalar_t>* factorization = MemoryManager::safe_new<TuckerTensor<scalar_t>>(ndims);
  factorization->total_timer_.start();

  // Compute the threshold
  scalar_t tensorNorm = X->norm2();
  scalar_t thresh = epsilon*epsilon*tensorNorm/X->N();
  std::cout << "\tAutoST-HOSVD::Tensor Norm: "
      << std::sqrt(tensorNorm) << "...\n";
  std::cout << "\tAutoST-HOSVD::Relative Threshold: "
      << thresh << "...\n";

  const Tensor<scalar_t>* Y = X;
  // For each dimension...
  for(int n=0; n<X->N(); n++){
    if(!useQR){
      // Compute the Gram matrix
      // S = Y_n*Y_n'
      std::cout << "\tAutoST-HOSVD::Starting Gram(" << n << ")...\n";
      factorization->gram_timer_[n].start();
      Matrix<scalar_t>* S = computeGram(Y,n);
      factorization->gram_timer_[n].stop();
      std::cout << "\tAutoST-HOSVD::Gram(" << n << ") time: "
          << factorization->gram_timer_[n].duration() << "s\n";
      // Compute the leading eigenvectors of S
      // call dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
      std::cout << "\tAutoST-HOSVD::Starting Evecs(" << n << ")...\n";
      factorization->eigen_timer_[n].start();
      computeEigenpairs(S, factorization->eigenvalues[n],
          factorization->U[n], thresh, flipSign);
      factorization->eigen_timer_[n].stop();
      {
        const int nrows = factorization->U[n]->nrows();
        factorization->singularValues[n] = MemoryManager::safe_new_array<scalar_t>(nrows);
        for (int i = 0; i < nrows; ++i) {
          if (factorization->eigenvalues[n][i] < static_cast<scalar_t>(0)) {
            factorization->eigenvalues[n][i] = static_cast<scalar_t>(0);
            factorization->singularValues[n][i] = static_cast<scalar_t>(0);
          } else {
            factorization->singularValues[n][i] = std::sqrt(factorization->eigenvalues[n][i]);
          }
        }
      }
      std::cout << std::endl;
      std::cout << "\tAutoST-HOSVD::EVECS(" << n << ") time: "
          << factorization->eigen_timer_[n].duration() << "s\n";

      // Free the Gram matrix
      MemoryManager::safe_delete(S);
    }
    else{
      std::cout << "\tAutoST-HOSVD::Starting LQ(" << n << ")...\n";
      factorization->LQ_timer_[n].start();
      Matrix<scalar_t>* L = computeLQ(Y, n);
      factorization->LQ_timer_[n].stop();
      std::cout << "\tAutoST-HOSVD::LQ(" << n << ") time: "
          << factorization->LQ_timer_[n].duration() << "s\n";
      std::cout << "\tAutoST-HOSVD::Starting SVD(" << n << ")...\n";
      factorization->svd_timer_[n].start();
      computeSVD(L, factorization->singularValues[n],
          factorization->U[n], thresh);
      factorization->svd_timer_[n].stop();
      std::cout << "\tAutoST-HOSVD::SVD(" << n << ") time: "
          << factorization->svd_timer_[n].duration() << "s\n";
      std::cout << std::endl;
      MemoryManager::safe_delete(L);
    }

    // Perform the tensor times matrix multiplication
    std::cout << "\tAutoST-HOSVD::Starting TTM(" << n << ")...\n";
    factorization->ttm_timer_[n].start();
    Tensor<scalar_t>* temp = ttm(Y,n,factorization->U[n],true);
    factorization->ttm_timer_[n].stop();
    std::cout << "\tAutoST-HOSVD::TTM(" << n << ") time: "
        << factorization->ttm_timer_[n].duration() << "s\n";
    if(n > 0) {
      MemoryManager::safe_delete<const Tensor<scalar_t>>(Y);
    }
    Y = temp;

    size_t nnz = Y->getNumElements();
    std::cout << "Local tensor size after STHOSVD iteration "
        << n << ": " << Y->size() << ", or ";
    Tucker::printBytes(nnz*sizeof(scalar_t));
  }
  

  factorization->G = const_cast<Tensor<scalar_t>*>(Y);
  factorization->total_timer_.stop();
  return factorization;
}


template <class scalar_t>
const struct TuckerTensor<scalar_t>* STHOSVD(const Tensor<scalar_t>* X,
    const SizeArray* reducedI, bool useQR, bool flipSign)
{
  if(X == 0) {
    throw std::runtime_error("Tucker::STHOSVD(const Tensor<scalar_t>* X, const SizeArray* reducedI, bool flipSign): X is a null pointer");
  }
  if(X->getNumElements() == 0) {
    throw std::runtime_error("Tucker::STHOSVD(const Tensor<scalar_t>* X, const SizeArray* reducedI, bool flipSign): X has no entries");
  }
  if(X->N() != reducedI->size()) {
    std::ostringstream oss;
    oss << "Tucker::STHOSVD(const Tensor<scalar_t>* X, const SizeArray* reducedI, "
        << "bool flipSign): X->N() = " << X->N()
        << " != reducedI->size() = " << reducedI->size();
    throw std::runtime_error(oss.str());
  }
  for(int i=0; i<reducedI->size(); i++) {
    if((*reducedI)[i] <= 0) {
      std::ostringstream oss;
      oss << "Tucker::STHOSVD(const Tensor<scalar_t>* X, const SizeArray* reducedI, "
      << "bool flipSign): reducedI[" << i << "] = " << (*reducedI)[i]
      << " <= 0";
      throw std::runtime_error(oss.str());
    }
    if((*reducedI)[i] > X->size(i)) {
      std::ostringstream oss;
      oss << "Tucker::STHOSVD(const Tensor<scalar_t>* X, const SizeArray* reducedI, "
      << "bool flipSign): reducedI[" << i << "] = " << (*reducedI)[i]
      << " > X->size(" << i << ") = " << X->size(i);
      throw std::runtime_error(oss.str());
    }
  }

  // Create a struct to store the factorization
  struct TuckerTensor<scalar_t>* factorization = MemoryManager::safe_new<TuckerTensor<scalar_t>>(X->N());
  factorization->total_timer_.start();

  const Tensor<scalar_t>* Y = X;
  // For each dimension...
  for(int n=0; n<X->N(); n++)
  {
    if(useQR){
      factorization->LQ_timer_[n].start();
      Matrix<scalar_t>* L = computeLQ(Y, n);
      factorization->LQ_timer_[n].stop();
      factorization->svd_timer_[n].start();
      computeSVD(L, factorization->singularValues[n],
          factorization->U[n], (*reducedI)[n]);
      factorization->svd_timer_[n].stop();
      MemoryManager::safe_delete(L);
    }
    else{
      // Compute the Gram matrix
      // S = Y_n*Y_n'
      factorization->gram_timer_[n].start();
      Matrix<scalar_t>* S = computeGram(Y,n);
      factorization->gram_timer_[n].stop();

      // Compute the leading eigenvectors of S
      // call dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
      factorization->eigen_timer_[n].start();
      computeEigenpairs(S, factorization->eigenvalues[n],
          factorization->U[n], (*reducedI)[n], flipSign);
      factorization->eigen_timer_[n].stop();

      MemoryManager::safe_delete(S);
    }
    // Perform the tensor times matrix multiplication
    factorization->ttm_timer_[n].start();
    Tensor<scalar_t>* temp = ttm(Y,n,factorization->U[n],true);
    factorization->ttm_timer_[n].stop();
    if(n > 0) {
      MemoryManager::safe_delete<const Tensor<scalar_t>>(Y);
    }
    Y = temp;
  }

  factorization->G = const_cast<Tensor<scalar_t>*>(Y);
  factorization->total_timer_.stop();
  return factorization;
}


template <class scalar_t>
Tensor<scalar_t>* ttm(const Tensor<scalar_t>* X, const int n,
    const Matrix<scalar_t>* U, bool Utransp)
{
  if(X == 0) {
    throw std::runtime_error("Tucker::ttm(const Tensor<scalar_t>* X, const int n, const Matrix<scalar_t>* U, bool Utransp): X is a null pointer");
  }
  if(X->getNumElements() == 0) {
    throw std::runtime_error("Tucker::ttm(const Tensor<scalar_t>* X, const int n, const Matrix<scalar_t>* U, bool Utransp): X has no entries");
  }
  if(U == 0) {
    throw std::runtime_error("Tucker::ttm(const Tensor<scalar_t>* X, const int n, const Matrix<scalar_t>* U, bool Utransp): U is a null pointer");
  }
  if(U->getNumElements() == 0) {
    throw std::runtime_error("Tucker::ttm(const Tensor<scalar_t>* X, const int n, const Matrix<scalar_t>* U, bool Utransp): U has no entries");
  }
  if(n < 0 || n >= X->N()) {
    std::ostringstream oss;
    oss << "Tucker::ttm(const Tensor<scalar_t>* X, const int n, const Matrix<scalar_t>* U, "
        << "bool Utransp): n = " << n << " is not in the range [0,"
        << X->N() << ")";
    throw std::runtime_error(oss.str());
  }
  if(!Utransp && U->ncols() != X->size(n)) {
    std::ostringstream oss;
    // TODO: amk Oct 17 2016 Finish adding exceptions to this file
  }

  // Compute the number of rows for the resulting "matrix"
  int nrows;
  if(Utransp)
    nrows = U->ncols();
  else
    nrows = U->nrows();

  // Allocate space for the new tensor
  SizeArray I(X->N());
  for(int i=0; i<I.size(); i++) {
    if(i != n) {
      I[i] = X->size(i);
    }
    else {
      I[i] = nrows;
    }
  }
  Tensor<scalar_t>* Y = MemoryManager::safe_new<Tensor<scalar_t>>(I);

  // Call TTM
  ttm(X, n, U, Y, Utransp);

  // Return the tensor
  return Y;
}


template <class scalar_t>
Tensor<scalar_t>* ttm(const Tensor<scalar_t>* const X, const int n,
    const scalar_t* const Uptr, const int dimU,
    const int strideU, bool Utransp)
{
  // Allocate space for the new tensor
  SizeArray I(X->N());
  for(int i=0; i<I.size(); i++) {
    if(i != n) {
      I[i] = X->size(i);
    }
    else {
      I[i] = dimU;
    }
  }
  Tensor<scalar_t>* Y = MemoryManager::safe_new<Tensor<scalar_t>>(I);

  // Call TTM
  ttm(X, n, Uptr, strideU, Y, Utransp);

  // Return the tensor
  return Y;
}


template <class scalar_t>
void ttm(const Tensor<scalar_t>* const X, const int n,
    const Matrix<scalar_t>* const U, Tensor<scalar_t>* Y, bool Utransp)
{
  // Check that the input is valid
  assert(U != 0);
  if(Utransp) {
    assert(U->nrows() == X->size(n));
    assert(U->ncols() == Y->size(n));
  }
  else {
    assert(U->ncols() == X->size(n));
    assert(U->nrows() == Y->size(n));
  }
  ttm(X, n, U->data(), U->nrows(), Y, Utransp);
}


template <class scalar_t>
void ttm(const Tensor<scalar_t>* const X, const int n,
    const scalar_t* const Uptr, const int strideU,
    Tensor<scalar_t>* Y, bool Utransp)
{
  // Check that the input is valid
  assert(X != 0);
  assert(Uptr != 0);
  assert(Y != 0);
  assert(n >= 0 && n < X->N());
  for(int i=0; i<X->N(); i++) {
    if(i != n) {
      assert(X->size(i) == Y->size(i));
    }
  }

  // Obtain the number of rows and columns of U
  int Unrows, Uncols;
  if(Utransp) {
    Unrows = X->size(n);
    Uncols = Y->size(n);
  }
  else {
    Uncols = X->size(n);
    Unrows = Y->size(n);
  }

  // n = 0 is a special case
  // Y_0 is stored column major
  if(n == 0)
  {
    // Compute number of columns of Y_n
    // Technically, we could divide the total number of entries by n,
    // but that seems like a bad decision
    size_t ncols = X->size().prod(1,X->N()-1);

    if(ncols > std::numeric_limits<int>::max()) {
      std::ostringstream oss;
      oss << "Error in Tucker::ttm: " << ncols
          << " is larger than std::numeric_limits<int>::max() ("
          << std::numeric_limits<int>::max() << ")";
      throw std::runtime_error(oss.str());
    }

    // Call matrix matrix multiply
    // call gemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
    // C := alpha*op( A )*op( B ) + beta*C
    // A, B and C are matrices, with op( A ) an m by k matrix,
    // op( B ) a k by n matrix and C an m by n matrix.
    char transa;
    char transb = 'N';
    int m = Y->size(n);
    int blas_n = (int)ncols;
    int k = X->size(n);
    int lda = strideU;
    int ldb = k;
    int ldc = m;
    scalar_t alpha = 1;
    scalar_t beta = 0;

    if(Utransp) {
      transa = 'T';
    } else {
      transa = 'N';
    }
    gemm(&transa, &transb, &m, &blas_n, &k, &alpha, Uptr,
        &lda, X->data(), &ldb, &beta, Y->data(), &ldc);
  }
  else
  {
    // Count the number of columns
    size_t ncols = X->size().prod(0,n-1);

    // Count the number of matrices
    size_t nmats = X->size().prod(n+1,X->N()-1,1);

    if(ncols > std::numeric_limits<int>::max()) {
      std::ostringstream oss;
      oss << "Error in Tucker::ttm: " << ncols
          << " is larger than std::numeric_limits<int>::max() ("
          << std::numeric_limits<int>::max() << ")";
      throw std::runtime_error(oss.str());
    }

    // For each matrix...
    for(size_t i=0; i<nmats; i++) {
      // Call matrix matrix multiply
      // call dgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)
      // C := alpha*op( A )*op( B ) + beta*C
      // A, B and C are matrices, with op( A ) an m by k matrix,
      // op( B ) a k by n matrix and C an m by n matrix.
      char transa = 'N';
      char transb;
      int m = (int)ncols;
      int blas_n = Y->size(n);
      int k;
      int lda = (int)ncols;
      int ldb = strideU;
      int ldc = (int)ncols;
      scalar_t alpha = 1;
      scalar_t beta = 0;
      if(Utransp) {
        transb = 'N';
        k = Unrows;
      } else {
        transb = 'T';
        k = Uncols;
      }
      gemm(&transa, &transb, &m, &blas_n, &k, &alpha,
          X->data()+i*k*m, &lda, Uptr, &ldb, &beta,
          Y->data()+i*m*blas_n, &ldc);
    }
  }
}


template <class scalar_t>
MetricData<scalar_t>* computeSliceMetrics(const Tensor<scalar_t>* Y, const int mode, const int metrics)
{
  // If there are no slices, calling this function was a bad idea
  int numSlices = Y->size(mode);
  if(numSlices <= 0) {
    std::ostringstream oss;
    oss << "Tucker::computeSliceMetrics(const Tensor<scalar_t>* Y, const int mode, const int metrics): "
        << "numSlices = " << numSlices << " <= 0";
    throw std::runtime_error(oss.str());
  }

  // Allocate memory for the result
  MetricData<scalar_t>* result = MemoryManager::safe_new<MetricData<scalar_t>>(metrics, numSlices);

  // Initialize the result
  scalar_t* delta;
  int* nArray;
  if((metrics & MEAN) || (metrics & VARIANCE)) {
    delta = MemoryManager::safe_new_array<scalar_t>(numSlices);
    nArray = MemoryManager::safe_new_array<int>(numSlices);
  }
  for(int i=0; i<numSlices; i++) {
    if(metrics & MIN) {
      result->getMinData()[i] = std::numeric_limits<scalar_t>::max();
    }
    if(metrics & MAX) {
      result->getMaxData()[i] = std::numeric_limits<scalar_t>::lowest();
    }
    if(metrics & SUM) {
      result->getSumData()[i] = 0;
    }
    if((metrics & MEAN) || (metrics & VARIANCE)) {
      result->getMeanData()[i] = 0;
      nArray[i] = 0;
    }
    if(metrics & VARIANCE) {
      result->getVarianceData()[i] = 0;
    }
  } // end for(int i=0; i<numSlices; i++)

  if(Y->getNumElements() == 0) {
    if((metrics & MEAN) || (metrics & VARIANCE)) {
      MemoryManager::safe_delete_array<scalar_t>(delta,numSlices);
      MemoryManager::safe_delete_array<int>(nArray,numSlices);
    }
    return result;
  }

  // Compute the result
  int ndims = Y->N();
  size_t numContig = Y->size().prod(0,mode-1,1); // Number of contiguous elements in a slice
  size_t numSetsContig = Y->size().prod(mode+1,ndims-1,1); // Number of sets of contiguous elements per slice
  size_t distBetweenSets = Y->size().prod(0,mode); // Distance between sets of contiguous elements

  const scalar_t* dataPtr;
  int slice;
  size_t i, c;
  #pragma omp parallel for default(shared) private(slice,i,c,dataPtr)
  for(slice=0; slice<numSlices; slice++)
  {
    dataPtr = Y->data() + slice*numContig;
    for(c=0; c<numSetsContig; c++)
    {
      for(i=0; i<numContig; i++)
      {
        if(metrics & MIN) {
          result->getMinData()[slice] = std::min(result->getMinData()[slice],dataPtr[i]);
        }
        if(metrics & MAX) {
          result->getMaxData()[slice] = std::max(result->getMaxData()[slice],dataPtr[i]);
        }
        if(metrics & SUM) {
          result->getSumData()[slice] += dataPtr[i];
        }
        if((metrics & MEAN) || (metrics & VARIANCE)) {
          delta[slice] = dataPtr[i] - result->getMeanData()[slice];
          nArray[slice]++;
          result->getMeanData()[slice] += (delta[slice]/nArray[slice]);
        }
        if(metrics & VARIANCE) {
          result->getVarianceData()[slice] +=
              (delta[slice]*(dataPtr[i]-result->getMeanData()[slice]));
        }
      } // end for(i=0; i<numContig; i++)
      dataPtr += distBetweenSets;
    } // end for(c=0; c<numSetsContig; c++)
  } // end for(slice=0; slice<numSlices; slice++)

  if((metrics & MEAN) || (metrics & VARIANCE)) {
    MemoryManager::safe_delete_array<scalar_t>(delta,numSlices);
    MemoryManager::safe_delete_array<int>(nArray,numSlices);
  }
  if(metrics & VARIANCE) {
    size_t sizeOfSlice = numContig*numSetsContig;
    for(int i=0; i<numSlices; i++) {
      result->getVarianceData()[i] /= (scalar_t)sizeOfSlice;
    }
  }

  return result;
}


// Shift is applied before scale
// We divide by scaleVals, not multiply
template <class scalar_t>
void transformSlices(Tensor<scalar_t>* Y, int mode, const scalar_t* scales, const scalar_t* shifts)
{
  // If the tensor has no entries, no transformation is necessary
  size_t numEntries = Y->getNumElements();
  if(numEntries == 0)
    return;

  // Compute the result
  int ndims = Y->N();
  int numSlices = Y->size(mode);
  size_t numContig = Y->size().prod(0,mode-1,1); // Number of contiguous elements in a slice
  size_t numSetsContig = Y->size().prod(mode+1,ndims-1,1); // Number of sets of contiguous elements per slice
  size_t distBetweenSets = Y->size().prod(0,mode); // Distance between sets of contiguous elements

  scalar_t* dataPtr;
  int slice;
  size_t i, c;
  #pragma omp parallel for default(shared) private(slice,i,c,dataPtr)
  for(slice=0; slice<numSlices; slice++)
  {
    dataPtr = Y->data() + slice*numContig;
    for(c=0; c<numSetsContig; c++)
    {
      for(i=0; i<numContig; i++)
        dataPtr[i] = (dataPtr[i] + shifts[slice]) / scales[slice];
      dataPtr += distBetweenSets;
    }
  }
}


template <class scalar_t>
void normalizeTensorMinMax(Tensor<scalar_t>* Y, int mode, const char* scale_file)
{
  MetricData<scalar_t>* metrics = computeSliceMetrics(Y, mode, MAX+MIN);
  int sizeOfModeDim = Y->size(mode);
  scalar_t* scales = MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  scalar_t* shifts = MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scales[i] = metrics->getMaxData()[i] - metrics->getMinData()[i];
    shifts[i] = -metrics->getMinData()[i];
  }
  transformSlices(Y,mode,scales,shifts);
  if(scale_file) writeScaleShift(mode,sizeOfModeDim,scales,shifts,scale_file);
  MemoryManager::safe_delete_array(scales,sizeOfModeDim);
  MemoryManager::safe_delete_array(shifts,sizeOfModeDim);
  MemoryManager::safe_delete(metrics);
}

// \todo This function is not being tested at all
template <class scalar_t>
void normalizeTensorMax(Tensor<scalar_t>* Y, int mode, const char* scale_file)
{
  Tucker::MetricData<scalar_t>* metrics = computeSliceMetrics(Y, mode,
      Tucker::MIN + Tucker::MAX);
  int sizeOfModeDim = Y->size(mode);
  scalar_t* scales = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  scalar_t* shifts = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scalar_t scaleval = std::max(std::abs(metrics->getMinData()[i]),
        std::abs(metrics->getMaxData()[i]));
    scales[i] = scaleval;
    shifts[i] = 0;
  }
  transformSlices(Y,mode,scales,shifts);
  if(scale_file) writeScaleShift(mode,sizeOfModeDim,scales,shifts,scale_file);
  Tucker::MemoryManager::safe_delete_array(scales,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete_array(shifts,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete(metrics);
}


template <class scalar_t>
void normalizeTensorStandardCentering(Tensor<scalar_t>* Y, int mode, scalar_t stdThresh, const char* scale_file)
{
  MetricData<scalar_t>* metrics = computeSliceMetrics(Y, mode, MEAN+VARIANCE);
  int sizeOfModeDim = Y->size(mode);
  scalar_t* scales = MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  scalar_t* shifts = MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scales[i] = sqrt(metrics->getVarianceData()[i]);
    shifts[i] = -metrics->getMeanData()[i];

    if(scales[i] < stdThresh) {
      scales[i] = 1;
    }
  }
  transformSlices(Y,mode,scales,shifts);
  if(scale_file) writeScaleShift(mode,sizeOfModeDim,scales,shifts,scale_file);
  MemoryManager::safe_delete_array(scales,sizeOfModeDim);
  MemoryManager::safe_delete_array(shifts,sizeOfModeDim);
  MemoryManager::safe_delete(metrics);
}

// \todo This function is not being tested
template <class scalar_t>
void writeScaleShift(const int mode, const int sizeOfModeDim, const scalar_t* scales,
    const scalar_t* shifts, const char* scale_file)
{
  std::ofstream outStream(scale_file);

  outStream << mode << std::endl;

  // Set output precision to match scalar_t representation (8 or 16)
  outStream << std::fixed << std::setprecision(std::numeric_limits<scalar_t>::digits);
  for(int i=0; i<sizeOfModeDim; i++)
  {
    outStream << scales[i] << " " << shifts[i] << std::endl;
  }

  outStream.close();
}

// \todo This function is not being tested
template <class scalar_t>
void readTensorBinary(Tensor<scalar_t>* Y, const char* filename)
{
  // Count the number of filenames
  std::ifstream inStream(filename);

  std::string temp;
  int nfiles = 0;
  while(inStream >> temp) {
    nfiles++;
  }

  inStream.close();

  if(nfiles == 1) {
    importTensorBinary(Y,temp.c_str());
  }
  else {
    int ndims = Y->N();
    if(nfiles != Y->size(ndims-1)) {
      std::ostringstream oss;
      oss << "Tucker::readTensorBinary(Tensor<scalar_t>* Y, const char* filename: "
          << "The number of filenames you provided is "
          << nfiles << ", but the dimension of the tensor's last mode is "
          << Y->size(ndims-1);

      throw std::runtime_error(oss.str());
    }
    importTimeSeries(Y,filename);
  }
}


template <class scalar_t>
Tensor<scalar_t>* importTensor(const char* filename)
{
  // Open file
  std::ifstream ifs;
  ifs.open(filename);
  assert(ifs.is_open());

  // Read the type of object
  // If the type is not "tensor", that's bad
  std::string tensorStr;
  ifs >> tensorStr;
  assert(tensorStr == "tensor" || tensorStr == "matrix");

  // Read the number of dimensions
  int ndims;
  ifs >> ndims;

  // Create a SizeArray of that length
  SizeArray sz(ndims);

  // Read the dimensions
  for(int i=0; i<ndims; i++) {
    ifs >> sz[i];
  }

  // Create a tensor using that SizeArray
  Tensor<scalar_t>* t = MemoryManager::safe_new<Tensor<scalar_t>>(sz);

  // Read the entries of the tensor
  size_t numEntries = sz.prod();
  scalar_t * data = t->data();
  for(size_t i=0; i<numEntries; i++) {
    ifs >> data[i];
  }

  // Close the file
  ifs.close();

  // Return the tensor
  return t;
}


template <class scalar_t>
void importTensorBinary(Tensor<scalar_t>* t, const char* filename)
{
  // Get the maximum file size we can read
  const std::streamoff MAX_OFFSET =
      std::numeric_limits<std::streamoff>::max();
//  std::cout << "The maximum file size is " << MAX_OFFSET << " bytes\n";

  // Open file
  std::ifstream ifs;
  ifs.open(filename, std::ios::in | std::ios::binary);
  assert(ifs.is_open());

  // Get the size of the file
  std::streampos begin, end, size;
  begin = ifs.tellg();
  ifs.seekg(0, std::ios::end);
  end = ifs.tellg();
  size = end - begin;
  //std::cout << "Reading " << size << " bytes...\n";

  // Assert that this size is consistent with the number of tensor entries
  size_t numEntries = t->getNumElements();
  //std::cout << "Size is: "<< size << "bytes" << std::endl;
  //std::cout << "should be " << numEntries*sizeof(double) << "bytes." << std::endl;
  assert(size == numEntries*sizeof(scalar_t));

  // Read the file
  scalar_t* data = t->data();
  ifs.seekg(0, std::ios::beg);
  ifs.read((char*)data,size);

  // Close the file
  ifs.close();
}

// \todo This function never gets tested
template <class scalar_t>
void importTimeSeries(Tensor<scalar_t>* Y, const char* filename)
{
   // Open the file
   std::ifstream ifs;
   ifs.open(filename);

   // Define data layout parameters
   int ndims = Y->N();

   int nsteps = Y->size(ndims-1);
   scalar_t* dataPtr = Y->data();
   size_t count = Y->size().prod(0,ndims-2);
   assert(count <= std::numeric_limits<int>::max());

   for(int step=0; step<nsteps; step++) {
     std::string stepFilename;
     ifs >> stepFilename;
     std::cout << "Reading file " << stepFilename << std::endl;

     std::ifstream bifs;
     bifs.open(stepFilename.c_str(), std::ios::in | std::ios::binary);
     assert(bifs.is_open());

     // Get the size of the file
     std::streampos begin, end, size;
     begin = bifs.tellg();
     bifs.seekg(0, std::ios::end);
     end = bifs.tellg();
     size = end - begin;

     // Assert that this size is consistent with the number of tensor entries
     //HKsize_t numEntries = Y->getNumElements();
     //HKassert(size == numEntries*sizeof(scalar_t));
     assert(size == count*sizeof(scalar_t));

     // Read the file
     bifs.seekg(0, std::ios::beg);
     bifs.read((char*)dataPtr,size);

     bifs.close();

     // Increment the pointer
     dataPtr += count;
   }

   ifs.close();
}


template <class scalar_t>
Matrix<scalar_t>* importMatrix(const char* filename)
{
  // Open file
  std::ifstream ifs;
  ifs.open(filename);
  assert(ifs.is_open());

  // Read the type of object
  // If the type is not "tensor", that's bad
  std::string tensorStr;
  ifs >> tensorStr;
  assert(tensorStr == "tensor" || tensorStr == "matrix");

  // Read the number of dimensions
  int ndims;
  ifs >> ndims;
  assert(ndims == 2);

  // Read the dimensions
  int nrows, ncols;
    ifs >> nrows >> ncols;

  // Create a matrix
  Matrix<scalar_t>* m = MemoryManager::safe_new<Matrix<scalar_t>>(nrows,ncols);

  // Read the entries of the tensor
  size_t numEntries = nrows*ncols;
  scalar_t * data = m->data();
  for(size_t i=0; i<numEntries; i++) {
    ifs >> data[i];
  }

  // Close the file
  ifs.close();

  // Return the tensor
  return m;
}


template <class scalar_t>
SparseMatrix<scalar_t>* importSparseMatrix(const char* filename)
{
  // Open file
  std::ifstream ifs;
  ifs.open(filename);
  assert(ifs.is_open());

  // Read the type of object
  // If the type is not "sptensor", that's bad
  std::string tensorStr;
  ifs >> tensorStr;
  assert(tensorStr == "sptensor");

  // Read the number of dimensions
  int ndims;
  ifs >> ndims;
  assert(ndims == 2);

  // Read the dimensions
  int nrows, ncols, nnz;
    ifs >> nrows >> ncols >> nnz;

  // Create a matrix
  SparseMatrix<scalar_t>* m = MemoryManager::safe_new<SparseMatrix<scalar_t>>(nrows,ncols,nnz);

  // Read the entries of the tensor
  int* rows = m->rows();
  int* cols = m->cols();
  scalar_t* vals = m->vals();
  for(size_t i=0; i<nnz; i++) {
    ifs >> rows[i] >> cols[i] >> vals[i];
    rows[i]--;
    cols[i]--;
  }

  // Close the file
  ifs.close();

  // Return the sparse matrix
  return m;
}

// \todo This function never gets tested
template <class scalar_t>
void writeTensorBinary(const Tensor<scalar_t>* Y, const char* filename)
{
  // Count the number of filenames
   std::ifstream inStream(filename);

   std::string temp;
   int nfiles = 0;
   while(inStream >> temp) {
     nfiles++;
   }

   inStream.close();

   if(nfiles == 1) {
     exportTensorBinary(Y,temp.c_str());
   }
   else {
     int ndims = Y->N();
     if(nfiles != Y->size(ndims-1)) {
       std::ostringstream oss;
       oss << "Tucker::writeTensorBinary(const Tensor<scalar_t>* Y, const char* filename: "
           << "The number of filenames you provided is "
           << nfiles << ", but the dimension of the tensor's last mode is "
           << Y->size(ndims-1);

       throw std::runtime_error(oss.str());
     }
     exportTimeSeries(Y,filename);
   }
}

// \todo This function never gets tested
template <class scalar_t>
void exportTensor(const Tensor<scalar_t>* Y, const char* filename)
{
  // Open the file
  std::ofstream ofs;
  ofs.open(filename);

  // Write the type of object
  ofs << "tensor\n";

  // Write the number of dimensions of the tensor
  int ndims = Y->size().size();
  ofs << ndims << std::endl;

  // Write the size of each dimension
  for(int i=0; i<ndims; i++) {
    ofs << Y->size(i) << " ";
  }
  ofs << std::endl;

  // Write the elements of the tensor
  size_t numEntries = Y->size().prod();
  const scalar_t* data = Y->data();
  for(size_t i=0; i<numEntries; i++) {
    ofs << data[i] << std::endl;
  }

  // Close the file
  ofs.close();
}


template <class scalar_t>
void exportTensorBinary(const Tensor<scalar_t>* Y, const char* filename)
{
  // Get the maximum file size we can write
  const std::streamoff MAX_OFFSET =
      std::numeric_limits<std::streamoff>::max();
//  std::cout << "The maximum file size is " << MAX_OFFSET << " bytes\n";

  // Determine how many bytes we are writing
  size_t numEntries = Y->getNumElements();
//  std::cout << "Writing " << numEntries*sizeof(scalar_t) << " bytes...\n";

  // Open file
  std::ofstream ofs;
  ofs.open(filename, std::ios::out | std::ios::binary);
  assert(ofs.is_open());

  // Write the file
  const scalar_t* data = Y->data();
  ofs.write((char*)data,numEntries*sizeof(scalar_t));

  // Close the file
  ofs.close();
}

// \todo This function never gets tested
template <class scalar_t>
void exportTimeSeries(const Tensor<scalar_t>* Y, const char* filename)
{
  // Open the file
  std::ifstream ifs;
  ifs.open(filename);

  // Determine how many bytes we are writing per file
  int N = Y->N();
  size_t numEntriesPerTimestep = Y->size().prod(0,N-2);

  int nsteps = Y->size(N-1);
  size_t offset = 0;
  for(int step=0; step<nsteps; step++) {
    std::string stepFilename;
    ifs >> stepFilename;
    std::cout << "Writing file " << stepFilename << std::endl;

    std::ofstream ofs;
    ofs.open(stepFilename.c_str(), std::ios::out | std::ios::binary);
    assert(ofs.is_open());

    const scalar_t* data = Y->data() + offset;
    ofs.write((char*)data,numEntriesPerTimestep*sizeof(scalar_t));
    ofs.close();

    offset += numEntriesPerTimestep;
  }
}


template <class scalar_t>
void premultByDiag(const Vector<scalar_t>* diag, Matrix<scalar_t>* mat)
{
  scalar_t* mydata = mat->data();
  int myrows = mat->nrows();
  int mycols = mat->ncols();

  assert(myrows == diag->nrows());

  for(int r=0; r<myrows; r++) {
    for(int c=0; c<mycols; c++) {
      mydata[r+c*myrows] *= (*diag)[r];
    }
  }
}

template <class scalar_t>
void padToSquare(Matrix<scalar_t>*& R)
{
  int one = 1;
  int Rncols = R->ncols();
  int Rnrows = R->nrows();
  Tucker::Matrix<scalar_t>* squareR = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(Rncols, Rncols);
  int sizeOfSquareR = Rncols*Rncols;
  for(int i=0; i<Rncols; i++){
    //copy the top part over
    Tucker::copy(&Rnrows, R->data()+i*Rnrows, &one, squareR->data()+i*Rncols, &one); 
    //padding with zeros
    for(int j=i*Rncols+Rnrows; j<(i+1)*Rncols; j++){
      squareR->data()[j] = 0;
    }
  }
  Tucker::MemoryManager::safe_delete(R);
  R = squareR;
}
template <class scalar_t>
Tensor<scalar_t> *padTensorAlongMode(const Tensor<scalar_t>* X, int n, int p) {
  const std::string method_signature = "padTensorAlongMode(const Tensor<scalar_t> *X, int n, int p)";

  if (X == nullptr) {
    std::ostringstream str;
    str << method_signature << ": pointer to tensor X cannot be null";
    throw std::invalid_argument(str.str());
  }

  const int d = X->N();

  if (n < 0 || n >= d) {
    std::ostringstream str;
    str << method_signature << ": mode index n is out of range for dimensionality of tensor X";
    throw std::invalid_argument(str.str());
  }

  if (p <= 0) {
    std::ostringstream str;
    str << method_signature << ": number of additional zero slices p must be positive";
    throw std::invalid_argument(str.str());
  }

  const int nrow1 = X->size().prod(0, n);
  const int nrow2 = X->size().prod(0, n - 1) * p;
  const int nrow = nrow1 + nrow2;
  const int ncol = X->size().prod(n + 1, d - 1);

  Tensor<scalar_t> *X_new;
  {
    SizeArray size(d);
    for (int k = 0; k < d; ++k) {
        size[k] = (n == k) ? X->size(k) + p : X->size(k);
    }
    X_new = MemoryManager::safe_new<Tensor<scalar_t>>(size);
  }
  for (int j = 0; j < ncol; ++j) {
    const scalar_t zero = 0;
    const int incr0 = 0;
    const int incr1 = 1;

    copy(&nrow1, X->data() + j * nrow1, &incr1, X_new->data() + j * nrow, &incr1);
    copy(&nrow2, &zero, &incr0, X_new->data() + j * nrow + nrow1, &incr1);
  }

  return X_new;
}

template <class scalar_t>
Tensor<scalar_t> *concatenateTensorsAlongMode(const Tensor<scalar_t>* X, const Tensor<scalar_t>* Y, int n) {
  const std::string method_signature = "concatenateTensorsAlongMode(const Tensor<scalar_t>* X, const Tensor<scalar_t>* Y, int n)";

  if (X == nullptr || Y == nullptr) {
    std::ostringstream str;
    str << method_signature << ": pointers to tensors X and Y cannot be null";
    throw std::invalid_argument(str.str());
  }

  const int d = X->N();
  if (Y->N() != d) {
    std::ostringstream str;
    str << method_signature << ": tensors X and Y must have the same dimensionality";
    throw std::invalid_argument(str.str());
  }

  if (n < 0 || n >= d) {
    std::ostringstream str;
    str << method_signature << ": mode index n is out of range for dimensionality of tensors X and Y";
    throw std::invalid_argument(str.str());
  }

  SizeArray size(d);
  for (int k = 0; k < d; ++k) {
    if (k != n) {
      if (X->size(k) != Y->size(k)) {
        std::ostringstream str;
        str << method_signature << ": tensors X and Y must have the same modes sizes except along mode n";
        throw std::invalid_argument(str.str());
      }
      size[k] = X->size(k);
    } else {
      size[k] = X->size(k) + Y->size(k);
    }
  }

  Tensor<scalar_t> *Z = MemoryManager::safe_new<Tensor<scalar_t>>(size);

  const int nrowx = X->size().prod(0, n);
  const int nrowy = Y->size().prod(0, n);
  const int nrowz = nrowx + nrowy;
  const int ncolz = X->size().prod(n + 1, d - 1);

  for (int j = 0; j < ncolz; ++j) {
    const int incr = 1;
    copy(&nrowx, X->data() + j * nrowx, &incr, Z->data() + j * nrowz, &incr);
    copy(&nrowy, Y->data() + j * nrowy, &incr, Z->data() + j * nrowz + nrowx, &incr);
  }

  return Z;
}

// Explicit instantiations to build static library for both single and double precision
template Matrix<float>* computeGram(const Tensor<float>* Y, const int n);
template void computeGram(const Tensor<float>*, const int, float*, const int);
template void computeEigenpairs(Matrix<float>*, float*&, const bool);
template void computeEigenpairs(Matrix<float>*, float*&, Matrix<float>*&, const int, const bool);
template void computeEigenpairs(Matrix<float>*, float*&, Matrix<float>*&, const float, const bool);
template const struct TuckerTensor<float>* STHOSVD(const Tensor<float>*, const float, bool, bool);
template const struct TuckerTensor<float>* STHOSVD(const Tensor<float>*, const SizeArray*, bool, bool);
template Tensor<float>* ttm(const Tensor<float>*, const int, const Matrix<float>*, bool);
template Tensor<float>* ttm(const Tensor<float>* const, const int, const float* const, const int, const int, bool);
template void ttm(const Tensor<float>* const, const int, const Matrix<float>* const, Tensor<float>*, bool);
template void ttm(const Tensor<float>* const, const int, const float* const, const int, Tensor<float>*, bool);
template MetricData<float>* computeSliceMetrics(const Tensor<float>*, const int, const int);
template void transformSlices(Tensor<float>*, int, const float*, const float*);
template void normalizeTensorMinMax(Tensor<float>*, int, const char*);
template void normalizeTensorMax(Tensor<float>*, int, const char*);
template void normalizeTensorStandardCentering(Tensor<float>*, int, float, const char*);
template void writeScaleShift(const int, const int, const float*, const float*, const char*);
template void readTensorBinary(Tensor<float>* Y, const char* filename);
template Tensor<float>* importTensor(const char*);
template void importTensorBinary(Tensor<float>*, const char*);
template void importTimeSeries(Tensor<float>*, const char*);
template Matrix<float>* importMatrix(const char*);
template SparseMatrix<float>* importSparseMatrix(const char*);
template void writeTensorBinary(const Tensor<float>*, const char*);
template void exportTensor(const Tensor<float>*, const char*);
template void exportTensorBinary(const Tensor<float>*, const char*);
template void exportTimeSeries(const Tensor<float>*, const char*);
template void premultByDiag(const Vector<float>*, Matrix<float>*);
template void padToSquare(Matrix<float>*&);
template void combineColumnMajorBlocks(const Tensor<float>*, Matrix<float>*, const int, const int, const int);
template Matrix<float>* computeLQ(const Tensor<float>*, const int);
template void computeLQ(const Tensor<float>*, const int, Matrix<float>*);
template void computeSVD(Matrix<float>*, float*, Matrix<float>*);
template void computeSVD(Matrix<float>*, float*&, Matrix<float>*&, const float);
template void computeSVD(Matrix<float>*, float*&, Matrix<float>*&, const int);
template Tensor<float> *padTensorAlongMode(const Tensor<float>*, int, int);
template Tensor<float> *concatenateTensorsAlongMode(const Tensor<float> *, const Tensor<float> *, int);

template Matrix<double>* computeGram(const Tensor<double>* Y, const int n);
template void computeGram(const Tensor<double>*, const int, double*, const int);
template void computeEigenpairs(Matrix<double>*, double*&, const bool);
template void computeEigenpairs(Matrix<double>*, double*&, Matrix<double>*&, const int, const bool);
template void computeEigenpairs(Matrix<double>*, double*&, Matrix<double>*&, const double, const bool);
template const struct TuckerTensor<double>* STHOSVD(const Tensor<double>*, const double, bool, bool);
template const struct TuckerTensor<double>* STHOSVD(const Tensor<double>*, const SizeArray*, bool, bool);
template Tensor<double>* ttm(const Tensor<double>*, const int, const Matrix<double>*, bool);
template Tensor<double>* ttm(const Tensor<double>* const, const int, const double* const, const int, const int, bool);
template void ttm(const Tensor<double>* const, const int, const Matrix<double>* const, Tensor<double>*, bool);
template void ttm(const Tensor<double>* const, const int, const double* const, const int, Tensor<double>*, bool);
template MetricData<double>* computeSliceMetrics(const Tensor<double>*, const int, const int);
template void transformSlices(Tensor<double>*, int, const double*, const double*);
template void normalizeTensorMinMax(Tensor<double>*, int, const char*);
template void normalizeTensorMax(Tensor<double>*, int, const char*);
template void normalizeTensorStandardCentering(Tensor<double>*, int, double, const char*);
template void writeScaleShift(const int, const int, const double*, const double*, const char*);
template void readTensorBinary(Tensor<double>* Y, const char* filename);
template Tensor<double>* importTensor(const char*);
template void importTensorBinary(Tensor<double>*, const char*);
template void importTimeSeries(Tensor<double>*, const char*);
template Matrix<double>* importMatrix(const char*);
template SparseMatrix<double>* importSparseMatrix(const char*);
template void writeTensorBinary(const Tensor<double>*, const char*);
template void exportTensor(const Tensor<double>*, const char*);
template void exportTensorBinary(const Tensor<double>*, const char*);
template void exportTimeSeries(const Tensor<double>*, const char*);
template void premultByDiag(const Vector<double>*, Matrix<double>*);
template void padToSquare(Matrix<double>*&);
template void combineColumnMajorBlocks(const Tensor<double>*, Matrix<double>*, const int, const int, const int);
template Matrix<double>* computeLQ(const Tensor<double>*, const int);
template void computeLQ(const Tensor<double>*, const int, Matrix<double>*);
template void computeSVD(Matrix<double>*, double*, Matrix<double>*);
template void computeSVD(Matrix<double>*, double*&, Matrix<double>*&, const double);
template void computeSVD(Matrix<double>*, double*&, Matrix<double>*&, const int);
template Tensor<double> *padTensorAlongMode(const Tensor<double>*, int, int);
template Tensor<double> *concatenateTensorsAlongMode(const Tensor<double> *, const Tensor<double> *, int);

} // end namespace Tucker
