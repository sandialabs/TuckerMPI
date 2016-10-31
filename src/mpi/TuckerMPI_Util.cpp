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
 * \brief Stores utility routines
 *
 * @author Alicia Klinvex
 */

#include "assert.h"
#include "TuckerMPI_Util.hpp"

namespace TuckerMPI {

bool isPackForGramNecessary(int n, const Map* origMap, const Map* redistMap)
{
  // Get the local number of rows of Y_n
  int localNumRows = origMap->getLocalNumEntries();

  // If the local number of rows is 1, no packing is necessary
  if(localNumRows <= 1) {
    return false;
  }

  // If n is 0, the local data is already column-major and no packing is necessary
  if(n == 0) {
    return false;
  }

  return true;
}

// Pack the data for redistribution
// Y_n is block-row distributed; we are packing
// so that Y_n will be block-column distributed
const double* packForGram(const Tensor* Y, int n, const Map* redistMap)
{
  const int ONE = 1;

  assert(isPackForGramNecessary(n, Y->getDistribution()->getMap(n,true), redistMap));

  // Get the local number of rows of Y_n
  int localNumRows = Y->getLocalSize(n);

  // Get the number of columns of Y_n
  int globalNumCols = redistMap->getGlobalNumEntries();

  // Get the number of dimensions
  int ndims = Y->getNumDimensions();

  // Get the number of MPI processes
  int nprocs = Y->getDistribution()->getProcessorGrid()->getNumProcs(n,true);

  // Allocate memory for packed data
  double* sendData = Tucker::safe_new<double>(Y->getLocalNumEntries());

  // Local data is row-major
  if(n == ndims-1) {
    const double* YnData = Y->getLocalTensor()->data();

    int offset=0;
    for(int b=0; b<nprocs; b++) {
      int n = redistMap->getNumEntries(b);
      for(int r=0; r<localNumRows; r++) {
        dcopy_(&n, YnData+redistMap->getOffset(b)+globalNumCols*r, &ONE,
               sendData+offset, &ONE);
        offset += n;
      }
    }
  }
  else {
    // Get the size of the tensor
    const Tucker::SizeArray& sz = Y->getLocalSize();

    // Get information about the local blocks
    int numLocalBlocks = sz.prod(n+1,ndims-1);
    int ncolsPerLocalBlock = sz.prod(0,n-1);

    const double* src = Y->getLocalTensor()->data();

    // Make local data column major
    for(int b=0; b<numLocalBlocks; b++) {
      double* dest = sendData+b*localNumRows*ncolsPerLocalBlock;
      // Copy one row at a time
      for(int r=0; r<localNumRows; r++) {
        dcopy_(&ncolsPerLocalBlock, src, &ONE, dest, &localNumRows);
        src += ncolsPerLocalBlock;
        dest += 1;
      }
    }
  }

  return sendData;
}


const Matrix* redistributeTensorForGram(const Tensor* Y, int n,
    Tucker::Timer* pack_timer, Tucker::Timer* alltoall_timer,
    Tucker::Timer* unpack_timer)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // Get the original Tensor map
  const Map* oldMap = Y->getDistribution()->getMap(n,false);

  // Get the communicator
  const MPI_Comm& comm = Y->getDistribution()->getProcessorGrid()->getColComm(n,false);

  // Get the number of MPI processes in this communicator
  int numProcs;
  MPI_Comm_size(comm,&numProcs);

  // If the communicator only has one MPI process, no redistribution is needed
  if(numProcs < 2) {
    return 0;
  }

  // Get the dimensions of the redistributed matrix Y_n
  int ndims = Y->getNumDimensions();
  const Tucker::SizeArray& sz = Y->getLocalSize();
  int nrows = Y->getGlobalSize(n);
  int ncols = sz.prod(0,n-1,1)*sz.prod(n+1,ndims-1,1);

  // Create a matrix to store the redistributed Y_n
  // Y_n has a block row distribution
  // We want it to have a block column distribution
  Matrix* redistY;
  try {
    redistY = new Matrix(nrows,ncols,comm,false);
  }
  catch(std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  // Get the column map of the redistributed Y_n
  const Map* redistMap = redistY->getMap();

  // Get the number of local rows of the matrix Y_n
  int nLocalRows = Y->getLocalSize(n);

  // Get the number of local columns of the matrix Y_n
  int nLocalCols = redistMap->getLocalNumEntries();

  // Compute the number of entries this rank will send to others,
  // along with their displacements
  int* sendCounts = Tucker::safe_new<int>(numProcs);
  int* sendDispls = Tucker::safe_new<int>(numProcs+1); sendDispls[0] = 0;
  for(int i=0; i<numProcs; i++) {
    sendCounts[i] = nLocalRows * redistMap->getNumEntries(i);
    sendDispls[i+1] = sendDispls[i] + sendCounts[i];
  }

  // Compute the number of entries this rank will receive from others,
  // along with their displacements
  int* recvCounts = Tucker::safe_new<int>(numProcs);
  int* recvDispls = Tucker::safe_new<int>(numProcs+1); recvDispls[0] = 0;
  for(int i=0; i<numProcs; i++) {
    recvCounts[i] = nLocalCols * oldMap->getNumEntries(i);
    recvDispls[i+1] = recvDispls[i] + recvCounts[i];
  }

  // Pack the data, if packing is necessary
  bool isPackingNecessary = isPackForGramNecessary(n, oldMap, redistMap);
  const double* sendBuf;
  if(isPackingNecessary) {
    if(pack_timer) pack_timer->start();
    sendBuf = packForGram(Y, n, redistMap);
    if(pack_timer) pack_timer->stop();
  }
  else {
    // If Y has no entries, we're not sending anything
    if(Y->getLocalNumEntries() == 0)
      sendBuf = 0;
    else
      sendBuf = Y->getLocalTensor()->data();
  }

  // Determine whether the data needs to be unpacked
  bool isUnpackingNecessary = isUnpackForGramNecessary(n, ndims, oldMap, redistMap);
  double* recvBuf;
  if(isUnpackingNecessary) {
    recvBuf = Tucker::safe_new<double>(redistY->getLocalNumEntries());
  }
  else {
    // If redistY has no entries, we're not receiving anything
    if(redistY->getLocalNumEntries() == 0)
      recvBuf = 0;
    else
      recvBuf = redistY->getLocalMatrix()->data();
  }

//  std::cout << rank << ": " << numProcs << std::endl;
//  for(int i=0; i<numProcs; i++) {
//    std::cout << rank << ": " << sendCounts[i] << " " << sendDispls[i]
//              << " " << recvCounts[i] << " " << recvDispls[i] << std::endl;
//  }

  // Perform the all-to-all communication
  if(alltoall_timer) alltoall_timer->start();
  MPI_Alltoallv((void*)sendBuf, sendCounts, sendDispls, MPI_DOUBLE,
      recvBuf, recvCounts, recvDispls, MPI_DOUBLE, comm);
  if(alltoall_timer) alltoall_timer->stop();

  if(isUnpackingNecessary) {
    if(unpack_timer) unpack_timer->start();
    unpackForGram(n, ndims, redistY, recvBuf, oldMap);
    if(unpack_timer) unpack_timer->stop();
  }

  // Free memory
  delete[] sendCounts; delete[] sendDispls;
  delete[] recvCounts; delete[] recvDispls;
  if(isPackingNecessary && Y->getLocalNumEntries() > 0) delete[] sendBuf;
  if(isUnpackingNecessary) delete[] recvBuf;

  return redistY;
}


bool isUnpackForGramNecessary(int n, int ndims, const Map* origMap, const Map* redistMap)
{
  // Get the local number of columns
  int nLocalCols = redistMap->getLocalNumEntries();

  // If the number of local columns is 1, no unpacking is necessary
  if(nLocalCols <= 1) {
    return false;
  }

  // If this is the last dimension, the data is row-major and no unpacking is necessary
  if(n == ndims-1) {
    return false;
  }

  return true;
}


void unpackForGram(int n, int ndims, Matrix* redistMat,
    const double* dataToUnpack, const Map* origMap)
{
  const int ONE = 1;

  assert(isUnpackForGramNecessary(n, ndims, origMap, redistMat->getMap()));

  // Get the size of the matrix
  int nLocalCols = redistMat->getLocalNumCols();

  // Get the number of MPI processes
  const MPI_Comm& comm = origMap->getComm();
  int nprocs;
  MPI_Comm_size(comm,&nprocs);

  double* dest = redistMat->getLocalMatrix()->data();
  for(int c=0; c<nLocalCols; c++) {
    for(int b=0; b<nprocs; b++) {
      int nLocalRows=origMap->getNumEntries(b);
      const double* src = dataToUnpack + origMap->getOffset(b)*nLocalCols + c*nLocalRows;
      dcopy_(&nLocalRows, src, &ONE, dest, &ONE);
      dest += nLocalRows;
    }
  }
}

const Tucker::Matrix* localRankKForGram(const Matrix* Y, int n, int ndims)
{
  int nrows = Y->getLocalNumRows();
  Tucker::Matrix* localResult;
  try {
    localResult = new Tucker::Matrix(nrows, nrows);
  }
  catch(std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  char uplo = 'U';
  int ncols = Y->getLocalNumCols();
  double alpha = 1;
  const double* A = Y->getLocalMatrix()->data();
  double beta = 0;
  double* C = localResult->data();
  int ldc = nrows;

  if(n < ndims-1) {
    // Local matrix is column major
    char trans = 'N';
    int lda = nrows;
    dsyrk_(&uplo, &trans, &nrows, &ncols, &alpha, A, &lda, &beta, C, &ldc);
  }
  else {
    // Local matrix is row major
    char trans = 'T';
    int lda = ncols;
    dsyrk_(&uplo, &trans, &nrows, &ncols, &alpha, A, &lda, &beta, C, &ldc);
  }

  return localResult;
}

void localGEMMForGram(const double* Y1, int nrowsY1, int n,
    const Tensor* Y2, double* result)
{
  int ndims = Y2->getNumDimensions();
  int numLocalRows = Y2->getLocalSize(n);
  int numGlobalRows = Y2->getGlobalSize(n);
  const Tucker::SizeArray& sz = Y2->getLocalSize();
  int numCols = sz.prod(0,n-1,1)*sz.prod(n+1,ndims-1,1);

  if(n == 0) {
    // Data is stored column-major
    char transa = 'N';
    char transb = 'T';
    int crows = nrowsY1;
    int ccols = numLocalRows;
    int interDim = numCols;
    double alpha = 1;
    const double* Aptr = Y1;
    int lda = nrowsY1;
    const double* Bptr = Y2->getLocalTensor()->data();
    int ldb = numLocalRows;
    double beta = 0;
    int ldc = numGlobalRows;

    dgemm_(&transa, &transb, &crows, &ccols, &interDim,
        &alpha, Aptr, &lda, Bptr, &ldb, &beta, result, &ldc);
  }
  else if(n == ndims-1) {
    // Data is row-major
    char transa = 'T';
    char transb = 'N';
    int crows = nrowsY1;
    int ccols = numLocalRows;
    int interDim = numCols;
    double alpha = 1;
    const double* Aptr = Y1;
    int lda = numCols;
    const double* Bptr = Y2->getLocalTensor()->data();
    int ldb = numCols;
    double beta = 0;
    int ldc = numGlobalRows;

    dgemm_(&transa, &transb, &crows, &ccols, &interDim,
        &alpha, Aptr, &lda, Bptr, &ldb, &beta, result, &ldc);
  }
  else {
    // Data is a series of row-major blocks
    int numBlocks = sz.prod(n+1,ndims-1);
    int colsPerBlock = sz.prod(0,n-1);

    char transa = 'T';
    char transb = 'N';
    int crows = nrowsY1;
    int ccols = numLocalRows;
    int interDim = colsPerBlock;
    double alpha = 1;
    const double* Aptr = Y1;
    int lda = colsPerBlock;
    const double* Bptr = Y2->getLocalTensor()->data();
    int ldb = colsPerBlock;
    double beta;
    int ldc = numGlobalRows;

    for(int b=0; b<numBlocks; b++) {
      if(b == 0) {
        beta = 0;
      }
      else {
        beta = 1;
      }

      // Call dgemm
      dgemm_(&transa, &transb, &crows, &ccols, &interDim,
          &alpha, Aptr, &lda, Bptr, &ldb, &beta, result, &ldc);

      // Update pointers
      Aptr += (nrowsY1*colsPerBlock);
      Bptr += (numLocalRows*colsPerBlock);
    }
  }
}

Tucker::Matrix* reduceForGram(const Tucker::Matrix* U)
{
  // Get the dimensions of U
  int nrows = U->nrows();
  int ncols = U->ncols();
  int count = nrows*ncols;

  // Create a matrix to store the result
  Tucker::Matrix* reducedU;
  try {
    reducedU = new Tucker::Matrix(nrows,ncols);
  }
  catch(std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  MPI_Allreduce((void*)U->data(), reducedU->data(), count,
      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  return reducedU;
}

} // end namespace TuckerMPI
