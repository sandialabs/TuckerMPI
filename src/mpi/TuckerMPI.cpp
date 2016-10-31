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
 * \brief Contains the functions for reading and writing tensors,
 * as well as the functions for computing the Gram matrix,
 * performing a tensor-times-matrix computation, and computing
 * a %Tucker decomposition
 *
 * @author Alicia Klinvex
 */

#include "Tucker.hpp"
#include "TuckerMPI.hpp"
#include "TuckerMPI_Util.hpp"
#include "TuckerMPI_TuckerTensor.hpp"
#include "mpi.h"
#include "math.h"
#include "assert.h"
#include <cmath>
#include <chrono>
#include <fstream>

namespace TuckerMPI
{

Tensor* ttm(const Tensor* X, const int n,
    const Tucker::Matrix* const U, bool Utransp,
    Tucker::Timer* mult_timer, Tucker::Timer* pack_timer,
    Tucker::Timer* reduce_scatter_timer,
    Tucker::Timer* reduce_timer)
{
  // Compute the number of rows for the resulting "matrix"
  int nrows;
  if(Utransp)
    nrows = U->ncols();
  else
    nrows = U->nrows();

  // Get the size of the new tensor
  int ndims = X->getNumDimensions();
  Tucker::SizeArray newSize(ndims);
  for(int i=0; i<ndims; i++) {
    if(i == n) {
      newSize[i] = nrows;
    }
    else {
      newSize[i] = X->getGlobalSize(i);
    }
  }

  // Create a distribution object for it
  Distribution* dist = new Distribution(newSize,
          X->getDistribution()->getProcessorGrid()->size());

  // Create the new tensor
  Tensor* Y;
  try {
    Y = new Tensor(dist);
  }
  catch(std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  // Get the local part of the tensor
  const Tucker::Tensor* localX = X->getLocalTensor();
  Tucker::Tensor* localY = Y->getLocalTensor();

  // Determine whether there are multiple MPI processes along this dimension
  int Pn = X->getDistribution()->getProcessorGrid()->getNumProcs(n,false);
  if(Pn == 1)
  {
    if(!X->getDistribution()->ownNothing()) {
      // Compute the TTM
      if(mult_timer) mult_timer->start();
      Tucker::ttm(localX, n, U, localY, Utransp);
      if(mult_timer) mult_timer->stop();
    }
  }
  else
  {
    // Get the local communicator
    const MPI_Comm& comm = X->getDistribution()->getProcessorGrid()->getColComm(n,false);

    // Determine whether we must block the result
    // If the temporary storage is bigger than the tensor, we block instead
    int K = U->ncols();
    int Jn = U->nrows();

    int uGlobalRows = Y->getGlobalSize(n);
    int uGlobalCols = X->getGlobalSize(n);

    const Map* xMap = X->getDistribution()->getMap(n,false);
    const Map* yMap = Y->getDistribution()->getMap(n,false);

    const double* Uptr;
    assert(U->getNumElements() > 0);
    if(Utransp)
      Uptr = U->data() + xMap->getGlobalIndex(0);
    else
      Uptr = U->data() + xMap->getGlobalIndex(0)*uGlobalRows;

    int stride;
    if(Utransp)
      stride = uGlobalCols;
    else
      stride = uGlobalRows;

    // If the required memory is small, we can do a single reduce_scatter
    if(K < std::ceil(Jn/Pn)) {
      // Compute the TTM
      Tucker::Tensor* localResult;
      if(X->getDistribution()->ownNothing()) {
        Tucker::SizeArray sz(ndims);
        for(int i=0; i<ndims; i++) {
          sz[i] = X->getLocalSize(i);
        }
        sz[n] = Y->getGlobalSize(n);
        localResult = new Tucker::Tensor(sz);
        localResult->initialize();
      }
      else {
        if(mult_timer) mult_timer->start();
        localResult = Tucker::ttm(localX, n, Uptr,
                  uGlobalRows, stride, Utransp);
        if(mult_timer) mult_timer->stop();
      }

      // Pack the data
      if(pack_timer) pack_timer->start();
      packTensor(localResult,n,yMap);
      if(pack_timer) pack_timer->stop();

      // Perform a reduce-scatter
      const double* sendBuf;
      if(localResult->getNumElements() > 0)
        sendBuf = localResult->data();
      else
        sendBuf = 0;
      double* recvBuf;
      if(localY->getNumElements() > 0)
        recvBuf = localY->data();
      else
        recvBuf = 0;
      int nprocs;
      MPI_Comm_size(comm,&nprocs);
      int* recvCounts = Tucker::safe_new<int>(nprocs);
      int multiplier = Y->getLocalSize().prod(0,n-1,1)*Y->getLocalSize().prod(n+1,ndims-1,1);
      for(int i=0; i<nprocs; i++) {
        recvCounts[i] = multiplier*(yMap->getNumEntries(i));
      }

      if(reduce_scatter_timer) reduce_scatter_timer->start();
      MPI_Reduce_scatter((void*)sendBuf, recvBuf, recvCounts, MPI_DOUBLE,
          MPI_SUM, comm);
      if(reduce_scatter_timer) reduce_scatter_timer->stop();
    }
    else {
      for(int root=0; root<Pn; root++) {
        int uLocalRows = yMap->getNumEntries(root);

        if(uLocalRows == 0) {
          continue;
        }

        // Compute the local TTM
        Tucker::Tensor* localResult;
        if(X->getDistribution()->ownNothing()) {
          Tucker::SizeArray sz(ndims);
          for(int i=0; i<ndims; i++) {
            sz[i] = X->getLocalSize(i);
          }
          sz[n] = uLocalRows;
          localResult = new Tucker::Tensor(sz);
          localResult->initialize();
        }
        else {
          if(mult_timer) mult_timer->start();
          localResult = Tucker::ttm(localX, n, Uptr, uLocalRows, stride, Utransp);
          if(mult_timer) mult_timer->stop();
        }

        // Combine the local results with a reduce operation
        const double* sendBuf;
        if(localResult->getNumElements() > 0)
          sendBuf = localResult->data();
        else
          sendBuf = 0;
        double* recvBuf;
        if(localY->getNumElements() > 0)
          recvBuf = localY->data();
        else
          recvBuf = 0;
        int count = localResult->getNumElements();


        if(count > 0) {
          if(reduce_timer) reduce_timer->start();
          MPI_Reduce((void*)sendBuf, recvBuf, count, MPI_DOUBLE, MPI_SUM,
              root, comm);
          if(reduce_timer)reduce_timer->stop();
        }


        // Free memory
        delete localResult;

        // Increment the data pointer
        if(Utransp)
          Uptr += (uLocalRows*stride);
        else
          Uptr += uLocalRows;
      } // end for i = 0 .. Pn-1
    } // end if K >= Jn/Pn
  } // end if Pn != 1

  // Return the result
  return Y;
}

/**
 * \test TuckerMPI_old_gram_test_file.cpp
 */
Tucker::Matrix* oldGram(const Tensor* Y, const int n,
    Tucker::Timer* mult_timer, Tucker::Timer* shift_timer,
    Tucker::Timer* allreduce_timer, Tucker::Timer* allgather_timer)
{
  // Size of Y
  int numLocalRows = Y->getLocalSize(n);
  int numGlobalRows = Y->getGlobalSize(n);

  // Create the matrix to return
  Tucker::Matrix* gram;
  try {
    gram = new Tucker::Matrix(numGlobalRows,numGlobalRows);
  }
  catch(std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  // Get the row and column communicators
  const MPI_Comm& rowComm =
      Y->getDistribution()->getProcessorGrid()->getRowComm(n,false);
  const MPI_Comm& colComm =
      Y->getDistribution()->getProcessorGrid()->getColComm(n,false);
  int numRowProcs, numColProcs;
  MPI_Comm_size(rowComm,&numRowProcs);
  MPI_Comm_size(colComm,&numColProcs);

  // Create buffer for all-reduce
  double* allRedBuf;
  if(numLocalRows > 0)
    allRedBuf = Tucker::safe_new<double>(numGlobalRows*numLocalRows);
  else
    allRedBuf = 0;

  if(numLocalRows > 0) {
    const MPI_Comm& colCommSqueezed =
        Y->getDistribution()->getProcessorGrid()->getColComm(n,true);

    // Stores the local Gram result
    Tucker::Matrix* localMatrix;

    // Get information about the column distribution
    int myColRankSqueezed, numColProcsSqueezed;
    MPI_Comm_rank(colCommSqueezed, &myColRankSqueezed);
    MPI_Comm_size(colCommSqueezed, &numColProcsSqueezed);

    if(!Y->getDistribution()->ownNothing()) {
      if(numColProcsSqueezed == 1) {
        // Local Gram matrix computation
        if(mult_timer) mult_timer->start();
        localMatrix = Tucker::computeGram(Y->getLocalTensor(),n);
        if(mult_timer) mult_timer->stop();
      }
      else
      {
        // Create a matrix to store the local result
        try {
          localMatrix = new Tucker::Matrix(numGlobalRows,numLocalRows);
        }
        catch(std::exception& e) {
          std::cout << "Exception: " << e.what() << std::endl;
        }

        // Determine the amount of data being received
        int ndims = Y->getNumDimensions();
        const Tucker::SizeArray& sz = Y->getLocalSize();
        int maxNumRows = Y->getDistribution()->getMap(n,true)->getMaxNumEntries();
        int numCols = sz.prod(0,n-1,1)*sz.prod(n+1,ndims-1,1);
        int maxEntries = maxNumRows*numCols;

        // Create buffer for receiving data
        double* recvBuf = Tucker::safe_new<double>(maxEntries);

        // Send data to the next proc in column
        MPI_Request* sendRequests =
            Tucker::safe_new<MPI_Request>(numColProcsSqueezed);
        int numToSend = Y->getLocalNumEntries();
        int tag = 0;
        int sendDest = (myColRankSqueezed+1)%numColProcsSqueezed;
        if(shift_timer) shift_timer->start();
        MPI_Isend((void*)Y->getLocalTensor()->data(), numToSend, MPI_DOUBLE,
            sendDest, tag, colCommSqueezed, sendRequests+sendDest);
        if(shift_timer) shift_timer->stop();

        // Receive information from the previous proc in column
        MPI_Request* recvRequests =
            Tucker::safe_new<MPI_Request>(numColProcsSqueezed);
        int recvSource =
            (numColProcsSqueezed+myColRankSqueezed-1)%numColProcsSqueezed;
        int numRowsToReceive =
            Y->getDistribution()->getMap(n,true)->getNumEntries(recvSource);
        int numToReceive = numRowsToReceive*numCols;

        if(shift_timer) shift_timer->start();
        MPI_Irecv(recvBuf, numToReceive, MPI_DOUBLE,
            recvSource, tag, colCommSqueezed, recvRequests+recvSource);
        if(shift_timer) shift_timer->stop();

        // Local computation (dsyrk)
        double* Cptr = localMatrix->data() +
            Y->getDistribution()->getMap(n,true)->getOffset(myColRankSqueezed);
        int stride = numGlobalRows;

        if(mult_timer) mult_timer->start();
        Tucker::computeGram(Y->getLocalTensor(), n, Cptr, stride);
        if(mult_timer) mult_timer->stop();

        MPI_Status stat;
        while(recvSource != myColRankSqueezed) {
          // Wait to receive from previous proc
          if(shift_timer) shift_timer->start();
          MPI_Wait(recvRequests+recvSource, &stat);

          // Send data to next proc in column
          sendDest = (sendDest+1)%numColProcsSqueezed;
          if(sendDest != myColRankSqueezed) {
            MPI_Isend((void*)Y->getLocalTensor()->data(), numToSend, MPI_DOUBLE,
                sendDest, tag, colCommSqueezed, sendRequests+sendDest);
          }
          if(shift_timer) shift_timer->stop();

          // Local computation (dgemm)
          Cptr = localMatrix->data() +
              Y->getDistribution()->getMap(n,true)->getOffset(recvSource);
          if(mult_timer) mult_timer->start();
          localGEMMForGram(recvBuf, numRowsToReceive, n, Y, Cptr);
          if(mult_timer) mult_timer->stop();

          // Request more data
          if(shift_timer) shift_timer->start();
          recvSource = (numColProcsSqueezed+recvSource-1)%numColProcsSqueezed;
          if(recvSource != myColRankSqueezed) {
            numRowsToReceive =
                    Y->getDistribution()->getMap(n,true)->getNumEntries(recvSource);
            numToReceive = numRowsToReceive*numCols;
            MPI_Irecv(recvBuf, numToReceive, MPI_DOUBLE,
                recvSource, tag, colCommSqueezed, recvRequests+recvSource);
          }
          if(shift_timer) shift_timer->stop();
        }

        // Wait for all data to be sent
        MPI_Status* sendStatuses = Tucker::safe_new<MPI_Status>(numColProcsSqueezed);
        if(shift_timer) shift_timer->start();
        if(myColRankSqueezed > 0) {
          MPI_Waitall(myColRankSqueezed, sendRequests, sendStatuses);
        }
        if(myColRankSqueezed < numColProcsSqueezed-1) {
          MPI_Waitall(numColProcsSqueezed-(myColRankSqueezed+1),
              sendRequests+myColRankSqueezed+1, sendStatuses+myColRankSqueezed+1);
        }
        if(shift_timer) shift_timer->stop();

        delete[] recvBuf;
        delete[] sendRequests;
        delete[] recvRequests;
        delete[] sendStatuses;
      }
    }
    else {
      try {
        localMatrix = new Tucker::Matrix(numGlobalRows,numLocalRows);
      }
      catch(std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
      }

      localMatrix->initialize();
    }

    // All reduce across row communicator
    if(numRowProcs > 1) {
      if(allreduce_timer) allreduce_timer->start();
      MPI_Allreduce(localMatrix->data(), allRedBuf, numLocalRows*numGlobalRows,
          MPI_DOUBLE, MPI_SUM, rowComm);
      if(allreduce_timer) allreduce_timer->stop();
    }
    else {
      size_t nnz = localMatrix->getNumElements();
      for(size_t i=0; i<nnz; i++) {
        allRedBuf[i] = localMatrix->data()[i];
      }
    }

    delete localMatrix;
  }
  else {
  }

  if(numColProcs > 1) {
    // All-gather across column communicator
    int* recvcounts = Tucker::safe_new<int>(numColProcs);
    int* displs = Tucker::safe_new<int>(numColProcs);
    for(int i=0; i<numColProcs; i++) {
      recvcounts[i] = Y->getDistribution()->getMap(n,false)->getNumEntries(i)*numGlobalRows;
      displs[i] = Y->getDistribution()->getMap(n,false)->getOffset(i)*numGlobalRows;
    }
    if(allgather_timer) allgather_timer->start();
    MPI_Allgatherv(allRedBuf, numLocalRows*numGlobalRows, MPI_DOUBLE,
        gram->data(), recvcounts, displs, MPI_DOUBLE, colComm);
    if(allgather_timer) allgather_timer->stop();

    // Free memory
    delete[] recvcounts;
    delete[] displs;
  }
  else {
    size_t nnz = gram->getNumElements();
    for(size_t i=0; i<nnz; i++)
      gram->data()[i] = allRedBuf[i];
  }

  // Free memory
  delete[] allRedBuf;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  if(rank == 0) {
    if(mult_timer)
      std::cout << "\t\tGram(" << n << ")::Local Matmul time: " << mult_timer->duration() << "s\n";
    if(shift_timer)
      std::cout << "\t\tGram(" << n << ")::Circ-shift time: " << shift_timer->duration() << "s\n";
    if(allreduce_timer)
      std::cout << "\t\tGram(" << n << ")::All-reduce time: " << allreduce_timer->duration() << "s\n";
    if(allgather_timer)
      std::cout << "\t\tGram(" << n << ")::All-gather time: " << allgather_timer->duration() << "s\n";
  }

  return gram;
}

/**
 * \test TuckerMPI_new_gram_test_file.cpp
 */
Tucker::Matrix* newGram(const Tensor* Y, const int n,
    Tucker::Timer* mult_timer, Tucker::Timer* pack_timer,
    Tucker::Timer* alltoall_timer, Tucker::Timer* unpack_timer,
    Tucker::Timer* allreduce_timer)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // Get the number of dimensions
  int ndims = Y->getNumDimensions();

  // Get the communicator
  const MPI_Comm& comm = Y->getDistribution()->getProcessorGrid()->getColComm(n,false);

  // Get the number of MPI processes in this communicator
  int numProcs;
  MPI_Comm_size(comm,&numProcs);

  // If the communicator only has one MPI process, no redistribution is needed
  const Tucker::Matrix* localGram;
  if(numProcs > 1) {
    bool myColEmpty = false;
    for(int i=0; i<ndims; i++) {
      if(i==n) continue;
      if(Y->getLocalSize(i) == 0) {
        myColEmpty = true;
        break;
      }
    }

    if(myColEmpty) {
      int nGlobalRows = Y->getGlobalSize(n);
      Tucker::Matrix* temp = new Tucker::Matrix(nGlobalRows,nGlobalRows);
      temp->initialize();
      localGram = temp;
    }
    else {
      // Redistribute the data
      const Matrix* redistributedY = redistributeTensorForGram(Y, n,
          pack_timer, alltoall_timer, unpack_timer);

      // Call symmetric rank-k update
      if(redistributedY->getLocalNumEntries() > 0) {
        if(mult_timer) mult_timer->start();
        localGram = localRankKForGram(redistributedY, n, ndims);
        if(mult_timer) mult_timer->stop();
      }
      else {
        int nGlobalRows = Y->getGlobalSize(n);
        Tucker::Matrix* temp = new Tucker::Matrix(nGlobalRows,nGlobalRows);
        temp->initialize();
        localGram = temp;
      }
    }
  }
  else {
    if(Y->getDistribution()->ownNothing()) {
      int nGlobalRows = Y->getGlobalSize(n);
      Tucker::Matrix* temp = new Tucker::Matrix(nGlobalRows,nGlobalRows);
      temp->initialize();
      localGram = temp;
    }
    else {
      if(mult_timer) mult_timer->start();
      localGram = Tucker::computeGram(Y->getLocalTensor(),n);
      if(mult_timer) mult_timer->stop();
    }
  }

  // Perform the all-reduce
  if(allreduce_timer) allreduce_timer->start();
  Tucker::Matrix* gramMat = reduceForGram(localGram);
  if(allreduce_timer) allreduce_timer->stop();

  if(rank == 0) {
    if(mult_timer)
      std::cout << "\t\tGram(" << n << ")::Local Matmul time: " << mult_timer->duration() << "s\n";
    if(pack_timer)
      std::cout << "\t\tGram(" << n << ")::Pack time: " << pack_timer->duration() << "s\n";
    if(alltoall_timer)
      std::cout << "\t\tGram(" << n << ")::All-to-all time: " << alltoall_timer->duration() << "s\n";
    if(unpack_timer)
      std::cout << "\t\tGram(" << n << ")::Unpack time: " << unpack_timer->duration() << "s\n";
    if(allreduce_timer)
      std::cout << "\t\tGram(" << n << ")::All-reduce time: " << allreduce_timer->duration() << "s\n";
  }

  // Return the gram matrix
  return gramMat;
}

void packTensor(Tucker::Tensor* Y, int n, const Map* map)
{
  // If Y has no entries, there's nothing to pack
  size_t nentries = Y->getNumElements();
  if(nentries == 0)
    return;

  // Get the number of dimensions
  int ndim = Y->N();

  // If n is the last dimension, the data is already packed
  // (because the data is stored in row-major order)
  if(n == ndim-1) {
    return;
  }

  const int inc = 1;

  // Allocate memory
  // TODO: I'm sure there's a more space-efficient way than this
  int numEntries = Y->getNumElements();
  double* tempMem = Tucker::safe_new<double>(numEntries);

  // Get communicator corresponding to this dimension
  const MPI_Comm& comm = map->getComm();

  // Get number of MPI processes
  int nprocs;
  MPI_Comm_size(comm,&nprocs);

  // Get the leading dimension of this tensor unfolding
  const Tucker::SizeArray& sa = Y->size();
  int leadingDim = sa.prod(0,n-1,1);

  // Get the number of global rows of this tensor unfolding
  int nGlobalRows = map->getGlobalNumEntries();

  // Get pointer to tensor data
  double* tenData = Y->data();

  // Set the stride
  int stride = leadingDim*nGlobalRows;

  int tempMemOffset = 0;
  for(int rank=0; rank<nprocs; rank++)
  {
    // Get number of local rows
    int nLocalRows = map->getNumEntries(rank);

    // Number of contiguous elements to copy
    int blockSize = leadingDim*nLocalRows;

    // Get offset of this row
    int rowOffset = map->getOffset(rank);

    for(int tensorOffset = rowOffset*leadingDim;
        tensorOffset < numEntries;
        tensorOffset += stride)
    {
      int RANK;
      MPI_Comm_rank(MPI_COMM_WORLD,&RANK);

      // Copy block to destination
      dcopy_(&blockSize, tenData+tensorOffset, &inc,
          tempMem+tempMemOffset, &inc);

      // Update the offset
      tempMemOffset += blockSize;
    }
  }

  // Copy data from temporary memory back to tensor
  dcopy_(&numEntries, tempMem, &inc, tenData, &inc);

  delete[] tempMem;
}

const TuckerTensor* STHOSVD(const Tensor* const X,
    const double epsilon, bool useOldGram, bool flipSign)
{
  // Get this rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ndims = X->getNumDimensions();

  // Create a struct to store the factorization
  TuckerTensor* factorization;
  try {
    factorization = new TuckerTensor(ndims);
  }
  catch(std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  // Compute the threshold
  double tensorNorm = X->norm2();
  double thresh = epsilon*epsilon*tensorNorm/ndims;
  if(rank == 0) {
    std::cout << "\tAutoST-HOSVD::Relative Threshold: "
        << thresh << "...\n";
  }

  // Barrier for timing
  MPI_Barrier(MPI_COMM_WORLD);
  factorization->total_timer_.start();

  const Tensor* Y = X;

  // For each dimension...
  for(int n=0; n<ndims; n++)
  {
    // Compute the Gram matrix
    // S = Y_n*Y_n'
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting Gram(" << n << ")...\n";
    }
    factorization->gram_timer_[n].start();
    Tucker::Matrix* S;
    if(useOldGram) {
      S = oldGram(Y, n,
          &factorization->gram_matmul_timer_[n],
          &factorization->gram_shift_timer_[n],
          &factorization->gram_allreduce_timer_[n],
          &factorization->gram_allgather_timer_[n]);
    }
    else {
      S = newGram(Y, n,
          &factorization->gram_matmul_timer_[n],
          &factorization->gram_pack_timer_[n],
          &factorization->gram_alltoall_timer_[n],
          &factorization->gram_unpack_timer_[n],
          &factorization->gram_allreduce_timer_[n]);
    }
    factorization->gram_timer_[n].stop();
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Gram(" << n << ") time: "
          << factorization->gram_timer_[n].duration() << "s\n";
    }

    // Compute the relevant eigenpairs
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting Evecs(" << n << ")...\n";
    }
    factorization->eigen_timer_[n].start();
    Tucker::computeEigenpairs(S, factorization->eigenvalues[n],
        factorization->U[n], thresh, flipSign);
    factorization->eigen_timer_[n].stop();
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::EVECS(" << n << ") time: "
          << factorization->eigen_timer_[n].duration() << "s\n";
      std::cout << "\t\tEvecs(" << n << ")::Local EV time: "
          << factorization->eigen_timer_[n].duration() << "s\n";
    }

    // Free the Gram matrix
    delete S;

    // Perform the tensor times matrix multiplication
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting TTM(" << n << ")...\n";
    }
    factorization->ttm_timer_[n].start();
    Tensor* temp = ttm(Y,n,factorization->U[n],true,
        &factorization->ttm_matmul_timer_[n],
        &factorization->ttm_pack_timer_[n],
        &factorization->ttm_reduce_timer_[n],
        &factorization->ttm_reducescatter_timer_[n]);
    factorization->ttm_timer_[n].stop();
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::TTM(" << n << ") time: "
          << factorization->ttm_timer_[n].duration() << "s\n";
    }

    if(n > 0) {
      delete Y;
    }
    Y = temp;
  }

  factorization->G = const_cast<Tensor*>(Y);
  factorization->total_timer_.stop();
  return factorization;
}

const TuckerTensor* STHOSVD(const Tensor* const X,
    const Tucker::SizeArray* const reducedI, bool useOldGram,
    bool flipSign)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int ndims = X->getNumDimensions();
  assert(ndims == reducedI->size());

  // Create a struct to store the factorization
  TuckerTensor* factorization;
  try {
    factorization = new TuckerTensor(ndims);
  }
  catch(std::exception& e) {
    std::cout << "Exception: " << e.what() << std::endl;
  }

  // Barrier for timing
  MPI_Barrier(MPI_COMM_WORLD);
  factorization->total_timer_.start();

  const Tensor* Y = X;

  // For each dimension...
  for(int n=0; n<ndims; n++)
  {
    // Compute the Gram matrix
    // S = Y_n*Y_n'
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting Gram(" << n << ")...\n";
    }
    Tucker::Matrix* S;
    factorization->gram_timer_[n].start();
    if(useOldGram) {
      S = oldGram(Y, n,
          &factorization->gram_matmul_timer_[n],
          &factorization->gram_shift_timer_[n],
          &factorization->gram_allreduce_timer_[n],
          &factorization->gram_allgather_timer_[n]);
    }
    else {
      S = newGram(Y, n,
          &factorization->gram_matmul_timer_[n],
          &factorization->gram_pack_timer_[n],
          &factorization->gram_alltoall_timer_[n],
          &factorization->gram_unpack_timer_[n],
          &factorization->gram_allreduce_timer_[n]);
    }
    factorization->gram_timer_[n].stop();
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Gram(" << n << ") time: "
          << factorization->gram_timer_[n].duration() << "s\n";
    }

    // Compute the leading eigenvectors of S
    // call dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting Evecs(" << n << ")...\n";
    }
    factorization->eigen_timer_[n].start();
    Tucker::computeEigenpairs(S, factorization->eigenvalues[n],
        factorization->U[n], (*reducedI)[n], flipSign);
    factorization->eigen_timer_[n].stop();
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::EVECS(" << n << ") time: "
          << factorization->eigen_timer_[n].duration() << "s\n";
    }

    delete S;

    // Perform the tensor times matrix multiplication
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting TTM(" << n << ")...\n";
    }
    factorization->ttm_timer_[n].start();
    Tensor* temp = ttm(Y,n,factorization->U[n],true,
            &factorization->ttm_matmul_timer_[n],
            &factorization->ttm_pack_timer_[n],
            &factorization->ttm_reduce_timer_[n],
            &factorization->ttm_reducescatter_timer_[n]);
    factorization->ttm_timer_[n].stop();
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::TTM(" << n << ") time: "
          << factorization->ttm_timer_[n].duration() << "s\n";
    }
    if(n > 0) {
      delete Y;
    }
    Y = temp;
  }

  factorization->G = const_cast<Tensor*>(Y);
  factorization->total_timer_.stop();
  return factorization;
}

Tucker::MetricData* computeSliceMetrics(const Tensor* const Y,
    int mode, int metrics)
{
  // If I don't own any slices, I don't have any work to do.
  if(Y->getLocalSize(mode) == 0) {
    return 0;
  }

  // Compute the local result
  Tucker::MetricData* result =
      Tucker::computeSliceMetrics(Y->getLocalTensor(), mode, metrics);

  // Get the row communicator
  const MPI_Comm& comm =
      Y->getDistribution()->getProcessorGrid()->getRowComm(mode,false);
  int nprocs;
  MPI_Comm_size(comm,&nprocs);

  if(nprocs > 1)
  {
    // Compute the global result
    int numSlices = Y->getLocalSize(mode);
    double* sendBuf = Tucker::safe_new<double>(numSlices);
    double* recvBuf = Tucker::safe_new<double>(numSlices);
    if(metrics & Tucker::MIN) {
      for(int i=0; i<numSlices; i++) sendBuf[i] = result->getMinData()[i];
      MPI_Allreduce(sendBuf, recvBuf, numSlices, MPI_DOUBLE_PRECISION,
          MPI_MIN, comm);
      for(int i=0; i<numSlices; i++) result->getMinData()[i] = recvBuf[i];
    }
    if(metrics & Tucker::MAX) {
      for(int i=0; i<numSlices; i++) sendBuf[i] = result->getMaxData()[i];
      MPI_Allreduce(sendBuf, recvBuf, numSlices, MPI_DOUBLE_PRECISION,
          MPI_MAX, comm);
      for(int i=0; i<numSlices; i++) result->getMaxData()[i] = recvBuf[i];
    }
    if(metrics & Tucker::SUM) {
      for(int i=0; i<numSlices; i++) sendBuf[i] = result->getSumData()[i];
      MPI_Allreduce(sendBuf, recvBuf, numSlices, MPI_DOUBLE_PRECISION,
          MPI_SUM, comm);
      for(int i=0; i<numSlices; i++) result->getSumData()[i] = recvBuf[i];
    }
    // If X is partitioned into X_A and X_B,
    // mean_X = (n_A mean_A + n_B mean_B) / (n_A + n_B)
    if((metrics & Tucker::MEAN) || (metrics & Tucker::VARIANCE)) {
      // Compute the size of my local slice
      int ndims = Y->getNumDimensions();
      const Tucker::SizeArray& localSize = Y->getLocalSize();
      int localSliceSize = localSize.prod(0,mode-1,1) *
          localSize.prod(mode+1,ndims-1,1);

      // Compute the size of the global slice
      const Tucker::SizeArray& globalSize = Y->getGlobalSize();
      int globalSliceSize = globalSize.prod(0,mode-1,1) *
          globalSize.prod(mode+1,ndims-1,1);

      for(int i=0; i<numSlices; i++) {
        sendBuf[i] = result->getMeanData()[i] * localSliceSize;
      }

      MPI_Allreduce(sendBuf, recvBuf, numSlices,
          MPI_DOUBLE_PRECISION, MPI_SUM, comm);

      double* meanDiff;
      if(metrics & Tucker::VARIANCE) {
        meanDiff = Tucker::safe_new<double>(numSlices);
        for(int i=0; i<numSlices; i++) {
          meanDiff[i] = result->getMeanData()[i] -
              recvBuf[i] / globalSliceSize;
        }
      }

      for(int i=0; i<numSlices; i++) {
        result->getMeanData()[i] = recvBuf[i] / globalSliceSize;
      }

      if(metrics & Tucker::VARIANCE) {
        for(int i=0; i<numSlices; i++) {
          // Source of this equation:
          // http://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
          sendBuf[i] = localSliceSize*result->getVarianceData()[i] +
              localSliceSize*meanDiff[i]*meanDiff[i];
        }

        MPI_Allreduce(sendBuf, recvBuf, numSlices,
            MPI_DOUBLE_PRECISION, MPI_SUM, comm);

        for(int i=0; i<numSlices; i++) {
          result->getVarianceData()[i] = recvBuf[i] / globalSliceSize;
        }
      }
    }

    delete[] sendBuf;
    delete[] recvBuf;
  }

  return result;
}

// Shift is applied before scale
// We divide by scaleVals, not multiply
void transformSlices(Tensor* Y, int mode, const double* scales, const double* shifts)
{
  Tucker::transformSlices(Y->getLocalTensor(), mode, scales, shifts);
}

void normalizeTensorStandardCentering(Tensor* Y, int mode, double stdThresh)
{
  // I don't have to do any work because I don't own any data
  if(Y->getLocalSize(mode) == 0)
    return;

  Tucker::MetricData* metrics =
      computeSliceMetrics(Y, mode, Tucker::MEAN+Tucker::VARIANCE);
  int sizeOfModeDim = Y->getLocalSize(mode);
  double* scales = Tucker::safe_new<double>(sizeOfModeDim);
  double* shifts = Tucker::safe_new<double>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scales[i] = sqrt(metrics->getVarianceData()[i]);
    shifts[i] = -metrics->getMeanData()[i];
    if(std::abs(scales[i]) < stdThresh) {
      scales[i] = 1;
    }
  }
  transformSlices(Y,mode,scales,shifts);
  delete[] scales;
  delete[] shifts;
  delete metrics;
}

void normalizeTensorMinMax(Tensor* Y, int mode)
{
  // I don't have to do any work because I don't own any data
  if(Y->getLocalSize(mode) == 0)
    return;

  Tucker::MetricData* metrics = computeSliceMetrics(Y, mode,
      Tucker::MAX+Tucker::MIN);
  int sizeOfModeDim = Y->getLocalSize(mode);
  double* scales = Tucker::safe_new<double>(sizeOfModeDim);
  double* shifts = Tucker::safe_new<double>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scales[i] = metrics->getMaxData()[i] - metrics->getMinData()[i];
    shifts[i] = -metrics->getMinData()[i];
  }
  transformSlices(Y,mode,scales,shifts);
  delete[] scales;
  delete[] shifts;
  delete metrics;
}

void normalizeTensorMax(Tensor* Y, int mode)
{
  // I don't have to do any work because I don't own any data
  if(Y->getLocalSize(mode) == 0)
    return;

  Tucker::MetricData* metrics = computeSliceMetrics(Y, mode,
      Tucker::MIN + Tucker::MAX);
  int sizeOfModeDim = Y->getLocalSize(mode);
  double* scales = Tucker::safe_new<double>(sizeOfModeDim);
  double* shifts = Tucker::safe_new<double>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    double scaleval = std::max(std::abs(metrics->getMinData()[i]),
        std::abs(metrics->getMaxData()[i]));
    scales[i] = scaleval;
    shifts[i] = 0;
  }
  transformSlices(Y,mode,scales,shifts);
  delete[] scales;
  delete[] shifts;
  delete metrics;
}

void readTensorBinary(std::string& filename, Tensor& Y)
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
    importTensorBinary(temp.c_str(),&Y);
  }
  else {
    int ndims = Y.getNumDimensions();
    if(nfiles != Y.getGlobalSize(ndims-1)) {
      int rank;
      MPI_Comm_rank(MPI_COMM_WORLD,&rank);
      if(rank == 0) {
        std::cerr << "ERROR: The number of filenames you provided is "
            << nfiles << ", but the dimension of the tensor's last mode is "
            << Y.getGlobalSize(ndims-1) << ".\nCalling MPI_Abort...\n";
      }
      MPI_Abort(MPI_COMM_WORLD,1);
    }
    importTimeSeries(filename.c_str(),&Y);
  }
}

//! \todo This function should report when the file was not sufficiently large
void importTensorBinary(const char* filename, Tensor* Y)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if(rank == 0) {
    std::cout << "Reading file " << filename << std::endl;
  }

  if(Y->getDistribution()->ownNothing()) {
    return;
  }

  int ndims = Y->getNumDimensions();

  // Define data layout parameters
  int* starts = Tucker::safe_new<int>(ndims);
  int* lsizes = Tucker::safe_new<int>(ndims);
  int* gsizes = Tucker::safe_new<int>(ndims);

  for(int i=0; i<ndims; i++) {
    starts[i] = Y->getDistribution()->getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y->getLocalSize(i);
    gsizes[i] = Y->getGlobalSize(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray(ndims, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, MPI_DOUBLE, &view);
  MPI_Type_commit(&view);

  // Open the file
  MPI_File fh;
  const MPI_Comm& comm = Y->getDistribution()->getComm(true);
  int ret = MPI_File_open(comm, (char*)filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
  }

  // Set the view
  MPI_Offset disp = 0;
  MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);

  // Read the file
  int count = Y->getLocalNumEntries();
  MPI_Status status;
  ret = MPI_File_read_all(fh, Y->getLocalTensor()->data(),
      count, MPI_DOUBLE, &status);
  int nread;
  MPI_Get_count (&status, MPI_DOUBLE, &nread);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not read file " << filename << std::endl;
  }

  // Close the file
  MPI_File_close(&fh);

  // Free the datatype
  MPI_Type_free(&view);

  // Free other memory
  delete[] starts;
  delete[] lsizes;
  delete[] gsizes;
}

// This function assumes that Y has already been allocated
// and its values need to be filled in
void importTensorBinary(const char* filename, Tucker::Tensor* Y)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int ret;
  MPI_File fh;

  if(rank == 0) {
    std::cout << "Reading file " << filename << std::endl;
  }

  // Open the file
  ret = MPI_File_open(MPI_COMM_WORLD, (char*)filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS && rank == 0) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
  }

  // Read the file
  int count = Y->size().prod();
  double * data = Y->data();
  MPI_Status status;
  ret = MPI_File_read(fh, data, count, MPI_DOUBLE, &status);
  if(ret != MPI_SUCCESS && rank == 0) {
    std::cerr << "Error: Could not read file " << filename << std::endl;
  }

  // Close the file
  MPI_File_close(&fh);
}

void importTimeSeries(const char* filename, Tensor* Y)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  // Open the file
  std::ifstream ifs;
  ifs.open(filename);

  if(Y->getDistribution()->ownNothing()) {
    return;
  }

  // Define data layout parameters
  int ndims = Y->getNumDimensions();
  int* starts = Tucker::safe_new<int>(ndims-1);
  int* lsizes = Tucker::safe_new<int>(ndims-1);
  int* gsizes = Tucker::safe_new<int>(ndims-1);

  for(int i=0; i<ndims-1; i++) {
    starts[i] = Y->getDistribution()->getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y->getLocalSize(i);
    gsizes[i] = Y->getGlobalSize(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray(ndims-1, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, MPI_DOUBLE, &view);
  MPI_Type_commit(&view);

  int nsteps = Y->getGlobalSize(ndims-1);
  const Map* stepMap = Y->getDistribution()->getMap(ndims-1,true);
  const MPI_Comm& stepComm = Y->getDistribution()->getProcessorGrid()->getRowComm(ndims-1,true);
  double* dataPtr = Y->getLocalTensor()->data();
  int count = Y->getLocalSize().prod(0,ndims-2);

  for(int step=0; step<nsteps; step++) {
    std::string stepFilename;
    ifs >> stepFilename;
    if(rank == 0) {
      std::cout << "Reading file " << stepFilename << std::endl;
    }

    int LO = stepMap->getLocalIndex(step);
    if(LO < 0) {
      // This file doesn't contain any information I should own
      continue;
    }

    // Open the file
    MPI_File fh;
    int ret = MPI_File_open(stepComm, (char*)stepFilename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    if(ret != MPI_SUCCESS && rank == 0) {
      std::cerr << "Error: Could not open file " << stepFilename << std::endl;
    }

    // Set the view
    MPI_Offset disp = 0;
    MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);

    // Read the file
    MPI_Status status;
    ret = MPI_File_read_all(fh, dataPtr,
        count, MPI_DOUBLE, &status);
    if(ret != MPI_SUCCESS && rank == 0) {
      std::cerr << "Error: Could not read file " << stepFilename << std::endl;
      exit(1);
    }

    // Close the file
    MPI_File_close(&fh);

    // Increment the pointer
    dataPtr += count;
  }

  // Free the datatype
  MPI_Type_free(&view);

  // Free other memory
  delete[] starts;
  delete[] lsizes;
  delete[] gsizes;
}

void writeTensorBinary(std::string& filename, const Tensor& Y)
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
     exportTensorBinary(temp.c_str(),&Y);
   }
   else {
     int ndims = Y.getNumDimensions();
     if(nfiles != Y.getGlobalSize(ndims-1)) {
       int rank;
       MPI_Comm_rank(MPI_COMM_WORLD,&rank);
       if(rank == 0) {
         std::cerr << "ERROR: The number of filenames you provided is "
             << nfiles << ", but the dimension of the tensor's last mode is "
             << Y.getGlobalSize(ndims-1) << ".\nCalling MPI_Abort...\n";
       }
       MPI_Abort(MPI_COMM_WORLD,1);
     }
     exportTimeSeries(filename.c_str(),&Y);
   }
}

void exportTensorBinary(const char* filename, const Tensor* Y)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if(rank == 0) {
    std::cout << "Writing file " << filename << std::endl;
  }

  if(Y->getDistribution()->ownNothing()) {
    return;
  }

  int ndims = Y->getNumDimensions();

  // Define data layout parameters
  int* starts = Tucker::safe_new<int>(ndims);
  int* lsizes = Tucker::safe_new<int>(ndims);
  int* gsizes = Tucker::safe_new<int>(ndims);

  for(int i=0; i<ndims; i++) {
    starts[i] = Y->getDistribution()->getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y->getLocalSize(i);
    gsizes[i] = Y->getGlobalSize(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray(ndims, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, MPI_DOUBLE, &view);
  MPI_Type_commit(&view);

  // Open the file
  MPI_File fh;
  const MPI_Comm& comm = Y->getDistribution()->getComm(true);
  int ret = MPI_File_open(comm, (char*)filename,
      MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS && rank == 0) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
  }

  // Set the view
  MPI_Offset disp = 0;
  MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);

  // Write the file
  int count = Y->getLocalNumEntries();
  MPI_Status status;
  ret = MPI_File_write_all(fh, (double*)Y->getLocalTensor()->data(), count,
      MPI_DOUBLE, &status);
  if(ret != MPI_SUCCESS && rank == 0) {
    std::cerr << "Error: Could not write to file " << filename << std::endl;
  }

  // Close the file
  MPI_File_close(&fh);

  // Free the datatype
  MPI_Type_free(&view);

  // Free other memory
  delete[] starts;
  delete[] lsizes;
  delete[] gsizes;
}

void exportTensorBinary(const char* filename, const Tucker::Tensor* Y)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int ret;
  MPI_File fh;

  if(rank == 0) {
    std::cout << "Writing file " << filename << std::endl;
  }

  // Send core to binary file
  ret = MPI_File_open(MPI_COMM_SELF, (char*)filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
  if(ret != MPI_SUCCESS && rank == 0) {
    std::cerr << "Error: Could not open file " << filename << std::endl;
  }

  // Write the tensor to a binary file
  int nentries = Y->size().prod();
  const double* entries = Y->data();
  MPI_Status status;
  ret = MPI_File_write(fh, (double*)entries, nentries, MPI_DOUBLE, &status);
  if(ret != MPI_SUCCESS && rank == 0) {
    std::cerr << "Error: Could not write file " << filename << std::endl;
  }

  // Close the file
  MPI_File_close(&fh);
}

void exportTimeSeries(const char* filename, const Tensor* Y)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if(Y->getDistribution()->ownNothing()) {
    return;
  }

  // Open the file
  std::ifstream ifs;
  ifs.open(filename);

  // Define data layout parameters
  int ndims = Y->getNumDimensions();
  int* starts = Tucker::safe_new<int>(ndims-1);
  int* lsizes = Tucker::safe_new<int>(ndims-1);
  int* gsizes = Tucker::safe_new<int>(ndims-1);

  for(int i=0; i<ndims-1; i++) {
    starts[i] = Y->getDistribution()->getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y->getLocalSize(i);
    gsizes[i] = Y->getGlobalSize(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray(ndims-1, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, MPI_DOUBLE, &view);
  MPI_Type_commit(&view);

  int nsteps = Y->getGlobalSize(ndims-1);
  const Map* stepMap = Y->getDistribution()->getMap(ndims-1,true);
  const MPI_Comm& stepComm = Y->getDistribution()->getProcessorGrid()->getRowComm(ndims-1,true);
  const double* dataPtr = Y->getLocalTensor()->data();
  int count = Y->getLocalSize().prod(0,ndims-2);
  for(int step=0; step<nsteps; step++) {
    std::string stepFilename;
    ifs >> stepFilename;
    if(rank == 0) {
      std::cout << "Writing file " << stepFilename << std::endl;
    }

    int LO = stepMap->getLocalIndex(step);
    if(LO < 0) {
      // This file doesn't contain any information I should own
      continue;
    }

    // Open the file
    MPI_File fh;
    int ret = MPI_File_open(stepComm, (char*)stepFilename.c_str(),
        MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    if(ret != MPI_SUCCESS && rank == 0) {
      std::cerr << "Error: Could not open file " << stepFilename << std::endl;
    }

    // Set the view
    MPI_Offset disp = 0;
    MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);

    // Write the file
    MPI_Status status;
    ret = MPI_File_write_all(fh, (void*)dataPtr, count,
        MPI_DOUBLE, &status);
    if(ret != MPI_SUCCESS && rank == 0) {
      std::cerr << "Error: Could not write to file " << stepFilename << std::endl;
    }

    // Close the file
    MPI_File_close(&fh);

    // Increment the pointer
    dataPtr += count;
  }

  // Free the datatype
  MPI_Type_free(&view);

  // Free other memory
  delete[] starts;
  delete[] lsizes;
  delete[] gsizes;
}

} // end namespace TuckerMPI
