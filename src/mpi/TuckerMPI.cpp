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
#include "TuckerMPI_ttm.hpp"
#include "mpi.h"
#include "assert.h"
#include <cmath>
#include <chrono>
#include <fstream>
#include <random>
#include <iomanip>

namespace TuckerMPI
{
//There isn't a check for this since we might handle this case later but for now
//it is assumed that the processor grid is never bigger than the tensor in any mode.
Tucker::Matrix* LQ(const Tensor* Y, const int n, Tucker::Timer* tsqr_timer,
    Tucker::Timer* local_qr_timer, Tucker::Timer* redistribute_timer,
    Tucker::Timer* localqr_dcopy_timer, Tucker::Timer* localqr_decompose_timer, 
    Tucker::Timer* localqr_transpose_timer){
  int one = 1;
  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
  int globalnp;
  MPI_Comm_size(MPI_COMM_WORLD, &globalnp);
  // const Matrix* redistYn = redistributeTensorForGram(Y, n, pack_timer, alltoall_timer, unpack_timer);
  if(redistribute_timer) redistribute_timer->start();
  const Matrix* redistYn = redistributeTensorForGram(Y, n);
  if(redistribute_timer) redistribute_timer->stop();
  //R of the transpose of the local unfolding in column major.
  Tucker::Matrix* R;
  int Rnrows;
  int Rncols;
  if(local_qr_timer) local_qr_timer->start();
  if(!redistYn){
    Tucker::Matrix* L = Tucker::computeLQ(Y->getLocalTensor(), n);
    Rnrows = L->nrows();
    Rncols = L->ncols(); 
    //Do an explicit transpose of tempT.
    R = Tucker::MemoryManager::safe_new<Tucker::Matrix>(Rnrows, Rncols);
    int sizeOfTempT = Rnrows*Rncols;
    for(int i=0; i<Rncols; i++){
      dcopy_(&Rnrows, L->data()+i*Rnrows, &one, R->data()+i, &Rnrows);
    }
    Tucker::MemoryManager::safe_delete<Tucker::Matrix>(L);
  }
  else{
    bool isLastMode = n == Y->getNumDimensions()-1;
    R = localQR(redistYn, isLastMode, localqr_dcopy_timer, localqr_decompose_timer, localqr_transpose_timer);
    Rnrows = R->nrows();
    Rncols = R->ncols();
    Tucker::MemoryManager::safe_delete<const Matrix>(redistYn);  
  }
  if(local_qr_timer) local_qr_timer->stop();
  int sizeOfR = Rnrows*Rncols;
  Tucker::Matrix* L = Tucker::MemoryManager::safe_new<Tucker::Matrix>(Rnrows, Rncols);
  Tucker::Matrix* tempB;
  int treeDepth = (int)ceil(log2(globalnp));
  MPI_Status status;
  if(tsqr_timer) tsqr_timer->start();
  for(int i=0; i < treeDepth; i++){
    if(globalRank % (int)pow(2, i+1) == 0){
      if(globalRank+ pow(2, i) < globalnp){
        tempB = Tucker::MemoryManager::safe_new<Tucker::Matrix>(Rncols, Rncols);
        MPI_Recv(tempB->data(), Rncols*Rncols, MPI_DOUBLE, globalRank+pow(2, i), globalRank+pow(2, i), MPI_COMM_WORLD, &status);
        
        //TODO: delete this?? make tempT square if it is short and fat so that it has enough space to store the R.
        // if(tempTnrows < tempTncols){
        //   std::cout << "Processor["<<globalRank<<"] in iteration["<<i<<"] enter special copy block because R is in shape "<< tempTnrows << " by "<< tempTncols<< std::endl;
        //   Tucker::Matrix* squareTempT = Tucker::MemoryManager::safe_new<Tucker::Matrix>(tempTncols, tempTncols);
        //   squareTempT->initialize();
        //   //copy one row at a time.
        //   for(int j=0; j<tempTnrows; j++){
        //     dcopy_(&tempTncols, tempT->data()+j, &tempTnrows, squareTempT->data()+j, &tempTncols);
        //   }
        //   std::cout << "Processor["<<globalRank<<"] in iteration["<<i<<"] matrix A for dtpqrt:"<< std::endl;
        //   for(int a=0; a<tempTncols; a++){
        //     for(int b=0; b<tempTncols;b++){
        //       std::cout << squareTempT->data()[a+b*tempTncols] << ", ";
        //     }
        //     std::cout << std::endl;
        //   }
        //   std::cout << std::endl;
        //   Tucker::MemoryManager::safe_delete<Tucker::Matrix>(tempT);
        //   tempT = squareTempT;
        //   tempTnrows = tempT->nrows();
        // }
        int nb = (Rncols > 32)? 32 : Rncols;
        double* T = Tucker::MemoryManager::safe_new_array<double>(nb*Rncols);
        double* work = Tucker::MemoryManager::safe_new_array<double>(nb*Rncols);
        int info;
        Tucker::dtpqrt_(&Rncols, &Rncols, &Rncols, &nb, R->data(), &Rnrows, tempB->data(),
          &Rncols, T, &nb, work, &info);
        Tucker::MemoryManager::safe_delete<Tucker::Matrix>(tempB);
        Tucker::MemoryManager::safe_delete_array<double>(work, nb*Rncols);
        Tucker::MemoryManager::safe_delete_array<double>(T, nb*Rncols);
      }
    }
    else if(globalRank % (int)pow(2, i) == 0){
      MPI_Send(R->data(), sizeOfR, MPI_DOUBLE, globalRank-pow(2, i), globalRank, MPI_COMM_WORLD);
    }
  }
  if(tsqr_timer) tsqr_timer->stop();
  if(globalRank == 0){
    //add zeros at the lower triangle
    for(int c=0; c<R->ncols(); c++){
      for(int r=c+1; r<R->nrows(); r++){
        R->data()[r+c*R->nrows()] = 0;
      }
    }
    //transpose
    for(int i=0; i<Rncols; i++){
      dcopy_(&Rnrows, R->data()+i*Rnrows, &one, L->data()+i, &Rnrows); 
    }
    Tucker::MemoryManager::safe_delete<Tucker::Matrix>(R);
    //std::cout << L->prettyPrint();
  }
  return L;
}

/**
 * \test TuckerMPI_old_gram_test_file.cpp
 */
template <class scalar_t>
Tucker::Matrix<scalar_t>* oldGram(const Tensor<scalar_t>* Y, const int n,
    Tucker::Timer* mult_timer, Tucker::Timer* shift_timer,
    Tucker::Timer* allreduce_timer, Tucker::Timer* allgather_timer)
{
  // Size of Y
  int numLocalRows = Y->getLocalSize(n);
  int numGlobalRows = Y->getGlobalSize(n);

  // Create the matrix to return
  Tucker::Matrix<scalar_t>* gram =
      Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(numGlobalRows,numGlobalRows);

  // Get the row and column communicators
  const MPI_Comm& rowComm =
      Y->getDistribution()->getProcessorGrid()->getRowComm(n,false);
  const MPI_Comm& colComm =
      Y->getDistribution()->getProcessorGrid()->getColComm(n,false);
  int numRowProcs, numColProcs;
  MPI_Comm_size(rowComm,&numRowProcs);
  MPI_Comm_size(colComm,&numColProcs);

  // Create buffer for all-reduce
  scalar_t* allRedBuf;
  if(numLocalRows > 0)
    allRedBuf = Tucker::MemoryManager::safe_new_array<scalar_t>(numGlobalRows*numLocalRows);
  else
    allRedBuf = 0;

  if(numLocalRows > 0) {
    const MPI_Comm& colCommSqueezed =
        Y->getDistribution()->getProcessorGrid()->getColComm(n,true);

    // Stores the local Gram result
    Tucker::Matrix<scalar_t>* localMatrix;

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
        localMatrix = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(numGlobalRows,numLocalRows);

        // Determine the amount of data being received
        int ndims = Y->getNumDimensions();
        const Tucker::SizeArray& sz = Y->getLocalSize();
        int maxNumRows = Y->getDistribution()->getMap(n,true)->getMaxNumEntries();
        size_t numCols = sz.prod(0,n-1,1)*sz.prod(n+1,ndims-1,1);
        size_t maxEntries = maxNumRows*numCols;

        // Create buffer for receiving data
        scalar_t* recvBuf = Tucker::MemoryManager::safe_new_array<scalar_t>(maxEntries);

        // Send data to the next proc in column
        MPI_Request* sendRequests =
            Tucker::MemoryManager::safe_new_array<MPI_Request>(numColProcsSqueezed);
        size_t numToSend = Y->getLocalNumEntries();
        assert(numToSend <= std::numeric_limits<int>::max());
        int tag = 0;
        int sendDest = (myColRankSqueezed+1)%numColProcsSqueezed;
        if(shift_timer) shift_timer->start();
        MPI_Isend_(Y->getLocalTensor()->data(), (int)numToSend, sendDest, tag, 
            colCommSqueezed, sendRequests+sendDest);
        if(shift_timer) shift_timer->stop();

        // Receive information from the previous proc in column
        MPI_Request* recvRequests =
            Tucker::MemoryManager::safe_new_array<MPI_Request>(numColProcsSqueezed);
        int recvSource =
            (numColProcsSqueezed+myColRankSqueezed-1)%numColProcsSqueezed;
        int numRowsToReceive =
            Y->getDistribution()->getMap(n,true)->getNumEntries(recvSource);
        size_t numToReceive = numRowsToReceive*numCols;
        assert(numToReceive <= std::numeric_limits<int>::max());

        if(shift_timer) shift_timer->start();
        MPI_Irecv_(recvBuf, (int)numToReceive, recvSource, tag, colCommSqueezed, 
            recvRequests+recvSource);
        if(shift_timer) shift_timer->stop();

        // Local computation (dsyrk)
        scalar_t* Cptr = localMatrix->data() +
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
            MPI_Isend_(Y->getLocalTensor()->data(), (int)numToSend, 
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
            assert(numToReceive <= std::numeric_limits<int>::max());
            MPI_Irecv_(recvBuf, numToReceive, recvSource, tag, colCommSqueezed, 
                recvRequests+recvSource);
          }
          if(shift_timer) shift_timer->stop();
        }

        // Wait for all data to be sent
        MPI_Status* sendStatuses = Tucker::MemoryManager::safe_new_array<MPI_Status>(numColProcsSqueezed);
        if(shift_timer) shift_timer->start();
        if(myColRankSqueezed > 0) {
          MPI_Waitall(myColRankSqueezed, sendRequests, sendStatuses);
        }
        if(myColRankSqueezed < numColProcsSqueezed-1) {
          MPI_Waitall(numColProcsSqueezed-(myColRankSqueezed+1),
              sendRequests+myColRankSqueezed+1, sendStatuses+myColRankSqueezed+1);
        }
        if(shift_timer) shift_timer->stop();

        Tucker::MemoryManager::safe_delete_array(recvBuf,maxEntries);
        Tucker::MemoryManager::safe_delete_array(sendRequests,numColProcsSqueezed);
        Tucker::MemoryManager::safe_delete_array(recvRequests,numColProcsSqueezed);
        Tucker::MemoryManager::safe_delete_array(sendStatuses,numColProcsSqueezed);
      }
    }
    else {
      localMatrix = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(numGlobalRows,numLocalRows);
      localMatrix->initialize();
    }

    // All reduce across row communicator
    if(numRowProcs > 1) {
      if(allreduce_timer) allreduce_timer->start();
      MPI_Allreduce_(localMatrix->data(), allRedBuf, numLocalRows*numGlobalRows,
          MPI_SUM, rowComm);
      if(allreduce_timer) allreduce_timer->stop();
    }
    else {
      size_t nnz = localMatrix->getNumElements();
      for(size_t i=0; i<nnz; i++) {
        allRedBuf[i] = localMatrix->data()[i];
      }
    }

    Tucker::MemoryManager::safe_delete(localMatrix);
  }
  else {
  }

  if(numColProcs > 1) {
    // All-gather across column communicator
    int* recvcounts = Tucker::MemoryManager::safe_new_array<int>(numColProcs);
    int* displs = Tucker::MemoryManager::safe_new_array<int>(numColProcs);
    for(int i=0; i<numColProcs; i++) {
      recvcounts[i] = Y->getDistribution()->getMap(n,false)->getNumEntries(i)*numGlobalRows;
      displs[i] = Y->getDistribution()->getMap(n,false)->getOffset(i)*numGlobalRows;
    }
    if(allgather_timer) allgather_timer->start();
    MPI_Allgatherv_(allRedBuf, numLocalRows*numGlobalRows, gram->data(), recvcounts, 
        displs, colComm);
    if(allgather_timer) allgather_timer->stop();

    // Free memory
    Tucker::MemoryManager::safe_delete_array<int>(recvcounts,numColProcs);
    Tucker::MemoryManager::safe_delete_array<int>(displs,numColProcs);
  }
  else {
    size_t nnz = gram->getNumElements();
    for(size_t i=0; i<nnz; i++)
      gram->data()[i] = allRedBuf[i];
  }

  // Free memory
  Tucker::MemoryManager::safe_delete_array(allRedBuf,numGlobalRows*numLocalRows);

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
template <class scalar_t>
Tucker::Matrix<scalar_t>* newGram(const Tensor<scalar_t>* Y, const int n,
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
  const Tucker::Matrix<scalar_t>* localGram;
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
      Tucker::Matrix<scalar_t>* temp = 
        Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(nGlobalRows,nGlobalRows);
      temp->initialize();
      localGram = temp;
    }
    else {
      // Redistribute the data
      const Matrix<scalar_t>* redistributedY = redistributeTensorForGram(Y, n,
          pack_timer, alltoall_timer, unpack_timer);

      // Call symmetric rank-k update
      if(redistributedY->getLocalNumEntries() > 0) {
        if(mult_timer) mult_timer->start();
        localGram = localRankKForGram(redistributedY, n, ndims);
        if(mult_timer) mult_timer->stop();
      }
      else {
        int nGlobalRows = Y->getGlobalSize(n);
        Tucker::Matrix<scalar_t>* temp = 
          Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(nGlobalRows,nGlobalRows);
        temp->initialize();
        localGram = temp;
      }
      Tucker::MemoryManager::safe_delete(redistributedY);
    } // end if(!myColEmpty)
  }
  else {
    if(Y->getDistribution()->ownNothing()) {
      int nGlobalRows = Y->getGlobalSize(n);
      Tucker::Matrix<scalar_t>* temp = 
        Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(nGlobalRows,nGlobalRows);
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
  Tucker::Matrix<scalar_t>* gramMat = reduceForGram(localGram);
  if(allreduce_timer) allreduce_timer->stop();
  Tucker::MemoryManager::safe_delete(localGram);

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

// \todo STHOSVD is never tested with the new Gram computation
template <class scalar_t>
const TuckerTensor<scalar_t>* STHOSVD(const Tensor<scalar_t>* const X,
    const scalar_t epsilon, int* modeOrder, bool useOldGram, bool flipSign,
    bool useLQ)
{
  // Get this rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ndims = X->getNumDimensions();

  // Create a struct to store the factorization
  TuckerTensor<scalar_t>* factorization = 
    Tucker::MemoryManager::safe_new<TuckerTensor<scalar_t>>(ndims);

  // Compute the threshold
  scalar_t tensorNorm = X->norm2();
  scalar_t thresh = epsilon*epsilon*tensorNorm/ndims;
  if(rank == 0) {
    std::cout << "\tAutoST-HOSVD::Tensor Norm: "
        << std::sqrt(tensorNorm) << "...\n";
    std::cout << "\tAutoST-HOSVD::Relative Threshold: "
        << thresh << "...\n";
  }

  // Compute the nnz of the largest tensor piece being stored by any process
  size_t max_lcl_nnz_x = 1;
  for(int i=0; i<ndims; i++) {
    max_lcl_nnz_x *= X->getDistribution()->getMap(i,false)->getMaxNumEntries();
  }

  // Barrier for timing
  MPI_Barrier(MPI_COMM_WORLD);
  factorization->total_timer_.start();

  const Tensor<scalar_t>* Y = X;

  // For each dimension...
  for(int n=0; n<ndims; n++)
  {
    int mode = modeOrder? modeOrder[n] : n;
    if(useLQ){
      if(rank == 0) std::cout << "\tAutoST-HOSVD::Starting LQ(" << mode << ")...\n";
      factorization->LQ_timer_[mode].start();
      Tucker::Matrix<scalar_t>* L = LQ(Y, mode, &factorization->LQ_tsqr_timer_[mode], &factorization->LQ_localqr_timer_[mode], 
        &factorization->LQ_redistribute_timer_[mode], &factorization->LQ_dcopy_timer_[mode],
        &factorization->LQ_decompose_timer_[mode], &factorization->LQ_transpose_timer_[mode]);
      factorization->LQ_timer_[mode].stop();
      int SizeOfL = L->nrows()*L->ncols();
      factorization->LQ_bcast_timer_[mode].start();
      MPI_Bcast(L->data(), SizeOfL, 0, MPI_COMM_WORLD);
      factorization->LQ_bcast_timer_[mode].stop();
      if(rank == 0) std::cout << "\tAutoST-HOSVD::Starting computeSVD(" << mode << ")...\n";
      factorization->svd_timer_[mode].start();
      Tucker::computeSVD(L, factorization->singularValues[mode], factorization->U[mode], thresh);
      factorization->svd_timer_[mode].stop();
      Tucker::MemoryManager::safe_delete(L);
    }
    else{
      // Compute the Gram matrix
      // S = Y_n*Y_n'
      Tucker::Matrix<scalar_t>* S;
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::Starting Gram(" << mode << ")...\n";
      }
      factorization->gram_timer_[mode].start();
      if(useOldGram) {
        S = oldGram(Y, mode,
            &factorization->gram_matmul_timer_[mode],
            &factorization->gram_shift_timer_[mode],
            &factorization->gram_allreduce_timer_[mode],
            &factorization->gram_allgather_timer_[mode]);
      }
      else {
        S = newGram(Y, mode,
            &factorization->gram_matmul_timer_[mode],
            &factorization->gram_pack_timer_[mode],
            &factorization->gram_alltoall_timer_[mode],
            &factorization->gram_unpack_timer_[mode],
            &factorization->gram_allreduce_timer_[mode]);
      }
      factorization->gram_timer_[mode].stop();
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::Gram(" << mode << ") time: "
            << factorization->gram_timer_[mode].duration() << "s\n";
      }
      // Compute the relevant eigenpairs
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::Starting Evecs(" << mode << ")...\n";
      }
      factorization->eigen_timer_[mode].start();
      Tucker::computeEigenpairs(S, factorization->eigenvalues[mode],
        factorization->U[mode], thresh, flipSign);
      factorization->eigen_timer_[mode].stop();
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::EVECS(" << mode << ") time: "
            << factorization->eigen_timer_[mode].duration() << "s\n";
        std::cout << "\t\tEvecs(" << mode << ")::Local EV time: "
            << factorization->eigen_timer_[mode].duration() << "s\n";
      }
      // Free the Gram matrix
      Tucker::MemoryManager::safe_delete(S);
    }


    // Perform the tensor times matrix multiplication
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting TTM(" << mode << ")...\n";
    }
    factorization->ttm_timer_[mode].start();
    Tensor<scalar_t>* temp = ttm(Y,mode,factorization->U[mode],true,
        &factorization->ttm_matmul_timer_[mode],
        &factorization->ttm_pack_timer_[mode],
        &factorization->ttm_reducescatter_timer_[mode],
        &factorization->ttm_reduce_timer_[mode],
        max_lcl_nnz_x);
    factorization->ttm_timer_[mode].stop();
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::TTM(" << mode << ") time: "
          << factorization->ttm_timer_[mode].duration() << "s\n";
    }

    if(n > 0) {
      Tucker::MemoryManager::safe_delete(Y);
    }
    Y = temp;

    if(rank == 0) {
      size_t local_nnz = Y->getLocalNumEntries();
      size_t global_nnz = Y->getGlobalNumEntries();
      std::cout << "Local tensor size after STHOSVD iteration "
          << mode << ": " << Y->getLocalSize() << ", or ";
      Tucker::printBytes(local_nnz*sizeof(scalar_t));
      std::cout << "Global tensor size after STHOSVD iteration "
          << mode << ": " << Y->getGlobalSize() << ", or ";
      Tucker::printBytes(global_nnz*sizeof(scalar_t));
    }
  }

  factorization->G = const_cast<Tensor<scalar_t>*>(Y);
  factorization->total_timer_.stop();
  return factorization;
}

// \todo This function is never tested
template <class scalar_t>
const TuckerTensor<scalar_t>* STHOSVD(const Tensor<scalar_t>* const X,
    const Tucker::SizeArray* const reducedI, int* modeOrder, bool useOldGram,
    bool flipSign, bool useLQ)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int ndims = X->getNumDimensions();
  assert(ndims == reducedI->size());

  // Create a struct to store the factorization
  TuckerTensor<scalar_t>* factorization = 
    Tucker::MemoryManager::safe_new<TuckerTensor<scalar_t>>(ndims);

  // Compute the nnz of the largest tensor piece being stored by any process
  size_t max_lcl_nnz_x = 1;
  for(int i=0; i<ndims; i++) {
    max_lcl_nnz_x *= X->getDistribution()->getMap(i,false)->getMaxNumEntries();
  }

  // Barrier for timing
  MPI_Barrier(MPI_COMM_WORLD);
  factorization->total_timer_.start();

  const Tensor<scalar_t>* Y = X;

  // For each dimension...
  for(int n=0; n<ndims; n++)
  {
    int mode = modeOrder ? modeOrder[n] : n;
    if(useLQ){
      Tucker::Matrix<scalar_t>* L;
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::Starting LQ(" << mode << ")...\n";
      }
      factorization->LQ_timer_[mode].start();
      L = LQ(Y, mode, 
        &factorization->LQ_tsqr_timer_[mode], 
        &factorization->LQ_localqr_timer_[mode], 
        &factorization->LQ_redistribute_timer_[mode], 
        &factorization->LQ_dcopy_timer_[mode],
        &factorization->LQ_decompose_timer_[mode], 
        &factorization->LQ_transpose_timer_[mode]);
      factorization->LQ_timer_[mode].stop();
      int SizeOfL = L->nrows()*L->ncols();
      factorization->LQ_bcast_timer_[mode].start();
      MPI_Bcast(L->data(), SizeOfL, 0, MPI_COMM_WORLD);
      factorization->LQ_bcast_timer_[mode].stop();
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::Starting computeSVD(" << mode << ")...\n";
      }
      factorization->svd_timer_[mode].start();
      Tucker::computeSVD(L, factorization->singularValues[mode], factorization->U[mode], (*reducedI)[mode]);
      factorization->svd_timer_[mode].stop();
      Tucker::MemoryManager::safe_delete(L);
    }
    else{
      // Compute the Gram matrix
      // S = Y_n*Y_n'
      Tucker::Matrix<scalar_t>* S;
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::Starting Gram(" << mode << ")...\n";
      }
      factorization->gram_timer_[mode].start();
      if(useOldGram) {
        S = oldGram(Y, mode,
            &factorization->gram_matmul_timer_[mode],
            &factorization->gram_shift_timer_[mode],
            &factorization->gram_allreduce_timer_[mode],
            &factorization->gram_allgather_timer_[mode]);
      }
      else {
        S = newGram(Y, mode,
            &factorization->gram_matmul_timer_[mode],
            &factorization->gram_pack_timer_[mode],
            &factorization->gram_alltoall_timer_[mode],
            &factorization->gram_unpack_timer_[mode],
            &factorization->gram_allreduce_timer_[mode]);
      }
      factorization->gram_timer_[mode].stop();
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::Gram(" << mode << ") time: "
            << factorization->gram_timer_[mode].duration() << "s\n";
      }

      // Compute the leading eigenvectors of S
      // call dsyev(jobz, uplo, n, a, lda, w, work, lwork, info)
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::Starting Evecs(" << mode << ")...\n";
      }
      factorization->eigen_timer_[mode].start();
      Tucker::computeEigenpairs(S, factorization->eigenvalues[mode],
          factorization->U[mode], (*reducedI)[mode], flipSign);
      factorization->eigen_timer_[mode].stop();
      
      Tucker::MemoryManager::safe_delete(S);
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::EVECS(" << mode << ") time: "
            << factorization->eigen_timer_[mode].duration() << "s\n";
      }
    }

    // Perform the tensor times matrix multiplication
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting TTM(" << mode << ")...\n";
    }
    factorization->ttm_timer_[mode].start();
    Tensor<scalar_t>* temp = ttm(Y,mode,factorization->U[mode],true,
            &factorization->ttm_matmul_timer_[mode],
            &factorization->ttm_pack_timer_[mode],
            &factorization->ttm_reducescatter_timer_[mode],
            &factorization->ttm_reduce_timer_[mode],
            max_lcl_nnz_x);
    factorization->ttm_timer_[mode].stop();
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::TTM(" << mode << ") time: "
          << factorization->ttm_timer_[mode].duration() << "s\n";
    }
    if(n > 0) {
      Tucker::MemoryManager::safe_delete(Y);
    }
    Y = temp;

    if(rank == 0) {
      size_t local_nnz = Y->getLocalNumEntries();
      size_t global_nnz = Y->getGlobalNumEntries();
      std::cout << "Local tensor size after STHOSVD iteration "
          << mode << ": " << Y->getLocalSize() << ", or ";
      Tucker::printBytes(local_nnz*sizeof(scalar_t));
      std::cout << "Global tensor size after STHOSVD iteration "
          << mode << ": " << Y->getGlobalSize() << ", or ";
      Tucker::printBytes(global_nnz*sizeof(scalar_t));
    }
  }

  factorization->G = const_cast<Tensor<scalar_t>*>(Y);
  factorization->total_timer_.stop();
  return factorization;
}

template <class scalar_t>
Tucker::MetricData<scalar_t>* computeSliceMetrics(const Tensor<scalar_t>* const Y,
    int mode, int metrics)
{
  // If I don't own any slices, I don't have any work to do.
  if(Y->getLocalSize(mode) == 0) {
    return 0;
  }

  // Compute the local result
  Tucker::MetricData<scalar_t>* result =
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
    scalar_t* sendBuf = Tucker::MemoryManager::safe_new_array<scalar_t>(numSlices);
    scalar_t* recvBuf = Tucker::MemoryManager::safe_new_array<scalar_t>(numSlices);
    if(metrics & Tucker::MIN) {
      for(int i=0; i<numSlices; i++) sendBuf[i] = result->getMinData()[i];
      MPI_Allreduce_(sendBuf, recvBuf, numSlices, MPI_MIN, comm);
      for(int i=0; i<numSlices; i++) result->getMinData()[i] = recvBuf[i];
    }
    if(metrics & Tucker::MAX) {
      for(int i=0; i<numSlices; i++) sendBuf[i] = result->getMaxData()[i];
      MPI_Allreduce_(sendBuf, recvBuf, numSlices, MPI_MAX, comm);
      for(int i=0; i<numSlices; i++) result->getMaxData()[i] = recvBuf[i];
    }
    if(metrics & Tucker::SUM) {
      for(int i=0; i<numSlices; i++) sendBuf[i] = result->getSumData()[i];
      MPI_Allreduce_(sendBuf, recvBuf, numSlices, MPI_SUM, comm);
      for(int i=0; i<numSlices; i++) result->getSumData()[i] = recvBuf[i];
    }
    // If X is partitioned into X_A and X_B,
    // mean_X = (n_A mean_A + n_B mean_B) / (n_A + n_B)
    if((metrics & Tucker::MEAN) || (metrics & Tucker::VARIANCE)) {
      // Compute the size of my local slice
      int ndims = Y->getNumDimensions();
      const Tucker::SizeArray& localSize = Y->getLocalSize();
      size_t localSliceSize = localSize.prod(0,mode-1,1) *
          localSize.prod(mode+1,ndims-1,1);

      // Compute the size of the global slice
      const Tucker::SizeArray& globalSize = Y->getGlobalSize();
      size_t globalSliceSize = globalSize.prod(0,mode-1,1) *
          globalSize.prod(mode+1,ndims-1,1);

      for(int i=0; i<numSlices; i++) {
        sendBuf[i] = result->getMeanData()[i] * (scalar_t)localSliceSize;
      }

      MPI_Allreduce_(sendBuf, recvBuf, numSlices,
          MPI_SUM, comm);

      scalar_t* meanDiff;
      if(metrics & Tucker::VARIANCE) {
        meanDiff = Tucker::MemoryManager::safe_new_array<scalar_t>(numSlices);
        for(int i=0; i<numSlices; i++) {
          meanDiff[i] = result->getMeanData()[i] -
              recvBuf[i] / (scalar_t)globalSliceSize;
        }
      }

      for(int i=0; i<numSlices; i++) {
        result->getMeanData()[i] = recvBuf[i] / (scalar_t)globalSliceSize;
      }

      if(metrics & Tucker::VARIANCE) {
        for(int i=0; i<numSlices; i++) {
          // Source of this equation:
          // http://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
          sendBuf[i] = (scalar_t)localSliceSize*result->getVarianceData()[i] +
              (scalar_t)localSliceSize*meanDiff[i]*meanDiff[i];
        }

        Tucker::MemoryManager::safe_delete_array<scalar_t>(meanDiff,numSlices);

        MPI_Allreduce_(sendBuf, recvBuf, numSlices,
            MPI_SUM, comm);

        for(int i=0; i<numSlices; i++) {
          result->getVarianceData()[i] = recvBuf[i] / (scalar_t)globalSliceSize;
        }
      } // end if(metrics & Tucker::VARIANCE)
    } // end if((metrics & Tucker::MEAN) || (metrics & Tucker::VARIANCE))

    Tucker::MemoryManager::safe_delete_array(sendBuf,numSlices);
    Tucker::MemoryManager::safe_delete_array(recvBuf,numSlices);
  } // end if(nprocs > 1)

  return result;
}

// Shift is applied before scale
// We divide by scaleVals, not multiply
template <class scalar_t>
void transformSlices(Tensor<scalar_t>* Y, int mode, const scalar_t* scales, const scalar_t* shifts)
{
  Tucker::transformSlices(Y->getLocalTensor(), mode, scales, shifts);
}

template <class scalar_t>
void normalizeTensorStandardCentering(Tensor<scalar_t>* Y, int mode, scalar_t stdThresh)
{
  // I don't have to do any work because I don't own any data
  if(Y->getLocalSize(mode) == 0)
    return;

  Tucker::MetricData<scalar_t>* metrics =
      computeSliceMetrics(Y, mode, Tucker::MEAN+Tucker::VARIANCE);
  int sizeOfModeDim = Y->getLocalSize(mode);
  scalar_t* scales = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  scalar_t* shifts = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scales[i] = sqrt(metrics->getVarianceData()[i]);
    shifts[i] = -metrics->getMeanData()[i];
    if(std::abs(scales[i]) < stdThresh) {
      scales[i] = 1;
    }
  }
  transformSlices(Y,mode,scales,shifts);
  Tucker::MemoryManager::safe_delete_array(scales,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete_array(shifts,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete(metrics);
}

template <class scalar_t>
void normalizeTensorMinMax(Tensor<scalar_t>* Y, int mode)
{
  // I don't have to do any work because I don't own any data
  if(Y->getLocalSize(mode) == 0)
    return;

  Tucker::MetricData<scalar_t>* metrics = computeSliceMetrics(Y, mode,
      Tucker::MAX+Tucker::MIN);
  int sizeOfModeDim = Y->getLocalSize(mode);
  scalar_t* scales = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  scalar_t* shifts = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scales[i] = metrics->getMaxData()[i] - metrics->getMinData()[i];
    shifts[i] = -metrics->getMinData()[i];
  }
  transformSlices(Y,mode,scales,shifts);
  Tucker::MemoryManager::safe_delete_array(scales,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete_array(shifts,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete(metrics);
}

// \todo This function is never tested
template <class scalar_t>
void normalizeTensorMax(Tensor<scalar_t>* Y, int mode)
{
  // I don't have to do any work because I don't own any data
  if(Y->getLocalSize(mode) == 0)
    return;

  Tucker::MetricData<scalar_t>* metrics = computeSliceMetrics(Y, mode,
      Tucker::MIN + Tucker::MAX);
  int sizeOfModeDim = Y->getLocalSize(mode);
  scalar_t* scales = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  scalar_t* shifts = Tucker::MemoryManager::safe_new_array<scalar_t>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scalar_t scaleval = std::max(std::abs(metrics->getMinData()[i]),
        std::abs(metrics->getMaxData()[i]));
    scales[i] = scaleval;
    shifts[i] = 0;
  }
  transformSlices(Y,mode,scales,shifts);
  Tucker::MemoryManager::safe_delete_array(scales,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete_array(shifts,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete(metrics);
}

template <class scalar_t>
const Tensor<scalar_t>* reconstructSingleSlice(const TuckerTensor<scalar_t>* fact,
    const int mode, const int sliceNum)
{
  assert(mode >= 0 && mode < fact->N);
  assert(sliceNum >= 0 && sliceNum < fact->U[mode]->nrows());

  // Process mode first, in order to minimize storage requirements

  // Copy row of matrix
  int nrows = fact->U[mode]->nrows();
  int ncols = fact->U[mode]->ncols();
  Tucker::Matrix<scalar_t> tempMat(1,ncols);
  const scalar_t* olddata = fact->U[mode]->data();
  scalar_t* rowdata = tempMat.data();
  for(int j=0; j<ncols; j++)
    rowdata[j] = olddata[j*nrows+sliceNum];
  Tensor<scalar_t>* ten = ttm(fact->G, mode, &tempMat);

  for(int i=0; i<fact->N; i++)
  {
    Tucker::Matrix<scalar_t>* tempMat;
    if(i == mode)
      continue;  
    
    Tensor<scalar_t>* temp = ttm(ten, i, fact->U[i]);

    Tucker::MemoryManager::safe_delete(ten);
    ten = temp;
  }

  return ten;
}

template <class scalar_t>
void readTensorBinary(std::string& filename, Tensor<scalar_t>& Y)
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
template <class scalar_t>
void importTensorBinary(const char* filename, Tensor<scalar_t>* Y)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if(Y->getDistribution()->ownNothing()) {
    return;
  }

  int ndims = Y->getNumDimensions();

  // Define data layout parameters
  int* starts = Tucker::MemoryManager::safe_new_array<int>(ndims);
  int* lsizes = Tucker::MemoryManager::safe_new_array<int>(ndims);
  int* gsizes = Tucker::MemoryManager::safe_new_array<int>(ndims);

  for(int i=0; i<ndims; i++) {
    starts[i] = Y->getDistribution()->getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y->getLocalSize(i);
    gsizes[i] = Y->getGlobalSize(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray_<scalar_t>(ndims, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, &view);
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
  MPI_File_set_view_<scalar_t>(fh, disp, view, "native", MPI_INFO_NULL);

  // Read the file
  size_t count = Y->getLocalNumEntries();
  assert(count <= std::numeric_limits<int>::max());
  if(rank == 0 && 8*count > std::numeric_limits<int>::max()) {
    std::cout << "WARNING: We are attempting to call MPI_File_read_all to read ";
    Tucker::printBytes(8*count);
    std::cout << "Depending on your MPI implementation, this may fail "
              << "because you are trying to read over 2.1 GB.\nIf MPI_File_read_all"
              << " crashes, please try again with a more favorable processor grid.\n";
  }

  MPI_Status status;
  ret = MPI_File_read_all_(fh, Y->getLocalTensor()->data(),
      (int)count, &status);
  int nread;
  MPI_Get_count_<scalar_t>(&status, &nread);
  if(ret != MPI_SUCCESS) {
    std::cerr << "Error: Could not read file " << filename << std::endl;
  }

  // Close the file
  MPI_File_close(&fh);

  // Free the datatype
  MPI_Type_free(&view);

  // Free other memory
  Tucker::MemoryManager::safe_delete_array(starts,ndims);
  Tucker::MemoryManager::safe_delete_array(lsizes,ndims);
  Tucker::MemoryManager::safe_delete_array(gsizes,ndims);
}

// This function assumes that Y has already been allocated
// and its values need to be filled in
template <class scalar_t>
void importTensorBinary(const char* filename, Tucker::Tensor<scalar_t>* Y)
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
  size_t count = Y->size().prod();
  assert(count <= std::numeric_limits<int>::max());
  scalar_t * data = Y->data();
  MPI_Status status;
  ret = MPI_File_read_(fh, data, (int)count, &status);
  if(ret != MPI_SUCCESS && rank == 0) {
    std::cerr << "Error: Could not read file " << filename << std::endl;
  }

  // Close the file
  MPI_File_close(&fh);
}

template <class scalar_t>
void importTimeSeries(const char* filename, Tensor<scalar_t>* Y)
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
  int* starts = Tucker::MemoryManager::safe_new_array<int>(ndims-1);
  int* lsizes = Tucker::MemoryManager::safe_new_array<int>(ndims-1);
  int* gsizes = Tucker::MemoryManager::safe_new_array<int>(ndims-1);

  for(int i=0; i<ndims-1; i++) {
    starts[i] = Y->getDistribution()->getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y->getLocalSize(i);
    gsizes[i] = Y->getGlobalSize(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray_<scalar_t>(ndims-1, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, &view);
  MPI_Type_commit(&view);

  int nsteps = Y->getGlobalSize(ndims-1);
  const Map* stepMap = Y->getDistribution()->getMap(ndims-1,true);
  const MPI_Comm& stepComm = Y->getDistribution()->getProcessorGrid()->getRowComm(ndims-1,true);
  scalar_t* dataPtr = Y->getLocalTensor()->data();
  size_t count = Y->getLocalSize().prod(0,ndims-2);
  assert(count <= std::numeric_limits<int>::max());
  if(rank == 0 && 8*count > std::numeric_limits<int>::max()) {
    std::cout << "WARNING: We are attempting to call MPI_File_read_all to read ";
    Tucker::printBytes(8*count);
    std::cout << "Depending on your MPI implementation, this may fail "
              << "because you are trying to read over 2.1 GB.\nIf MPI_File_read_all"
              << " crashes, please try again with a more favorable processor grid.\n";
  }

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
    MPI_File_set_view_<scalar_t>(fh, disp, view, "native", MPI_INFO_NULL);

    // Read the file
    MPI_Status status;
    ret = MPI_File_read_all_(fh, dataPtr,
        (int)count, &status);
    if(ret != MPI_SUCCESS && rank == 0) {
      std::cerr << "Error: Could not read file " << stepFilename << std::endl;
      exit(1);
    }

    // Close the file
    MPI_File_close(&fh);

    // Increment the pointer
    dataPtr += count;
  }

  ifs.close();

  // Free the datatype
  MPI_Type_free(&view);

  // Free other memory
  Tucker::MemoryManager::safe_delete_array<int>(starts,ndims-1);
  Tucker::MemoryManager::safe_delete_array<int>(lsizes,ndims-1);
  Tucker::MemoryManager::safe_delete_array<int>(gsizes,ndims-1);
}

template <class scalar_t>
void writeTensorBinary(std::string& filename, const Tensor<scalar_t>& Y)
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

template <class scalar_t>
void exportTensorBinary(const char* filename, const Tensor<scalar_t>* Y)
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
  int* starts = Tucker::MemoryManager::safe_new_array<int>(ndims);
  int* lsizes = Tucker::MemoryManager::safe_new_array<int>(ndims);
  int* gsizes = Tucker::MemoryManager::safe_new_array<int>(ndims);

  for(int i=0; i<ndims; i++) {
    starts[i] = Y->getDistribution()->getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y->getLocalSize(i);
    gsizes[i] = Y->getGlobalSize(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray_<scalar_t>(ndims, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, &view);
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
  MPI_File_set_view_<scalar_t>(fh, disp, view, "native", MPI_INFO_NULL);

  // Write the file
  size_t count = Y->getLocalNumEntries();
  assert(count <= std::numeric_limits<int>::max());
  MPI_Status status;
  ret = MPI_File_write_all_(fh, Y->getLocalTensor()->data(), (int)count,
      &status);
  if(ret != MPI_SUCCESS && rank == 0) {
    std::cerr << "Error: Could not write to file " << filename << std::endl;
  }

  // Close the file
  MPI_File_close(&fh);

  // Free the datatype
  MPI_Type_free(&view);

  // Free other memory
  Tucker::MemoryManager::safe_delete_array<int>(starts,ndims);
  Tucker::MemoryManager::safe_delete_array<int>(lsizes,ndims);
  Tucker::MemoryManager::safe_delete_array<int>(gsizes,ndims);
}

template <class scalar_t>
void exportTensorBinary(const char* filename, const Tucker::Tensor<scalar_t>* Y)
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
  size_t nentries = Y->size().prod();
  assert(nentries <= std::numeric_limits<int>::max());
  const scalar_t* entries = Y->data();
  MPI_Status status;
  ret = MPI_File_write_(fh, entries, (int)nentries, &status);
  if(ret != MPI_SUCCESS && rank == 0) {
    std::cerr << "Error: Could not write file " << filename << std::endl;
  }

  // Close the file
  MPI_File_close(&fh);
}

template <class scalar_t>
void exportTimeSeries(const char* filename, const Tensor<scalar_t>* Y)
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
  int* starts = Tucker::MemoryManager::safe_new_array<int>(ndims-1);
  int* lsizes = Tucker::MemoryManager::safe_new_array<int>(ndims-1);
  int* gsizes = Tucker::MemoryManager::safe_new_array<int>(ndims-1);

  for(int i=0; i<ndims-1; i++) {
    starts[i] = Y->getDistribution()->getMap(i,true)->getGlobalIndex(0);
    lsizes[i] = Y->getLocalSize(i);
    gsizes[i] = Y->getGlobalSize(i);
  }

  // Create the datatype associated with this layout
  MPI_Datatype view;
  MPI_Type_create_subarray_<scalar_t>(ndims-1, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, &view);
  MPI_Type_commit(&view);

  int nsteps = Y->getGlobalSize(ndims-1);
  const Map* stepMap = Y->getDistribution()->getMap(ndims-1,true);
  const MPI_Comm& stepComm = Y->getDistribution()->getProcessorGrid()->getRowComm(ndims-1,true);
  const scalar_t* dataPtr = Y->getLocalTensor()->data();
  size_t count = Y->getLocalSize().prod(0,ndims-2);
  assert(count <= std::numeric_limits<int>::max());
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
    MPI_File_set_view_<scalar_t>(fh, disp, view, "native", MPI_INFO_NULL);

    // Write the file
    MPI_Status status;
    ret = MPI_File_write_all_(fh, dataPtr, (int)count, &status);
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
  Tucker::MemoryManager::safe_delete_array(starts,ndims-1);
  Tucker::MemoryManager::safe_delete_array(lsizes,ndims-1);
  Tucker::MemoryManager::safe_delete_array(gsizes,ndims-1);
}

Tensor<scalar_t>* generateTensor(int seed, TuckerTensor<scalar_t>* fact, Tucker::SizeArray* proc_dims, 
  Tucker::SizeArray* tensor_dims, Tucker::SizeArray* core_dims,
  scalar_t noise){
  if(proc_dims->size() != tensor_dims->size()){
    throw std::runtime_error("TuckerMPI::generateTensor(): processor grid dimension doesn't match that of the output tensor");
  }
  if(tensor_dims->size() != core_dims->size()){
    throw std::runtime_error("TuckerMPI::generateTensor(): output tensor dimension doesn't match that of the core tensor");
  }
  if(core_dims->size() != fact->N){
    throw std::runtime_error("TuckerMPI::generateTensor(): core tensor grid dimension doesn't match that of the TuckerTensor");
  }
  int rank;
  int nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  /////////////////////////////////////////////////////////////////
  // Generate the seeds for each MPI process and scatter them    //
  // Given the input seed, the seed that each processor use will //
  // be the same each run yet different from each other.         //
  /////////////////////////////////////////////////////////////////
  int myseed;
  if(rank == 0) {
    unsigned* seeds = Tucker::MemoryManager::safe_new_array<unsigned>(nprocs);
    srand(seed);
    for(int i=0; i<nprocs; i++) {
      seeds[i] = rand();
    }
    MPI_Scatter(seeds,1,MPI_INT,&myseed,1,MPI_INT,0,MPI_COMM_WORLD);
    Tucker::MemoryManager::safe_delete_array<unsigned>(seeds,nprocs);
  }
  else {
    MPI_Scatter(NULL,1,MPI_INT,&myseed,1,MPI_INT,0,MPI_COMM_WORLD);
  }
  std::default_random_engine generator(myseed);
  std::normal_distribution<scalar_t> distribution;
  //GENERATE CORE TENSOR//
  //distribution for the core
  TuckerMPI::Distribution<scalar_t>* dist =
      Tucker::MemoryManager::safe_new<TuckerMPI::Distribution<scalar_t>>(*core_dims, *proc_dims);
  fact->G = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor<scalar_t>>(dist);
  size_t nnz = dist->getLocalDims().prod();
  scalar_t* dataptr = fact->G->getLocalTensor()->data();
  for(size_t i=0; i<nnz; i++) {
    dataptr[i] = distribution(generator);
  }
  //GENERATE FACTOR MATRICES//
  for(int d=0; d<fact->N; d++) {
    if(rank == 0) std::cout << "Generating factor matrix " << d << "...\n";
    int nrows = (*tensor_dims)[d];
    int ncols = (*core_dims)[d];
    fact->U[d] = Tucker::MemoryManager::safe_new<Tucker::Matrix<scalar_t>>(nrows,ncols);
    nnz = nrows*ncols;
    dataptr = fact->U[d]->data();
    if(rank == 0) {
      for(size_t i=0; i<nnz; i++) {
        dataptr[i] = distribution(generator);
      }
    }

    MPI_Bcast(dataptr,nnz,0,MPI_COMM_WORLD);
  }
  //TTM between factor matrices and core//
  TuckerMPI::Tensor<scalar_t>* product = fact->G;
  for(int d=0; d<fact->N; d++) {
    TuckerMPI::Tensor<scalar_t>* temp = TuckerMPI::ttm(product, d, fact->U[d]);
    if(product != fact->G) {
      Tucker::MemoryManager::safe_delete(product);
    }
    product = temp;
  }
  /////////////////////////////////////////////////////////////////////
  // Compute the norm of the global tensor                           //
  // \todo This could be more efficient; see Bader/Kolda for details //
  /////////////////////////////////////////////////////////////////////
  scalar_t normM = std::sqrt(product->norm2());
  ///////////////////////////////////////////////////////////////////
  // Compute the estimated norm of the noise matrix                //
  // The average of each element squared is the standard deviation //
  // squared, so this quantity should be sqrt(nnz * stdev^2)       //
  ///////////////////////////////////////////////////////////////////
  nnz = tensor_dims->prod();
  scalar_t normN = std::sqrt(nnz);
  scalar_t alpha = noise*normM/normN;
  //add noise to product
  dataptr = product->getLocalTensor()->data();
  nnz = dist->getLocalDims().prod();
  for(size_t i=0; i<nnz; i++) {
    dataptr[i] += alpha*distribution(generator);
  }
  return product;
}
// Explicit instantiations to build static library for both single and double precision
template Tucker::Matrix<float>* oldGram(const Tensor<float>*, const int,
    Tucker::Timer*, Tucker::Timer*, Tucker::Timer*, Tucker::Timer*);
template Tucker::Matrix<float>* newGram(const Tensor<float>*, const int,
    Tucker::Timer*, Tucker::Timer*, Tucker::Timer*, Tucker::Timer*, Tucker::Timer*);
template const TuckerTensor<float>* STHOSVD(const Tensor<float>* const, const float, 
    bool, bool);
template const TuckerTensor<float>* STHOSVD(const Tensor<float>* const, 
    const Tucker::SizeArray* const, bool, bool);
template Tucker::MetricData<float>* computeSliceMetrics(const Tensor<float>* const,
    int, int);
template void transformSlices(Tensor<float>*, int, const float*, const float*);
template void normalizeTensorStandardCentering(Tensor<float>*, int, float);
template void normalizeTensorMinMax(Tensor<float>*, int);
template void normalizeTensorMax(Tensor<float>*, int);
template const Tensor<float>* reconstructSingleSlice(const TuckerTensor<float>*,
    const int, const int);
template void readTensorBinary(std::string&, Tensor<float>&);
template void importTensorBinary(const char*, Tucker::Tensor<float>*);
template void importTimeSeries(const char*, Tensor<float>*);
template void writeTensorBinary(std::string&, const Tensor<float>&);
template void exportTensorBinary(const char*, const Tensor<float>*);
template void exportTensorBinary(const char*, const Tucker::Tensor<float>*);
template void exportTimeSeries(const char*, const Tensor<float>*);

template Tucker::Matrix<double>* oldGram(const Tensor<double>*, const int,
    Tucker::Timer*, Tucker::Timer*, Tucker::Timer*, Tucker::Timer*);
template Tucker::Matrix<double>* newGram(const Tensor<double>*, const int,
    Tucker::Timer*, Tucker::Timer*, Tucker::Timer*, Tucker::Timer*, Tucker::Timer*);
template const TuckerTensor<double>* STHOSVD(const Tensor<double>* const, const double, 
    bool, bool);
template const TuckerTensor<double>* STHOSVD(const Tensor<double>* const, 
    const Tucker::SizeArray* const, bool, bool);
template Tucker::MetricData<double>* computeSliceMetrics(const Tensor<double>* const,
    int, int);
template void transformSlices(Tensor<double>*, int, const double*, const double*);
template void normalizeTensorStandardCentering(Tensor<double>*, int, double);
template void normalizeTensorMinMax(Tensor<double>*, int);
template void normalizeTensorMax(Tensor<double>*, int);
template const Tensor<double>* reconstructSingleSlice(const TuckerTensor<double>*,
    const int, const int);
template void readTensorBinary(std::string&, Tensor<double>&);
template void importTensorBinary(const char*, Tucker::Tensor<double>*);
template void importTimeSeries(const char*, Tensor<double>*);
template void writeTensorBinary(std::string&, const Tensor<double>&);
template void exportTensorBinary(const char*, const Tensor<double>*);
template void exportTensorBinary(const char*, const Tucker::Tensor<double>*);
template void exportTimeSeries(const char*, const Tensor<double>*);

} // end namespace TuckerMPI
