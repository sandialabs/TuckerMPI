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
//Compute the LQ of the local matrix, transpose it, do TSQR and then transpose it back to an L.
Tucker::Matrix* LQ(const Tensor* Y, const int n, bool useButterflyTSQR, Tucker::Timer* tsqr_timer,
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
    Rncols = L->nrows();
    Rnrows = L->ncols(); 
    //Do an explicit transpose of R.
    R = Tucker::MemoryManager::safe_new<Tucker::Matrix>(Rnrows, Rncols);
    for(int i=0; i<L->ncols(); i++){
      dcopy_(&Rncols, L->data()+i*Rncols, &one, R->data()+i, &Rnrows);
    }
    Tucker::MemoryManager::safe_delete<Tucker::Matrix>(L);
  }
  else{
    bool isLastMode = n == Y->getNumDimensions()-1;
    R = localQR(redistYn, isLastMode, localqr_dcopy_timer, localqr_decompose_timer, localqr_transpose_timer);
    Rnrows = R->nrows();
    Rncols = R->ncols();
    Tucker::MemoryManager::safe_delete(redistYn);  
  }
  if(local_qr_timer) local_qr_timer->stop();

  if(tsqr_timer) tsqr_timer->start();
  Tucker::Matrix* L = Tucker::MemoryManager::safe_new<Tucker::Matrix>(Rncols, Rncols);
  if(useButterflyTSQR){
    ButterflyTSQR(R, L);
  }
  else{
    TSQR(R);
    if(globalRank == 0){
      //add zeros at the lower triangle
      for(int c=0; c<R->ncols(); c++){
        for(int r=c+1; r<R->nrows(); r++){
          R->data()[r+c*R->nrows()] = 0;
        }
      }
      //transpose
      for(int i=0; i<Rncols; i++){
        dcopy_(&Rncols, R->data()+i*Rncols, &one, L->data()+i, &Rncols); 
      }
      Tucker::MemoryManager::safe_delete(R);
    }
    //bcast
  }
  if(tsqr_timer) tsqr_timer->stop();
  return L;
}

Tucker::Matrix* TSQR(Tucker::Matrix* R){
  int one = 1;
  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
  int globalnp;
  MPI_Comm_size(MPI_COMM_WORLD, &globalnp);
  int Rnrows = R->nrows();
  int Rncols = R->ncols();
  int sizeOfR = R->nrows()*R->ncols();
  
  Tucker::Matrix* tempB;
  int treeDepth = (int)ceil(log2(globalnp));
  MPI_Status status;
  for(int i=0; i < treeDepth; i++){
    if(globalRank % (int)pow(2, i+1) == 0){
      if(globalRank+ pow(2, i) < globalnp){
        int tempBnrows;
        MPI_Recv(&tempBnrows, 1, MPI_INT, globalRank+pow(2, i), globalRank+pow(2, i), MPI_COMM_WORLD, &status);
        tempB = Tucker::MemoryManager::safe_new<Tucker::Matrix>(tempBnrows, Rncols);
        MPI_Recv(tempB->data(), tempBnrows*Rncols, MPI_DOUBLE, globalRank+pow(2, i), globalRank+pow(2, i), MPI_COMM_WORLD, &status);
        int nb = (Rncols > 32)? 32 : Rncols;
        double* T = Tucker::MemoryManager::safe_new_array<double>(nb*Rncols);
        double* work = Tucker::MemoryManager::safe_new_array<double>(nb*Rncols);
        int info;
        // Edge case
        // padd with rows of zeros to make R square
        if(R->nrows() < R->ncols()){
          Tucker::Matrix* squareR = Tucker::MemoryManager::safe_new<Tucker::Matrix>(Rncols, Rncols);
          int sizeOfSquareR = Rncols*Rncols;
          for(int i=0; i<Rncols; i++){
            dcopy_(&Rnrows, R->data()+i*Rnrows, &one, squareR->data()+i*Rncols, &one); //copy the top part over
            for(int j=i*Rncols+Rnrows; j<(i+1)*Rncols; j++){// padd the bottom with zeros
              squareR->data()[j] = 0;
            }
          }
          Tucker::MemoryManager::safe_delete(R);
          R = squareR;
          Rnrows = R->nrows();//Rnrows might change here
        }
        Tucker::dtpqrt_(&tempBnrows, &Rncols, &tempBnrows, &nb, R->data(), &Rncols, tempB->data(),
        &tempBnrows, T, &nb, work, &info);
        Tucker::MemoryManager::safe_delete(tempB);
        Tucker::MemoryManager::safe_delete_array<double>(work, nb*Rncols);
        Tucker::MemoryManager::safe_delete_array<double>(T, nb*Rncols);
      }
    }
    else if(globalRank % (int)pow(2, i) == 0){
      sizeOfR = R->nrows()* R->ncols();
      Rnrows = R->nrows();
      MPI_Send(&Rnrows, 1, MPI_INT, globalRank-pow(2, i), globalRank, MPI_COMM_WORLD);
      MPI_Send(R->data(), sizeOfR, MPI_DOUBLE, globalRank-pow(2, i), globalRank, MPI_COMM_WORLD);
    }
  }
}

Tucker::Matrix* ButterflyTSQR(Tucker::Matrix* R, Tucker::Matrix* L){
  int one = 1;
  int globalRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &globalRank);
  int globalnp;
  MPI_Comm_size(MPI_COMM_WORLD, &globalnp);
  int treeDepth = (int)floor(log2(globalnp)); //depth of the binary TSQR tree
  MPI_Status status;
  int cutOff = pow(2, treeDepth);//largest power of 2 that is smaller than globalnp, also the index of the first element not participating in the butterfly
  //This means globalnp is not a power of 2. 
  //If so, do pairs of send-receive to address the processors with rank >= cutOff not participating in the butterfly TSQR.
  if(globalnp > cutOff){
    if(globalRank >= cutOff){
      int target = globalRank - cutOff;
      int sizeOfR = R->nrows()* R->ncols();
      int Rnrows = R->nrows();
      MPI_Send(&Rnrows, 1, MPI_INT, target, globalRank, MPI_COMM_WORLD);
      MPI_Send(R->data(), sizeOfR, MPI_DOUBLE, target, globalRank, MPI_COMM_WORLD);
    }
    else if(globalRank + cutOff < globalnp){
      int target = globalRank + cutOff;
      int tempBnrows;
      int Rncols = R->ncols();
      int Rnrows = R->nrows();
      MPI_Recv(&tempBnrows, 1, MPI_INT, target, target, MPI_COMM_WORLD, &status);
      Tucker::Matrix* tempB = Tucker::MemoryManager::safe_new<Tucker::Matrix>(tempBnrows, Rncols);
      MPI_Recv(tempB->data(), tempBnrows*Rncols, MPI_DOUBLE, target, target, MPI_COMM_WORLD, &status);
      //padd the top R with zeros if necessary before the dtpqrt
      if(Rncols > Rnrows){
        Tucker::Matrix* squareR = Tucker::MemoryManager::safe_new<Tucker::Matrix>(Rncols, Rncols);
        int sizeOfSquareR = Rncols*Rncols;
        for(int i=0; i<Rncols; i++){
          dcopy_(&Rnrows, R->data()+i*Rnrows, &one, squareR->data()+i*Rncols, &one); //copy the top part over
          for(int j=i*Rncols+Rnrows; j<(i+1)*Rncols; j++){// padd the bottom with zeros
            squareR->data()[j] = 0;
          }
        }
        Tucker::MemoryManager::safe_delete(R);
        R = squareR;
        squareR = NULL;
        Rnrows = R->nrows();
      }
      int nb = (Rncols > 32)? 32 : Rncols;
      double* T = Tucker::MemoryManager::safe_new_array<double>(nb*Rncols);
      double* work = Tucker::MemoryManager::safe_new_array<double>(nb*Rncols);
      int info;
      Tucker::dtpqrt_(&tempBnrows, &Rncols, &tempBnrows, &nb, R->data(), &Rncols, tempB->data(),
      &tempBnrows, T, &nb, work, &info);
      Tucker::MemoryManager::safe_delete(tempB);
      Tucker::MemoryManager::safe_delete_array<double>(work, nb*Rncols);
      Tucker::MemoryManager::safe_delete_array<double>(T, nb*Rncols);
    }
  }
  //Butterfly TSQR part
  for(int i=0; i < treeDepth; i++){

  }
  //
  if(globalnp > cutOff){
    if(globalRank >= cutOff){
      int target = globalRank - cutOff;
      int sizeOfR = R->nrows()* R->ncols();
      int Rnrows = R->nrows();
      MPI_Recv(&Rnrows, 1, MPI_INT, target, target, MPI_COMM_WORLD, &status);
      MPI_Recv(R->data(), sizeOfR, MPI_DOUBLE, target, target, MPI_COMM_WORLD, &status);
    }
    else if(globalRank + cutOff < globalnp){
      int target = globalRank - cutOff;
      int sizeOfR = R->nrows()* R->ncols();
      int Rnrows = R->nrows();
      MPI_Send(&Rnrows, 1, MPI_INT, target, globalRank, MPI_COMM_WORLD);
      MPI_Send(R->data(), sizeOfR, MPI_DOUBLE, target, globalRank, MPI_COMM_WORLD);
    }
  }
  int Rncols = R->ncols(); int Rnrows = R->nrows();
  for(int c=0; c<Rncols; c++){
    for(int r=c+1; r<Rnrows; r++){
      R->data()[r+c*Rnrows] = 0;
    }
  }
  //transpose
  for(int i=0; i<Rncols; i++){
    dcopy_(&Rncols, R->data()+i*Rncols, &one, L->data()+i, &Rncols); 
  }
  Tucker::MemoryManager::safe_delete(R);

  
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
  Tucker::Matrix* gram =
      Tucker::MemoryManager::safe_new<Tucker::Matrix>(numGlobalRows,numGlobalRows);

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
    allRedBuf = Tucker::MemoryManager::safe_new_array<double>(numGlobalRows*numLocalRows);
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
        localMatrix = Tucker::MemoryManager::safe_new<Tucker::Matrix>(numGlobalRows,numLocalRows);

        // Determine the amount of data being received
        int ndims = Y->getNumDimensions();
        const Tucker::SizeArray& sz = Y->getLocalSize();
        int maxNumRows = Y->getDistribution()->getMap(n,true)->getMaxNumEntries();
        size_t numCols = sz.prod(0,n-1,1)*sz.prod(n+1,ndims-1,1);
        size_t maxEntries = maxNumRows*numCols;

        // Create buffer for receiving data
        double* recvBuf = Tucker::MemoryManager::safe_new_array<double>(maxEntries);

        // Send data to the next proc in column
        MPI_Request* sendRequests =
            Tucker::MemoryManager::safe_new_array<MPI_Request>(numColProcsSqueezed);
        size_t numToSend = Y->getLocalNumEntries();
        assert(numToSend <= std::numeric_limits<int>::max());
        int tag = 0;
        int sendDest = (myColRankSqueezed+1)%numColProcsSqueezed;
        if(shift_timer) shift_timer->start();
        MPI_Isend((void*)Y->getLocalTensor()->data(), (int)numToSend, MPI_DOUBLE,
            sendDest, tag, colCommSqueezed, sendRequests+sendDest);
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
        MPI_Irecv(recvBuf, (int)numToReceive, MPI_DOUBLE,
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
            MPI_Isend((void*)Y->getLocalTensor()->data(), (int)numToSend, MPI_DOUBLE,
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
            MPI_Irecv(recvBuf, (int)numToReceive, MPI_DOUBLE,
                recvSource, tag, colCommSqueezed, recvRequests+recvSource);
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

        Tucker::MemoryManager::safe_delete_array<double>(recvBuf,maxEntries);
        Tucker::MemoryManager::safe_delete_array<MPI_Request>(sendRequests,numColProcsSqueezed);
        Tucker::MemoryManager::safe_delete_array<MPI_Request>(recvRequests,numColProcsSqueezed);
        Tucker::MemoryManager::safe_delete_array<MPI_Status>(sendStatuses,numColProcsSqueezed);
      }
    }
    else {
      localMatrix = Tucker::MemoryManager::safe_new<Tucker::Matrix>(numGlobalRows,numLocalRows);
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

    Tucker::MemoryManager::safe_delete<Tucker::Matrix>(localMatrix);
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
    MPI_Allgatherv(allRedBuf, numLocalRows*numGlobalRows, MPI_DOUBLE,
        gram->data(), recvcounts, displs, MPI_DOUBLE, colComm);
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
  Tucker::MemoryManager::safe_delete_array<double>(allRedBuf,numGlobalRows*numLocalRows);

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
      Tucker::Matrix* temp = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nGlobalRows,nGlobalRows);
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
        Tucker::Matrix* temp = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nGlobalRows,nGlobalRows);
        temp->initialize();
        localGram = temp;
      }
      Tucker::MemoryManager::safe_delete<const Matrix>(redistributedY);
    } // end if(!myColEmpty)
  }
  else {
    if(Y->getDistribution()->ownNothing()) {
      int nGlobalRows = Y->getGlobalSize(n);
      Tucker::Matrix* temp = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nGlobalRows,nGlobalRows);
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
  Tucker::MemoryManager::safe_delete<const Tucker::Matrix>(localGram);

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
const TuckerTensor* STHOSVD(const Tensor* const X,
    const double epsilon, int* modeOrder, bool useOldGram, bool flipSign,
    bool useLQ)
{
  // Get this rank
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int ndims = X->getNumDimensions();

  // Create a struct to store the factorization
  TuckerTensor* factorization = Tucker::MemoryManager::safe_new<TuckerTensor>(ndims);

  // Compute the threshold
  double tensorNorm = X->norm2();
  double thresh = epsilon*epsilon*tensorNorm/ndims;
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

  const Tensor* Y = X;

  // For each dimension...
  for(int n=0; n<ndims; n++)
  {
    int mode = modeOrder? modeOrder[n] : n;
    if(useLQ){
      if(rank == 0) std::cout << "\tAutoST-HOSVD::Starting LQ(" << mode << ")...\n";
      factorization->LQ_timer_[mode].start();
      Tucker::Matrix* L = LQ(Y, mode, &factorization->LQ_tsqr_timer_[mode], &factorization->LQ_localqr_timer_[mode], 
        &factorization->LQ_redistribute_timer_[mode], &factorization->LQ_dcopy_timer_[mode],
        &factorization->LQ_decompose_timer_[mode], &factorization->LQ_transpose_timer_[mode]);
      factorization->LQ_timer_[mode].stop();
      int SizeOfL = L->nrows()*L->ncols();
      factorization->LQ_bcast_timer_[mode].start();
      MPI_Bcast(L->data(), SizeOfL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      factorization->LQ_bcast_timer_[mode].stop();
      if(rank == 0) std::cout << "\tAutoST-HOSVD::Starting computeSVD(" << mode << ")...\n";
      factorization->svd_timer_[mode].start();
      Tucker::computeSVD(L, factorization->singularValues[mode], factorization->U[mode], thresh);
      factorization->svd_timer_[mode].stop();
      Tucker::MemoryManager::safe_delete<Tucker::Matrix>(L);
    }
    else{
      // Compute the Gram matrix
      // S = Y_n*Y_n'
      Tucker::Matrix* S;
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
      Tucker::MemoryManager::safe_delete<Tucker::Matrix>(S);
    }


    // Perform the tensor times matrix multiplication
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting TTM(" << mode << ")...\n";
    }
    factorization->ttm_timer_[mode].start();
    Tensor* temp = ttm(Y,mode,factorization->U[mode],true,
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
      Tucker::MemoryManager::safe_delete<const Tensor>(Y);
    }
    Y = temp;

    if(rank == 0) {
      size_t local_nnz = Y->getLocalNumEntries();
      size_t global_nnz = Y->getGlobalNumEntries();
      std::cout << "Local tensor size after STHOSVD iteration "
          << mode << ": " << Y->getLocalSize() << ", or ";
      Tucker::printBytes(local_nnz*sizeof(double));
      std::cout << "Global tensor size after STHOSVD iteration "
          << mode << ": " << Y->getGlobalSize() << ", or ";
      Tucker::printBytes(global_nnz*sizeof(double));
    }
  }

  factorization->G = const_cast<Tensor*>(Y);
  factorization->total_timer_.stop();
  return factorization;
}

// \todo This function is never tested
const TuckerTensor* STHOSVD(const Tensor* const X,
    const Tucker::SizeArray* const reducedI, int* modeOrder, bool useOldGram,
    bool flipSign, bool useLQ)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  int ndims = X->getNumDimensions();
  assert(ndims == reducedI->size());

  // Create a struct to store the factorization
  TuckerTensor* factorization = Tucker::MemoryManager::safe_new<TuckerTensor>(ndims);

  // Compute the nnz of the largest tensor piece being stored by any process
  size_t max_lcl_nnz_x = 1;
  for(int i=0; i<ndims; i++) {
    max_lcl_nnz_x *= X->getDistribution()->getMap(i,false)->getMaxNumEntries();
  }

  // Barrier for timing
  MPI_Barrier(MPI_COMM_WORLD);
  factorization->total_timer_.start();

  const Tensor* Y = X;

  // For each dimension...
  for(int n=0; n<ndims; n++)
  {
    int mode = modeOrder ? modeOrder[n] : n;
    if(useLQ){
      Tucker::Matrix* L;
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
      MPI_Bcast(L->data(), SizeOfL, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      factorization->LQ_bcast_timer_[mode].stop();
      if(rank == 0) {
        std::cout << "\tAutoST-HOSVD::Starting computeSVD(" << mode << ")...\n";
      }
      factorization->svd_timer_[mode].start();
      Tucker::computeSVD(L, factorization->singularValues[mode], factorization->U[mode], (*reducedI)[mode]);
      factorization->svd_timer_[mode].stop();
      Tucker::MemoryManager::safe_delete<Tucker::Matrix>(L);
    }
    else{
      // Compute the Gram matrix
      // S = Y_n*Y_n'
      Tucker::Matrix* S;
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
      
      Tucker::MemoryManager::safe_delete<Tucker::Matrix>(S);
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
    Tensor* temp = ttm(Y,mode,factorization->U[mode],true,
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
      Tucker::MemoryManager::safe_delete<const Tensor>(Y);
    }
    Y = temp;

    if(rank == 0) {
      size_t local_nnz = Y->getLocalNumEntries();
      size_t global_nnz = Y->getGlobalNumEntries();
      std::cout << "Local tensor size after STHOSVD iteration "
          << mode << ": " << Y->getLocalSize() << ", or ";
      Tucker::printBytes(local_nnz*sizeof(double));
      std::cout << "Global tensor size after STHOSVD iteration "
          << mode << ": " << Y->getGlobalSize() << ", or ";
      Tucker::printBytes(global_nnz*sizeof(double));
    }
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
    double* sendBuf = Tucker::MemoryManager::safe_new_array<double>(numSlices);
    double* recvBuf = Tucker::MemoryManager::safe_new_array<double>(numSlices);
    if(metrics & Tucker::MIN) {
      for(int i=0; i<numSlices; i++) sendBuf[i] = result->getMinData()[i];
      MPI_Allreduce(sendBuf, recvBuf, numSlices, MPI_DOUBLE,
          MPI_MIN, comm);
      for(int i=0; i<numSlices; i++) result->getMinData()[i] = recvBuf[i];
    }
    if(metrics & Tucker::MAX) {
      for(int i=0; i<numSlices; i++) sendBuf[i] = result->getMaxData()[i];
      MPI_Allreduce(sendBuf, recvBuf, numSlices, MPI_DOUBLE,
          MPI_MAX, comm);
      for(int i=0; i<numSlices; i++) result->getMaxData()[i] = recvBuf[i];
    }
    if(metrics & Tucker::SUM) {
      for(int i=0; i<numSlices; i++) sendBuf[i] = result->getSumData()[i];
      MPI_Allreduce(sendBuf, recvBuf, numSlices, MPI_DOUBLE,
          MPI_SUM, comm);
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
        sendBuf[i] = result->getMeanData()[i] * (double)localSliceSize;
      }

      MPI_Allreduce(sendBuf, recvBuf, numSlices,
          MPI_DOUBLE, MPI_SUM, comm);

      double* meanDiff;
      if(metrics & Tucker::VARIANCE) {
        meanDiff = Tucker::MemoryManager::safe_new_array<double>(numSlices);
        for(int i=0; i<numSlices; i++) {
          meanDiff[i] = result->getMeanData()[i] -
              recvBuf[i] / (double)globalSliceSize;
        }
      }

      for(int i=0; i<numSlices; i++) {
        result->getMeanData()[i] = recvBuf[i] / (double)globalSliceSize;
      }

      if(metrics & Tucker::VARIANCE) {
        for(int i=0; i<numSlices; i++) {
          // Source of this equation:
          // http://stats.stackexchange.com/questions/10441/how-to-calculate-the-variance-of-a-partition-of-variables
          sendBuf[i] = (double)localSliceSize*result->getVarianceData()[i] +
              (double)localSliceSize*meanDiff[i]*meanDiff[i];
        }

        Tucker::MemoryManager::safe_delete_array<double>(meanDiff,numSlices);

        MPI_Allreduce(sendBuf, recvBuf, numSlices,
            MPI_DOUBLE, MPI_SUM, comm);

        for(int i=0; i<numSlices; i++) {
          result->getVarianceData()[i] = recvBuf[i] / (double)globalSliceSize;
        }
      } // end if(metrics & Tucker::VARIANCE)
    } // end if((metrics & Tucker::MEAN) || (metrics & Tucker::VARIANCE))

    Tucker::MemoryManager::safe_delete_array<double>(sendBuf,numSlices);
    Tucker::MemoryManager::safe_delete_array<double>(recvBuf,numSlices);
  } // end if(nprocs > 1)

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
  double* scales = Tucker::MemoryManager::safe_new_array<double>(sizeOfModeDim);
  double* shifts = Tucker::MemoryManager::safe_new_array<double>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scales[i] = sqrt(metrics->getVarianceData()[i]);
    shifts[i] = -metrics->getMeanData()[i];
    if(std::abs(scales[i]) < stdThresh) {
      scales[i] = 1;
    }
  }
  transformSlices(Y,mode,scales,shifts);
  Tucker::MemoryManager::safe_delete_array<double>(scales,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete_array<double>(shifts,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete<Tucker::MetricData>(metrics);
}

void normalizeTensorMinMax(Tensor* Y, int mode)
{
  // I don't have to do any work because I don't own any data
  if(Y->getLocalSize(mode) == 0)
    return;

  Tucker::MetricData* metrics = computeSliceMetrics(Y, mode,
      Tucker::MAX+Tucker::MIN);
  int sizeOfModeDim = Y->getLocalSize(mode);
  double* scales = Tucker::MemoryManager::safe_new_array<double>(sizeOfModeDim);
  double* shifts = Tucker::MemoryManager::safe_new_array<double>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    scales[i] = metrics->getMaxData()[i] - metrics->getMinData()[i];
    shifts[i] = -metrics->getMinData()[i];
  }
  transformSlices(Y,mode,scales,shifts);
  Tucker::MemoryManager::safe_delete_array<double>(scales,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete_array<double>(shifts,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete<Tucker::MetricData>(metrics);
}

// \todo This function is never tested
void normalizeTensorMax(Tensor* Y, int mode)
{
  // I don't have to do any work because I don't own any data
  if(Y->getLocalSize(mode) == 0)
    return;

  Tucker::MetricData* metrics = computeSliceMetrics(Y, mode,
      Tucker::MIN + Tucker::MAX);
  int sizeOfModeDim = Y->getLocalSize(mode);
  double* scales = Tucker::MemoryManager::safe_new_array<double>(sizeOfModeDim);
  double* shifts = Tucker::MemoryManager::safe_new_array<double>(sizeOfModeDim);
  for(int i=0; i<sizeOfModeDim; i++) {
    double scaleval = std::max(std::abs(metrics->getMinData()[i]),
        std::abs(metrics->getMaxData()[i]));
    scales[i] = scaleval;
    shifts[i] = 0;
  }
  transformSlices(Y,mode,scales,shifts);
  Tucker::MemoryManager::safe_delete_array<double>(scales,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete_array<double>(shifts,sizeOfModeDim);
  Tucker::MemoryManager::safe_delete<Tucker::MetricData>(metrics);
}

const Tensor* reconstructSingleSlice(const TuckerTensor* fact,
    const int mode, const int sliceNum)
{
  assert(mode >= 0 && mode < fact->N);
  assert(sliceNum >= 0 && sliceNum < fact->U[mode]->nrows());

  // Process mode first, in order to minimize storage requirements

  // Copy row of matrix
  int nrows = fact->U[mode]->nrows();
  int ncols = fact->U[mode]->ncols();
  Tucker::Matrix tempMat(1,ncols);
  const double* olddata = fact->U[mode]->data();
  double* rowdata = tempMat.data();
  for(int j=0; j<ncols; j++)
    rowdata[j] = olddata[j*nrows+sliceNum];
  Tensor* ten = ttm(fact->G, mode, &tempMat);

  for(int i=0; i<fact->N; i++)
  {
    Tucker::Matrix* tempMat;
    if(i == mode)
      continue;  
    
    Tensor* temp = ttm(ten, i, fact->U[i]);

    Tucker::MemoryManager::safe_delete<Tensor>(ten);
    ten = temp;
  }

  return ten;
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
  ret = MPI_File_read_all(fh, Y->getLocalTensor()->data(),
      (int)count, MPI_DOUBLE, &status);
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
  Tucker::MemoryManager::safe_delete_array<int>(starts,ndims);
  Tucker::MemoryManager::safe_delete_array<int>(lsizes,ndims);
  Tucker::MemoryManager::safe_delete_array<int>(gsizes,ndims);
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
  size_t count = Y->size().prod();
  assert(count <= std::numeric_limits<int>::max());
  double * data = Y->data();
  MPI_Status status;
  ret = MPI_File_read(fh, data, (int)count, MPI_DOUBLE, &status);
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
  MPI_Type_create_subarray(ndims-1, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, MPI_DOUBLE, &view);
  MPI_Type_commit(&view);

  int nsteps = Y->getGlobalSize(ndims-1);
  const Map* stepMap = Y->getDistribution()->getMap(ndims-1,true);
  const MPI_Comm& stepComm = Y->getDistribution()->getProcessorGrid()->getRowComm(ndims-1,true);
  double* dataPtr = Y->getLocalTensor()->data();
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
    MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);

    // Read the file
    MPI_Status status;
    ret = MPI_File_read_all(fh, dataPtr,
        (int)count, MPI_DOUBLE, &status);
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
  size_t count = Y->getLocalNumEntries();
  assert(count <= std::numeric_limits<int>::max());
  MPI_Status status;
  ret = MPI_File_write_all(fh, (double*)Y->getLocalTensor()->data(), (int)count,
      MPI_DOUBLE, &status);
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
  size_t nentries = Y->size().prod();
  assert(nentries <= std::numeric_limits<int>::max());
  const double* entries = Y->data();
  MPI_Status status;
  ret = MPI_File_write(fh, (double*)entries, (int)nentries, MPI_DOUBLE, &status);
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
  MPI_Type_create_subarray(ndims-1, gsizes, lsizes, starts,
      MPI_ORDER_FORTRAN, MPI_DOUBLE, &view);
  MPI_Type_commit(&view);

  int nsteps = Y->getGlobalSize(ndims-1);
  const Map* stepMap = Y->getDistribution()->getMap(ndims-1,true);
  const MPI_Comm& stepComm = Y->getDistribution()->getProcessorGrid()->getRowComm(ndims-1,true);
  const double* dataPtr = Y->getLocalTensor()->data();
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
    MPI_File_set_view(fh, disp, MPI_DOUBLE, view, "native", MPI_INFO_NULL);

    // Write the file
    MPI_Status status;
    ret = MPI_File_write_all(fh, (void*)dataPtr, (int)count,
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
  Tucker::MemoryManager::safe_delete_array<int>(starts,ndims-1);
  Tucker::MemoryManager::safe_delete_array<int>(lsizes,ndims-1);
  Tucker::MemoryManager::safe_delete_array<int>(gsizes,ndims-1);
}

Tensor* generateTensor(int seed, TuckerTensor* fact, Tucker::SizeArray* proc_dims, 
  Tucker::SizeArray* tensor_dims, Tucker::SizeArray* core_dims,
  double noise){
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
  std::normal_distribution<double> distribution;
  //GENERATE CORE TENSOR//
  //distribution for the core
  TuckerMPI::Distribution* dist =
      Tucker::MemoryManager::safe_new<TuckerMPI::Distribution>(*core_dims, *proc_dims);
  fact->G = Tucker::MemoryManager::safe_new<TuckerMPI::Tensor>(dist);
  size_t nnz = dist->getLocalDims().prod();
  double* dataptr = fact->G->getLocalTensor()->data();
  for(size_t i=0; i<nnz; i++) {
    dataptr[i] = distribution(generator);
  }
  //GENERATE FACTOR MATRICES//
  for(int d=0; d<fact->N; d++) {
    if(rank == 0) std::cout << "Generating factor matrix " << d << "...\n";
    int nrows = (*tensor_dims)[d];
    int ncols = (*core_dims)[d];
    fact->U[d] = Tucker::MemoryManager::safe_new<Tucker::Matrix>(nrows,ncols);
    nnz = nrows*ncols;
    dataptr = fact->U[d]->data();
    if(rank == 0) {
      for(size_t i=0; i<nnz; i++) {
        dataptr[i] = distribution(generator);
      }
    }

    MPI_Bcast(dataptr,nnz,MPI_DOUBLE,0,MPI_COMM_WORLD);
  }
  //TTM between factor matrices and core//
  TuckerMPI::Tensor* product = fact->G;
  for(int d=0; d<fact->N; d++) {
    TuckerMPI::Tensor* temp = TuckerMPI::ttm(product, d, fact->U[d]);
    if(product != fact->G) {
      Tucker::MemoryManager::safe_delete<TuckerMPI::Tensor>(product);
    }
    product = temp;
  }
  /////////////////////////////////////////////////////////////////////
  // Compute the norm of the global tensor                           //
  // \todo This could be more efficient; see Bader/Kolda for details //
  /////////////////////////////////////////////////////////////////////
  double normM = std::sqrt(product->norm2());
  ///////////////////////////////////////////////////////////////////
  // Compute the estimated norm of the noise matrix                //
  // The average of each element squared is the standard deviation //
  // squared, so this quantity should be sqrt(nnz * stdev^2)       //
  ///////////////////////////////////////////////////////////////////
  nnz = tensor_dims->prod();
  double normN = std::sqrt(nnz);
  double alpha = noise*normM/normN;
  //add noise to product
  dataptr = product->getLocalTensor()->data();
  nnz = dist->getLocalDims().prod();
  for(size_t i=0; i<nnz; i++) {
    dataptr[i] += alpha*distribution(generator);
  }
  return product;
}
} // end namespace TuckerMPI
