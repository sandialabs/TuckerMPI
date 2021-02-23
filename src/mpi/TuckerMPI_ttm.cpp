/*
 * TuckerMPI_ttm.cpp
 *
 *  Created on: Nov 2, 2016
 *      Author: amklinv
 */

#include "TuckerMPI_ttm.hpp"

namespace TuckerMPI
{

template <class scalar_t>
Tensor<scalar_t>* ttm(const Tensor<scalar_t>* X, const int n,
    const Tucker::Matrix<scalar_t>* const U, bool Utransp,
    Tucker::Timer* mult_timer, Tucker::Timer* pack_timer,
    Tucker::Timer* reduce_scatter_timer,
    Tucker::Timer* reduce_timer, size_t nnz_limit)
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
  Distribution* dist = Tucker::MemoryManager::safe_new<Distribution>(newSize,
          X->getDistribution()->getProcessorGrid()->size());

  // Create the new tensor
  Tensor<scalar_t>* Y = Tucker::MemoryManager::safe_new<Tensor<scalar_t>>(dist);

  // Get the local part of the tensor
  const Tucker::Tensor<scalar_t>* localX = X->getLocalTensor();
  Tucker::Tensor<scalar_t>* localY = Y->getLocalTensor();

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
    int K = nrows;
    int Jn = Utransp ? U->nrows() : U->ncols();

    int uGlobalRows = Y->getGlobalSize(n);
    int uGlobalCols = X->getGlobalSize(n);

    const Map* xMap = X->getDistribution()->getMap(n,false);
    const Map* yMap = Y->getDistribution()->getMap(n,false);

    const scalar_t* Uptr;
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

    // We can do the TTM either by doing a single reduce_scatter, or a
    // series of reductions.
    // Reduce_scatter tends to be faster, so we try to use it if the
    // memory requirements are not prohibitive.

    // Compute the nnz of the largest tensor piece being stored by any process
    size_t max_lcl_nnz_x = 1;
    for(int i=0; i<ndims; i++) {
      max_lcl_nnz_x *= X->getDistribution()->getMap(i,false)->getMaxNumEntries();
    }

    // Compute the nnz required for the reduce_scatter
    size_t nnz_reduce_scatter = 1;
    for(int i=0; i<ndims; i++) {
      if(i == n)
        nnz_reduce_scatter *= Y->getGlobalSize(n);
      else
        nnz_reduce_scatter *= X->getDistribution()->getMap(i,false)->getMaxNumEntries();
    }

    // If the required memory is small, we can do a single reduce_scatter
    if(nnz_reduce_scatter <= std::max(max_lcl_nnz_x,nnz_limit)) {
      // Compute the TTM
      Tucker::Tensor<scalar_t>* localResult;
      if(X->getDistribution()->ownNothing()) {
        Tucker::SizeArray sz(ndims);
        for(int i=0; i<ndims; i++) {
          sz[i] = X->getLocalSize(i);
        }
        sz[n] = Y->getGlobalSize(n);
        localResult = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(sz);
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
      packForTTM(localResult,n,yMap);
      if(pack_timer) pack_timer->stop();

      // Perform a reduce-scatter
      const scalar_t* sendBuf;
      if(localResult->getNumElements() > 0)
        sendBuf = localResult->data();
      else
        sendBuf = 0;
      scalar_t* recvBuf;
      if(localY->getNumElements() > 0)
        recvBuf = localY->data();
      else
        recvBuf = 0;
      int nprocs;
      MPI_Comm_size(comm,&nprocs);
      int* recvCounts = Tucker::MemoryManager::safe_new_array<int>(nprocs);
      size_t multiplier = Y->getLocalSize().prod(0,n-1,1)*Y->getLocalSize().prod(n+1,ndims-1,1);
      for(int i=0; i<nprocs; i++) {
        size_t temp = multiplier*(yMap->getNumEntries(i));
        assert(temp <= std::numeric_limits<int>::max());
        recvCounts[i] = (int)temp;
      }

      if(reduce_scatter_timer) reduce_scatter_timer->start();
      MPI_Reduce_scatter_(sendBuf, recvBuf, recvCounts, MPI_SUM, comm);
      if(reduce_scatter_timer) reduce_scatter_timer->stop();
      Tucker::MemoryManager::safe_delete_array(recvCounts,nprocs);
      Tucker::MemoryManager::safe_delete(localResult);
    } // end if(K < std::ceil(Jn/Pn))
    else {
      for(int root=0; root<Pn; root++) {
        int uLocalRows = yMap->getNumEntries(root);

        if(uLocalRows == 0) {
          continue;
        }

        // Compute the local TTM
        Tucker::Tensor<scalar_t>* localResult;
        if(X->getDistribution()->ownNothing()) {
          Tucker::SizeArray sz(ndims);
          for(int i=0; i<ndims; i++) {
            sz[i] = X->getLocalSize(i);
          }
          sz[n] = uLocalRows;
          localResult = Tucker::MemoryManager::safe_new<Tucker::Tensor<scalar_t>>(sz);
          localResult->initialize();
        }
        else {
          if(mult_timer) mult_timer->start();
          localResult = Tucker::ttm(localX, n, Uptr, uLocalRows, stride, Utransp);
          if(mult_timer) mult_timer->stop();
        }

        // Combine the local results with a reduce operation
        const scalar_t* sendBuf;
        if(localResult->getNumElements() > 0)
          sendBuf = localResult->data();
        else
          sendBuf = 0;
        scalar_t* recvBuf;
        if(localY->getNumElements() > 0)
          recvBuf = localY->data();
        else
          recvBuf = 0;
        size_t count = localResult->getNumElements();
        assert(count <= std::numeric_limits<int>::max());

        if(count > 0) {
          if(reduce_timer) reduce_timer->start();
          MPI_Reduce_(sendBuf, recvBuf, (int)count, MPI_SUM, root, comm);
          if(reduce_timer) reduce_timer->stop();
        }

        // Free memory
        Tucker::MemoryManager::safe_delete(localResult);

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

// Explicit instantiations to build static library for both single and double precision
template Tensor<float>* ttm(const Tensor<float>* X, const int n,
    const Tucker::Matrix<float>* const U, bool Utransp,
    Tucker::Timer* mult_timer, Tucker::Timer* pack_timer,
    Tucker::Timer* reduce_scatter_timer,
    Tucker::Timer* reduce_timer, size_t nnz_limit);
template Tensor<double>* ttm(const Tensor<double>* X, const int n,
    const Tucker::Matrix<double>* const U, bool Utransp,
    Tucker::Timer* mult_timer, Tucker::Timer* pack_timer,
    Tucker::Timer* reduce_scatter_timer,
    Tucker::Timer* reduce_timer, size_t nnz_limit);

} // end namespace TuckerMPI
