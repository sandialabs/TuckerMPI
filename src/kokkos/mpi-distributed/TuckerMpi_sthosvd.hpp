#ifndef TUCKER_KOKKOS_MPI_STHOSVD_HPP_
#define TUCKER_KOKKOS_MPI_STHOSVD_HPP_

#include "Tucker_BlasWrapper.hpp"
#include "TuckerOnNode_sthosvd.hpp"
#include "TuckerMpi_CoreTensorTruncator.hpp"
#include "TuckerMpi_Tensor.hpp"
#include "TuckerMpi_MPIWrapper.hpp"
#include "TuckerMpi_Matrix.hpp"
#include <Kokkos_Core.hpp>

namespace TuckerMpi{
namespace impl{

template<class ViewT>
size_t prod(const ViewT & sz,
	    const int low, const int high,
	    const int defaultReturnVal = -1)
{
  if(low < 0 || high >= sz.extent(0)) {
    // std::cerr << "ERROR: prod(" << low << "," << high
    // 	      << ") is invalid because indices must be in the range [0,"
    // 	      << sz.extent(0) << ").  Returning " << defaultReturnVal << std::endl;
    return defaultReturnVal;
  }

  if(low > high) { return defaultReturnVal; }
  size_t result = 1;
  for(int j = low; j <= high; j++){ result *= sz[j]; }
  return result;
}

}//impl

template <class scalar_t>
auto localRankKForGram(Matrix<scalar_t> Y, int n, int ndims)
{
  int nrows = Y.getLocalNumRows();
  Kokkos::View<scalar_t**, Kokkos::LayoutLeft> localResult("loc", nrows, nrows);

  char uplo = 'U';
  int ncols = Y.getLocalNumCols();
  scalar_t alpha = 1;
  const scalar_t* A = Y.getLocalMatrix().data();
  scalar_t beta = 0;
  scalar_t* C = localResult.data();
  int ldc = nrows;
  if(n < ndims-1) {
    // Local matrix is column major
    char trans = 'N';
    int lda = nrows;
    Tucker::syrk(&uplo, &trans, &nrows, &ncols, &alpha, A, &lda, &beta, C, &ldc);
  }
  else {
    // Local matrix is row major
    char trans = 'T';
    int lda = ncols;
    Tucker::syrk(&uplo, &trans, &nrows, &ncols, &alpha, A, &lda, &beta, C, &ldc);
  }

  return localResult;
}

bool isUnpackForGramNecessary(int n, int ndims, const Map* origMap, const Map* redistMap){
  // Get the local number of columns
  int nLocalCols = redistMap->getLocalNumEntries();
  // If the number of local columns is 1, no unpacking is necessary
  if(nLocalCols <= 1) { return false; }
  // If this is the last dimension, the data is row-major and no unpacking is necessary
  if(n == ndims-1) { return false; }
  return true;
}

bool isPackForGramNecessary(int n, const Map* origMap, const Map* redistMap){
  // Get the local number of rows of Y_n
  int localNumRows = origMap->getLocalNumEntries();
  // If the local number of rows is 1, no packing is necessary
  if(localNumRows <= 1) { return false; }
  // If n is 0, the local data is already column-major and no packing is necessary
  if(n == 0) { return false; }
  return true;
}


template <class scalar_t>
void unpackForGram(int n, int ndims,
		   Matrix<scalar_t> redistMat,
		   const std::vector<scalar_t> & dataToUnpack,
		   const Map* origMap)
{
  const int ONE = 1;
  assert(isUnpackForGramNecessary(n, ndims, origMap, redistMat.getMap()));
  // Get the size of the matrix
  int nLocalCols = redistMat.getLocalNumCols();

  // Get the number of MPI processes
  const MPI_Comm& comm = origMap->getComm();
  int nprocs;
  MPI_Comm_size(comm,&nprocs);

  scalar_t* dest = redistMat.getLocalMatrix().data();
  for(int c=0; c<nLocalCols; c++) {
    for(int b=0; b<nprocs; b++) {
      int nLocalRows=origMap->getNumEntries(b);
      const scalar_t* src = dataToUnpack.data()
	+ origMap->getOffset(b)*nLocalCols + c*nLocalRows;
      Tucker::copy(&nLocalRows, src, &ONE, dest, &ONE);
      dest += nLocalRows;
    }
  }
}


// Pack the data for redistribution
// Y_n is block-row distributed; we are packing
// so that Y_n will be block-column distributed
template <class scalar_t, class ...Ps>
auto packForGram(Tensor<scalar_t, Ps...> Y, int n, const Map* redistMap)
{
  const int ONE = 1;
  assert(isPackForGramNecessary(n, Y.getDistribution().getMap(n,true), redistMap));

  // Get the local number of rows of Y_n
  int localNumRows = Y.getLocalSize(n);
  // Get the number of columns of Y_n
  int globalNumCols = redistMap->getGlobalNumEntries();
  // Get the number of dimensions
  int ndims = Y.getNumDimensions();
  // Get the number of MPI processes
  int nprocs = Y.getDistribution().getProcessorGrid().getNumProcs(n,true);
  // Allocate memory for packed data
  std::vector<scalar_t> sendData(Y.getLocalNumEntries());

  // Local data is row-major
  //after packing the local data should have block column pattern where each block is row major.
  if(n == ndims-1) {
    const scalar_t* YnData = Y.getLocalTensor().data().data();
    int offset=0;
    for(int b=0; b<nprocs; b++) {
      int n = redistMap->getNumEntries(b);
      for(int r=0; r<localNumRows; r++) {
        Tucker::copy(&n, YnData+redistMap->getOffset(b)+globalNumCols*r, &ONE,
		     &sendData[0]+offset, &ONE);
        offset += n;
      }
    }
  }
  else
  {
    auto sz = Y.getLocalSize();
    // Get information about the local blocks
    size_t numLocalBlocks = impl::prod(sz, n+1,ndims-1);
    size_t ncolsPerLocalBlock = impl::prod(sz, 0,n-1);
    assert(ncolsPerLocalBlock <= std::numeric_limits<int>::max());

    const scalar_t* src = Y.getLocalTensor().data().data();

    // Make local data column major
    for(size_t b=0; b<numLocalBlocks; b++) {
      scalar_t* dest = &sendData[0] + b*localNumRows*ncolsPerLocalBlock;
      // Copy one row at a time
      for(int r=0; r<localNumRows; r++) {
        int temp = (int)ncolsPerLocalBlock;
        Tucker::copy(&temp, src, &ONE, dest, &localNumRows);
        src += ncolsPerLocalBlock;
        dest += 1;
      }
    }
  }

  return sendData;
}

template <class ScalarType, class ...Properties>
auto redistributeTensorForGram(Tensor<ScalarType, Properties...> & Y, int n)
{
  using result_t = ::TuckerMpi::Matrix<ScalarType>;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  // Get the original Tensor map
  const Map* oldMap = Y.getDistribution().getMap(n,false);
  const MPI_Comm& comm = Y.getDistribution().getProcessorGrid().getColComm(n,false);

  // Get the number of MPI processes in this communicator
  int numProcs;
  MPI_Comm_size(comm,&numProcs);
  // If the communicator only has one MPI process, no redistribution is needed
  if(numProcs < 2) { return result_t(); }

  // Get the dimensions of the redistributed matrix Y_n
  const int ndims = Y.rank();
  const auto & sz = Y.getLocalSize();
  const int nrows = Y.getGlobalSize(n);
  size_t ncols = impl::prod(sz, 0,n-1,1) * impl::prod(sz, n+1,ndims-1,1);
  assert(ncols <= std::numeric_limits<int>::max());

  // Create a matrix to store the redistributed Y_n
  // Y_n has a block row distribution
  // We want it to have a block column distribution
  result_t recvY(nrows, (int)ncols, comm, false);

  // Get the column map of the redistributed Y_n
  const Map* redistMap = recvY.getMap();
  int nLocalRows = Y.getLocalSize(n);
  // Get the number of local columns of the redistributed matrix Y_n
  // Same as: int nLocalCols = recvY->getLocalNumCols();
  int nLocalCols = redistMap->getLocalNumEntries();

  // Compute the number of entries this rank will send to others, along with displacements
  int sendCounts[numProcs];
  int sendDispls[numProcs+1]; sendDispls[0] = 0;
  for(int i=0; i<numProcs; i++) {
    sendCounts[i] = nLocalRows * redistMap->getNumEntries(i);
    sendDispls[i+1] = sendDispls[i] + sendCounts[i];
  }

  // Compute the number of entries this rank will receive from others,
  // along with their displacements
  int recvCounts[numProcs];
  int recvDispls[numProcs+1];
  recvDispls[0] = 0;
  for(int i=0; i<numProcs; i++) {
    recvCounts[i] = nLocalCols * oldMap->getNumEntries(i);
    recvDispls[i+1] = recvDispls[i] + recvCounts[i];
  }

  bool isPackingNecessary = isPackForGramNecessary(n, oldMap, redistMap);
  std::vector<ScalarType> sendBuf;
  if(isPackingNecessary) {
    sendBuf = packForGram(Y, n, redistMap);
  }
  else{
    if(Y.getLocalNumEntries() != 0){
      sendBuf.resize(Y.getLocalNumEntries());
      Tucker::copy_view_to_stdvec(Y.getLocalTensor().data(), sendBuf);
    }
  }

  std::vector<ScalarType> recvBuf;
  if(recvY.getLocalNumEntries() != 0) {
    recvBuf.resize(recvY.getLocalNumEntries());
    Tucker::copy_view_to_stdvec(recvY.getLocalMatrix(), recvBuf);
  }

  MPI_Alltoallv_(sendBuf.data(), sendCounts, sendDispls,
		 recvBuf.data(), recvCounts, recvDispls, comm);

  // Determine whether the data needs to be unpacked
  bool isUnpackingNecessary = isUnpackForGramNecessary(n, ndims, oldMap, redistMap);
  result_t redistY;
  if(isUnpackingNecessary) {
    redistY = result_t(nrows, (int)ncols, comm, false);
    unpackForGram(n, ndims, redistY, recvBuf, oldMap);
  }
  else {
    redistY = recvY;
  }

  return redistY;
}


template <class ScalarType>
auto reduceForGram(Kokkos::View<ScalarType**, Kokkos::LayoutLeft> U)
{
  int nrows = U.extent(0);
  int ncols = U.extent(1);
  int count = nrows*ncols;
  Kokkos::View<ScalarType**, Kokkos::LayoutLeft> reducedU("redU", nrows, ncols);
  MPI_Allreduce_(U.data(), reducedU.data(), count, MPI_SUM, MPI_COMM_WORLD);
  return reducedU;
}


template <class ScalarType, class ...Properties>
auto newGram(Tensor<ScalarType, Properties...> Y, int n)
{
  using local_gram_t = Kokkos::View<ScalarType**, Kokkos::LayoutLeft>;

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int ndims = Y.getNumDimensions();
  const MPI_Comm& comm = Y.getDistribution().getProcessorGrid().getColComm(n, false);
  int numProcs;
  MPI_Comm_size(comm, &numProcs);

  // If the communicator only has one MPI process, no redistribution is needed
  local_gram_t localGram;
  if(numProcs > 1)
  {
    bool myColEmpty = false;
    for(int i=0; i<ndims; i++) {
      if(i==n) continue;
      if(Y.getLocalSize(i) == 0) {
        myColEmpty = true;
        break;
      }
    }

    if(myColEmpty) {
      const int nGlobalRows = Y.getGlobalSize(n);
      Kokkos::resize(localGram, nGlobalRows, nGlobalRows);
    }
    else {
      // Redistribute the data
      auto redistributedY = redistributeTensorForGram(Y, n);
      if(redistributedY.getLocalNumEntries() > 0) {
        localGram = localRankKForGram(redistributedY, n, ndims);
      }
      else {
        int nGlobalRows = Y.getGlobalSize(n);
	Kokkos::resize(localGram, nGlobalRows, nGlobalRows);
      }
    } // end if(!myColEmpty)
  }
  else{
    throw std::runtime_error("Missing");
  }

  return reduceForGram(localGram);
}


template <class ScalarType, class ...Properties, class TruncatorType>
auto STHOSVD(const Tensor<ScalarType, Properties...> & X,
	     TruncatorType && truncator,
	     const std::vector<int> & modeOrder,
	     bool useOldGram,
	     bool flipSign,
	     bool useLQ,
	     bool useButterflyTSQR)
{
  using tensor_type  = Tensor<ScalarType, Properties...>;
  using memory_space = typename tensor_type::traits::memory_space;

  // preconditions for now
  if (useLQ || useButterflyTSQR || useOldGram){
    throw std::runtime_error("cannot use LQ or TSQR yet");
  }

  // ---------------------
  // prepare
  // ---------------------
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  const int rank = X.rank();

  // Compute the nnz of the largest tensor piece being stored by any process
  size_t max_lcl_nnz_x = 1;
  for(int i=0; i<rank; i++) {
    max_lcl_nnz_x *= X.getDistribution().getMap(i,false)->getMaxNumEntries();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ---------------------
  // core loop
  // ---------------------
  tensor_type Y = X;
  for (std::size_t n=0; n<1 /*X.rank()*/; n++)
  {
    const int mode = modeOrder.empty() ? n : modeOrder[n];

    /*
     * GRAM
     */
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting Gram(" << mode << ")...\n";
    }
    auto S = newGram(Y, mode);
    if (mpiRank == 0){
      Tucker::write_view_to_stream(std::cout, S);
    }

    /*
     * Eigenvaulues
     */
    if(rank == 0) {
      std::cout << "\n\tAutoST-HOSVD::Eigen{vals,vecs}(" << mode << ")...\n";
    }
    auto currEigvals = TuckerOnNode::impl::compute_eigenvalues(S, flipSign);
    // FIXME: append eigvals
    if (mpiRank ==0){
      Tucker::write_view_to_stream(std::cout, currEigvals);
    }

    /*
     * Truncation
     */
    if(rank == 0) {
      std::cout << "\n\tAutoST-HOSVD::Truncating\n";
    }
    const std::size_t numEvecs = truncator(mode, currEigvals);

    using eigvec_rank2_view_t = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, memory_space>;
    eigvec_rank2_view_t currEigVecs("currEigVecs", S.extent(0), numEvecs);
    const int nToCopy = S.extent(0)*numEvecs;
    const int ONE = 1;
    Tucker::copy(&nToCopy, S.data(), &ONE, currEigVecs.data(), &ONE);
    // FIXME: append data
    if (mpiRank ==0){
      Tucker::write_view_to_stream(std::cout, currEigVecs);
    }

    /*
     * TTM
     */
    // Perform the tensor times matrix multiplication
    if(rank == 0) {
      std::cout << "\tAutoST-HOSVD::Starting TTM(" << mode << ")...\n";
    }

  }//end loop

  return int{};
}

}
#endif
