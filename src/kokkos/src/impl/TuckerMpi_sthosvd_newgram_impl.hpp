#ifndef TUCKER_KOKKOS_MPI_STHOSVD_NEW_GRAM_IMPL_HPP_
#define TUCKER_KOKKOS_MPI_STHOSVD_NEW_GRAM_IMPL_HPP_

#include "TuckerMpi_Matrix.hpp"
#include "TuckerMpi_Tensor.hpp"
#include "TuckerMpi_ttm.hpp"
#include "TuckerOnNode_sthosvd.hpp"
#include "TuckerMpi_TuckerTensor.hpp"
#include "Tucker_BlasWrapper.hpp"
#include "Tucker_ComputeEigValsEigVecs.hpp"

#include <Kokkos_Core.hpp>

namespace TuckerMpi{
namespace impl{

template <class scalar_t>
auto local_rank_k_for_gram(Matrix<scalar_t> Y, int n, int ndims)
{
  using C_view_type = Kokkos::View<scalar_t**, Kokkos::LayoutLeft>;

  const int nrows = Y.getLocalNumRows();
  C_view_type C("loc", nrows, nrows);
  auto C_h = Kokkos::create_mirror(C);

  char uplo = 'U';
  int ncols = Y.getLocalNumCols();
  scalar_t alpha = 1;
  auto YlocalMat = Y.getLocalMatrix();
  auto YlocalMat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), YlocalMat);

  const scalar_t* A = YlocalMat_h.data();
  scalar_t beta = 0;
  scalar_t* Cptr = C_h.data();
  int ldc = nrows;
  if(n < ndims-1) {
    // Local matrix is column major
    char trans = 'N';
    int lda = nrows;
    Tucker::syrk(&uplo, &trans, &nrows, &ncols, &alpha, A, &lda, &beta, Cptr, &ldc);
  }
  else {
    // Local matrix is row major
    char trans = 'T';
    int lda = ncols;
    Tucker::syrk(&uplo, &trans, &nrows, &ncols, &alpha, A, &lda, &beta, Cptr, &ldc);
  }

  Kokkos::deep_copy(C, C_h);
  return C;
}

bool is_unpack_for_gram_necessary(int n, int ndims, const Map* origMap, const Map* redistMap)
{
  int nLocalCols = redistMap->getLocalNumEntries();
  // If the number of local columns is 1, no unpacking is necessary
  if(nLocalCols <= 1) { return false; }
  // If this is the last dimension, the data is row-major and no unpacking is necessary
  if(n == ndims-1) { return false; }
  return true;
}

bool is_pack_for_gram_necessary(int n, const Map* origMap, const Map* redistMap)
{
  int localNumRows = origMap->getLocalNumEntries();
  // If the local number of rows is 1, no packing is necessary
  if(localNumRows <= 1) { return false; }
  // If n is 0, the local data is already column-major and no packing is necessary
  if(n == 0) { return false; }
  return true;
}


template <class scalar_t>
void unpack_for_gram(int n, int ndims,
		     Matrix<scalar_t> redistMat,
		     const scalar_t * dataToUnpack,
		     const Map* origMap)
{

  int nLocalCols = redistMat.getLocalNumCols();
  const MPI_Comm& comm = origMap->getComm();
  int nprocs;
  MPI_Comm_size(comm,&nprocs);

  auto matrixView_d = redistMat.getLocalMatrix();
  auto matrixView_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixView_d);
  scalar_t* dest = matrixView_h.data();
  const int ONE = 1;
  for(int c=0; c<nLocalCols; c++) {
    for(int b=0; b<nprocs; b++) {
      int nLocalRows=origMap->getNumEntries(b);
      const scalar_t* src = dataToUnpack + origMap->getOffset(b)*nLocalCols + c*nLocalRows;
      Tucker::copy(&nLocalRows, src, &ONE, dest, &ONE);
      dest += nLocalRows;
    }
  }
  Kokkos::deep_copy(matrixView_d, matrixView_h);
}


// Pack the data for redistribution
// Y_n is block-row distributed; we are packing
// so that Y_n will be block-column distributed
template <class scalar_t, class ...Ps>
auto pack_for_gram(Tensor<scalar_t, Ps...> & Y, int n, const Map* redistMap)
{
  const int ONE = 1;
  assert(is_pack_for_gram_necessary(n, Y.getDistribution().getMap(n,true), redistMap));

  // Get the local number of rows of Y_n
  int localNumRows = Y.localExtent(n);
  // Get the number of columns of Y_n
  int globalNumCols = redistMap->getGlobalNumEntries();
  // Get the number of dimensions
  int ndims = Y.rank();
  // Get the number of MPI processes
  int nprocs = Y.getDistribution().getProcessorGrid().getNumProcs(n,true);
  // Allocate memory for packed data
  std::vector<scalar_t> sendData(Y.localSize());

  // Local data is row-major
  //after packing the local data should have block column pattern where each block is row major.
  if(n == ndims-1) {
    auto localTensorView_d = Y.localTensor().data();
    auto localTensorView_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localTensorView_d);
    const scalar_t* YnData = localTensorView_h.data();
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
    auto sz = Y.localDimensionsOnHost();
    // Get information about the local blocks
    size_t numLocalBlocks = impl::prod(sz, n+1,ndims-1);
    size_t ncolsPerLocalBlock = impl::prod(sz, 0,n-1);
    assert(ncolsPerLocalBlock <= std::numeric_limits<int>::max());

    auto localTensorView_d = Y.localTensor().data();
    auto localTensorView_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localTensorView_d);
    const scalar_t* src = localTensorView_h.data();

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
auto redistribute_tensor_for_gram(Tensor<ScalarType, Properties...> & Y, int n)
{
  using result_t = ::TuckerMpi::Matrix<ScalarType>;

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
  const auto & sz = Y.localDimensionsOnHost();
  const int nrows = Y.globalExtent(n);
  size_t ncols = impl::prod(sz, 0,n-1,1) * impl::prod(sz, n+1,ndims-1,1);
  assert(ncols <= std::numeric_limits<int>::max());

  // Create a matrix to store the redistributed Y_n
  // Y_n has a block row distribution
  // We want it to have a block column distribution
  result_t recvY(nrows, (int)ncols, comm, false);

  // Get the column map of the redistributed Y_n
  const Map* redistMap = recvY.getMap();
  int nLocalRows = Y.localExtent(n);
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

  bool isPackingNecessary = is_pack_for_gram_necessary(n, oldMap, redistMap);
  std::vector<ScalarType> sendBuf;
  if(isPackingNecessary) {
    sendBuf = pack_for_gram(Y, n, redistMap);
  }
  else{
    if(Y.localSize() != 0){
      sendBuf.resize(Y.localSize());
      Tucker::impl::copy_view_to_stdvec(Y.localTensor().data(), sendBuf);
    }
  }

  ScalarType * recvBuf;
  auto matrixView_d = recvY.getLocalMatrix();
  auto matrixView_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixView_d);
  if(recvY.localSize() == 0) {
    recvBuf = 0;
  }
  else {
    recvBuf = matrixView_h.data();
  }
  MPI_Alltoallv_(sendBuf.data(), sendCounts, sendDispls,
		 recvBuf, recvCounts, recvDispls, comm);
  Kokkos::deep_copy(matrixView_d, matrixView_h);

  bool isUnpackingNecessary = is_unpack_for_gram_necessary(n, ndims, oldMap, redistMap);
  if(isUnpackingNecessary) {
    result_t redistY(nrows, (int)ncols, comm, false);
    unpack_for_gram(n, ndims, redistY, recvBuf, oldMap);
    return redistY;
  }
  else {
    return recvY;
  }
}


template <class ScalarType>
auto reduce_for_gram(Kokkos::View<ScalarType**, Kokkos::LayoutLeft> U)
{
  int nrows = U.extent(0);
  int ncols = U.extent(1);
  int count = nrows*ncols;

  auto U_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), U);
  Kokkos::View<ScalarType**, Kokkos::LayoutLeft> reducedU("redU", nrows, ncols);
  auto redU_h = Kokkos::create_mirror(reducedU);
  MPI_Allreduce_(U_h.data(), redU_h.data(), count, MPI_SUM, MPI_COMM_WORLD);
  Kokkos::deep_copy(reducedU, redU_h);

  return reducedU;
}


template <class ScalarType, class GramViewType, class ...Properties>
void local_gram_without_data_redistribution(Tensor<ScalarType, Properties...> & Y,
					    int n,
					    GramViewType & localGram)
{
  if(Y.getDistribution().ownNothing()) {
    const int nGlobalRows = Y.globalExtent(n);
    Kokkos::resize(localGram, nGlobalRows, nGlobalRows);
  }
  else {
    localGram = TuckerOnNode::compute_gram(Y.localTensor(), n);
  }
}

template <class ScalarType, class GramViewType, class ...Properties>
void local_gram_after_data_redistribution(Tensor<ScalarType, Properties...> & Y,
					  int n,
					  GramViewType & localGram)
{

  bool myColEmpty = false;
  for(int i=0; i<Y.rank(); i++) {
    if(i==n) continue;
    if(Y.localExtent(i) == 0) {
      myColEmpty = true;
      break;
    }
  }

  if(myColEmpty) {
    const int nGlobalRows = Y.globalExtent(n);
    Kokkos::resize(localGram, nGlobalRows, nGlobalRows);
  }
  else {
    auto redistributedY = redistribute_tensor_for_gram(Y, n);
    if(redistributedY.localSize() > 0) {
      localGram = local_rank_k_for_gram(redistributedY, n, Y.rank());
    }
    else {
      int nGlobalRows = Y.globalExtent(n);
      Kokkos::resize(localGram, nGlobalRows, nGlobalRows);
    }
  } // end if(!myColEmpty)
}

template <class ScalarType, class ...Properties>
auto new_gram(Tensor<ScalarType, Properties...> & Y, int n)
{
  using local_gram_t = Kokkos::View<ScalarType**, Kokkos::LayoutLeft>;
  local_gram_t localGram;

  const MPI_Comm& comm = Y.getDistribution().getProcessorGrid().getColComm(n, false);
  int numProcs;
  MPI_Comm_size(comm, &numProcs);
  if(numProcs > 1)
  {
    local_gram_after_data_redistribution(Y, n, localGram);
  }
  else{
    local_gram_without_data_redistribution(Y, n, localGram);
  }

  return reduce_for_gram(localGram);
}

template <class ScalarType, class ...Properties, class TruncatorType>
[[nodiscard]] auto sthosvd_newgram(Tensor<ScalarType, Properties...> X,
				   TruncatorType && truncator,
				   const std::vector<int> & modeOrder,
				   bool flipSign)
{
  using tensor_type         = Tensor<ScalarType, Properties...>;
  using memory_space        = typename tensor_type::traits::memory_space;
  using tucker_tensor_type  = ::TuckerMpi::TuckerTensor<tensor_type>;
  using slicing_info_view_t = Kokkos::View<::Tucker::impl::PerModeSliceInfo*, Kokkos::HostSpace>;

  // ---------------------
  // prepare
  // ---------------------
  int mpiRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);

  // Compute the nnz of the largest tensor piece being stored by any process
  size_t max_lcl_nnz_x = 1;
  for(int i=0; i<X.rank(); i++) {
    max_lcl_nnz_x *= X.getDistribution().getMap(i,false)->getMaxNumEntries();
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // ---------------------
  // core loop
  // ---------------------
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> eigvals;
  Kokkos::View<ScalarType*, Kokkos::LayoutLeft, memory_space> factors;
  slicing_info_view_t perModeSlicingInfo("pmsi", X.rank());
  tensor_type Y = X;
  for (std::size_t n=0; n<X.rank(); n++)
  {
    const int mode = modeOrder.empty() ? n : modeOrder[n];

    if(mpiRank == 0) {
      std::cout << "\n---------------------------------------------\n";
      std::cout << "--- AutoST-HOSVD::Starting Mode(" << n << ") --- \n";
      std::cout << "---------------------------------------------\n";
    }

    /*
     * GRAM
     */
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Gram(" << mode << ") \n";
    }
    auto S = new_gram(Y, mode);
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    if (mpiRank == 0){
      std::cout << "\n";
      Tucker::write_view_to_stream(std::cout, S);
      std::cout << "\n";
    }
#endif

    /*
     * eigenvalues and eigenvectors
     */
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Eigen{vals,vecs}(" << mode << ")...\n";
    }
    auto currEigvals = Tucker::impl::compute_and_sort_descending_eigvals_and_eigvecs_inplace(S, flipSign);
    TuckerOnNode::impl::appendEigenvaluesAndUpdateSliceInfo(mode, eigvals, currEigvals,
							    perModeSlicingInfo(mode));
    //#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    if (mpiRank == 0){
      std::cout << "\n";
      Tucker::write_view_to_stream(std::cout, currEigvals);
      std::cout << "\n";
    }
    //#endif

    /*
     * Truncation
     */
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Truncating\n";
    }
    const std::size_t numEvecs = truncator(mode, currEigvals);
    auto currEigVecs = Kokkos::subview(S, Kokkos::ALL, std::pair<std::size_t,std::size_t>{0, numEvecs});
    TuckerOnNode::impl::appendFactorsAndUpdateSliceInfo(mode, factors, currEigVecs, perModeSlicingInfo(mode));
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
    if (mpiRank ==0){
      std::cout << "\n";
      Tucker::write_view_to_stream(std::cout, currEigVecs);
      std::cout << "\n";
    }
#endif

    /*
     * TTM
     */
    if(mpiRank == 0) {
      std::cout << "  AutoST-HOSVD::Starting TTM(" << mode << ")...\n";
    }
    tensor_type temp = ::TuckerMpi::ttm(Y, mode, currEigVecs, true, max_lcl_nnz_x);

    // need to do = {} first, otherwise Y=temp throws because Y = temp
    // is assigning tensors with different distributions
    Y = {};
    Y = temp;
    MPI_Barrier(MPI_COMM_WORLD);

    if(mpiRank == 0) {
      const size_t local_nnz = Y.localSize();
      const size_t global_nnz = Y.globalSize();

      std::cout << "Local tensor size after STHOSVD iteration  " << mode << ": ";
      Tucker::write_view_to_stream_inline(std::cout, Y.localDimensionsOnHost());
      std::cout << ", or ";
      Tucker::print_bytes_to_stream(std::cout, local_nnz*sizeof(ScalarType));

      std::cout << "Global tensor size after STHOSVD iteration " << mode << ": ";
      Tucker::write_view_to_stream_inline(std::cout, Y.globalDimensionsOnHost());
      std::cout << ", or ";
      Tucker::print_bytes_to_stream(std::cout, global_nnz*sizeof(ScalarType));
    }

  }//end loop

  return tucker_tensor_type(Y, eigvals, factors, perModeSlicingInfo);
}

}}
#endif
