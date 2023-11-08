#ifndef IMPL_TUCKERMPI_NEWGRAM_IMPL_HPP_
#define IMPL_TUCKERMPI_NEWGRAM_IMPL_HPP_

#include "TuckerMpi_Tensor.hpp"
#include "TuckerMpi_ttm.hpp"
#include "TuckerOnNode_compute_gram.hpp"
#include "Tucker_boilerplate_view_io.hpp"
#include "Tucker_print_bytes.hpp"
#include "./TuckerOnNode_TensorGramEigenvalues.hpp"
#include "./Tucker_TuckerTensorSliceHelpers.hpp"
#include "./TuckerMpi_Matrix.hpp"
#include "Tucker_TuckerTensor.hpp"
#include "./Tucker_BlasWrapper.hpp"
#include "./Tucker_ComputeEigValsEigVecs.hpp"
#include <Kokkos_Core.hpp>

namespace TuckerMpi{
namespace impl{

template <class ScalarType, class MemSpace>
auto local_rank_k_for_gram_host(Matrix<ScalarType, MemSpace> Y, int n, int ndims)
{
  using C_view_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemSpace>;

  const int nrows = Y.getLocalNumRows();
  C_view_type C("loc", nrows, nrows);
  auto C_h = Kokkos::create_mirror(C);

  char uplo = 'U';
  int ncols = Y.getLocalNumCols();
  ScalarType alpha = 1;
  auto YlocalMat = Y.getLocalMatrix();
  auto YlocalMat_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), YlocalMat);

  const ScalarType* A = YlocalMat_h.data();
  ScalarType beta = 0;
  ScalarType* Cptr = C_h.data();
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

template <class ScalarType, class MemSpace>
auto local_rank_k_for_gram(Matrix<ScalarType, MemSpace> Y, int n, int ndims)
{
  using C_view_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemSpace>;
  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutLeft, MemSpace,
				Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  const int nrows = Y.getLocalNumRows();
  const int ncols = Y.getLocalNumCols();

  C_view_type C("C_local_rank_k_for_gram", nrows, nrows);
  auto YlocalMat = Y.getLocalMatrix();
  ScalarType alpha = 1;
  ScalarType beta = 0;
  if(n < ndims-1) {
    Tucker::impl::syrk_kokkos("U", "N", alpha, YlocalMat, beta, C);
  }
  else {
    umv_type Aview(YlocalMat.data(), ncols, nrows);
    Tucker::impl::syrk_kokkos("U", "T", alpha, Aview, beta, C);
  }

  return C;
}

inline
bool is_unpack_for_gram_necessary(int n, int ndims,
				  const Map* origMap,
				  const Map* redistMap)
{
  const int nLocalCols = redistMap->getLocalNumEntries();
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

inline
bool is_pack_for_gram_necessary(int n,
				const Map* origMap,
				const Map* redistMap)
{
  const int localNumRows = origMap->getLocalNumEntries();
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

template <class ScalarType, class MemSpace>
void unpack_for_gram(int n, int ndims,
		     Matrix<ScalarType, MemSpace> redistMat,
		     const ScalarType * dataToUnpack,
		     const Map* origMap)
{

  int nLocalCols = redistMat.getLocalNumCols();
  const MPI_Comm& comm = origMap->getComm();
  int nprocs;
  MPI_Comm_size(comm,&nprocs);

  auto matrixView_d = redistMat.getLocalMatrix();
  auto matrixView_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matrixView_d);
  ScalarType* dest = matrixView_h.data();
  const int ONE = 1;

  for(int c=0; c<nLocalCols; c++) {
    for(int b=0; b<nprocs; b++) {
      int nLocalRows=origMap->getNumEntries(b);
      const ScalarType* src = dataToUnpack + origMap->getOffset(b)*nLocalCols + c*nLocalRows;
      Tucker::copy(&nLocalRows, src, &ONE, dest, &ONE);
      dest += nLocalRows;
    }
  }
  Kokkos::deep_copy(matrixView_d, matrixView_h);
}


// Pack the data for redistribution
// Y_n is block-row distributed; we are packing
// so that Y_n will be block-column distributed
template <class ScalarType, class ...Ps>
auto pack_for_gram_fallback_copy_host(Tensor<ScalarType, Ps...> & Y,
				      int n,
				      const Map* redistMap)
{
  const int ONE = 1;
  assert(is_pack_for_gram_necessary(n, Y.getDistribution().getMap(n), redistMap));

  int localNumRows = Y.localExtent(n);
  int globalNumCols = redistMap->getGlobalNumEntries();
  int ndims = Y.rank();
  int nprocs = Y.getDistribution().getProcessorGrid().getNumProcs(n);

  auto localTensorView_d = Y.localTensor().data();
  auto localTensorView_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localTensorView_d);

  // Allocate memory for packed data
  std::vector<ScalarType> sendData(Y.localSize());

  // Local data is row-major
  //after packing the local data should have block column pattern where each block is row major.
  if(n == ndims-1)
  {

    int offset=0;
    for(int b=0; b<nprocs; b++) {
      int n = redistMap->getNumEntries(b);
      const ScalarType* YnData = localTensorView_h.data();
      for(int r=0; r<localNumRows; r++){
        Tucker::copy(&n, YnData + redistMap->getOffset(b)+globalNumCols*r,
		     &ONE, &sendData[0]+offset, &ONE);
        offset += n;
      }
    }
  }
  else{

    auto sz = Y.localDimensionsOnHost();
    std::size_t numLocalBlocks = impl::prod(sz, n+1,ndims-1);
    std::size_t ncolsPerLocalBlock = impl::prod(sz, 0,n-1);
    assert(ncolsPerLocalBlock <= std::numeric_limits<std::size_t>::max());

    const ScalarType* src = localTensorView_h.data();
    // Make local data column major
    for(std::size_t b=0; b<numLocalBlocks; b++) {
      ScalarType* dest = &sendData[0] + b*localNumRows*ncolsPerLocalBlock;
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

template <class ScalarType, class ...Ps>
std::vector<ScalarType> pack_for_gram(Tensor<ScalarType, Ps...> & Y,
				      int n,
				      const Map* redistMap)
{
  assert(is_pack_for_gram_necessary(n, Y.getDistribution().getMap(n), redistMap));

  int localNumRows = Y.localExtent(n);
  int globalNumCols = redistMap->getGlobalNumEntries();
  int ndims = Y.rank();
  int nprocs = Y.getDistribution().getProcessorGrid().getNumProcs(n);

  auto localTensorView_d = Y.localTensor().data();
  auto localTensorView_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localTensorView_d);

  // Allocate memory for packed data
  std::vector<ScalarType> sendData(Y.localSize());

  using senddata_umview_t = Kokkos::View<
    ScalarType*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>
    >;
  senddata_umview_t sendDataView(sendData.data(), sendData.size());

  using offsets_umview_t = Kokkos::View<
    const int*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>
    >;
  auto & offsetsStdVec = redistMap->getOffsets();
  offsets_umview_t offsetsView(offsetsStdVec.data(), offsetsStdVec.size());


  using host_exe = Kokkos::DefaultHostExecutionSpace;
  using policy_t = Kokkos::RangePolicy<host_exe>;

  // Local data is row-major
  //after packing the local data should have block column pattern where each block is row major.
  if(n == ndims-1)
  {
    namespace KE = Kokkos::Experimental;
    // FIXME: this loop should be improved
    auto itFrom = KE::begin(localTensorView_h);
    auto itDest = KE::begin(sendDataView);
    std::size_t offset = 0;
    for(std::size_t b=0; b<(std::size_t)nprocs; b++)
    {
      std::size_t n = redistMap->getNumEntries(b);
      for(int r=0; r<localNumRows; r++)
      {
	policy_t policy(host_exe(), 0, n);
	Kokkos::parallel_for(policy,
			     KOKKOS_LAMBDA(std::size_t i){
			       *(itDest + offset + i) = *(itFrom + offsetsView(b) + globalNumCols*r +i);
			   });
        offset += n;
      }
    }

  }
  else
  {
    namespace KE = Kokkos::Experimental;
    // FIXME: this loop should be improved
    auto sz = Y.localDimensionsOnHost();
    std::size_t numLocalBlocks = impl::prod(sz, n+1,ndims-1);
    std::size_t ncolsPerLocalBlock = impl::prod(sz, 0,n-1);
    assert(ncolsPerLocalBlock <= std::numeric_limits<std::size_t>::max());

    auto itFrom = KE::begin(localTensorView_h);
    for(std::size_t b=0; b<numLocalBlocks; b++)
    {
      auto itDest = KE::begin(sendDataView) + b*localNumRows*ncolsPerLocalBlock;
      for(std::size_t r=0; r<(std::size_t)localNumRows; r++)
      {
	policy_t policy(host_exe(), 0, ncolsPerLocalBlock);
	Kokkos::parallel_for(policy,
			     KOKKOS_LAMBDA(std::size_t i){
			       *(itDest + i*localNumRows) = *(itFrom + i);
			     });

	itFrom += ncolsPerLocalBlock;
	itDest += 1;
      }
    }
  }

  return sendData;
}

template <class ScalarType, class ...Properties>
auto redistribute_tensor_for_gram(Tensor<ScalarType, Properties...> & Y, int n)
{
  using tensor_type       = Tensor<ScalarType, Properties...>;
  using memory_space      = typename tensor_type::traits::memory_space;
  using matrix_result_t = ::TuckerMpi::impl::Matrix<ScalarType, memory_space>;

  // Get the original Tensor map
  const Map* oldMap = Y.getDistribution().getMap(n);
  const MPI_Comm& comm = Y.getDistribution().getProcessorGrid().getColComm(n);

  // Get the number of MPI processes in this communicator
  int numProcs;
  MPI_Comm_size(comm,&numProcs);
  // If the communicator only has one MPI process, no redistribution is needed
  if(numProcs < 2) { return matrix_result_t(); }

  // Get the dimensions of the redistributed matrix Y_n
  const int ndims = Y.rank();
  const auto & sz = Y.localDimensionsOnHost();
  const int nrows = Y.globalExtent(n);
  std::size_t ncols = impl::prod(sz, 0,n-1,1) * impl::prod(sz, n+1,ndims-1,1);
  assert(ncols <= std::numeric_limits<std::size_t>::max());

  // Create a matrix to store the redistributed Y_n
  // Y_n has a block row distribution
  // We want it to have a block column distribution
  matrix_result_t recvY(nrows, (int)ncols, comm, false);

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
#if defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)
    sendBuf = pack_for_gram_fallback_copy_host(Y, n, redistMap);
#else
    sendBuf = pack_for_gram(Y, n, redistMap);
#endif
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
    matrix_result_t redistY(nrows, (int)ncols, comm, false);
    unpack_for_gram(n, ndims, redistY, recvBuf, oldMap);
    return redistY;
  }
  else {
    return recvY;
  }
}

template <class ScalarType, class ...Properties, class ...ViewProps>
void local_gram_after_data_redistribution(Tensor<ScalarType, Properties...> & Y,
					  const int n,
					  Kokkos::View<ScalarType**, Kokkos::LayoutLeft, ViewProps...> & localGram)
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

    if(redistributedY.localSize() > 0)
    {
#if defined(TUCKER_ENABLE_FALLBACK_VIA_HOST)
      localGram = local_rank_k_for_gram_host(redistributedY, n, Y.rank());
#else
      localGram = local_rank_k_for_gram(redistributedY, n, Y.rank());
#endif
    }
    else {
      int nGlobalRows = Y.globalExtent(n);
      Kokkos::resize(localGram, nGlobalRows, nGlobalRows);
    }
  } // end if(!myColEmpty)
}


template <class ScalarType, class ...Properties, class ...ViewProps>
void local_gram_without_data_redistribution(Tensor<ScalarType, Properties...> & Y,
					    const int n,
			    Kokkos::View<ScalarType**, Kokkos::LayoutLeft, ViewProps...> & localGram)
{
  if(Y.getDistribution().ownNothing()) {
    const int nGlobalRows = Y.globalExtent(n);
    Kokkos::resize(localGram, nGlobalRows, nGlobalRows);
  }
  else {
    localGram = ::TuckerOnNode::compute_gram(Y.localTensor(), n);
  }
}

}}
#endif  // IMPL_TUCKERMPI_NEWGRAM_IMPL_HPP_
