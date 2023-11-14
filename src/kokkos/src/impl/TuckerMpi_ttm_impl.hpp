#ifndef IMPL_TUCKERMPI_TTM_IMPL_HPP_
#define IMPL_TUCKERMPI_TTM_IMPL_HPP_

#include "TuckerMpi_prod_impl.hpp"
#include "TuckerOnNode_ttm.hpp"
#include "Kokkos_StdAlgorithms.hpp"
#include <cmath>
#include <unistd.h>

namespace TuckerMpi{
namespace impl{

template <class ScalarType, class ...TensorProperties>
void packForTTM(TuckerOnNode::Tensor<ScalarType, TensorProperties...> Y,
		int n,
		const Map* map)
{
  using tensor_type = TuckerOnNode::Tensor<ScalarType, TensorProperties...>;
  using mem_space = typename tensor_type::traits::memory_space;
  namespace KE = Kokkos::Experimental;

  if(Y.size() == 0){ return; }

  // If n is the last dimension, the data is already packed
  // (because the data is stored in row-major order)
  if(n == Y.rank()-1) { return; }

  // Create empty view
  size_t numEntries = Y.size();
  Kokkos::View<ScalarType*, mem_space> tempMem("tempMem", numEntries);

  const MPI_Comm& comm = map->getComm();
  int nprocs;
  MPI_Comm_size(comm,&nprocs);

  // Get the leading dimension of this tensor unfolding
  auto sa = Y.dimensionsOnHost();
  size_t leadingDim = impl::prod(sa, 0,n-1,1);

  // Get the number of global rows of this tensor unfolding
  int nGlobalRows = map->getGlobalNumEntries();

  auto y_data_view = Y.data();

  size_t stride = leadingDim*nGlobalRows;
  size_t tempMemOffset = 0;
  const int inc = 1;
  for(int rank=0; rank<nprocs; rank++)
  {
    int nLocalRows = map->getNumEntries(rank);
    size_t blockSize = leadingDim*nLocalRows;
    int rowOffset = map->getOffset(rank);
    for(size_t tensorOffset = rowOffset*leadingDim;
        tensorOffset < numEntries;
        tensorOffset += stride)
    {
      int tbs = (int)blockSize;

      auto it_first_from = KE::begin(y_data_view)+tensorOffset;
      auto it_first_to = KE::begin(tempMem)+tempMemOffset;
      Kokkos::parallel_for(tbs, KOKKOS_LAMBDA (const int i){
        const int shift = inc*i;
        *(it_first_to + shift) = *(it_first_from + shift);
      });

      tempMemOffset += blockSize;
    }
  }

  // Copy data from temporary memory back to tensor
  KE::copy(typename mem_space::execution_space(), tempMem, y_data_view);
}


template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
void ttm_impl_use_single_reduce_scatter(const int mpiRank,
                                       Tensor<ScalarType, TensorProperties...> & X,
                                       Tensor<ScalarType, TensorProperties...> & Y,
                                       int n,
                                       Kokkos::View<ScalarType**, ViewProperties...> & U,
                                       bool Utransp)
{
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
  if (mpiRank == 0){ std::cout << "MPITTM: use single reduce scatter \n"; }
#endif

  using result_type = Tensor<ScalarType, TensorProperties...>;
  using local_tensor_type = typename result_type::traits::onnode_tensor_type;

  using U_view = Kokkos::View<ScalarType**, ViewProperties...>;
  using U_mem_space = typename U_view::memory_space;
  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutStride,
                                U_mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  auto & Xdist = X.getDistribution();
  const MPI_Comm& comm = Xdist.getProcessorGrid().getColComm(n);
  const Map* xMap = Xdist.getMap(n);
  const Map* yMap = Y.getDistribution().getMap(n);

  int uGlobalRows = Y.globalExtent(n);
  int uGlobalCols = X.globalExtent(n);
  const int stride = Utransp ? uGlobalCols : uGlobalRows;

  ScalarType* Uptr;
  assert(U.size() > 0);
  if(Utransp){
    Uptr = U.data() + xMap->getGlobalIndex(0);
  }
  else{
    Uptr = U.data() + xMap->getGlobalIndex(0)*uGlobalRows;
  }

  auto localX = X.localTensor();
  auto localY = Y.localTensor();
  const int ndims = X.rank();
  local_tensor_type localResult;
  if(X.getDistribution().ownNothing()) {
    std::vector<int> sz(ndims);
    for(int i=0; i<ndims; i++) {
      sz[i] = X.localExtent(i);
    }
    sz[n] = Y.globalExtent(n);
    localResult = local_tensor_type(sz);
  }
  else
  {
    std::vector<int> I(localX.rank());
    for(int i=0; i<(int)I.size(); i++) {
      I[i] = (i != n) ? localX.extent(i) : uGlobalRows;
    }
    localResult = local_tensor_type(I);

    const std::size_t Unrows = (Utransp) ? localX.extent(n) : localResult.extent(n);
    const std::size_t Uncols = (Utransp) ? localResult.extent(n) : localX.extent(n);
    Kokkos::LayoutStride layout(Unrows, 1, Uncols, stride);
    umv_type Aum(Uptr, layout);
    TuckerOnNode::ttm(localX, n, Aum, localResult, Utransp);
  }

  packForTTM(localResult, n, yMap);

  ScalarType *sendBuf = nullptr;
  if (localResult.size() > 0)
    sendBuf = localResult.data().data();

  ScalarType* recvBuf = nullptr;
  if (localY.size() > 0)
    recvBuf = localY.data().data();

  int nprocs;
  MPI_Comm_size(comm, &nprocs);
  int recvCounts[nprocs];
  auto Ylsz = Y.localDimensionsOnHost();
  size_t multiplier = impl::prod(Ylsz,0,n-1,1) * impl::prod(Ylsz, n+1,ndims-1,1);

  for(int i=0; i<nprocs; i++) {
    size_t temp = multiplier*(yMap->getNumEntries(i));
    recvCounts[i] = (int)temp;
  }
  MPI_Reduce_scatter_(sendBuf, recvBuf, recvCounts, MPI_SUM, comm);
}



template <class ScalarType, class ...TensorProperties, class ...ViewProperties>
void ttm_impl_use_series_of_reductions(const int mpiRank,
                                       Tensor<ScalarType, TensorProperties...> & X,
                                       Tensor<ScalarType, TensorProperties...> & Y,
                                       int n,
                                       Kokkos::View<ScalarType**, ViewProperties...> & U,
                                       bool Utransp)
{
#if defined(TUCKER_ENABLE_DEBUG_PRINTS)
  if (mpiRank == 0){ std::cout << "MPITTM: use series of reductions \n"; }
#endif

  using result_type = Tensor<ScalarType, TensorProperties...>;
  using local_tensor_type = typename result_type::traits::onnode_tensor_type;

  using U_view = Kokkos::View<ScalarType**, ViewProperties...>;
  using U_mem_space = typename U_view::memory_space;
  using umv_type = Kokkos::View<ScalarType**, Kokkos::LayoutStride,
                                U_mem_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  auto & Xdist = X.getDistribution();
  const MPI_Comm& comm = Xdist.getProcessorGrid().getColComm(n);
  const Map* xMap = Xdist.getMap(n);

  int uGlobalRows = Y.globalExtent(n);
  int uGlobalCols = X.globalExtent(n);
  const int stride = Utransp ? uGlobalCols : uGlobalRows;

  ScalarType* Uptr;
  assert(U.size() > 0);
  if(Utransp){
    Uptr = U.data() + xMap->getGlobalIndex(0);
  }
  else{
    Uptr = U.data() + xMap->getGlobalIndex(0)*uGlobalRows;
  }

  auto localX = X.localTensor();
  auto localY = Y.localTensor();
  auto localYview_h = Kokkos::create_mirror_view(localY.data());
  const Map* yMap = Y.getDistribution().getMap(n);
  const int Pn = Xdist.getProcessorGrid().getNumProcs(n);
  for(int root=0; root<Pn; root++)
  {
    int uLocalRows = yMap->getNumEntries(root);
    if(uLocalRows == 0) { continue; }

    // Compute the local TTM
    const int ndims = X.rank();
    local_tensor_type localResult;
    if(Xdist.ownNothing()) {
      std::vector<int> sz(ndims);
      for(int i=0; i<ndims; i++) { sz[i] = X.localExtent(i); }
      sz[n] = uLocalRows;
      localResult = local_tensor_type(sz);
    }
    else
	  {
      std::vector<int> I(localX.rank());
      for(int i=0; i<(int)I.size(); i++) {
        I[i] = (i != n) ? localX.extent(i) : uLocalRows;
      }
      localResult = local_tensor_type(I);

      const std::size_t Unrows = (Utransp) ? localX.extent(n) : localResult.extent(n);
      const std::size_t Uncols = (Utransp) ? localResult.extent(n) : localX.extent(n);
      Kokkos::LayoutStride layout(Unrows, 1, Uncols, stride);
      umv_type Aum(Uptr, layout);
      TuckerOnNode::ttm(localX, n, Aum, localResult, Utransp);
    }

    // Combine the local results with a reduce operation
    Kokkos::View<ScalarType*, Kokkos::LayoutRight, Kokkos::HostSpace> sendBuf;
    if(localResult.size() > 0){
      Kokkos::realloc(sendBuf, localResult.size());
      auto localResult_v_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localResult.data());
      namespace KE = Kokkos::Experimental;
      KE::copy(Kokkos::DefaultHostExecutionSpace(), localResult_v_h, sendBuf);
    }
    ScalarType* recvBuf = (localY.size() > 0) ? localYview_h.data() : nullptr;
    size_t count = localResult.size();
    assert(count <= std::numeric_limits<std::size_t>::max());
    if(count > 0) {
      MPI_Reduce_(sendBuf.data(), recvBuf, (int)count, MPI_SUM, root, comm);
    }

    if(Utransp){ Uptr += (uLocalRows*stride);}
    else{ Uptr += uLocalRows; }
  } // end for root

  Kokkos::deep_copy(localY.data(), localYview_h);
}

}} // end namespace TuckerMpi::impl

#endif  // IMPL_TUCKERMPI_TTM_IMPL_HPP_
